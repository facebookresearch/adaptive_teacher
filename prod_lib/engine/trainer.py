#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import time
from collections import OrderedDict
from typing import Dict

import detectron2.utils.comm as comm
import numpy as np
import torch
from detectron2.engine import SimpleTrainer
from detectron2.structures import BitMasks, Boxes, Instances, Keypoints
from detectron2.utils.events import get_event_storage
from d2go.projects.unbiased_teacher.engine.trainer import UnbiasedTeacherTrainer
from d2go.projects.unbiased_teacher.utils.probe import probe

import copy

logger = logging.getLogger(__name__)

class DAobjTrainer(UnbiasedTeacherTrainer):
    """
    A trainer for Teacher-Student mutual learning following this paper:
    "Unbiased Teacher for Semi-Supervised Object Detection"

    It assumes that every step, you:

    For Teacher:
    1. Perform a forward pass on a weakly augmented unlabeled data from the data_loader.
    2. Generate pseudo-labels on the weakly augmented unlabeled data

    For Student:
    1. Perform a forward pass on a strongly augmented unlabeled data from the data_loader.
    2. Perform a forward pass on a labeled data from the data_loader.
    1. Use pseudo-labels generated from the Teacher as target and compute the
       loss on a strongly augmented unlabeled data
    2. Compute the gradients with the above losses on labeled and unlabeled data.
    3. Update the Student model with the optimizer.
    4. EMA update the Teacher model
    """

    # def __init__(self, cfg, model, model_teacher, data_loader, optimizer):
    #     """
    #     Args:
    #         model: a torch Module. Takes a data from data_loader and returns a
    #             dict of losses.
    #         data_loader: an iterable. Contains data to be used to call model.
    #         optimizer: a torch optimizer.
    #     """
    #     super().__init__(model, data_loader, optimizer)

    #     self.cfg = cfg
    #     self.model_teacher = model_teacher

    def run_step(self):
        assert (
            self.model.training
        ), "Student model was changed to eval mode during training"
        start = time.perf_counter()

        data = next(self._data_loader_iter)
        # q (queue): strongly augmented, k (key): weakly augmented
        #TODO Need to further use the weak samples for domain adaptation
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
        data_time = time.perf_counter() - start

        if (
            self.cfg.UNBIASEDTEACHER.BURN_IN_STEP != 0
            and self.iter < self.cfg.UNBIASEDTEACHER.BURN_IN_STEP
        ):
            # Burn-In stage. Supervisedly train the Student model.
            losses, loss_dict, record_dict = self.burn_in(label_data_q, label_data_k)
        else:
            # Copy the Student model to the Teacher (using keep_rate = 0)
            if self.iter == self.cfg.UNBIASEDTEACHER.BURN_IN_STEP:
                logger.info("Copying Student weights to the Teacher .....")
                self._update_teacher_model(keep_rate=0.0)
            elif (
                self.iter - self.cfg.UNBIASEDTEACHER.BURN_IN_STEP
            ) % self.cfg.UNBIASEDTEACHER.TEACHER_UPDATE_ITER == 0:
                self._update_teacher_model(
                    keep_rate=self.cfg.UNBIASEDTEACHER.EMA.KEEP_RATE
                )

            # Teacher-Student Mutual Learning
            losses, loss_dict, record_dict = self.teacher_student_learning(
                label_data_q, label_data_k, unlabel_data_q, unlabel_data_k
            )

        self.optimizer.zero_grad()
        losses.backward()

        self._write_metrics(record_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

    def burn_in(self, label_data_q, label_data_k):
        """
        Perform Burn-In stage with labeled data
        """
        # combine label_data_q + label_data_k
        label_data_q.extend(label_data_k)

        record_dict, _, _, _ = self.model(label_data_q, branch="supervised")

        # weight losses
        loss_dict = self.weight_losses(record_dict)
        losses = sum(loss_dict.values())
        return losses, loss_dict, record_dict

    def teacher_student_learning(
        self, label_data_q, label_data_k, unlabel_data_q, unlabel_data_k
    ):
        """
        Perform Teacher-Student Mutual Learning with labeled and unlabeled data
        """
        # q (queue): strongly augmented, k (key): weakly augmented
        record_dict = {}

        ######################## For probe #################################
        # import pdb; pdb. set_trace() 
        gt_unlabel_k = self.get_label(unlabel_data_k)

        # 0. remove potential ground-truth labels in the unlabeled data
        unlabel_data_q = self.remove_label(unlabel_data_q)
        unlabel_data_k = self.remove_label(unlabel_data_k)

        # 1. generate the pseudo-label using teacher model
        # TODO: why is the Teacher not in .eval() mode?
        with torch.no_grad():
            (
                _,
                proposals_rpn_unsup_k,
                proposals_roih_unsup_k,
                _,
            ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")

        

        ######################## For probe #################################
        # import pdb; pdb. set_trace()
        # analysis_pred, _ = self.probe.compute_num_box(gt_unlabel_k,proposals_roih_unsup_k,'pred')
        # record_dict.update(analysis_pred)
        # 2. Pseudo-labeling
        # Pseudo-labeling for RPN head (bbox location/objectness)
        joint_proposal_dict = {}

        ## No need this
        joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k

        (
            pesudo_proposals_rpn_unsup_k,
            nun_pseudo_bbox_rpn,
        ) = self.process_pseudo_label(
            proposals_rpn_unsup_k,
            self.cfg.UNBIASEDTEACHER.BBOX_THRESHOLD,
            self.cfg.UNBIASEDTEACHER.MASK_THRESHOLD,
            self.cfg.UNBIASEDTEACHER.KEYPOINT_THRESHOLD,
            "rpn",
            "thresholding",
        )
        joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k
        ## No need this end


        # Pseudo-labeling for ROI head (bbox location/objectness)
        pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
            proposals_roih_unsup_k,
            self.cfg.UNBIASEDTEACHER.BBOX_THRESHOLD,
            self.cfg.UNBIASEDTEACHER.MASK_THRESHOLD,
            self.cfg.UNBIASEDTEACHER.KEYPOINT_THRESHOLD,
            "roih",
            "thresholding",
        )
        joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k

        ######################## For probe #################################
        analysis_pred, _ = self.probe.compute_num_box(gt_unlabel_k,pesudo_proposals_roih_unsup_k,'pred')
        record_dict.update(analysis_pred)

        # Probe for analysis (usually for research development)
        if self.cfg.UNBIASEDTEACHER.PROBE:
            record_dict = probe(
                self.cfg,
                proposals_roih_unsup_k,
                unlabel_data_k,
                pesudo_proposals_roih_unsup_k,
                record_dict,
            )

        # 3. add pseudo-label to unlabeled data
        unlabel_data_q = self.add_label(
            unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
        )
        unlabel_data_k = self.add_label(
            unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
        )

        # all_label_data = label_data_q + label_data_k
        if self.cfg.UNBIASEDTEACHER.ISAUG == "No":
            all_label_data = label_data_k
            all_unlabel_data = unlabel_data_k
        else:
            all_label_data = label_data_q + label_data_k
            all_unlabel_data = unlabel_data_q

        # 4. input both strongly and weakly augmented labeled data into student model
        # all_unlabel_data = unlabel_data_q
        record_all_label_data, _, _, _ = self.model(all_label_data, branch="supervised")
        record_dict.update(record_all_label_data)

        # 5. input strongly augmented unlabeled data into model
        record_all_unlabel_data, _, _, _ = self.model(
            all_unlabel_data, branch="supervised-pseudo"
        )

        # rename unsupervised loss
        # NOTE: names of the recorded output from model are hard-coded
        #  we rename them accordingly for unlabeled data
        new_record_all_unlabel_data = {}
        for key in record_all_unlabel_data.keys():
            new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[key]
        record_dict.update(new_record_all_unlabel_data)

        # 6. input weakly labeled data (source) and weakly unlabeled data (target) to student model
        # give sign to the target data

        for i_index in range(len(unlabel_data_k)):
            # unlabel_data_item = {}
            for k, v in unlabel_data_k[i_index].items():
                # label_data_k[i_index][k + "_unlabeled"] = v
                label_data_k[i_index][k + "_unlabeled"] = v
            # unlabel_data_k[i_index] = unlabel_data_item

        all_domain_data = label_data_k
        # all_domain_data = label_data_k + unlabel_data_k
        record_all_domain_data, _, _, _ = self.model(all_domain_data, branch="domain")
        record_dict.update(record_all_domain_data)

        # 7. distill teacher
        # for distill back to teacher
        with torch.no_grad():
            (
                _,
                proposals_rpn_unsup_dis,
                proposals_roih_unsup_dis,
                _,
            ) = self.model(unlabel_data_k, branch="unsup_data_weak")


        pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
            proposals_roih_unsup_dis,
            self.cfg.UNBIASEDTEACHER.BBOX_THRESHOLD,
            self.cfg.UNBIASEDTEACHER.MASK_THRESHOLD,
            self.cfg.UNBIASEDTEACHER.KEYPOINT_THRESHOLD,
            "roih",
            "thresholding",
        )
        unlabel_data_k = self.remove_label(unlabel_data_k)
        unlabel_data_k = self.add_label(
            unlabel_data_k, pesudo_proposals_roih_unsup_k
        )
        record_distill_data, _, _, _ = self.model_teacher(
            unlabel_data_k, branch="supervised-pseudo"
        )
        new_record_all_distill_data = {}
        for key in record_distill_data.keys():
            new_record_all_distill_data[key + "_distill"] = record_distill_data[key]
        record_dict.update(new_record_all_distill_data)



        # weighting losses
        loss_dict = self.weight_losses(record_dict)

        #Add discriminator loss here
        #loss_dict.update(...)
        losses = sum(loss_dict.values())
        return losses, loss_dict, record_dict

    def weight_losses(self, record_dict):
        loss_dict = {}
        REGRESSION_LOSS_WEIGHT = 0
        for key in record_dict.keys():
            if key.startswith("loss"):
                if key == "loss_rpn_cls_pseudo":
                    loss_dict[key] = (
                        record_dict[key]
                        * self.cfg.UNBIASEDTEACHER.UNSUP_LOSS_WEIGHT_RPN_CLS
                    )
                elif (
                    key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo"
                ):  # set pseudo bbox regression to 0
                    loss_dict[key] = record_dict[key] * REGRESSION_LOSS_WEIGHT
                elif (
                    key == "loss_rpn_loc_distill" or key == "loss_box_reg_distill"
                ):  # set pseudo bbox regression to 0
                    loss_dict[key] = record_dict[key] * REGRESSION_LOSS_WEIGHT
                elif key.endswith("mask_pseudo"):  # unsupervised loss for segmentation
                    loss_dict[key] = (
                        record_dict[key]
                        * self.cfg.UNBIASEDTEACHER.UNSUP_LOSS_WEIGHT_MASK
                    )
                elif key.endswith("keypoint_pseudo"):  # unsupervised loss for keypoint
                    loss_dict[key] = (
                        record_dict[key]
                        * self.cfg.UNBIASEDTEACHER.UNSUP_LOSS_WEIGHT_KEYPOINT
                    )
                elif key.endswith("pseudo"):  # unsupervised loss
                    loss_dict[key] = (
                        record_dict[key] * self.cfg.UNBIASEDTEACHER.UNSUP_LOSS_WEIGHT
                    )
                elif (
                    key == "loss_D_img_s" or key == "loss_D_img_t"
                ):  # set weight for discriminator
                    # import pdb
                    # pdb.set_trace()
                    loss_dict[key] = record_dict[key] * self.cfg.UNBIASEDTEACHER.DIS_LOSS_WEIGHT #Need to modify defaults and yaml
                else:  # supervised loss
                    loss_dict[key] = record_dict[key] * 1
        return loss_dict

    def threshold_bbox(
        self,
        proposal_bbox_inst,
        thres=0.7,
        mask_thres=0.5,
        keypoint_thres=0.5,
        proposal_type="roih",
    ):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.pred_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.pred_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.pred_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

            if self.cfg.MODEL.MASK_ON and new_boxes:
                # put predicted output into gt_masks with thresholding
                new_masks = proposal_bbox_inst.pred_masks[valid_map].squeeze(1)
                new_masks = new_masks >= mask_thres
                new_proposal_inst.gt_masks = BitMasks(new_masks)
            if self.cfg.MODEL.KEYPOINT_ON and new_boxes:
                # we use the keypoint score as the basis for thresholding
                new_keypoints = proposal_bbox_inst.pred_keypoints[valid_map, :]
                invalid_keypoints = new_keypoints[:, :, 2] < keypoint_thres
                # (x, y, visibility): visibility flag = 0 -> not labeled (in which case x=y=0)
                new_keypoints[invalid_keypoints] = torch.FloatTensor([0, 0, 0]).to(
                    new_keypoints.device
                )
                new_proposal_inst.gt_keypoints = Keypoints(new_keypoints)

        return new_proposal_inst
    
