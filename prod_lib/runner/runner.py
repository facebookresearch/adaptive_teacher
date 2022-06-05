#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import logging
import os
from collections import OrderedDict
from functools import lru_cache

import d2go.utils.abnormal_checker as abnormal_checker
import detectron2.utils.comm as comm
from d2go.config import CONFIG_SCALING_METHOD_REGISTRY, temp_defrost
from d2go.data.dataset_mappers import D2GoDatasetMapper, build_dataset_mapper
from d2go.data.transforms.build import build_transform_gen
from d2go.data.utils import maybe_subsample_n_images
from d2go.modeling import build_model, kmeans_anchors, model_ema
from d2go.runner import GeneralizedRCNNRunner
from d2go.utils.flop_calculator import add_print_flops_callback
from d2go.utils.misc import get_tensorboard_log_dir
from d2go.utils.helper import TensorboardXWriter, D2Trainer
from detectron2.checkpoint import PeriodicCheckpointer
from detectron2.engine import hooks
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from torch.nn.parallel import DataParallel, DistributedDataParallel
from detectron2.evaluation import (
    DatasetEvaluators,
)
from detectron2.data import (
    MetadataCatalog,
)
from ..evaluation import (
    COCOEvaluator,
    PascalVOCDetectionEvaluator,
)
from d2go.projects.unbiased_teacher.checkpoint import EnsembleTSModel
from ..config.defaults import add_aut_config
# from ..config.defaults import add_ut_config
# from ..data.build import (
#     build_detection_semisup_train_loader_two_crops,
#     build_uru_detection_semisup_train_loader,
#     inject_uru_dataset,
# )
from d2go.projects.unbiased_teacher.data.build import (
    build_detection_semisup_train_loader_two_crops,
    build_uru_detection_semisup_train_loader,
)
from d2go.projects.unbiased_teacher.runner.runner import UnbiasedTeacherRunner
from d2go.projects.unbiased_teacher.data.dataset_mapper import DatasetMapperTwoCropSeparate  # noqa
from ..data import builtin  # noqa; for registering COCO unlabel dataset
from d2go.projects.unbiased_teacher.engine.trainer import UnbiasedTeacherTrainer
from d2go.projects.unbiased_teacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN  # noqa
from d2go.projects.unbiased_teacher.modeling.proposal_generator.rpn import PseudoLabRPN  # noqa
from d2go.projects.unbiased_teacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab  # noqa
from d2go.projects.unbiased_teacher.solver.build import ut_build_lr_scheduler

#For DA object detection
from ..engine.trainer import DAobjTrainer
from ..modeling.meta_arch.daobj_rcnn import DAobjTwoStagePseudoLabGeneralizedRCNN # noqa
#For VGG model architecture
from ..modeling.meta_arch.vgg import build_vgg_backbone,build_vgg_fpn_backbone  # noqa

ALL_TB_WRITERS = []


@lru_cache()
def _get_tbx_writer(log_dir):
    ret = TensorboardXWriter(log_dir)
    ALL_TB_WRITERS.append(ret)
    return ret

class BaseUnbiasedTeacherRunner(UnbiasedTeacherRunner):
    def get_default_cfg(self):
        cfg = super().get_default_cfg()
        add_aut_config(cfg)

        # add_pointrend_config(cfg)
        # cfg = CN(cfg)  # upgrade from D2's CfgNode to D2Go's CfgNode
        return cfg

    @staticmethod
    def get_evaluator(cfg, dataset_name, output_folder):
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["coco"]:
            # D2 is in the process of reducing the use of cfg.
            dataset_evaluators = COCOEvaluator(
                dataset_name,
                output_dir=output_folder,
                kpt_oks_sigmas=cfg.TEST.KEYPOINT_OKS_SIGMAS,
            )
        elif evaluator_type in ["pascal_voc"]:
            dataset_evaluators = PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type in ["pascal_voc_water"]:
            dataset_evaluators = PascalVOCDetectionEvaluator(dataset_name, target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
        else:
            dataset_evaluators = D2Trainer.build_evaluator(
                cfg, dataset_name, output_folder
            )
        if not isinstance(dataset_evaluators, DatasetEvaluators):
            dataset_evaluators = DatasetEvaluators([dataset_evaluators])
        return dataset_evaluators

# class DAobjUnbiasedTeacherRunner(UnbiasedTeacherRunner):
class DAobjUnbiasedTeacherRunner(BaseUnbiasedTeacherRunner):
    def get_default_cfg(self):
        cfg = super().get_default_cfg()

        # add_aut_config(cfg)
        # add_pointrend_config(cfg)
        # cfg = CN(cfg)  # upgrade from D2's CfgNode to D2Go's CfgNode
        return cfg

    def build_model(self, cfg, eval_only=False):
        """
        Build both Student and Teacher models

        Student: regular model
        Teacher: model that is updated by EMA
        """
        # build_model might modify the cfg, thus clone
        cfg = cfg.clone()

        model = build_model(cfg)
        model_teacher = build_model(cfg)

        if cfg.MODEL.FROZEN_LAYER_REG_EXP:
            raise NotImplementedError()

        if cfg.QUANTIZATION.QAT.ENABLED:
            raise NotImplementedError()

        if eval_only:
            raise NotImplementedError()

        return EnsembleTSModel(model_teacher, model)


    def do_train(self, cfg, model, resume):

        # NOTE: d2go's train_net applies DDP layer by default
        #  we need to strip it away and only put DDP on model_student
        if isinstance(model, (DistributedDataParallel, DataParallel)):
            model = model.module

        model_teacher, model_student = model.model_teacher, model.model_student

        if comm.get_world_size() > 1:
            model_student = DistributedDataParallel(
                model_student,
                device_ids=None
                if cfg.MODEL.DEVICE == "cpu"
                else [comm.get_local_rank()],
                broadcast_buffers=False,
                find_unused_parameters=cfg.MODEL.DDP_FIND_UNUSED_PARAMETERS,
            )

        add_print_flops_callback(cfg, model_student, disable_after_callback=True)

        optimizer = self.build_optimizer(cfg, model_student)
        scheduler = self.build_lr_scheduler(cfg, optimizer)

        checkpointer = self.build_checkpointer(
            cfg,
            model,
            save_dir=cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        checkpoint = checkpointer.resume_or_load(
            cfg.MODEL.WEIGHTS, resume=resume or cfg.UNBIASEDTEACHER.RESUME_FROM_ANOTHER
        )
        start_iter = (
            checkpoint.get("iteration", -1)
            if resume
            and checkpointer.has_checkpoint()
            or cfg.UNBIASEDTEACHER.RESUME_FROM_ANOTHER
            else -1
        )
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration (or iter zero if there's no checkpoint).
        start_iter += 1
        max_iter = cfg.SOLVER.MAX_ITER
        periodic_checkpointer = PeriodicCheckpointer(
            checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
        )

        # if resume from a pre-trained checkpoint, we modify the BURN_IN_STEP
        # so that the weights of the Student will be copied to the Teacher
        # at the 1st iteration when the training started
        if cfg.UNBIASEDTEACHER.RESUME_FROM_ANOTHER:
            cfg.defrost()
            cfg.UNBIASEDTEACHER.BURN_IN_STEP = start_iter
            cfg.freeze()

        data_loader = self.build_detection_train_loader(cfg)

        def _get_model_with_abnormal_checker(model):
            if not cfg.ABNORMAL_CHECKER.ENABLED:
                return model

            tbx_writer = _get_tbx_writer(get_tensorboard_log_dir(cfg.OUTPUT_DIR))
            writers = abnormal_checker.get_writers(cfg, tbx_writer)
            checker = abnormal_checker.AbnormalLossChecker(start_iter, writers)
            ret = abnormal_checker.AbnormalLossCheckerWrapper(model, checker)
            return ret

        trainer = DAobjTrainer(
            cfg,
            _get_model_with_abnormal_checker(model_student),
            _get_model_with_abnormal_checker(model_teacher),
            data_loader,
            optimizer,
        )

        trainer_hooks = [
            hooks.IterationTimer(),
            self._create_after_step_hook(
                cfg, model_student, optimizer, scheduler, periodic_checkpointer
            ),
            hooks.EvalHook(
                cfg.TEST.EVAL_PERIOD,
                lambda: self.do_test(cfg, model, train_iter=trainer.iter),
            ),
            kmeans_anchors.compute_kmeans_anchors_hook(self, cfg),
            self._create_qat_hook(cfg) if cfg.QUANTIZATION.QAT.ENABLED else None,
        ]

        if comm.is_main_process():
            tbx_writer = _get_tbx_writer(get_tensorboard_log_dir(cfg.OUTPUT_DIR))
            writers = [
                CommonMetricPrinter(max_iter),
                JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
                tbx_writer,
            ]
            trainer_hooks.append(
                hooks.PeriodicWriter(writers, period=cfg.WRITER_PERIOD)
            )
        trainer.register_hooks(trainer_hooks)
        trainer.train(start_iter, max_iter)

        trained_cfg = cfg.clone()
        with temp_defrost(trained_cfg):
            trained_cfg.MODEL.WEIGHTS = checkpointer.get_checkpoint_file()
        return {"model_final": trained_cfg}
