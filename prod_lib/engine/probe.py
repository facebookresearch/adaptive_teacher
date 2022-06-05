#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.structures import pairwise_iou

class OpenMatchTrainerProbe:
    def __init__(self, cfg):
        self.BOX_AP = 0.5
        self.NUM_CLASSES = cfg.MODEL.ROI_HEADS.NUM_CLASSES
       # self.bbox_stat_list = ['compute_fp_gtoutlier', 'compute_num_box', 'compute_ood_acc']
 
    def bbox_stat(self, unlabel_gt, unlabel_pseudo, name, bbox_stat_list):
        stats = {}
 
        sum_gpu_names = []
        for metric in bbox_stat_list:
            stats_per, sum_gpu_names_per = getattr(
                self, metric)(unlabel_gt, unlabel_pseudo, name)
            stats.update(stats_per)
            sum_gpu_names.extend(sum_gpu_names_per)
    
        return stats, sum_gpu_names
 
    def compute_fp_gtoutlier(self, unlabel_gt, unlabel_pseudo, name):
        num_gt_ood_object = 0
        num_gt_fp_ood_object = 0
        sum_iou = 0.0
        sum_gpu_names = []
        results = {}
    
        if len(unlabel_gt) != 0:
            for gt, pseudo in zip(unlabel_gt, unlabel_pseudo):
                # import pdb; pdb. set_trace() 
                if name == "pred":
                    pp_boxes = pseudo.pred_boxes
                elif name == "pseudo_conf" or name == "pseudo_ood":
                    # filter predicted ood box when evaluating this metric
                    pseudo = pseudo[pseudo.gt_classes != -1]
                    pp_boxes = pseudo.gt_boxes
    
                else:
                    raise ValueError("Unknown name for probe roi bbox.")
    
                if len(gt) != 0 and len(pseudo) != 0:
                    max_iou, max_idx = pairwise_iou(
                        gt.gt_boxes.to('cuda'), pp_boxes).max(1)
                    ood_idx = (gt.gt_classes == -1)
    
                    num_gt_ood_object += ood_idx.sum().item()
                    num_gt_fp_ood_object += (max_iou[ood_idx]
                                                > self.BOX_AP).sum().item()
                    sum_iou += max_iou[ood_idx].sum().item()
    
                elif len(gt) != 0 and len(pseudo) == 0:
                    ood_idx = (gt.gt_classes == -1)
                    num_gt_ood_object += ood_idx.shape[0]
    
            results = {'Analysis_'+name+'/num_gt_ood_object': num_gt_ood_object,
                        'Analysis_'+name+'/num_gt_fp_ood_object': num_gt_fp_ood_object,
                        'Analysis_'+name+'/sum_iou': sum_iou}
    
            sum_gpu_names.extend(list(results.keys()))
    
        return results, sum_gpu_names
 
    def compute_num_box(self, unlabel_gt, unlabel_pseudo, name):
        num_bbox = 0.0
        size_bbox = 0.0
        avg_conf = 0.0
    
        # measure in and out box for openset SS-OD
        num_bbox_in = 0.0
        num_bbox_out = 0.0
        num_bg = 0.0
    
        # when ground-truth is missing in unlabeled data
        if len(unlabel_gt) == 0:
            for pp_roi in unlabel_pseudo:
                if name == "pred":
                    pp_boxes = pp_roi.pred_boxes
                    pp_classes = pp_roi.pred_classes
                    pp_scores = pp_roi.scores
                elif name == "pseudo_conf" or name == "pseudo_ood":
                    pp_boxes = pp_roi.gt_boxes
                    pp_classes = pp_roi.gt_classes
                    pp_scores = pp_roi.scores
                elif name == "gt":
                    pp_boxes = pp_roi.gt_boxes
                    pp_classes = pp_roi.gt_classes
                else:
                    raise ValueError("Unknown name for probe roi bbox.")
                # all boxes (in + out boxes)
                if len(pp_roi) != 0:
                    # bbox number and size
                    num_bbox += len(pp_roi)
                    size_bbox += pp_boxes.area().mean().item()
                    # average box confidence
                    if name != "gt":
                        avg_conf += pp_scores.mean()
                else:
                    num_bbox += 0
                    size_bbox += torch.tensor(0).cuda()
            num_valid_img = len(unlabel_pseudo)
        else:
            # with ground-truth
            num_valid_img = 0
            for gt, pp_roi in zip(unlabel_gt, unlabel_pseudo):
    
                if name == "pred":
                    pp_boxes = pp_roi.pred_boxes
                    pp_classes = pp_roi.pred_classes
                    pp_scores = pp_roi.scores
                elif name == "pseudo_conf" or name == "pseudo_ood":
                    # filter out ood pseudo-box when doing analysis
                    pp_roi = pp_roi[pp_roi.gt_classes != -1]
    
                    pp_boxes = pp_roi.gt_boxes
                    pp_classes = pp_roi.gt_classes
                    pp_scores = pp_roi.scores
                elif name == "gt":
                    pp_boxes = pp_roi.gt_boxes
                    pp_classes = pp_roi.gt_classes
                else:
                    raise ValueError("Unknown name for probe roi bbox.")
    
                # all boxes (in + out boxes)
                if len(pp_roi) != 0:
                    # bbox number and size
                    num_bbox += len(pp_roi)
                    size_bbox += pp_boxes.area().mean().item()
    
                    # average box confidence
                    if name != "gt":
                        avg_conf += pp_scores.mean()
                else:
                    num_bbox += 0
                    size_bbox += torch.tensor(0).cuda()
    
                # in and out class
                if name == "gt":
                    pp_roi_in = pp_roi[pp_classes != -1]
                    num_bbox_in += len(pp_roi_in)
    
                    pp_roi_out = pp_roi[pp_classes == -1]
                    num_bbox_out += len(pp_roi_out)
                    num_valid_img += 1
    
    
                elif name == "pred" or name == "pseudo_conf" or name == "pseudo_ood":
    
                    if len(gt.gt_boxes.to('cuda'))>0 and len(pp_boxes) > 0:
                        max_iou, max_idx = pairwise_iou(gt.gt_boxes.to('cuda'), pp_boxes).max(0)
    
                        # for the ground-truth label for each pseudo-box
                        gtclass4pseudo = gt.gt_classes[max_idx]
                        matchgtbox = max_iou > 0.5
    
                        # compute the number of boxes (background, inlier, outlier)
                        num_bg += (~matchgtbox).sum().item()
                        num_bbox_in += (gtclass4pseudo[matchgtbox]
                                        != -1).sum().item()
                        num_bbox_out += (gtclass4pseudo[matchgtbox]
                                        == -1).sum().item()
                        num_valid_img += 1
    
                else:
                    raise ValueError("Unknown name for probe roi bbox.")
    
        box_probe = {}
        if num_valid_img >0 :
            box_probe["Analysis_" + name + "/Num_bbox"] = num_bbox / \
                num_valid_img
            box_probe["Analysis_" + name + "/Size_bbox"] = size_bbox / \
                num_valid_img
            box_probe["Analysis_" + name +
                    "/Num_bbox_inlier"] = num_bbox_in / num_valid_img
            box_probe["Analysis_" + name +
                    "/Num_bbox_outlier"] = num_bbox_out / num_valid_img
    
            if name != "gt":  # prediciton, background number
                box_probe["Analysis_" + name + "/Conf"] = avg_conf / \
                    num_valid_img
                box_probe["Analysis_" + name +
                        "/Num_bbox_background"] = num_bg / num_valid_img
                box_probe["Analysis_" + name +
                        "/background_fp_ratio"] = num_bg / num_bbox
                box_probe["Analysis_" + name +
                        "/background_tp_ratio"] = num_bbox_in / num_bbox
        else:
            box_probe["Analysis_" + name + "/Num_bbox"] = 0.0
            box_probe["Analysis_" + name + "/Size_bbox"] = 0.0
            box_probe["Analysis_" + name +
                    "/Num_bbox_inlier"] = 0.0
            box_probe["Analysis_" + name +
                    "/Num_bbox_outlier"] = 0.0
    
            if name != "gt":  # prediciton, background number
                box_probe["Analysis_" + name + "/Conf"] = 0.0
                box_probe["Analysis_" + name +
                        "/Num_bbox_background"] = 0.0
                box_probe["Analysis_" + name +
                        "/background_fp_ratio"] = num_bg / num_bbox
                box_probe["Analysis_" + name +
                        "/background_tp_ratio"] = num_bbox_in / num_bbox

        return box_probe, []
    
    def compute_ood_acc(self, unlabel_gt, unlabel_pseudo, name, BOX_IOU=0.5):
        results = {}
        sum_gpu_names = []
    
        if len(unlabel_gt) != 0:
            for metric in ['acc_outlier', 'recall_outlier']:
                for samples in ['_fg', '_all']:
                    for fraction_part in ['_nume', '_deno']:
                        results[metric+samples+fraction_part] = 0.0
    
            for gt, pred in zip(unlabel_gt, unlabel_pseudo):
                if name == "pred":
                    pp_boxes = pred.pred_boxes
                    pp_ood_scores = pred.ood_scores
    
                elif name == "pseudo_conf" or name == "pseudo_ood":
                    # assume these outlier are suppressed
                    pred = pred[pred.gt_classes != -1]
    
                    pp_boxes = pred.gt_boxes
                    pp_ood_scores = pred.ood_scores
    
                else:
                    raise ValueError("Unknown name for probe roi bbox.")
    
                if len(gt) != 0 and len(pred) != 0:
                    # find the most overlapped ground-truth box for each pseudo-box
                    max_iou, max_idx = pairwise_iou(
                        gt.gt_boxes.to('cuda'), pp_boxes).max(0)
    
                    # ignore background instances
                    find_fg_mask = max_iou > BOX_IOU
                    if find_fg_mask.sum() > 0:
                        gt_corres = gt[max_idx].gt_classes.to("cuda")
    
                        gt_outlier = (gt_corres[find_fg_mask] == -1)
                        pred_outlier = pp_ood_scores[find_fg_mask][:, 0] > 0.5
    
                        # accurcay of ood detection (foreground)
                        # acc_outlier_fg = (pred_outlier ==  gt_outlier).sum() /find_fg_mask.sum()
                        results['acc_outlier_fg_nume'] += (
                            pred_outlier == gt_outlier).sum()
                        results['acc_outlier_fg_deno'] += find_fg_mask.sum()
    
                        # recall of ood detection (foreground)
                        # recall_outlier_fg = (pred_outlier[gt_outlier] ==  gt_outlier[gt_outlier]).sum() /gt_outlier.sum()
                        results['recall_outlier_fg_nume'] += (
                            pred_outlier[gt_outlier] == gt_outlier[gt_outlier]).sum()
                        results['recall_outlier_fg_deno'] += gt_outlier.sum()
    
                    # Regard backgound gt as outlier
                    gt_corres = gt[max_idx].gt_classes.to("cuda")
                    # convert all background gt as outlier
                    gt_corres[~find_fg_mask] = -1
                    gt_outlier = gt_corres == -1
    
                    pred_outlier = pp_ood_scores[:, 0] > 0.5
    
                    # accurcay of ood detection (all)
                    # acc_outlier_all = (pred_outlier ==  gt_outlier).sum() /len(pred)
                    results['acc_outlier_all_nume'] += (
                        pred_outlier == gt_outlier).sum()
                    results['acc_outlier_all_deno'] += len(pred)
    
                    # recall of ood detection (all)
                    # recall_outlier_all = (pred_outlier[gt_outlier] ==  gt_outlier[gt_outlier]).sum() /gt_outlier.sum()
                    results['recall_outlier_all_nume'] += (
                        pred_outlier[gt_outlier] == gt_outlier[gt_outlier]).sum()
                    results['recall_outlier_all_deno'] += gt_outlier.sum()
    
            results = {'Analysis_'+name+'/'+k: v for k, v in results.items()}
    
            sum_gpu_names.extend(list(results.keys()))
    
        return results, sum_gpu_names
