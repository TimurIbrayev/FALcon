# A modification version from chainercv repository.
# (See https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_detection_voc.py)
from __future__ import division

import os
from collections import defaultdict
import numpy as np
from wetectron.structures.bounding_box import BoxList
from wetectron.structures.boxlist_ops import boxlist_iou
import torch

def do_voc_evaluation(sample_collection):
    result = eval_detection_voc(sample_collection,
                                iou_thresh=0.5,
                                use_07_metric=True)
    result_str = "mAP: {:.4f}\n".format(result["map"])
    for i, ap in enumerate(result["ap"]):
        if i == 0:  # skip background
            continue
        result_str += "{:<16}: {:.4f}\n".format(i, ap)
    return result


def eval_detection_voc(sample_collection, iou_thresh=0.5, use_07_metric=False):
    prec, rec = calc_detection_voc_prec_rec(sample_collection = sample_collection, iou_thresh = iou_thresh)
    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)
    return {"ap": ap, "map": np.nanmean(ap)}



def calc_detection_voc_prec_rec(sample_collection, iou_thresh=0.5):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.
    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
   """
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)

    for sample_id, sample_stats in sample_collection.items():
        # collect ALL predictions from sample_stats["predictions"]
        pred_bbox   = []
        pred_label  = []
        pred_score  = []
        for prediction in sample_stats["predictions"]:
            pred_bbox.append(prediction['xyxy_box'])
            pred_label.append(prediction['prediction_label'])
            #pred_score.append(prediction['final_glimpse_objectness'])
            pred_score.append(prediction['prediction_confidence'])
        pred_bbox = torch.stack(pred_bbox, dim=0).clone().detach().cpu().numpy()
        pred_label = np.array(pred_label)
        pred_score = np.array(pred_score)
        # collect ALL ground truths from sample_stats
        gt_bbox = sample_stats['gt_bboxes'].clone().detach().cpu().numpy()
        gt_bbox[:, 2] += gt_bbox[:, 0]
        gt_bbox[:, 3] += gt_bbox[:, 1]
        gt_label = sample_stats['gt_labels'].clone().detach().cpu().numpy()
        gt_difficult = torch.zeros_like(sample_stats['gt_labels']).clone().detach().cpu().numpy()
        if "gt_resized_wh" in sample_stats.keys():
            resized_img_size = sample_stats["gt_resized_wh"]
        else:
            resized_img_size = (256, 256)  # (image_width, image_height)

        # IF, our method has no predictions at any image, we store it as the prediction with:
        # prediction_label = -1, xyxy_box = [0,0,0,0], and final_glimpse_objectness = 0
        # As a result, ground truths are still accounted below, but predictions are ignored!
        unique_labels = np.unique(np.concatenate((pred_label, gt_label)).astype(int))
        if -1 in unique_labels:
            unique_labels = unique_labels[1:]

        for l in unique_labels:
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1

            iou = boxlist_iou(
                BoxList(pred_bbox_l, resized_img_size),
                BoxList(gt_bbox_l, resized_img_size),
            ).numpy()
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]
    return prec, rec


def calc_detection_voc_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.
    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.
    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap
