"""
Non-Maximum Suppression(NMS) ops for the NuScenes dataset
Three implemented NMS versions(blend_nms, no_blend_nms, blend_soft_nms)
TODO: to support more NMS versions
"""
import pdb

import numpy as np
import numba as nb
from typing import List
from utils.script import voxel_mask
from data.script.NUSC_CONSTANT import *
from geometry import NuscBox, norm_yaw_corners
from geometry.nusc_distance import iou_bev_s, iou_3d_s, giou_bev_s, giou_3d_s, d_eucl_s
from geometry.nusc_distance import iou_bev, iou_3d, giou_bev, giou_3d, d_eucl, a_giou_3d, a_giou_bev, a_iou_bev


def blend_nms(box_infos: dict, metrics: str, thre: float, voxel_mask_size: float, use_voxel_mask: bool = True) -> List[int]:
    """
    :param box_infos: dict, a collection of NuscBox info, keys must contain 'np_dets' and 'np_dets_bottom_corners'
    :param metrics: str, similarity metric for nms, five implemented metrics(iou_bev, iou_3d, giou_bev, giou_3d, d_eluc)
    :param thre: float, threshold of filter
    :param voxel_mask_size: float, the size of voxel mask
    :param use_voxel_mask: bool, whether to employ voxel mask to aviod invalid cost computation
    :return: keep box index, List[int]
    """
    assert metrics in ALL_METRIC, "unsupported NMS metrics"
    assert 'np_dets' in box_infos and 'np_dets_bottom_corners' in box_infos and 'np_dets_norm_corners' in box_infos, \
        'must contain specified keys'

    infos, norm_corners, corners = box_infos['np_dets'], box_infos['np_dets_norm_corners'], box_infos['np_dets_bottom_corners']
    sort_idxs, keep = np.argsort(-infos[:, -2]), []

    while sort_idxs.size > 0:
        i = sort_idxs[0]
        keep.append(i)
        # only one box left
        if sort_idxs.size == 1: break
        left, first = [
            {'np_dets_bottom_corners': corners[idx], 'np_dets_norm_corners': norm_corners[idx], 'np_dets': infos[idx]}
            for idx in [sort_idxs[1:], i]]

        # voxel mask
        if use_voxel_mask:
            nms_voxel_mask = voxel_mask(first['np_dets'], left['np_dets'], thre=voxel_mask_size)[0]
            # left dets all are far away from first det
            if nms_voxel_mask.sum() == 0:
                sort_idxs = sort_idxs[1:]
                continue
            left = {key: val[nms_voxel_mask] for key, val in left.items()}
        else:
            nms_voxel_mask = np.ones_like(sort_idxs[1:], dtype=bool)

        # the return value number varies by distinct metrics
        if metrics not in METRIC:
            distances = globals()[metrics](first, left)[0]
        else:
            distances = globals()[metrics](first, left)[1][0]

        # discard non-maximum dets
        nms_voxel_mask[nms_voxel_mask] = (distances > thre)
        sort_idxs = sort_idxs[1:][~nms_voxel_mask]

    return keep

def scale_nms(box_infos: dict, metrics: list, thres: list, factors: list, voxel_mask_size: dict, use_voxel_mask: bool = True) -> List[int]:
    """intergrate Scale NMS to takeover the size uncertainty under BEV

    Args:
        box_infos (dict): dict, a collection of NuscBox info, keys must contain 'np_dets', 'np_dets_bottom_corners', 'box_dets'
        metrics (dict): dict, similarity metric for nms, five implemented metrics(iou_bev, iou_3d, giou_bev, giou_3d, d_eluc)
        thres (dict): thresholds of filter, category-specfic
        factors (dict): scale factor for each category
        voxel_mask_size (dict): the size of voxel mask
        use_voxel_mask (bool): whether to employ voxel mask to aviod invalid cost computation

    Returns:
        List[int]: keep box index, List[int]
    """
    assert 'np_dets' in box_infos and 'box_dets' in box_infos, 'must contain specified keys'
    assert isinstance(thres, dict) and isinstance(factors, dict) and isinstance(metrics, dict), "Hyperparameters must be a dict in scale nms"

    # copy raw dets infos
    infos, boxes = box_infos['np_dets'].copy(), box_infos['box_dets']

    # step 1. transform raw dets infos by scale factors
    scale_bms = np.stack([box.corners(wlh_factor=factors[box.name_label])[:2, [2, 3, 7, 6]] for box in boxes])
    scale_bms = scale_bms.transpose((0, 2, 1))
    scale_norm_bms = norm_yaw_corners(scale_bms)

    # step 2. scale object size and bottoms by category-specific and NMS
    keep_idxs = []
    for cls_idx in CLASS_STR_TO_SEG_CLASS.keys():
        # obtain category-specific infos and scale
        objs_idxs = np.where(infos[:, -1] == cls_idx)[0]
        infos[objs_idxs, 3:6] *= factors[cls_idx]
        # construct infos for NMS
        nms_infos = {'np_dets': infos[objs_idxs, :],
                     'np_dets_bottom_corners': scale_bms[objs_idxs, :, :],
                     'np_dets_norm_corners': scale_norm_bms[objs_idxs, :]}
        keep_idx = blend_nms(box_infos=nms_infos, metrics=metrics[cls_idx], thre=thres[cls_idx],
                             voxel_mask_size=voxel_mask_size[cls_idx], use_voxel_mask=use_voxel_mask)
        keep_idxs.extend(objs_idxs[keep_idx])
    return keep_idxs
