"""
matching script
"""
import pdb
import numpy as np
from typing import Tuple
from data.script.NUSC_CONSTANT import *
from .math import expand_dims
from geometry.nusc_utils import mask_between_boxes


def mask_tras_dets(cls_num, det_labels, tra_labels) -> np.array:
    """
    mask invalid cost between tras and dets
    :return: np.array[bool], [cls_num, det_num, tra_num], True denotes valid (det label == tra label == cls idx)
    """
    det_num, tra_num = len(det_labels), len(tra_labels)
    cls_mask = np.ones(shape=(cls_num, det_num, tra_num)) * np.arange(cls_num)[:, None, None]
    # [det_num, tra_num], True denotes invalid(diff cls)
    same_mask, _ = mask_between_boxes(det_labels, tra_labels)
    # [det_num, tra_num], invalid idx assign -1
    tmp_labels = tra_labels[None, :].repeat(det_num, axis=0)
    tmp_labels[np.where(same_mask)] = -1
    return tmp_labels[None, :, :].repeat(cls_num, axis=0) == cls_mask, ~same_mask


def fast_compute_check(metrics: dict, second_metric: str) -> bool:
    """
    Whether cost matrix can be quickly constructed
    :param: dict, similarity metric for each class
    :param: str, similarity metric for second stage
    :return: bool, True -> fast computation
    """
    used_metrics = [m for _, m in metrics.items()] + [second_metric]
    assert len(used_metrics) != 0, 'must have metrics for association'
    giou_3d_fast = (len(set(used_metrics) - set(FAST_METRIC)) == 0)
    giou_norm_fast = (len(set(used_metrics) - set(FAST_NORM_METRIC)) == 0)
    return giou_3d_fast or giou_norm_fast


def reorder_metrics(metrics: dict) -> dict:
    """
    reorder metrics from {key(class, int): value(metrics, str)} to {key(metrics, str): value(class_labels, list)}
    :param metrics: dict, format: {key(class, int): value(metrics, str)}
    :return: dict, {key(metrics, str): value(class_labels, list)}
    """
    new_metrics = {}
    for cls, metric in metrics.items():
        if metric in new_metrics: new_metrics[metric].append(cls)
        else: new_metrics[metric] = [cls]
    return new_metrics


def spec_metric_mask(cls_list: list, det_labels: np.array, tra_labels: np.array) -> np.array:
    """
    mask matrix, merge all object instance index of the specific class
    :param cls_list: list, valid category list
    :param det_labels: np.array, class labels of detections
    :param tra_labels: np.array, class labels of trajectories
    :return: np.array[bool], True denotes invalid(the object's category is not specific)
    """
    det_num, tra_num = len(det_labels), len(tra_labels)
    metric_mask = np.ones((det_num, tra_num), dtype=bool)
    merge_det_idx = [idx for idx, cls in enumerate(det_labels) if cls in cls_list]
    merge_tra_idx = [idx for idx, cls in enumerate(tra_labels) if cls in cls_list]
    metric_mask[np.ix_(merge_det_idx, merge_tra_idx)] = False
    return metric_mask

def voxel_mask(det_infos: np.array, tra_infos: np.array, metric: str = 'eucl', thre: float = 3) -> np.array:
    """
    Voxel cost computation mask
    :param det_infos: np.array, [det_num, 14](x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), det_score, class_label)
    :param tra_infos: np.array, [tra_num, 14](x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), det_score, class_label)
    :param metric: str, metric used to calculate affinity between objects, Support metric : Euclidean, Manhattan
    :param thre: float, threshold used to determine the same voxel
    :return: mask, np.array[bool], True denotes valid, det_num x tra_num
    """
    assert metric in ['eucl', 'manh'], 'Unsupported metric'

    # expand dims
    if det_infos.ndim == 1: det_infos = det_infos[None, :]
    if tra_infos.ndim == 1: tra_infos = tra_infos[None, :]
    assert det_infos.shape[1] == 14 and tra_infos.shape[1] == 14, "dim must be 14"

    # get dets/tras infos
    det_num, tra_num = len(det_infos), len(tra_infos)
    dets_xyz, tras_xyz = expand_dims(det_infos[:, 0:3], tra_num, 1), expand_dims(tra_infos[:, 0:3], det_num, 0)

    if metric == 'eucl': dis_mat = np.linalg.norm(dets_xyz - tras_xyz, ord=2, axis=2)
    elif metric == 'manh': dis_mat = np.linalg.norm(dets_xyz - tras_xyz, ord=1, axis=2)

    return dis_mat < thre
def expand_category_cost(raw_cost: np.array, cls_num: int, det_labels: np.array, tra_labels: np.array) -> Tuple[np.array, np.array]:
    """
    integrate category-specific matching into cost matrix
    :param raw_cost: np.array, raw cost matrix, [det_num, tra_num]
    :param cls_num: int, number of category
    :param det_labels: np.array, class labels of detections
    :param tra_labels: np.array, class labels of trajectories
    :return: np.array, np.array, category-specific cost matrix and category mask, [cls_num, det_num, tra_num]
    """
    cls_cost = raw_cost[None, :, :].repeat(cls_num, axis=0)
    mask_result = mask_tras_dets(cls_num, det_labels, tra_labels)
    # mask_tras_dets returns a tuple (valid_mask, same_mask), we only need the first one
    if isinstance(mask_result, tuple):
        valid_mask = mask_result[0]
    else:
        valid_mask = mask_result
    cls_cost[~valid_mask] = 0
    return cls_cost, valid_mask


def visible_cam_mask(det_visible_cams: list, tra_visible_cams: list) -> np.array:
    """visible mask, cost is valid only when both det and tra are visible 
    at the same camera

    Args:
        det_visible_cams (list): list of list, [[cams id for det 1], [cams id for det 2], ...], cameras which can observe object
        tra_visible_cams (list): list of list, [[cams id for det 1], [cams id for det 2], ...], cameras which can observe object
    
    Notes:
        Args: Empty list refers to object(det/tracklet) is invisible in all camera.
        Returns: Two possible need to masked situations:
        - 1. det and tra are both invisible 
        - 2. det or tra is invisible 
    
    Returns:
        np.array: [cam_id, det_num, tra_num], True denotes valid
    """
    assert isinstance(det_visible_cams, list) and isinstance(tra_visible_cams, list)
    def fancy_idx(cam_ids: list):
        flatten_cam_ids = [item for sublist in cam_ids for item in sublist]
        det_ids = [det_idx for det_idx, cam_num in enumerate(cam_ids) for _ in range(len(cam_num))]
        return flatten_cam_ids, det_ids
    
    cam_num, det_num, tra_num = len(USED_ALLCAM), len(det_visible_cams), len(tra_visible_cams)
    det_vis_mask = np.zeros(shape=(cam_num, det_num, tra_num), dtype=bool)
    tra_vis_mask = np.zeros(shape=(cam_num, det_num, tra_num), dtype=bool)
    
    det_cam_idxs, det_idxs = fancy_idx(det_visible_cams)
    tra_cam_idxs, tra_idxs = fancy_idx(tra_visible_cams)
    det_vis_mask[det_cam_idxs, det_idxs, :] = True
    tra_vis_mask[tra_cam_idxs, :, tra_idxs] = True
    
    return np.logical_and(det_vis_mask, tra_vis_mask)

def concat_affinity(raw_affinity: np.array, aff_mask: np.array, mix_func: str) -> np.array:
    """integrate high dimensional affinity matrix to finial affinity matrix

    Args:
        raw_affinity (np.array): [dim, det_num, tra_num], High dimensional affinity matrix
        aff_mask (np.array): [dim, det_num, tra_num], true denotes affinity is valid
        mix_func (str): concat function, max/sum/avg

    Notes:
        all metrics low limit needed to warp to 0
    
    Returns:
        np.array: finial affinity matrix 
    """
    
    if mix_func == 'sum':
        return np.sum(raw_affinity, axis=0)
    elif mix_func == 'max':
        return np.max(raw_affinity, axis=0)
    elif mix_func == 'avg':
        return np.sum(raw_affinity, axis=0) / (np.sum(aff_mask, axis=0) + 1e-7)
    else:
        raise Exception('unsupported concat function.')