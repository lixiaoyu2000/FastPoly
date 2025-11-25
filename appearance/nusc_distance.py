"""
Similarity metric for computing appearance distance between tracklet and det(cost matrix) or det and det(NMS)
Two implemented Appearance Similarity metrics for NuScenes dataset

Parallel metrics: giou2d, iou2d

TODO: support more similarity metrics

Thanks:
- For GIoU part, Paper URL: GIOU2D(https://giou.stanford.edu/)

"""
import numpy as np
from geometry.nusc_utils import mask_between_boxes, logical_or_mask


def iou_2d(bboxs_a: dict, bboxs_b: dict) -> np.array:
    """parallel implementation of iou 2d on image plane
    'np_dets': np.array, [det_num, 14](x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), det_score, class_label)
    'np_dets_bbox2d': np.array, [det_num, 4], (x0, y0(top-left), x1, y1(bottom-right)), reproject bbox 2d

    Args:
        bboxs_a (dict): dict, a collection of NuscBox info, keys must contain 'np_dets' and 'np_dets_bbox2d'
        bboxs_b (dict): dict, a collection of NuscBox info, keys must contain 'np_dets' and 'np_dets_bbox2d'

    Returns:
        np.array: [det_num, tra_num], 2d iou between two bboxes collections
    """
    assert 'np_dets' in bboxs_a and 'np_dets_bbox2d' in bboxs_a, 'must contain specified keys'
    assert 'np_dets' in bboxs_b and 'np_dets_bbox2d' in bboxs_b, 'must contain specified keys'

    # load info
    infos_a, infos_b = bboxs_a['np_dets'], bboxs_b['np_dets']
    corners_a, corners_b = bboxs_a['np_dets_bbox2d'], bboxs_b['np_dets_bbox2d']

    # corner case, 1d array(collection only has one box) to 2d array
    if infos_a.ndim == 1: infos_a = infos_a[None, :]
    if infos_b.ndim == 1: infos_b = infos_b[None, :]
    assert infos_a.shape[1] == 14 and infos_b.shape[1] == 14, "dim must be 14"

    # mask matrix, True denotes different(invalid), False denotes same(valid)
    bool_mask, seq_mask = mask_between_boxes(infos_a[:, -1], infos_b[:, -1])
    bool_mask, seq_mask = logical_or_mask(bool_mask, seq_mask, bboxs_a, bboxs_b)

    # expend dims, [det_num, tra_num, 4(x0, y0, x1, y1)]
    corners_a = corners_a[:, None, :].repeat(len(corners_b), 1)
    corners_b = corners_b[None, :, :].repeat(len(corners_a), 0)

    # calu inter area, [det_num, tra_num]
    min_x = np.minimum(corners_a[:, :, 2], corners_b[:, :, 2])
    max_x = np.maximum(corners_a[:, :, 0], corners_b[:, :, 0])
    min_y = np.minimum(corners_a[:, :, 3], corners_b[:, :, 3])
    max_y = np.maximum(corners_a[:, :, 1], corners_b[:, :, 1])

    inter_x = np.maximum(np.zeros_like(min_x), (min_x - max_x))
    inter_y = np.maximum(np.zeros_like(min_y), (min_y - max_y))
    inter_area = inter_x * inter_y

    # calu union area
    bboxa_area = (corners_a[:, :, 2] - corners_a[:, :, 0]) * (corners_a[:, :, 3] - corners_a[:, :, 1])
    bboxb_area = (corners_b[:, :, 2] - corners_b[:, :, 0]) * (corners_b[:, :, 3] - corners_b[:, :, 1])
    union_area = bboxa_area + bboxb_area - inter_area

    # mask invalid value
    iou2d = inter_area / union_area
    iou2d[bool_mask] = -1

    return iou2d


def giou_2d(bboxs_a: dict, bboxs_b: dict) -> np.array:
    """parallel implementation of giou 2d on image plane
    'np_dets': np.array, [det_num, 14](x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), det_score, class_label)
    'np_dets_bbox2d': np.array, [det_num, 4], (x0, y0(top-left), x1, y1(bottom-right)), reproject bbox 2d

    Args:
        bboxs_a (dict): dict, a collection of NuscBox info, keys must contain 'np_dets' and 'np_dets_bbox2d'
        bboxs_b (dict): dict, a collection of NuscBox info, keys must contain 'np_dets' and 'np_dets_bbox2d'

    Returns:
        np.array: [det_num, tra_num], 2d iou between two bboxes collections
    """
    assert 'np_dets' in bboxs_a and 'np_dets_bbox2d' in bboxs_a, 'must contain specified keys'
    assert 'np_dets' in bboxs_b and 'np_dets_bbox2d' in bboxs_b, 'must contain specified keys'

    # load info
    infos_a, infos_b = bboxs_a['np_dets'], bboxs_b['np_dets']
    corners_a, corners_b = bboxs_a['np_dets_bbox2d'], bboxs_b['np_dets_bbox2d']

    # corner case, 1d array(collection only has one box) to 2d array
    if infos_a.ndim == 1: infos_a = infos_a[None, :]
    if infos_b.ndim == 1: infos_b = infos_b[None, :]
    assert infos_a.shape[1] == 14 and infos_b.shape[1] == 14, "dim must be 14"

    # mask matrix, True denotes different(invalid), False denotes same(valid)
    bool_mask, seq_mask = mask_between_boxes(infos_a[:, -1], infos_b[:, -1])
    bool_mask, seq_mask = logical_or_mask(bool_mask, seq_mask, bboxs_a, bboxs_b)

    # expend dims, [det_num, tra_num, 4(x0, y0, x1, y1)]
    corners_a = corners_a[:, None, :].repeat(len(corners_b), 1)
    corners_b = corners_b[None, :, :].repeat(len(corners_a), 0)

    # calu inter area, [det_num, tra_num]
    min_x = np.minimum(corners_a[:, :, 2], corners_b[:, :, 2])
    max_x = np.maximum(corners_a[:, :, 0], corners_b[:, :, 0])
    min_y = np.minimum(corners_a[:, :, 3], corners_b[:, :, 3])
    max_y = np.maximum(corners_a[:, :, 1], corners_b[:, :, 1])

    inter_x = np.maximum(np.zeros_like(min_x), (min_x - max_x))
    inter_y = np.maximum(np.zeros_like(min_y), (min_y - max_y))
    inter_area = inter_x * inter_y

    # calu union area
    bboxa_area = (corners_a[:, :, 2] - corners_a[:, :, 0]) * (corners_a[:, :, 3] - corners_a[:, :, 1])
    bboxb_area = (corners_b[:, :, 2] - corners_b[:, :, 0]) * (corners_b[:, :, 3] - corners_b[:, :, 1])
    union_area = bboxa_area + bboxb_area - inter_area

    # calu convex area
    union_x = np.maximum(corners_a[:, :, 2], corners_b[:, :, 2]) - np.minimum(corners_a[:, :, 0], corners_b[:, :, 0])
    union_y = np.maximum(corners_a[:, :, 3], corners_b[:, :, 3]) - np.minimum(corners_a[:, :, 1], corners_b[:, :, 1])
    convex_area = union_x * union_y

    # mask invalid value
    giou2d = inter_area / union_area - (convex_area - union_area) / convex_area
    giou2d[bool_mask] = -1

    iou2d = inter_area / union_area
    iou2d[bool_mask] = 0

    return giou2d, iou2d
