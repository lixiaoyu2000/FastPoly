"""
data format conversion and data concat on the NuScenes dataset
"""

import pdb
import numpy as np
from typing import Tuple, List
from geometry import NuscBox
from data.script.NUSC_CONSTANT import *


def concat_box_attr(nuscbox: NuscBox, *attrs) -> List:
    res = []
    for attr in attrs:
        tmp_attr = getattr(nuscbox, attr)
        if isinstance(tmp_attr, list):
            res += getattr(nuscbox, attr)
        elif isinstance(tmp_attr, (float, int)):
            res += [tmp_attr]
        elif isinstance(tmp_attr, np.ndarray):
            res += tmp_attr.tolist()
        elif isinstance(tmp_attr, tuple):
            res += list(tmp_attr)
        elif attr == 'detection_name':
            res += [CLASS_SEG_TO_STR_CLASS[tmp_attr]]
        else: raise Exception("unsupport data format to concat")
    return res


def concat_dict_attr(dictbox: dict, fix_posz: dict = None, fix_height: dict = None, *attrs) -> List:
    res = []
    fix_z = fix_posz.get(dictbox['detection_name'], False) if fix_posz else False
    fix_h = fix_height.get(dictbox['detection_name'], False) if fix_height else False
    
    for attr in attrs:
        if attr == 'detection_name':
            res += [CLASS_SEG_TO_STR_CLASS[dictbox[attr]]]
            continue
        elif attr == 'detection_score':
            res += [dictbox[attr]]
            continue
        
        # Apply fix_posz/fix_height if needed
        if fix_z and attr == 'translation':
            dictbox[attr] = dictbox[attr].copy()  # Avoid modifying original
            dictbox[attr][-1] = NUSC_AVG_H[dictbox['detection_name']] / 2
        if fix_h and attr == 'size':
            dictbox[attr] = dictbox[attr].copy()  # Avoid modifying original
            dictbox[attr][-1] = NUSC_AVG_H[dictbox['detection_name']]
        
        res += dictbox[attr]
    return res


def dictdet2array(dets: List[dict], fix_posz: dict = None, fix_height: dict = None, *attrs) -> Tuple[List, np.array]:
    listdets = [concat_dict_attr(det, fix_posz, fix_height, *attrs) for det in dets if det['detection_name'] in CLASS_SEG_TO_STR_CLASS]
    return listdets, np.array(listdets)


def arraydet2box(dets: np.array, ids: np.array = None, init_geo: bool = True, sample_token: str = None):
    # det -> (x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), det_score, class_label)
    if dets.ndim == 1: dets = dets[None, :]
    assert dets.shape[1] == 14, "The number of observed states must satisfy 14"

    NuscBoxes, boxes_bottom_corners, boxes_norm_corners = [], [], []
    for idx, det in enumerate(dets):
        # 如果提供了 sample_token，传递给 NuscBox（用于外观投影）
        box_kwargs = {'init_geo': init_geo}
        if sample_token is not None:
            box_kwargs['token'] = sample_token
            
        curr_box = NuscBox(center=det[0:3], size=det[3:6], rotation=det[8:12],
                            velocity=tuple(det[6:8].tolist() + [0.0]), score=det[12],
                            name=CLASS_STR_TO_SEG_CLASS[int(det[13])], **box_kwargs)
        if ids is not None: curr_box.tracking_id = int(ids[idx])
        NuscBoxes.append(curr_box)
        boxes_bottom_corners.append(curr_box.bottom_corners_)
        boxes_norm_corners.append(curr_box.norm_corners_)
    return np.array(NuscBoxes), np.array(boxes_bottom_corners), np.array(boxes_norm_corners)
