
"""
utils for appearance calculations on NuScenes dataset
"""
import pdb
import numpy as np
from geometry.nusc_utils import NuscBox
from typing import Tuple, List
from pyquaternion import Quaternion
from data.script.NUSC_CONSTANT import *

# 可选导入：nuscenes 相关模块
try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.geometry_utils import view_points
    import nuscenes.scripts.export_2d_annotations_as_json as export_2d
    APPEARANCE_AVAILABLE = True
except ImportError:
    APPEARANCE_AVAILABLE = False
    NuScenes = None
    view_points = None
    export_2d = None


def project_dets_to_images(nusc: NuScenes, sample_token: str, dets: np.array = None) -> dict:
    """Project 3D bounding box to pixel plane to get corner points
    TODO: parallelize projection process through matrix operations

    Args:
        nusc (NuScenes): nuscenes database
        sample_token (str): current frame sample token
        dets(np.array, optional): np.array[NuscBox]. Defaults to None.
    """
    if not APPEARANCE_AVAILABLE:
        raise ImportError("nuscenes-devkit is required for appearance features. Please install it: pip install nuscenes-devkit")
    if nusc is None:
        raise ValueError("nusc must be provided for appearance matching")
    if dets is None or len(dets) == 0: return {}

    # fill invalid value with -1
    sample_data = nusc.get('sample', sample_token)['data']
    det_num, cam_num = len(dets), len(USED_ALLCAM)
    corners3d, corners2d = -np.ones(shape=(det_num, cam_num, 8, 2)), -np.ones(shape=(det_num, cam_num, 4))
    is_visible_cam, visible_area = [[] for _ in range(det_num)], [[] for _ in range(det_num)]

    # project 3d bounding box to image plane
    for c_idx, cam_name in enumerate(USED_ALLCAM):
        sd_rec = nusc.get('sample_data', sample_data[cam_name])
        cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        cam_infos = {
            'intrinsic': np.array(cs_rec['camera_intrinsic']),
            'pose_rec': pose_rec,
            'cs_rec': cs_rec
        }
        for d_idx, det in enumerate(dets):
            corners3d[d_idx, c_idx], corners2d[d_idx, c_idx] = project_3d_to_2d(det, cam_infos)
            area = box_2d_area(corners2d[d_idx, c_idx])
            if area > 0:
                is_visible_cam[d_idx].append(c_idx)
                visible_area[d_idx].append(area)

    # get max project camera id, -1 refers to invalid
    max_visible_cam = -np.ones(det_num, dtype=int)
    for d_idx in range(det_num):
        all_visible_area = visible_area[d_idx]
        if len(all_visible_area) != 0:
            max_visible_cam[d_idx] = is_visible_cam[d_idx][np.argmax(all_visible_area)]

    # output appearance attributes
    appear_infos = {
        'np_dets_bbox2d': corners2d,
        'np_dets_bbox3d': corners3d,
        'maxarea_cam_id': max_visible_cam,
        'visible_cam_id': is_visible_cam,
        'visible_area': visible_area,
    }
    return appear_infos


def project_3d_to_2d(det: NuscBox, cam_infos: dict) -> Tuple[np.array, np.array]:
    """project nuscbox from global frame to image plane

    Args:
        det (NuscBox): detetion under NuscBox data format
        cam_infos (dict): {
            'intrinsic': camera intrinsics
            'pose_rec': camera extrinsics, from global frame to ego frame
            'cs_rec': camera extrinsics, from ego frame to camera frame
        }

    Returns:
        Tuple(np.array, np.array): bbox2d image corners, [4(x1, y1, x2, y2)]
                                   bbox3d image corners, [8(corner number), 2(x, y)]
    """
    box = det.copy()

    # global frame -> ego frame
    box.translate(-np.array(cam_infos['pose_rec']['translation']))
    box.rotate(Quaternion(cam_infos['pose_rec']['rotation']).inverse)

    # ego frame -> camera frame
    box.translate(-np.array(cam_infos['cs_rec']['translation']))
    box.rotate(Quaternion(cam_infos['cs_rec']['rotation']).inverse)

    # Filter out points after camera optical center(z < 0)
    corners_3d = box.corners()
    in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
    corners_3d = corners_3d[:, in_front]

    # project 3d corners to image plane
    if not APPEARANCE_AVAILABLE or view_points is None:
        raise ImportError("nuscenes-devkit is required for appearance features. Please install it: pip install nuscenes-devkit")
    corner_coords = view_points(corners_3d, cam_infos['intrinsic'], True).T[:, :2].tolist()

    # get 2d bbox corners
    if export_2d is None:
        raise ImportError("nuscenes-devkit is required for appearance features. Please install it: pip install nuscenes-devkit")
    corner_2d = export_2d.post_process_coords(corner_coords) if len(corner_coords) == 8 else None
    if corner_2d is None:
        corner_2d, corner_coords = [-1 for _ in range(4)], [[-1, -1] for _ in range(8)]

    return np.array(corner_coords, dtype=int), np.array(corner_2d, dtype=int)

def box_2d_area(box) -> int:
    """get bbox2d area in image plane

    Args:
        box (np.array): [4, ], (x0, y0(top-left), x1, y1(bottom-right))

    Returns:
        float: box area
    """
    return (box[2] - box[0]) * (box[3] - box[1]) if box[0] >= 0 else -1
