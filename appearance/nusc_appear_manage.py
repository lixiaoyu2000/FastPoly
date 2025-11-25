"""
Manage tracklet's appearance infos.
Project predict 3d boxes to image plane.

Besides, we don't integrate Kalman Filter to estimate
trajectory appearance state, which sounds unreasonable.
"""
import pdb
import numpy as np
from nuscenes.nuscenes import NuScenes
from data.script.NUSC_CONSTANT import *
from .nusc_utils import project_dets_to_images
from motion_module.nusc_object import FrameObject

class AppearObject:
    def __init__(self, sample_token: str, frame_id: int) -> None:
        """
        dim:
        pred_bbox2d: np.array, [6, 4], invisible camera corners filled with -1
        pred_bbox3d: np.array, [6, 8, 2], invisible camera corners filled with -1
        pred_camids: list, visible camera id
        pred_areas: list, visible bbox area
        pred_maxcam: int, -1 ref to tracklet is invisible
        """
        self.pred_bbox2d, self.pred_bbox3d = None, None
        self.pred_camids, self.pred_maxcam = None, None
        self.pred_areas, self.token, self.frame_id = None, None, None

    def __repr__(self) -> str:
        repr_str = ''
        if self.pred_maxcam == -1:
            return 'Tracklet is invisible at frame {} ({}).'.format(self.frame_id, self.token)
        else:
            # may be visibale in multi-camera
            for idx, cam_id in enumerate(self.pred_camids):
                repr_str += 'Tracklet is visible in {}, reproject bbox2d: {}, area: {}, reproject bbox3d: {}. '.format(USED_ALLCAM[cam_id],
                            self.pred_bbox2d[cam_id], self.pred_areas[idx], self.pred_bbox3d[cam_id])
            return repr_str + 'Max visible camera id: {}. Current Frame is {} ({})'.format(                            USED_ALLCAM[self.pred_maxcam], self.frame_id, self.token)


class AppearManagement:
    def __init__(self, nusc: NuScenes, sample_token: str, frame_id: int, det_infos: dict) -> None:
        self.nusc, self.token, self.frame_id = nusc, sample_token, frame_id
        self.frame_objects = {}
        self.initialize(det_infos)

    def initialize(self, det_infos: dict) -> None:
        """initialize traclet appearance infos, update frame_object
        Args:
            det (dict): detection infos under different data format.
            {
                'nusc_box': NuscBox,
                'np_array': np.array,
                'has_velo': bool, whether the detector has velocity info
            }
        """
        appear_infos = project_dets_to_images(self.nusc, self.token, np.array([det_infos['nusc_box']], dtype=object))
        self.addFrameObject(appear_infos, self.token, self.frame_id)

    def addFrameObject(self, appear_infos: dict, sample_token: str, frame_id: int) -> None:
        """add appear infos to frame objects

        Args:
            appear_infos (dict): project tracklet on multi-camera image planes
            {
                'np_dets_bbox2d': np.array, [1, 6(camera num), 4(x1, y1, x2, y2)], invalid values are filled with -1
                'np_dets_bbox3d': np.array, [1, 6(camera num), 8(corner), 2], invalid values are filled with -1, for visualization
                'maxarea_cam_id': np.array, [det_num], camera which has max reproject area
                'visible_cam_id': list of list, [[cams id for det 1]], cameras which can observe object
                'visible_area': list of list, reproject area
            }
            sample_token (str): current frame sample token
            frame_id (int): current frame id

        """
        app_obj = AppearObject(sample_token, frame_id)
        app_obj.token, app_obj.frame_id = sample_token, frame_id
        app_obj.pred_bbox2d, app_obj.pred_bbox3d = appear_infos['np_dets_bbox2d'][0], appear_infos['np_dets_bbox3d'][0]
        app_obj.pred_camids, app_obj.pred_maxcam = appear_infos['visible_cam_id'][0], appear_infos['maxarea_cam_id'][0]
        app_obj.pred_areas = appear_infos['visible_area'][0]

        # only for debug
        assert len(app_obj.pred_areas) == len(app_obj.pred_camids)

        self.frame_objects[frame_id] = app_obj

    def predict(self, timestamp: int, sample_token: str, pred_obj: FrameObject = None) -> None:
        """project current frame predicted 3d infos to multi-camera image planes

        Args:
            timestamp (int): current frame id
            sample_token (str): current frame sample token
            pred_obj (FrameObject): nusc box/infos predicted by filter. Defaults to None.
        """
        self.token, self.frame_id = sample_token, timestamp

        # project predict nuscbox on image plane
        pred_box = pred_obj.predict_box
        appear_infos = project_dets_to_images(self.nusc, self.token, np.array([pred_box], dtype=object))
        self.addFrameObject(appear_infos, self.token, self.frame_id)

    def __getitem__(self, item) -> AppearObject:
        return self.frame_objects[item]

    def __len__(self) -> int:
        return len(self.frame_objects)
