"""
object's trajectory. A trajectory is a collection of information for each frame(nusc_object.py)
Two core functions: state predict and state update.
The 'state' here is generalized, including attribute info, motion info, geometric info, score info, etc.

In general, Poly-MOT combines count-based and confidence-based strategy to manage trajectory lifecycle. 
Specifically, we use the count-based strategy to initialize and unregister trajectories, while using the 
score-based strategy to penalize mismatched trajectories
"""
import pdb
import numpy as np
from typing import TYPE_CHECKING, Optional
from .nusc_life_manage import LifeManagement
from .nusc_score_manage import ScoreManagement
from .nusc_geometry_manage import GeometryManagement
from motion_module.nusc_object import FrameObject
from motion_module.kalman_filter import LinearKalmanFilter, ExtendKalmanFilter

if TYPE_CHECKING:
    from nuscenes.nuscenes import NuScenes


class Trajectory:
    def __init__(self, 
                 timestamp: int, 
                 config: dict, 
                 track_id: int, 
                 det_infos: dict,
                 nusc: Optional['NuScenes'] = None,
                 sample_token: Optional[str] = None):
        # init basic infos
        self.cfg, self.tracking_id = config, track_id
        self.class_label, self.first_det = det_infos['np_array'][-1], det_infos
        self.init_timestamp, self.timestamp, self.token, self.nusc = timestamp, timestamp, sample_token, nusc
        # manage tracklet's attribute
        self.life_management = LifeManagement(timestamp, config, self.class_label, det_infos)
        # manage tracklet's score, predict/update/punish trackelet
        self.score_management = ScoreManagement(timestamp, config, self.class_label, det_infos)
        # manage for tracklet's motion/geometric infos
        KF_type = self.cfg['motion_model']['filter'][self.class_label]
        assert KF_type in ['LinearKalmanFilter', 'ExtendKalmanFilter'], "must use specific kalman filter"
        self.motion_management = globals()[KF_type](timestamp, sample_token, config, track_id, det_infos)

        # decouple size and z-position (constant) from motion management, reduce computational overhead
        if self.cfg['geometry_model']['use'][self.class_label]:
            self.geometry_management = GeometryManagement(timestamp, det_infos, self.motion_management[timestamp], self.cfg['geometry_model'])
        
        # only for quick iter
        self.last_update_det, self.last_update_stage = det_infos, 1 if not det_infos.get('low_conf', False) else 3
            
    
    def state_predict(self, timestamp: int, sample_token: Optional[str] = None) -> None:
        """
        predict trajectory's state
        :param timestamp: current frame id
        :param sample_token: current frame sample token
        """
        self.timestamp, self.token = timestamp, sample_token
        self.life_management.predict(timestamp)
        self.motion_management.predict(timestamp, sample_token)
        if self.cfg['geometry_model']['use'][self.class_label]:
            self.geometry_management.predict(timestamp, self.motion_management[timestamp])
        self.score_management.predict(timestamp, self.motion_management[timestamp])
    
    def state_update(self, timestamp: int, sample_token: Optional[str] = None, det: dict = None) -> None:
        """
        update trajectory's state
        :param timestamp: current frame id
        :param sample_token: current frame sample token
        :param det: dict, detection infos under different data format
            {
                'nusc_box': NuscBox,
                'np_array': np.array,
                'has_velo': bool, whether the detetor has velocity info
                'match_stage': stage where det is associated
            }
        """
        # only for quick iter
        if det is not None:
            self.last_update_det, self.last_update_stage = det, det.get('match_stage', None)
        
        self.timestamp, self.token = timestamp, sample_token
        self.motion_management.update(timestamp, sample_token, det)
        if self.cfg['geometry_model']['use'][self.class_label]:
            self.geometry_management.update(timestamp, self.motion_management[timestamp], det)
        self.score_management.update(timestamp, self.motion_management[timestamp], det)
        self.life_management.update(timestamp, self.score_management, det)
        
    def __getitem__(self, item) -> dict:
        return {
            'motion': self.motion_management[item],
        }
    
    def __len__(self) -> int:
        return len(self.motion_management)
    
    def __repr__(self) -> str:
        repr_str = 'tracklet status: {}, id: {}, score: {}, 3d state: {}, 2d state: None'
        return repr_str.format(self.life_management, self.tracking_id, 
                               self.score_management[self.timestamp],
                               self.motion_management[self.timestamp])