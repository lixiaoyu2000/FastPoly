"""
filte tracklet constant geometric info based on the linear kalman filter

The main significance of this management is to reduce computational overhead
"""

import pdb
from typing import Tuple, List

import numpy as np
from data.script.NUSC_CONSTANT import *
from motion_module import FrameObject, CA, CTRA, BICYCLE

class KalmanModel:
    def __init__(self, cfg):
        """
        state vector: [z, w, l, h]
        measure vector: [z, w, l, h]
        """
        self.SD, self.MD, self.idx, self.state = 4, 4, np.arange(2, 6), None
        self.Identity_SD, self.Identity_MD = np.mat(np.identity(self.SD)), np.mat(np.identity(self.MD))

    def getInitState(self, det_infos: dict, class_label: int):
        """
        init geometric infos (z-position, w, l, h)
        :param det_infos: dict
        det_infos (dict): detection infos under different data format.
            {
                'nusc_box': NuscBox,
                'np_array': np.array,
                'has_velo': bool, whether the detetor has velocity info
            }
        :return: state vector
        """
        # state transition
        self.F = self.getTransitionF()
        self.Q = self.getProcessNoiseQ()
        self.P = self.getInitCovP(class_label)

        # state to measurement transition
        self.R = self.getMeaNoiseR()
        self.H = self.getMeaStateH()

        init_state = np.array(det_infos['np_array'][self.idx])
        self.state = np.mat(init_state).T

    def getInitCovP(self, cls_label: int) -> np.mat:
        """
        init geometry infos' errorcov
        :param cls_label: int, the label of category, see detail in the nusc_config.yaml
        :return: geometry infos' errorcov
        """
        vector_p = [4, 4, 4, 4] if cls_label not in [0, 3] else [10, 10, 10, 10]

        return np.mat(np.diag([4, 4, 4, 4]))

    def getTransitionF(self) -> np.mat:
        """
        since each state is constant, the Transition Matrix is actually a identity matrix
        :return: np.mat, identity matrix
        """
        return self.Identity_SD

    def getMeaStateH(self) -> np.mat:
        """
        since the dim of measure vector is equal with the state vector's, H is also a identity matrix
        :return: np.mat, identity matrix
        """
        return self.Identity_MD

    def getProcessNoiseQ(self) -> np.mat:
        """set process noise(fix)
        """
        return np.mat(np.eye(self.SD))

    def getMeaNoiseR(self) -> np.mat:
        """set measure noise(fix)
        """
        return np.mat(np.eye(self.MD))

    def getOutputInfo(self, frame_obj, state, mode):
        """assign geometry infos to the FrameObject inplace
        """
        if mode == 'predict':
            frame_obj.predict_infos[self.idx], frame_box = state, frame_obj.predict_box
            frame_box.wlh, frame_box.center[-1] = np.array(state[1:]), state[0]
            frame_box.reset_box_infos()
            frame_obj.predict_bms, frame_obj.predict_norm_bms = frame_box.bottom_corners_, frame_box.norm_corners_
        elif mode == 'update':
            frame_obj.update_infos[self.idx], frame_box = state, frame_obj.update_box
            frame_box.wlh, frame_box.center[-1] = np.array(state[1:]), state[0]
            frame_box.reset_box_infos()
            frame_obj.update_bms, frame_obj.update_norm_bms = frame_box.bottom_corners_, frame_box.norm_corners_
        else:
            raise Exception('mode must be update or predict')

    def predict(self):
        """predict geometric state
        """
        # "no" state transition, since all states are constant, the identity matrix (self.F) is omitted
        self.P = self.P + self.Q
        return self.state

    def update(self, det):
        """update geometric state with detection
        """
        # update state and errorcov, the identity matrix (self.H) is omitted
        _res = np.mat(det['np_array'][self.idx]).T - self.state
        _S = self.P + self.R
        _KF_GAIN = self.P * _S.I

        self.state += _KF_GAIN * _res
        self.P = (self.Identity_SD - _KF_GAIN) * self.P

        return self.state


class MedianModel(KalmanModel):
    def __init__(self, cfg: dict, cls_label: int):
        super().__init__(cfg)
        self.window_size = cfg['window_size'][cls_label]
        self.history_state, self.state = None, None

    def getInitState(self, det_infos: dict, class_label: int):
        """init geometric infos (z-position, w, l, h)
        """
        self.state = det_infos['np_array'][self.idx].tolist()
        self.history_state = [self.state]

    def predict(self):
        """predict geometric state
        """
        return self.state

    def update(self, det):
        """update geometric state with detection
        """
        det_zwlh = det['np_array'][self.idx].tolist()

        if len(self.history_state) >= self.window_size:
            # self.history_state = np.concatenate((self.history_state[:, 1:], det_zwlh), axis=1)
            self.history_state.pop(0)
            self.history_state.append(det_zwlh)
        else:
            # self.history_state = np.concatenate((self.history_state, det_zwlh), axis=1)
            self.history_state.append(det_zwlh)

        self.state = np.median(self.history_state, axis=0).tolist()
        return self.state


class MeanModel(MedianModel):
    def __init__(self, cfg: dict, cls_label: int):
        super().__init__(cfg, cls_label)

    def getInitState(self, det_infos: dict, class_label: int):
        """init geometric infos (z-position, w, l, h)
        """
        self.state = det_infos['np_array'][self.idx].tolist()
        self.history_state = [self.state]

    def update(self, det):
        """update geometric state with detection
        """
        det_zwlh = det['np_array'][self.idx].tolist()

        if len(self.history_state) >= self.window_size:
            # self.history_state = np.concatenate((self.history_state[:, 1:], det_zwlh), axis=1)
            self.history_state.pop(0)
            self.history_state.append(det_zwlh)
        else:
            # self.history_state = np.concatenate((self.history_state, det_zwlh), axis=1)
            self.history_state.append(det_zwlh)

        self.state = np.mean(self.history_state, axis=0).tolist()

        return self.state


class GeometryManagement:
    """
    integrate a tiny linear kalman filter in GeometryManagement
    state vector: [z, w, l, h]
    measure vector: [z, w, l, h]
    all states are considered constant
    """
    def __init__(self, timestamp: int, det_infos: dict, frame_obj: FrameObject, cfg: dict):
        self.initstamp = self.last_timestamp = self.timestamp = timestamp
        self.const_idx, self.class_label = None, det_infos['np_array'][-1]
        self.model = globals()[cfg['filter'][self.class_label]](cfg, self.class_label)

        # init state and jacobian matrix
        self.initialize(det_infos, frame_obj)

    def initialize(self, det_infos: dict, frame_obj: FrameObject):
        # assign init size and z-position to the motion_management
        self.model.getInitState(det_infos, self.class_label)
        raw_det, self.const_idx = det_infos['np_array'], self.model.idx

        # self.model.getOutputInfo(frame_obj, raw_det[self.const_idx], 'predict')
        self.model.getOutputInfo(frame_obj, raw_det[self.const_idx], 'update')

    def predict(self, timestamp: int, predict_obj: FrameObject) -> None:
        """
        predict state and errorcov
        """
        self.timestamp = timestamp

        # predict geometric state with filter's intern method
        state = self.model.predict()

        # assign tracklet const states
        self.model.getOutputInfo(predict_obj, state, 'predict')

    def update(self, timestamp: int, update_obj: FrameObject, raw_det: dict = None) -> None:
        """Update trajectory geometric info inplace directly using filterd state

        Args:
            timestamp (int): current frame id
            update_obj (FrameObject): nusc box/infos updated by the motion filter
            raw_det (dict, optional): same as data format in the init function. Defaults to None.
        """
        if raw_det is None: return
        self.last_timestamp = timestamp

        # update geometric state with filter's intern method
        state = self.model.update(raw_det)

        # assign tracklet const states
        self.model.getOutputInfo(update_obj, state, 'update')


