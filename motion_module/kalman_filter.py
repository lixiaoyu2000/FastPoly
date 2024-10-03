"""
kalman filter for trajectory state(motion state) estimation
Two implemented KF version (LKF, EKF)
Three core functions for each model: state init, state predict and state update
Linear Kalman Filter for CA, CV Model, Extend Kalman Filter for CTRA, CTRV, Bicycle Model
Ref: https://en.wikipedia.org/wiki/Kalman_filter
"""
import pdb
import numpy as np

from .nusc_object import FrameObject
from .motion_model import CA, CTRA, BICYCLE, CV, CTRV
from pre_processing import arraydet2box
import math

class KalmanFilter:
    """kalman filter interface
    """
    def __init__(self, timestamp: int, config: dict, track_id: int, det_infos: dict) -> None:
        # init basic infos, no control input
        self.seq_id = det_infos['seq_id']
        self.initstamp = self.timestamp = timestamp
        self.tracking_id, self.class_label = track_id, det_infos['np_array'][-1]
        self.model = config['motion_model']['model'][self.class_label]
        self.dt = config['basic']['LiDAR_interval']
        self.has_velo, self.has_geofilter = config['basic']['has_velo'], config['geometry_model']['use'][self.class_label]
        # init FrameObject for each frame
        self.state, self.frame_objects = None, {}
    
    def initialize(self, det: dict) -> None:
        """initialize the filter parameters
        Args:
            det (dict): detection infos under different data format.
            {
                'nusc_box': NuscBox,
                'np_array': np.array,
                'has_velo': bool, whether the detetor has velocity info
            }
        """
        pass
    
    def predict(self, timestamp: int) -> None:
        """predict tracklet at each frame
        Args:
            timestamp (int): current frame id
        """
        pass
    
    def update(self, timestamp: int, det: dict = None) -> None:
        """update tracklet motion and geometric state
        Args:
            timestamp (int): current frame id
            det (dict, optional): same as self.init. Defaults to None.
        """
        pass
        
    def addFrameObject(self, timestamp: int, tra_info: dict, mode: str = None) -> None:
        """add predict/update tracklet state to the frameobjects, data 
        format is also implemented in this function.
        frame_objects: {
            frame_id: FrameObject
        }
        Args:
            timestamp (int): current frame id
            tra_info (dict): Trajectory state estimated by Kalman filter, 
            {
                'exter_state': np.array, for output file. 
                               [x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), tra_score, class_label]
                'inner_state': np.array, for state estimation. 
                               varies by motion model
                'cov_mat': np.mat, [2, 2], for score estimation.
            }
            mode (str, optional): stage of adding objects, 'update', 'predict'. Defaults to None.
        """
        # corner case, no tra info
        if mode is None: return
        
        # data format conversion
        inner_info, exter_info = tra_info['inner_state'], tra_info['exter_state']
        extra_info, init_geo = np.array([self.tracking_id, self.seq_id, timestamp]), False if self.has_geofilter else True
        box_info, bm_info, norm_bm_info = arraydet2box(exter_info, np.array([self.tracking_id]), init_geo)

        # update each frame infos 
        if mode == 'update':
            frame_object = self.frame_objects[timestamp]
            frame_object.update_bms, frame_object.update_norm_bms, frame_object.update_box = bm_info[0], norm_bm_info[0], box_info[0]
            frame_object.update_state, frame_object.update_infos = inner_info, np.append(exter_info, extra_info)
            frame_object.update_cov = tra_info['cov_mat']
        elif mode == 'predict':
            frame_object = FrameObject()
            frame_object.predict_bms, frame_object.predict_norm_bms, frame_object.predict_box = bm_info[0], norm_bm_info[0], box_info[0]
            frame_object.predict_state, frame_object.predict_infos = inner_info, np.append(exter_info, extra_info)
            frame_object.predict_cov = tra_info['cov_mat']
            self.frame_objects[timestamp] = frame_object
        else: raise Exception('mode must be update or predict')
    
    def getOutputInfo(self, state: np.mat) -> np.array:
        """convert state vector in the filter to the output format
        Note that, tra score will be process later
        Args:
            state (np.mat): [state dim, 1], predict or update state estimated by the filter

        Returns:
            np.array: [14(fix), 1], predict or update state under output file format
            output format: [x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), tra_score, class_label]
        """
        
        # return state vector except tra score and tra class
        inner_state = self.model.getOutputInfo(state)
        assert inner_state.shape[0] == 12, "The number of output states must satisfy 12"

        return np.append(inner_state, np.array([-1, self.class_label]))
    
    def __getitem__(self, item) -> FrameObject:
        return self.frame_objects[item]

    def __len__(self) -> int:
        return len(self.frame_objects)


class LinearKalmanFilter(KalmanFilter):
    """Linear Kalman Filter for linear motion model, such as CV and CA
    """
    def __init__(self, timestamp: int, config: dict, track_id: int, det_infos: dict) -> None:
        # init basic infos
        super(LinearKalmanFilter, self).__init__(timestamp, config, track_id, det_infos)
        # set motion model, default Constant Acceleration(CA) for LKF
        self.model = globals()[self.model](self.has_velo, self.has_geofilter, self.dt) if self.model in ['CV', 'CA'] \
                     else globals()['CA'](self.has_velo, self.has_geofilter, self.dt)
        # Transition and Observation Matrices are fixed in the LKF
        self.initialize(det_infos)
        
    def initialize(self, det_infos: dict) -> None:
        # state transition
        self.F = self.model.getTransitionF()
        self.Q = self.model.getProcessNoiseQ(self.class_label)
        self.SD = self.model.getStateDim()
        self.P = self.model.getInitCovP(self.class_label)
        
        # state to measurement transition
        self.R = self.model.getMeaNoiseR(self.class_label)
        self.H = self.model.getMeaStateH()

        # get different data format tracklet's state
        self.state = self.model.getInitState(det_infos)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': det_infos['np_array'],
            'cov_mat': self.P[:2, :2],
        }
        self.addFrameObject(self.timestamp, tra_infos, 'predict')
        self.addFrameObject(self.timestamp, tra_infos, 'update')
    
    def predict(self, timestamp: int) -> None:
        # predict state and errorcov
        self.state = self.F * self.state
        self.P = self.F * self.P * self.F.T + self.Q

        # convert the state in filter to the output format
        self.model.warpStateYawToPi(self.state)
        output_info = self.getOutputInfo(self.state)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': output_info,
            'cov_mat': self.P[:2, :2],
        }
        self.addFrameObject(timestamp, tra_infos, 'predict')
        
    def update(self, timestamp: int, det: dict = None) -> None:
        # corner case, no det for updating
        if det is None: return
        
        # update state and errorcov
        meas_info = self.model.getMeasureInfo(det)
        _res = meas_info - self.H * self.state
        self.model.warpResYawToPi(_res)
        _S = self.H * self.P * self.H.T + self.R
        _KF_GAIN = self.P * self.H.T * _S.I
        
        self.state += _KF_GAIN * _res
        self.P = (np.mat(np.identity(self.SD)) - _KF_GAIN * self.H) * self.P

        # output updated state to the result file
        self.model.warpStateYawToPi(self.state)
        output_info = self.getOutputInfo(self.state)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': output_info,
            'cov_mat': self.P[:2, :2],
        }
        self.addFrameObject(timestamp, tra_infos, 'update')
        
        
class ExtendKalmanFilter(KalmanFilter):
    def __init__(self, timestamp: int, config: dict, track_id: int, det_infos: dict) -> None:
        super().__init__(timestamp, config, track_id, det_infos)
        # set motion model, default Constant Acceleration and Turn Rate(CTRA) for EKF
        self.model = globals()[self.model](self.has_velo, self.has_geofilter, self.dt) if self.model in ['CTRV', 'CTRA', 'BICYCLE'] \
                     else globals()['CTRA'](self.has_velo, self.has_geofilter, self.dt)
        # Transition and Observation Matrices are changing in the EKF
        self.initialize(det_infos)
    
    def initialize(self, det_infos: dict) -> None:
        # init errorcov categoty-specific
        self.SD, self.MD = self.model.getStateDim(), self.model.getMeasureDim()
        self.Identity_MD, self.Identity_SD = np.mat(np.identity(self.MD)), np.mat(np.identity(self.SD))
        self.P = self.model.getInitCovP(self.class_label)
        
        # set noise matrix(fixed)
        self.Q = self.model.getProcessNoiseQ(self.class_label)
        self.R = self.model.getMeaNoiseR(self.class_label)

        # get different data format tracklet's state
        self.state = self.model.getInitState(det_infos)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': det_infos['np_array'],
            'cov_mat': self.P[:2, :2],
        }
        self.addFrameObject(self.timestamp, tra_infos, 'predict')
        self.addFrameObject(self.timestamp, tra_infos, 'update')
        
    def predict(self, timestamp: int) -> None:
        # get jacobian matrix F using the final estimated state of the previous frame
        self.F = self.model.getTransitionF(self.state)
        
        # state and errorcov transition
        self.state = self.model.stateTransition(self.state)

        self.P = self.F * self.P * self.F.T + self.Q
        # convert the state in filter to the output format
        self.model.warpStateYawToPi(self.state)
        output_info = self.getOutputInfo(self.state)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': output_info,
            'cov_mat': self.P[:2, :2],
        }
        self.addFrameObject(timestamp, tra_infos, 'predict')
    
    def update(self, timestamp: int, det: dict = None) -> None:
        # corner case, no det for updating
        if det is None: return
        
        # get measure infos for updating, and project state into meausre space
        meas_info = self.model.getMeasureInfo(det)
        state_info = self.model.StateToMeasure(self.state)
        
        # get state residual, and warp angle diff inplace
        _res = meas_info - state_info
        self.model.warpResYawToPi(_res)
        
        # get jacobian matrix H using the predict state
        self.H = self.model.getMeaStateH(self.state)
        
        # obtain KF gain and update state and errorcov
        _S = self.H * self.P * self.H.T + self.R
        # try:
        #     # obtain the cholesky factor U (upper matrix)
        #     _U = scipy.linalg.cholesky(a=_S, 
        #                                check_finite=False)     
            
        #     # speed up the inverse process (a * x = b -> U * U-1 = I)
        #     _U_INV = np.mat(scipy.linalg.solve_triangular(a=_U, 
        #                                                   b=self.Identity_MD, 
        #                                                   check_finite=False))
                    
        #     _KF_GAIN = self.P * self.H.T * _U_INV * _U_INV.T
        # except:
        #     pdb.set_trace()
        _KF_GAIN = self.P * self.H.T * _S.I
        _I_KH = self.Identity_SD - _KF_GAIN * self.H
        
        self.state += _KF_GAIN * _res
        # self.P = _I_KH * self.P * _I_KH.T + _KF_GAIN * self.R * _KF_GAIN.T
        self.P = _I_KH * self.P
        
        # output updated state to the result file
        self.model.warpStateYawToPi(self.state)
        output_info = self.getOutputInfo(self.state)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': output_info,
            'cov_mat': self.P[:2, :2],
        }
        self.addFrameObject(timestamp, tra_infos, 'update')
        
        
        
        
        
        
        
            
        
        
        
        
        
        
        
    
    
        
