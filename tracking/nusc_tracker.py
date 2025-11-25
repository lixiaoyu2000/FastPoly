"""
Tracker, Core of Poly-MOT.
Tracklet prediction and punishment, cost matrix construction, tracking id assignment, tracklet update and init, and output file

TODO: delete debug log in the release version
"""

import pdb
import time, json
from typing import TYPE_CHECKING, Optional

import numpy as np
from pre_processing import blend_nms, scale_nms
from .nusc_trajectory import Trajectory
from data.script.NUSC_CONSTANT import *
from utils.matching import Hungarian
from geometry.nusc_distance import iou_bev, iou_3d, giou_bev, giou_3d, d_eucl, giou_3d_s, a_giou_3d, a_giou_bev
from utils.script import mask_tras_dets, fast_compute_check, reorder_metrics, spec_metric_mask, voxel_mask, visible_cam_mask, concat_affinity, expand_category_cost
from geometry.nusc_utils import mask_between_boxes
# 导入外观特征距离函数（用于第二阶段外观匹配）
from appearance.nusc_distance import iou_2d, giou_2d
from appearance.nusc_utils import project_dets_to_images
from nuscenes.nuscenes import NuScenes


class Tracker:
    def __init__(self, config: dict, nusc: Optional['NuScenes'] = None) -> None:
       
        self.cfg, self.nusc = config, nusc

        # Hyper parameters
        self.set_association_config()
        self.is_debug = self.cfg['debug']['is_debug']
        self.cls_num, self.cam_num = self.cfg['basic']['CLASS_NUM'], len(USED_ALLCAM)
        self.punish_num, self.post_nms_cfg = config['output']['punish_num'], config['output']

        # speed-up tricks
        self.fast, self.re_metrics = fast_compute_check(self.metrics, self.second_metric), reorder_metrics(self.metrics)
        if self.fast:
            self.first_func = 'giou_3d' if 'giou_3d' in self.re_metrics else 'a_giou_3d'
        else:
            self.first_func = None

        # init, notice that no tentative trajectories in Poly-MOT best experiments.
        # self.xx_tras -> {tracking id(int): trajectory}
        self.active_tras, self.tentative_tras, self.dead_tras, self.valid_tras = {}, {}, {}, {}
        self.id_seed, self.frame_id, self.token, self.seq_id, self.det_infos, self.tra_infos = 0, None, None, None, None, None

    # noinspection PyAttributeOutsideInit
    def set_association_config(self) -> None:
        """
        set hyperparameters for data association module
        """
        # load cfg
        match_cfg = self.cfg['association']
        basic_cfg = match_cfg.get('basic', {})
        appear_cfg = match_cfg.get('appear', {})
        motion_cfg = match_cfg.get('motion', {})

        # basic info
        self.two_stage = basic_cfg.get('two_stage', False)
        # 注意：FastPoly 配置中使用 two_thre 作为第二阶段阈值，但字段名是 two_thre
        # 如果配置中有 second_thre 则使用，否则使用 two_thre（兼容性）
        self.s_thre = basic_cfg.get('second_thre', basic_cfg.get('two_thre', {}))
        self.s_cls_spec = basic_cfg.get('s_cls_spec', True)
        self.f_algo = basic_cfg.get('f_algorithm', 'Hungarian')
        # 如果配置中有 s_algorithm 则使用，否则使用 t_algorithm（兼容性）
        self.s_algo = basic_cfg.get('s_algorithm', basic_cfg.get('t_algorithm', 'Hungarian'))
        # only for quick iter
        self.f_thre = {int(cls_idx): float(f_thre) for cls_idx, f_thre in basic_cfg.get('first_thre', {}).items()}
        
        # voxel mask (从 basic 中读取)
        self.use_voxel_mask = basic_cfg.get('voxel_mask', False)
        self.voxel_mask_size = basic_cfg.get('voxel_mask_size', 5)

        # motion info
        self.metrics = motion_cfg.get('category_metrics', {})
        # 如果 motion 中没有 second_metric，尝试从 appear 中获取（兼容性）
        self.second_metric = motion_cfg.get('second_metric', appear_cfg.get('second_metric', 'giou_bev'))

        # appear info (用于第二阶段纯外观匹配)
        self.mix_func1 = appear_cfg.get('mix_func1', 'sum')
        self.mix_func2 = appear_cfg.get('mix_func2', 'sum')
        self.app_metric2 = appear_cfg.get('second_metric', 'giou_2d')
        
        assert self.app_metric2 in ['iou_2d', 'giou_2d'], 'unsupported appearance metric'

        # recall by tracker info (preprocessing.tracker_filter)
        pre_cfg = self.cfg.get('preprocessing', {}).get('tracker_filter', {})
        self.pre_metric = pre_cfg.get('metric', 'iou_2d')
        self.pre_mix_func = pre_cfg.get('mix_func', 'sum')
        self.pre_algo = pre_cfg.get('algorithm', 'Hungarian')
        self.pre_thre = pre_cfg.get('thre', {})
        self.pre_cls_spec = pre_cfg.get('cls_spec', True)
        assert self.pre_metric in ['iou_2d', 'giou_2d'], 'unsupported pre-processing recall metric'
        

    def reset(self) -> None:
        """
        Initialize Tracker for each new seq
        """
        self.active_tras, self.tentative_tras, self.dead_tras, self.valid_tras = {}, {}, {}, {}
        self.id_seed, self.frame_id, self.token, self.seq_id, self.det_infos, self.tra_infos = 0, None, None, None, None, None

    def tracking(self, data_info: dict) -> None:
        """
        :param data_info: the observation information(detection) of each frame
        :return: update data_info: {
            'np_track_res': np.array, [num, 17] add 'tracking_id', 'seq_id', 'frame_id'
            'box_track_res': np.array[NuscBox], [num,]
            'no_val_track_result'[optimal]: bool
        }
        """
        # step0. reset tracker for each seq
        if not self.cfg["basic"]["Multiprocessing"]:
            if data_info['is_first_frame']: self.reset()
            self.det_infos, self.seq_id = data_info, data_info['seq_id']
            self.frame_id, self.token = data_info['frame_id'], data_info['sample_token']
            # step1. predict all valid trajectories
            self.tras_predict()
            
        # step2. recall low confidence dets with vision tracker
        if self.two_stage: self.recall_low_dets(data_info)

        # step3. if there is no dets, we will punish all valid trajectories
        if self.det_infos['no_dets']:
            self.tras_punish(data_info)
            if self.post_nms_cfg['post_nms']: self.post_nms_tras(data_info)
            return

        # step4. associate current frame detections with existing trajectories
        tracking_ids = self.data_association(data_info)

        # step5. use observations(detection) to update the corresponding trajectories
        # and output unmatch trajectories(up to punish_num) states, and output new trajectories states
        dict_track_res = self.tras_update(tracking_ids, data_info)
        
        # corner case, all valid tracklets dead at the current frame
        if len(dict_track_res['np_track_res']) == 0: 
            data_info.update({'no_val_track_result': True})
            return 
            
        # step6. update and output tracking results
        data_info.update({
            'np_track_res': dict_track_res['np_track_res'],
            'box_track_res': dict_track_res['box_track_res'],
            'bm_track_res': dict_track_res['bm_track_res'],
            'norm_bm_track_res': dict_track_res['norm_bm_track_res'],
        })
        
        # whether to use post-predict to reduce FP prediction
        if self.post_nms_cfg['post_nms']: self.post_nms_tras(data_info)
        
        # only for debug
        # assert len(self.tentative_tras) == 0, "no tentative tracklet in the best performance version."

    def tras_predict(self) -> None:
        """
        State Prediction for all trajectories
        get self.tra_infos: {
            'np_tras': np.array, [valid_tra_num, 14]
            'np_tras_bottom_corners': np.array[NuscBox], [valid_tra_num,]
            'all_valid_ids': np.array, [valid_tra_num,]
            'all_valid_boxes': np.array[NuscBox], [valid_tra_num,]
            'tra_num': len(all_valid_ids)
        }
        """
        # debug, Check for tracklets with duplicate states
        if self.is_debug: self.debug()
        pred_infos, pred_bms, pred_norm_bms, pred_boxes, all_valid_ids = [], [], [], [], []

        # corner case(such as first frame..), no valid tracklets
        if len(self.valid_tras) == 0: return

        # predict tracklet for data association
        for tra_id, tra in self.valid_tras.items():
            # only for debug
            if self.is_debug: assert tra_id not in self.dead_tras

            # predict each valid tracklet state
            tra.state_predict(timestamp=self.frame_id, sample_token=self.token)
            pred_object = tra[self.frame_id]['motion']
            
            all_valid_ids.append(tra_id)
            pred_boxes.append(pred_object.predict_box)
            pred_infos.append(pred_object.predict_infos)
            pred_bms.append(pred_object.predict_bms)
            pred_norm_bms.append(pred_object.predict_norm_bms)
            
        # only for debug
        if self.is_debug:
            self.valid_tras = self.merge_valid_tras()
            assert len(all_valid_ids) == len(self.valid_tras)

        self.tra_infos = {
            'np_tras': np.array(pred_infos),  # info dim: 17, add 'tracking_id', 'seq_id', 'frame_id'
            'np_tras_bottom_corners': np.array(pred_bms),
            'np_tras_norm_corners': np.array(pred_norm_bms),
            'all_valid_ids': np.array(all_valid_ids),
            'all_valid_boxes': np.array(pred_boxes),
            'tra_num': len(all_valid_ids)
        }
        
        # 如果使用外观匹配（第二阶段），生成轨迹的外观信息
        if self.two_stage and self.second_metric in ['iou_2d', 'giou_2d']:
            tras_appear_infos = project_dets_to_images(self.nusc, self.token, np.array(pred_boxes))
            self.tra_infos.update({
                'np_tras_bbox2d': tras_appear_infos['np_dets_bbox2d'],
                'tras_visible_cams': tras_appear_infos['visible_cam_id']
            })

    def recall_low_dets(self, data_info: dict) -> None:
        """discard low-quality dets inplace 

        Args:
            data_info (dict): dict, data flow
        """
        # corner case, no low-quality and no valid tracklets
        if 'no_low_det' in data_info or len(self.valid_tras) == 0: return

        det_num, tra_num = data_info['low_det_num'], self.tra_infos['tra_num']
        det_labels, tra_labels = data_info['np_low_dets'][:, -1], self.tra_infos['np_tras'][:, -4]

        # Construct the appearance cost matrix between low-quality dets with all valid tracklets
        app_giou_cost, app_iou_cost = -np.ones((self.cam_num, det_num, tra_num)), np.zeros((self.cam_num, det_num, tra_num))
        tmp_det_infos, tmp_tra_infos = {'np_dets': data_info['np_low_dets']}, {'np_dets': self.tra_infos['np_tras'][:, :-3]}
        for cam_id in range(self.cam_num):
            tmp_tra_infos.update({'np_dets_bbox2d': self.tra_infos['np_tras_bbox2d'][:, cam_id, :]})
            tmp_det_infos.update({'np_dets_bbox2d': data_info['np_low_dets_bbox2d'][:, cam_id, :]})
            app_giou_cost[cam_id], app_iou_cost[cam_id] = giou_2d(tmp_det_infos, tmp_tra_infos)

        # obtain the appearance mask matrix
        valid_app_mask = np.logical_and(
            visible_cam_mask(data_info['low_visible_cam_id'], self.tra_infos['tras_visible_cams']),
            ~mask_between_boxes(det_labels, tra_labels)[0][None, :, :].repeat(self.cam_num, axis=0))
        app_giou_cost[~valid_app_mask], app_iou_cost[~valid_app_mask] = -1, 0

        # concat multi-camera affinity by mix function, 2-ndim cost matrix, [det_num, tra_num]
        pre_app_cost = (1 + app_giou_cost) if self.pre_metric == 'giou_2d' else app_iou_cost
        pre_app_cost = -concat_affinity(raw_affinity=pre_app_cost, aff_mask=valid_app_mask, mix_func=self.pre_mix_func)

        # whether to expand cost matrix to [cls_num, det_num, tra_num]
        if self.pre_cls_spec:
            pre_app_cost, pre_cls_mask = expand_category_cost(pre_app_cost, self.cls_num, det_labels, tra_labels)

        # matching, obtain the potential low-confidence candidate dets, core to recall low-quality dets
        m_low_dets, m_low_tras, _, _ = globals()[self.pre_algo](pre_app_cost, self.pre_thre)
        print('recall {} low-quality dets.'.format(len(m_low_dets)))
        if len(m_low_dets) == 0:
            data_info.update({
                'no_low_det': True,
                'low_det_num': 0,
            })
            return

        # discard low-quality dets inplace and keep valid cost matrix for further association
        data_info.update({
            'low_app_giou_cost': app_giou_cost[:, m_low_dets, :],
            'low_app_iou_cost': app_iou_cost[:, m_low_dets, :],
            'low_valid_mask': valid_app_mask[:, m_low_dets, :],
            'low_cls_mask': pre_cls_mask[:, m_low_dets, :] if self.pre_cls_spec else \
                            mask_tras_dets(self.cls_num, det_labels[m_low_dets], tra_labels),
            'low_det_num': len(m_low_dets),
            'low_det_tra': m_low_tras
        })
        for key in ['np_low_dets', 'box_low_dets', 'np_low_dets_bbox2d']:
            data_info[key] = data_info[key][m_low_dets]

    def tras_punish(self, data_info: dict) -> None:
        """
        handle the corner case where there is no detection at current frame
        also can be seen as "short-cut"
        :param data_info: dict, output file
        :return: update pure predict state, up to output punish_num frame
        update data_info: {
            'np_track_res': np.array, [num, 17] add 'tracking_id', 'seq_id', 'frame_id'
            'box_track_res': np.array[NuscBox], [num,]
        }
        """
        # no valid tras after predicting(valid ids is empty) or no valid tras at prev frame(None)
        if self.tra_infos is None or self.tra_infos['tra_num'] == 0: 
            data_info.update({'no_val_track_result': True})
            return
        
        # manage trajectory
        dict_track_res = self.tras_update(tracking_ids=np.array([]), data_info=data_info)
        if len(dict_track_res['np_track_res']) == 0: 
            data_info.update({'no_val_track_result': True})
            return 
            
        # output punishment tracking results
        data_info.update({
            'np_track_res': dict_track_res['np_track_res'],
            'box_track_res': dict_track_res['box_track_res'],
            'bm_track_res': dict_track_res['bm_track_res'],
            'norm_bm_track_res': dict_track_res['norm_bm_track_res'],
        })


    def tras_update(self, tracking_ids: np.array, data_info: dict) -> dict:
        """
        update the corresponding trajectories with observations, init new tras,
        and punish unmatched tracklets
        :param tracking_ids: np.array, tracking id of each detection
        :param data_info: dict, dets infos at the current frame
        :return: dict, valid estimated states(updated tras, new tras, valid unmatched tras)
            {
            'np_track_res': np.array, [valid_tra_num, 17],
            'box_track_res': np.array[NuscBox], [valid_tra_num,]
            'bm_track_res': np.array, [valid_tra_num, 4, 2]
            'prev_np_track_res': np.array, [prev_tra_num, 17], (optional)
            'prev_box_track_res': np.array[NuscBox], [prev_tra_num,], (optional)
            'prev_bm_track_res': np.array, [prev_tra_num, 4, 2], (optional)
            }
        """
        tracking_ids = tracking_ids.tolist()
        # only for debug
        if 'match_stage' in data_info:
            low_num = data_info.get('low_det_num', 0)
            if len(tracking_ids) != (self.det_infos['det_num'] + low_num):
                if self.is_debug:
                    import pdb; pdb.set_trace()
        else:
            assert len(tracking_ids) == self.det_infos['det_num']

        tmp_match_pair = {tra_id: det_id for det_id, tra_id in enumerate(tracking_ids)}

        np_res, box_res, bm_res, norm_bm_res = [], [], [], []
        prev_np_res, prev_box_res, prev_bm_res = [], [], []
        new_tras, ten_tras, act_tras = {}, {}, {}

        # iterative detections, use measurement(dets) to correct tracklet
        for det_idx, tra_id in enumerate(tracking_ids):
            # low-quality dets are not allowed to init new tras
            if tra_id < 0:
                continue

            # 处理低置信度检测
            low_det_idx = det_idx - self.det_infos['det_num']
            is_low_conf = det_idx >= self.det_infos['det_num']
            
            dict_det = {
                'nusc_box': data_info['box_dets'][det_idx] if not is_low_conf else \
                           data_info.get('box_low_dets', [])[low_det_idx] if 'box_low_dets' in data_info else \
                           data_info['box_dets'][det_idx],
                'np_array': data_info['np_dets'][det_idx] if not is_low_conf else \
                           data_info.get('np_low_dets', [])[low_det_idx] if 'np_low_dets' in data_info else \
                           data_info['np_dets'][det_idx],
                'match_stage': data_info['match_stage'][det_idx] if 'match_stage' in data_info else None,
                'has_velo': data_info['has_velo'],
                'seq_id': data_info['seq_id'],
                'low_conf': is_low_conf
            }

            # only for debug
            if self.is_debug:
                assert tra_id not in self.dead_tras

            if tra_id in self.valid_tras:
                # update exist trajectory
                tra = self.valid_tras[tra_id]
                tra.state_update(timestamp=self.frame_id, sample_token=self.token, det=dict_det)
            else:
                # only for debug
                # if self.is_debug: assert det_idx < self.det_infos['det_num']
                # init new trajectory
                tra = Trajectory(timestamp=self.frame_id,
                                 config=self.cfg,
                                 track_id=tra_id,
                                 det_infos=dict_det,
                                 nusc=self.nusc,
                                 sample_token=self.token)
                new_tras[tra_id] = tra

        # merge all tras, include exist trajectories and newly generated trajectory
        tmp_merge_tras = {**self.valid_tras, **new_tras}

        # iterative trajectories, punish and output
        for tra_id, tra in tmp_merge_tras.items():
            # update unmatched tracklets
            if tra_id not in tracking_ids:
                tra.state_update(timestamp=self.frame_id, sample_token=self.token, det=None)
            update_object = tra[self.frame_id]['motion']
            # only active tracklets' state are output to the result file
            if tra.life_management.state == 'active':
                act_tras[tra_id] = tra

                # update states in the current frame
                if update_object.update_infos is not None:
                    np_res.append(update_object.update_infos)
                    box_res.append(update_object.update_box)
                    bm_res.append(update_object.update_bms)
                    norm_bm_res.append(update_object.update_norm_bms)
                elif tra.life_management.time_since_update <= self.punish_num:
                    np_res.append(update_object.predict_infos)
                    box_res.append(update_object.predict_box)
                    bm_res.append(update_object.predict_bms)
                    norm_bm_res.append(update_object.predict_norm_bms)

                # previous states of trajectories converted from the tentative state
                if hasattr(tra.life_management, 'state_jump') and tra.life_management.state_jump:
                    # only for quick iter
                    # only for debug
                    if self.is_debug:
                        assert tra.first_det.get('low_conf', False), "only low conf can be tentative tracklets"
                    for prev_t in range(self.frame_id - 1, tra.init_timestamp - 1, -1):
                        prev_obj = tra[prev_t]['motion']
                        prev_np_res.append(prev_obj.update_infos)
                        prev_box_res.append(prev_obj.update_box)
                        prev_bm_res.append(prev_obj.update_bms)

            elif tra.life_management.state == 'tentative':
                # only for debug
                if self.is_debug:
                    assert tra.first_det.get('low_conf', False), "only low conf can be tentative tracklets"
                ten_tras[tra_id] = tra
            elif tra.life_management.state == 'dead':
                assert tra_id not in self.dead_tras
                self.dead_tras[tra_id] = tra
            else:
                raise Exception('Trajectory state only have three attributes')

        # reorganize active/dead/tentative trajectories
        self.active_tras, self.tentative_tras = act_tras, ten_tras
        self.valid_tras = {**self.active_tras, **self.tentative_tras}

        dict_track_res = {
            'np_track_res': np_res,
            'box_track_res': box_res,
            'bm_track_res': bm_res,
            'norm_bm_track_res': norm_bm_res
        }
        
        # 如果有历史帧输出，添加到结果中
        if len(prev_np_res) > 0:
            dict_track_res.update({
                'prev_np_track_res': prev_np_res,
                'prev_box_track_res': prev_box_res,
                'prev_bm_track_res': prev_bm_res
            })
        
        return dict_track_res   

    def data_association(self, data_info: dict = None) -> np.array:
        """
        Associate the track and the detection, and assign a tracking id to each detection
        :param data_info: dict, optional, used to update 'match_stage'
        :return: np.array, tracking ids of each detection
        """
        # corner case, no valid trajectory. quickly assign each det tracking id
        if len(self.valid_tras) == 0:
            ids = np.arange(self.id_seed, self.id_seed + self.det_infos['det_num'], dtype=int)
            self.id_seed += self.det_infos['det_num']
        else:
            cost_matrices = self.compute_cost(data_info)
            ids = self.matching_cost(cost_matrices, data_info)
        return ids

    def compute_cost(self, data_info: dict = None) -> dict:
        """
        Construct the cost matrix between the trajectory and the detection
        :param data_info: dict, optional, used to integrate low-confidence detections
        :return: dict, a collection of cost matrices,
        one-stage: np.array, [cls_num, det_num, tra_num] (only high-confidence dets),
        two-stage: np.array, [cls_num, det_num + low_det_num, tra_num] (unmatched high + all low dets)
        """
        assert self.tra_infos is not None and self.tra_infos['tra_num'] is not None
        det_num, tra_num = self.det_infos['det_num'], self.tra_infos['tra_num']
        det_labels, tra_labels = self.det_infos['np_dets'][:, -1], self.tra_infos['np_tras'][:, -4]
        
        # 获取低分检测数量
        low_det_num = data_info.get('low_det_num', 0) if data_info is not None else 0
        has_low_dets = low_det_num > 0 and 'no_low_det' not in (data_info or {})

        # [det_num, tra_num], True denotes valid (in the same voxel)
        if self.use_voxel_mask:
            if not isinstance(self.voxel_mask_size, dict):
                fast_voxel_2d_mask = voxel_mask(self.det_infos['np_dets'], self.tra_infos['np_tras'][:, :-3], thre=self.voxel_mask_size)
                fast_voxel_3d_mask = fast_voxel_2d_mask[None, :, :].repeat(self.cls_num, axis=0)
            else:
                fast_voxel_3d_mask = np.array([voxel_mask(self.det_infos['np_dets'],
                                                          self.tra_infos['np_tras'][:, :-3],
                                                          thre=self.voxel_mask_size[cls_idx])
                                               for cls_idx in range(self.cls_num)])
                fast_voxel_2d_mask = np.max(fast_voxel_3d_mask, axis=0)
        else:
            fast_voxel_2d_mask = np.ones((det_num, tra_num), dtype=bool)
            fast_voxel_3d_mask = np.ones((self.cls_num, det_num, tra_num), dtype=bool)

        # [cls_num, det_num, tra_num], True denotes valid (det label == tra label == cls idx)
        cls_3d_mask, cls_2d_mask = mask_tras_dets(self.cls_num, det_labels, tra_labels)
        # valid_3d_mask, valid_2d_mask = mask_tras_dets(self.cls_num, det_labels, tra_labels)

        # [cls_num, det_num, tra_num], final valid mask, True denotes valid
        valid_3d_mask = np.logical_and(fast_voxel_3d_mask, cls_3d_mask)
        valid_2d_mask = np.logical_and(fast_voxel_2d_mask, cls_2d_mask)

        # Densification, skip invalid affinity in cost computing process
        val_det_idx, val_tra_idx = np.where(valid_2d_mask)
        val_det_idx, val_tra_idx = np.unique(val_det_idx), np.unique(val_tra_idx)

        two_cost, first_cost = -np.ones((det_num, tra_num)) * np.inf, -np.ones((det_num, tra_num)) * np.inf

        if len(val_det_idx) == 0 or len(val_tra_idx) == 0:
            # corner case, no valid interaction
            # 确保 two_stage 也是 3D [cls_num, det_num, tra_num]
            if self.two_stage:
                two_stage_3d = two_cost[None, :, :].repeat(self.cls_num, axis=0)
                two_stage_return = 1 - two_stage_3d
            else:
                two_stage_return = None
            return {'one_stage': 1 - first_cost[None, :, :].repeat(self.cls_num, axis=0),
                    'two_stage': two_stage_return}

        # Construct valid det/tra infos
        tra_cost_infos = {'np_dets': self.tra_infos['np_tras'][val_tra_idx, :-3],
                          'np_dets_bottom_corners': self.tra_infos['np_tras_bottom_corners'][val_tra_idx],
                          'np_dets_norm_corners': self.tra_infos['np_tras_norm_corners'][val_tra_idx]}
        det_cost_infos = {'np_dets': self.det_infos['np_dets'][val_det_idx],
                          'np_dets_bottom_corners': self.det_infos['np_dets_bottom_corners'][val_det_idx],
                          'np_dets_norm_corners': self.det_infos['np_dets_norm_corners'][val_det_idx]}

        val_det_tra_idx = np.ix_(val_det_idx, val_tra_idx)

        if self.fast:
            # metrics only have giou_3d/giou_bev or a_giou_3d/a_giou_bev
            raw_two_cost, raw_first_cost = globals()[self.first_func](det_cost_infos, tra_cost_infos)
            two_cost[val_det_tra_idx] = raw_two_cost
            first_cost[val_det_tra_idx] = raw_first_cost
            first_cost = first_cost[None, :, :].repeat(self.cls_num, axis=0)

            if self.second_metric in self.re_metrics:
                first_cost[self.re_metrics[self.second_metric]] = two_cost
        else:
            if self.nusc is not None:
                # Step 1. 构建多相机外观成本矩阵 [cam_num, det_num, tra_num]
                val_det_num, val_tra_num = len(val_det_idx), len(val_tra_idx)
                app_giou_cost, app_iou_cost = -np.ones((self.cam_num, val_det_num, val_tra_num)), np.zeros((self.cam_num, val_det_num, val_tra_num))
                for cam_id in range(self.cam_num):
                    # 更新轨迹的 2D bbox 信息（当前相机）
                    tra_cost_infos_2d = {'np_dets': tra_cost_infos['np_dets'],
                                        'np_dets_bbox2d': self.tra_infos['np_tras_bbox2d'][val_tra_idx, cam_id, :]}
                    det_cost_infos_2d = {'np_dets': det_cost_infos['np_dets'],
                                        'np_dets_bbox2d': self.det_infos['np_dets_bbox2d'][val_det_idx, cam_id, :]}
                    tmp_giou, tmp_iou = giou_2d(det_cost_infos_2d, tra_cost_infos_2d)
                    # 将结果填充到完整矩阵中
                    app_giou_cost[cam_id] = tmp_giou
                    app_iou_cost[cam_id] = tmp_iou
                
                # Step 2. 获取外观掩码矩阵（可见性掩码）
                visible_mask = visible_cam_mask(
                    [self.det_infos['visible_cam_id'][i] for i in val_det_idx],
                    [self.tra_infos['tras_visible_cams'][i] for i in val_tra_idx]
                )
                diff_cls_mask = mask_between_boxes(det_labels[val_det_idx], tra_labels[val_tra_idx])[0]
                valid_app_mask = np.logical_and(visible_mask, ~diff_cls_mask[None, :, :].repeat(self.cam_num, axis=0))
                app_giou_cost[~valid_app_mask], app_iou_cost[~valid_app_mask] = -1, 0
                
                # Step 3. 融合多相机亲和度（高分检测的外观成本）
                app_two_cost = (1 + app_giou_cost) if self.app_metric2 == 'giou_2d' else app_iou_cost
                two_cost_full = concat_affinity(raw_affinity=app_two_cost, aff_mask=valid_app_mask, mix_func=self.mix_func1)
                two_cost[val_det_tra_idx] = two_cost_full
            else: pass
            # 第一阶段：运动匹配（只使用高分检测）
            first_cost = -np.ones((self.cls_num, det_num, tra_num)) * np.inf
            for metric, cls_list in self.re_metrics.items():
                cost1 = globals()[metric](det_cost_infos, tra_cost_infos)
                cost1 = cost1[1] if metric in METRIC else cost1
                first_cost[np.ix_(cls_list, val_det_idx, val_tra_idx)] = cost1

        # mask invalid value
        first_cost[np.where(~valid_3d_mask)] = -np.inf

        # 第二阶段：整合低分检测的外观成本
        if self.two_stage and has_low_dets:
            # 获取低分检测的外观成本（从 recall_low_dets 中计算好的）
            low_app_giou_cost = data_info['low_app_giou_cost']  # [cam_num, low_det_num, tra_num]
            low_app_iou_cost = data_info['low_app_iou_cost']
            low_valid_mask = data_info['low_valid_mask']  # [cam_num, low_det_num, tra_num]
            
            # 融合低分检测的多相机外观成本
            low_app_cost = (1 + low_app_giou_cost) if self.app_metric2 == 'giou_2d' else low_app_iou_cost
            low_two_cost = concat_affinity(raw_affinity=low_app_cost, aff_mask=low_valid_mask, mix_func=self.mix_func2)
            # low_two_cost 形状: [low_det_num, tra_num]
            
            # 整合高分检测和低分检测的外观成本
            # two_cost 当前形状: [det_num, tra_num]，需要扩展到 [det_num + low_det_num, tra_num]
            two_cost_combined = np.concatenate((two_cost, low_two_cost), axis=0)
            
            # 始终扩展为 3D [cls_num, det_num + low_det_num, tra_num]
            two_cost_combined = two_cost_combined[None, :, :].repeat(self.cls_num, axis=0)
            
            # 如果启用类别特定匹配，应用类别掩码
            if self.s_cls_spec:
                high_cls_mask, _ = mask_tras_dets(self.cls_num, det_labels, tra_labels)
                low_cls_mask = data_info.get('low_cls_mask', None)
                if low_cls_mask is None:
                    low_det_labels = data_info['np_low_dets'][:, -1]
                    low_cls_mask, _ = mask_tras_dets(self.cls_num, low_det_labels, tra_labels)
                
                valid_cls_mask = np.concatenate((high_cls_mask, low_cls_mask), axis=1)
                two_cost_combined[~valid_cls_mask] = 0
            
            two_stage_cost = -two_cost_combined
        elif self.two_stage:
            # 只有高分检测，没有低分检测
            # 始终扩展为 3D [cls_num, det_num, tra_num]
            two_cost_expanded = two_cost[None, :, :].repeat(self.cls_num, axis=0)
            
            # s_cls_spec 恒为 true，应用类别掩码
            high_cls_mask, _ = mask_tras_dets(self.cls_num, det_labels, tra_labels)
            two_cost_expanded[~high_cls_mask] = 0
            
            two_stage_cost = -two_cost_expanded
        else:
            two_stage_cost = None

        # Due to the execution speed of python,
        # construct the two-stage cost matrix under half-parallel framework is very tricky, 
        # we strongly recommend only use giou_bev as two-stage metric to build the cost matrix
        return {'one_stage': 1 - first_cost, 'two_stage': two_stage_cost}

    def matching_cost(self, cost_matrices: dict, data_info: dict = None) -> np.array:
        """
        Solve the matching pair according to the cost matrix
        :param cost_matrices: cost matrices between dets and tras construct in the one/two stage
        :param data_info: dict, optional, used to update 'match_stage'
        :return: np.array, tracking id of each detection
        """
        cost1, cost2 = cost_matrices['one_stage'], cost_matrices['two_stage']
        
        # 获取低分检测数量
        low_det_num = data_info.get('low_det_num', 0) if data_info is not None else 0
        has_low_dets = low_det_num > 0 and 'no_low_det' not in (data_info or {})
        
        # 初始化 match_stage（如果提供了 data_info）
        if data_info is not None:
            match_stage = -np.ones(shape=self.det_infos['det_num'] + low_det_num)
        else:
            match_stage = None
        
        # first-stage matching, only high score det with all tras
        # m_tras_1 is not the tracking id, but is the index of tracklet in the all valid trajectories
        m_dets_1, m_tras_1, um_dets_1, um_tras_1 = globals()[self.f_algo](cost1, self.f_thre)
        if match_stage is not None:
            match_stage[m_dets_1] = 1
        
        # only for debug
        if self.is_debug: 
            print('{} tracklets are be associated in the first stage'.format(len(m_dets_1)))
        
        # second-stage matching: unmatched high score dets + all low score dets with remain tras
        # 类似 RockTrack 的实现方式：直接在原始 cost2 上操作，使用 inf_cost2 限制匹配范围
        if self.two_stage and cost2 is not None:
            # cost2 始终是 3D [cls_num, det_num + low_det_num, tra_num] 或 [cls_num, det_num, tra_num]
            assert cost2.ndim == 3, f"cost2 must be 3D, but got {cost2.ndim}D with shape {cost2.shape}"
            
            # 构建第二阶段匹配的检测索引：未匹配的高分检测 + 所有低分检测
            if has_low_dets:
                # 低分检测的索引：从 det_num 到 det_num + low_det_num - 1
                low_det_indices = np.arange(self.det_infos['det_num'], self.det_infos['det_num'] + low_det_num)
                # 第二阶段匹配的检测：未匹配的高分检测 + 所有低分检测
                stage2_det_indices = np.concatenate([um_dets_1, low_det_indices]) if len(um_dets_1) > 0 else low_det_indices
            else:
                # 没有低分检测，只有未匹配的高分检测
                stage2_det_indices = um_dets_1
            
            # 空数组检查
            if len(stage2_det_indices) == 0 or len(um_tras_1) == 0:
                m_dets_2, m_tras_2 = [], []
            else:
                # 添加边界检查：确保索引在有效范围内
                max_det_idx = cost2.shape[1] - 1
                max_tra_idx = cost2.shape[2] - 1
                stage2_det_indices = np.array([idx for idx in stage2_det_indices if 0 <= idx <= max_det_idx])
                um_tras_1_filtered = np.array([idx for idx in um_tras_1 if 0 <= idx <= max_tra_idx])
                
                if len(stage2_det_indices) == 0 or len(um_tras_1_filtered) == 0:
                    m_dets_2, m_tras_2 = [], []
                else:
                    # 类似 RockTrack：使用 inf_cost2 限制匹配范围
                    # cost2 是 3D [cls_num, det_num + low_det_num, tra_num]
                    inf_cost2 = np.ones_like(cost2) * np.inf
                    # 为每个类别设置允许匹配的位置
                    for cls_idx in range(cost2.shape[0]):
                        inf_cost2[cls_idx][np.ix_(stage2_det_indices, um_tras_1_filtered)] = 0
                    m_dets_2, m_tras_2, _, _ = globals()[self.s_algo](cost2 + inf_cost2, self.s_thre)
            
            if match_stage is not None and len(m_dets_2) > 0:
                match_stage[m_dets_2] = 2
            
            m_dets_1 = list(m_dets_1) + list(m_dets_2)
            m_tras_1 = list(m_tras_1) + list(m_tras_2)
            
            # only for debug
            if self.is_debug: 
                print('{} tracklets are be associated in the second stage'.format(len(m_dets_2)))
        
        # 更新 data_info 中的 match_stage
        if data_info is not None and match_stage is not None:
            data_info['match_stage'] = match_stage.tolist()
        
        assert len(m_dets_1) == len(m_tras_1), "as the pair, number of the matched tras and dets must be equal"
        
        # corner case, no matching pair after matching
        if len(m_dets_1) == 0:
            ids = np.arange(self.id_seed, self.id_seed + self.det_infos['det_num'], dtype=int)
            self.id_seed += self.det_infos['det_num']
            # 如果有低分检测，为它们分配 -1（不初始化新轨迹）
            if has_low_dets:
                ids = np.concatenate([ids, -np.ones(low_det_num, dtype=int)])
            return ids
        else:
            ids, match_pairs = [], {key: value for key, value in zip(m_dets_1, m_tras_1)}
            all_valid_ids = self.tra_infos['all_valid_ids']
            
            # 处理高分检测
            for det_idx in range(self.det_infos['det_num']):
                if det_idx not in m_dets_1:
                    ids.append(self.id_seed)
                    self.id_seed += 1
                else:
                    ids.append(all_valid_ids[match_pairs[det_idx]])
            
            # 处理低分检测（如果有）
            if has_low_dets:
                # 使用 enumerate 获取 tra_idx（从0开始），对应 low_det_tra 的索引（与 RockTrack 一致）
                for tra_idx, low_idx in enumerate(range(self.det_infos['det_num'], self.det_infos['det_num'] + low_det_num)):
                    if low_idx not in m_dets_1:
                        # 低分检测未匹配，分配 -1（不初始化新轨迹）
                        ids.append(-1)
                    else:
                        # 低分检测匹配成功，验证一致性：使用 tra_idx 作为 low_det_tra 的索引
                        ids.append(all_valid_ids[match_pairs[low_idx]] if match_pairs[low_idx] == self.det_infos['low_det_tra'][tra_idx] else -1)

        return np.array(ids)

    def merge_valid_tras(self) -> dict:
        """
        Get all valid trajectories, 'valid' denotes that 'active' and 'tentative'
        :return: dict, merge active tracklets and tentative tracklets
        """
        return {**self.active_tras, **self.tentative_tras}

    def post_nms_tras(self, data_info) -> None:
        """
        use post-predict to reduce FP prediction
        :param data_info: the final tracking result at each frame
        :retrun: no return, but filter FP results in the data_info
        """
        if 'no_val_track_result' in data_info: return
        post_metric = self.post_nms_cfg['NMS_metric']
        post_thre = self.post_nms_cfg['NMS_thre']
        post_type = self.post_nms_cfg['NMS_type']
        tmp_tra_infos = {'np_dets': np.array(data_info['np_track_res'])[:, :-3],
                         'np_dets_bottom_corners': np.array(data_info['bm_track_res']),
                         'np_dets_norm_corners': np.array(data_info['norm_bm_track_res']),
                         'box_dets': np.array(data_info['box_track_res'])}
        
        # 根据 NMS 类型调用不同的函数
        if post_type == 'scale_nms':
            # scale_nms 需要额外的参数
            post_scale = self.post_nms_cfg.get('SCALE', {})
            post_voxel_mask_size_raw = self.post_nms_cfg.get('voxel_mask_size', None)
            # 如果 voxel_mask_size 是字典，直接使用
            if isinstance(post_voxel_mask_size_raw, dict) and len(post_voxel_mask_size_raw) > 0:
                post_voxel_mask_size = post_voxel_mask_size_raw
            elif isinstance(post_voxel_mask_size_raw, (int, float)):
                # 如果是数字，为所有类别设置相同的值
                post_voxel_mask_size = {cls_idx: post_voxel_mask_size_raw for cls_idx in CLASS_STR_TO_SEG_CLASS.keys()}
            else:
                # 如果未配置或为空，使用默认值 5（从 association.basic 获取，如果没有则使用默认值）
                default_voxel_size = self.use_voxel_mask and (self.voxel_mask_size if isinstance(self.voxel_mask_size, (int, float)) else 5) or 5
                post_voxel_mask_size = {cls_idx: default_voxel_size for cls_idx in CLASS_STR_TO_SEG_CLASS.keys()}
            post_use_voxel_mask = self.post_nms_cfg.get('voxel_mask', False)
            keep = scale_nms(box_infos=tmp_tra_infos, 
                           metrics=post_metric, 
                           thres=post_thre, 
                           factors=post_scale,
                           voxel_mask_size=post_voxel_mask_size,
                           use_voxel_mask=post_use_voxel_mask)
        else:
            # 其他 NMS 类型（blend_nms 等）
            keep = globals()[post_type](box_infos=tmp_tra_infos, metrics=post_metric, thre=post_thre)
        
        if len(keep) == 0:
            data_info.update({'no_val_track_result': True})
        else:
            data_info['np_track_res'] = np.array(data_info['np_track_res'])[keep].tolist()
            data_info['box_track_res'] = np.array(data_info['box_track_res'])[keep].tolist()
            data_info['bm_track_res'] = np.array(data_info['bm_track_res'])[keep].tolist()
            data_info['norm_bm_track_res'] = np.array(data_info['norm_bm_track_res'])[keep].tolist()

    def debug(self) -> None:
        """
        only for debug, check whether the trajectory status is repeated
        TODO: delete at public version
        """
        assert len(self.tentative_tras.keys() & self.active_tras.keys()) == 0
        assert len(self.active_tras.keys() & self.dead_tras.keys()) == 0
        assert len(self.tentative_tras.keys() & self.dead_tras.keys()) == 0
