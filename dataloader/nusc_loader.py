# -*- coding: utf-8 -*-
"""
dataloader of NuScenes dataset
Obtain the observation information(detection) of each frame iteratively
--------ATTENTION: Detector files must be in chronological order-------
"""

import pdb
import time, json, os, shutil

import numpy as np
from utils.io import load_file
from data.script.NUSC_CONSTANT import *
from pre_processing import dictdet2array, arraydet2box, blend_nms, scale_nms
from appearance.nusc_utils import project_dets_to_images
from nuscenes.nuscenes import NuScenes



class NuScenesloader:
    def __init__(self, detection_path, first_token_path, config, root_path=None, nusc=None):
        """
        :param detection_path: path of order detection file
        :param first_token_path: path of first frame token for each seq
        :param config: dict, hyperparameter setting
        :param root_path: path of nuscenes database (optional, needed for appearance features)
        :param nusc: NuScenes instance (optional, if provided, root_path will be ignored)
        """
        # detector -> {sample_token:[{det1_info}, {det2_info}], ...}， check the detailed "det_info" at nuscenes.org
        self.detector = load_file(detection_path)["results"]
        self.all_sample_token = list(self.detector.keys())
        self.seq_first_token = load_file(first_token_path)
        self.config, self.data_info = config, {}
        
        # 从 NMS_filter 子配置中读取参数
        nms_filter_cfg = config['preprocessing'].get('NMS_filter', {})
        self.SF_thre = {int(k): float(v) for k, v in nms_filter_cfg.get('SF_thre', {}).items()}
        self.NMS_thre = nms_filter_cfg.get('NMS_thre', 0.08)
        self.NMS_type = nms_filter_cfg.get('NMS_type', 'scale_nms')
        self.NMS_metric = nms_filter_cfg.get('NMS_metric', 'iou_bev')
        self.SCALE = nms_filter_cfg.get('SCALE', None) if self.NMS_type == 'scale_nms' else None
        self.seq_id = self.frame_id = 0

        self.nms_voxel_mask = nms_filter_cfg.get('voxel_mask', False)
        voxel_mask_size_raw = nms_filter_cfg.get('voxel_mask_size', 5)
        # 如果 voxel_mask_size 是字典，根据 NMS_type 处理
        if isinstance(voxel_mask_size_raw, dict):
            if self.NMS_type == 'scale_nms':
                self.voxel_mask_size = voxel_mask_size_raw
            else:
                # 对于非 scale_nms，使用字典的第一个值
                self.voxel_mask_size = list(voxel_mask_size_raw.values())[0] if voxel_mask_size_raw else 5
        else:
            self.voxel_mask_size = voxel_mask_size_raw

        # 读取 fix_posz/fix_height 配置
        pre_oth_cfg = config['preprocessing'].get('others', {})
        fix_posz_raw = pre_oth_cfg.get('fix_posz', {})
        fix_height_raw = pre_oth_cfg.get('fix_height', {})
        # 转换为类别名称字典格式
        self.fix_posz = {CLASS_STR_TO_SEG_CLASS[cls]: value for cls, value in fix_posz_raw.items()}
        self.fix_height = {CLASS_STR_TO_SEG_CLASS[cls]: value for cls, value in fix_height_raw.items()}

        # 加载 NuScenes 数据集（如果需要外观特征）
        self.use_appearance_stage = config.get('association', {}).get('basic', {}).get('two_stage', False) and \
                                     config.get('association', {}).get('appear', {}).get('second_metric', None) is not None
        self.use_tracker_filter = config.get('preprocessing', {}).get('tracker_filter', {}).get('use', False)
 
        if self.use_appearance_stage or self.use_tracker_filter:
            if nusc is not None:
                self.nusc = nusc
            elif root_path is not None:
                self.nusc = self.load_dataset(root_path, config['basic']['split'])
            else:
                raise ValueError("root_path or nusc must be provided when use_appearance_stage or use_tracker_filter is True")
        else:
            # 不需要外观特征时，不加载 NuScenes 数据集
            self.nusc = None
        # 调试标志（用于检查特定 token 的输出）
        self.hit = False

    @staticmethod
    def load_dataset(root_path: str, split: str) -> 'NuScenes':
        """load NuScenes dataset

        Args:
            root_path (str): root path of database
            split (str): split of nuscenes

        Returns:
            NuScenes: instance of nuscenes
        """
        
        assert split in ['val', 'test'], "set split must be val or test."
        version = 'v1.0-{}'.format('trainval' if split == 'val' else split)
        
        # load dataset
        nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
        return nusc

    def __getitem__(self, item) -> dict:
        """
        data_info(dict): {
            'is_first_frame': bool
            'timestamp': int
            'sample_token': str
            'seq_id': int
            'frame_id': int
            'has_velo': bool
            'no_dets': bool, corner case,
            'det_num': int,
            'box_dets': np.array[NuscBox], [det_num]
            
            # motion attributes
            'np_dets': np.array, [det_num, 14](x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), det_score, class_label)
            'np_dets_bottom_corners': np.array, [det_num, 4, 2]
            'np_dets_norm_corners': np.array, [det_num, 4, 2] (if available)
            
            # appearance attributes (if use_appearance_stage is True)
            'np_dets_bbox2d': np.array, [det_num, 6(camera num), 4(x1, y1, x2, y2)], invalid values are filled with -1
            'np_dets_bbox3d': np.array, [det_num, 6(camera num), 8(corner), 2], invalid values are filled with -1
            'maxarea_cam_id': np.array, [det_num], camera which has the max reproject area
            'visible_cam_id': list of list, [[cams id for det 1], [cams id for det 2], ...]
            'visible_area': list of list, reproject area
            
            # low confidence detections (if use_tracker_filter is True)
            'no_low_det': bool, True if no low confidence detections
            'np_low_dets': np.array, [low_det_num, 14]
            'box_low_dets': np.array[NuscBox], [low_det_num]
            'np_low_dets_bbox2d': np.array, [low_det_num, 6, 4]  # 用于外观匹配，不是图片路径
            'low_visible_cam_id': list of list  # 用于外观匹配，不是图片路径
            'low_det_num': int
        }
        """
        curr_token = self.all_sample_token[item]
        ori_dets = self.detector[curr_token]

        # assign seq and frame id
        if curr_token in self.seq_first_token:
            self.seq_id += 1
            self.frame_id = 1
        else: 
            self.frame_id += 1
        
        # only for debug: 只处理特定 token 的帧
        # 目标 token: 45eda670b5a840acbb730aac15e63b19
        # TARGET_TOKEN = '45eda670b5a840acbb730aac15e63b19'
        # if curr_token == TARGET_TOKEN:
        #     self.hit = True
        # elif not self.hit:
        #     # 跳过目标 token 之前的帧
        #     return {
        #         'is_first_frame': curr_token in self.seq_first_token,
        #         'timestamp': item,
        #         'sample_token': curr_token,
        #         'seq_id': self.seq_id,
        #         'frame_id': self.frame_id,
        #         'has_velo': self.config['basic']['has_velo'],
        #         'np_dets': np.zeros(0),
        #         'np_dets_bottom_corners': np.zeros(0),
        #         'np_dets_norm_corners': np.zeros(0),
        #         'box_dets': np.zeros(0),
        #         'no_dets': True,
        #         'det_num': 0,
        #     }

        # all categories are blended together and sorted by detection score
        # 使用 fix_posz/fix_height 参数
        list_dets, np_dets = dictdet2array(ori_dets, self.fix_posz, self.fix_height,
                                           'translation', 'size', 'velocity', 'rotation',
                                           'detection_score', 'detection_name')

        # RockTrack 流程：先 NMS，再分割高低置信度
        # 所有检测都参与 NMS（不先过滤）
        if len(np_dets) != 0:
        # NMS, "blend" ref to blend all categories together during NMS
            box_dets, np_dets_bottom_corners, np_dets_norm_corners = arraydet2box(np_dets, sample_token=curr_token)
            assert len(np_dets) == len(box_dets) == len(np_dets_bottom_corners) == len(np_dets_norm_corners)
            tmp_infos = {'np_dets': np_dets, 
                        'np_dets_bottom_corners': np_dets_bottom_corners,
                        'np_dets_norm_corners': np_dets_norm_corners,
                        'box_dets': box_dets}
            
            if self.NMS_type != 'scale_nms':
                keep = globals()[self.NMS_type](box_infos=tmp_infos,
                                                metrics=self.NMS_metric,
                                                thre=self.NMS_thre,
                                                voxel_mask_size=self.voxel_mask_size,
                                                use_voxel_mask=self.nms_voxel_mask)
            else:
                keep = scale_nms(box_infos=tmp_infos,
                                 metrics=self.NMS_metric,
                                 thres=self.NMS_thre,
                                 factors=self.SCALE,
                                 voxel_mask_size=self.voxel_mask_size,
                                 use_voxel_mask=self.nms_voxel_mask)
            keep_num = len(keep)
            
            # Score Filter based on category-specific thresholds (after NMS)
            if keep_num != 0:
                # All valid dets after NMS
                valid_np_dets, valid_box_dets, valid_bcs = np_dets[keep], box_dets[keep], np_dets_bottom_corners[keep]
                valid_sc_thre = np.array([self.SF_thre[int(det_cls)] for det_cls in valid_np_dets[:, -1]])
                
                # High confidence score dets
                high_dets_idx = np.where(valid_np_dets[:, -2] > valid_sc_thre)[0]
                high_keep_num = len(high_dets_idx)
                
                # Low confidence score dets
                low_dets_idx = np.where(valid_np_dets[:, -2] <= valid_sc_thre)[0]
                low_keep_num = len(low_dets_idx)
            else:
                high_keep_num = low_keep_num = 0
        else:
        # corner case, no det left
            keep = keep_num = high_keep_num = low_keep_num = 0

        print(f"\n Total {len(list_dets) - keep_num} bboxes are filtered; "
              f"{len(np_dets) - keep_num} during NMS, "
              f"Still {high_keep_num} high score bboxes left, "
              f"Still {low_keep_num} low score bboxes left, "
              f"seq id {self.seq_id}, frame id {self.frame_id}, "
              f"Total frame id {item + 1}.")

        # Available information for the current frame
        if keep_num == 0:
            data_info = {
                'is_first_frame': curr_token in self.seq_first_token,
                'timestamp': item,
                'sample_token': curr_token,
                'seq_id': self.seq_id,
                'frame_id': self.frame_id,
                'has_velo': self.config['basic']['has_velo'],
                'np_dets': np.zeros(0),
                'np_dets_bottom_corners': np.zeros(0),
                'box_dets': np.zeros(0),
                'no_dets': True,
                'det_num': 0,
            }
            if self.use_tracker_filter:
                data_info.update({'no_low_det': True})
            # 添加空的 norm_corners（即使没有检测）
            if 'np_dets_norm_corners' not in data_info:
                data_info['np_dets_norm_corners'] = np.zeros(0)
            return data_info

        # High confidence detections
        valid_np_dets, valid_box_dets, valid_bcs = np_dets[keep], box_dets[keep], np_dets_bottom_corners[keep]
        valid_ncs = np_dets_norm_corners[keep]  # 添加 norm_corners
        valid_sc_thre = np.array([self.SF_thre[int(det_cls)] for det_cls in valid_np_dets[:, -1]])
        high_dets_idx = np.where(valid_np_dets[:, -2] > valid_sc_thre)[0]
        low_dets_idx = np.where(valid_np_dets[:, -2] <= valid_sc_thre)[0]
        
        data_info = {
            'is_first_frame': curr_token in self.seq_first_token,
            'timestamp': item,
            'sample_token': curr_token,
            'seq_id': self.seq_id,
            'frame_id': self.frame_id,
            'has_velo': self.config['basic']['has_velo'],
            'np_dets': valid_np_dets[high_dets_idx] if len(high_dets_idx) > 0 else np.zeros(0),
            'np_dets_bottom_corners': valid_bcs[high_dets_idx] if len(high_dets_idx) > 0 else np.zeros(0),
            'np_dets_norm_corners': valid_ncs[high_dets_idx] if len(high_dets_idx) > 0 else np.zeros(0),
            'box_dets': valid_box_dets[high_dets_idx] if len(high_dets_idx) > 0 else np.zeros(0),
            'no_dets': len(high_dets_idx) == 0,
            'det_num': len(high_dets_idx),
        }
        
        # get appearance attributes of each high score det (if needed)
        if self.use_appearance_stage and len(high_dets_idx) > 0:
            det_vis_infos = project_dets_to_images(self.nusc, curr_token, data_info['box_dets'])
            data_info.update(det_vis_infos)
        
        # Low confidence score detections (for tracker_filter)
        if len(low_dets_idx) == 0 or not self.use_tracker_filter:
            data_info.update({'no_low_det': True})
            return data_info
        
        np_low_dets, box_low_dets = valid_np_dets[low_dets_idx], valid_box_dets[low_dets_idx]
        
        # get appearance attributes of each low score det (for tracker_filter matching)
        # 注意：这些外观特征（bbox2d, visible_cam_id）用于匹配，不是图片路径
        low_det_vis_infos = project_dets_to_images(self.nusc, curr_token, box_low_dets)
        data_info.update({
            'np_low_dets': np_low_dets,
            'box_low_dets': box_low_dets,
            'np_low_dets_bbox2d': low_det_vis_infos['np_dets_bbox2d'],
            'low_visible_cam_id': low_det_vis_infos['visible_cam_id'],
            'low_det_num': len(np_low_dets)
        })
        
        # only for debug: 保存特定 token 的数据到 JSON（用于可视化检查）
        # if curr_token == '45eda670b5a840acbb730aac15e63b19':
        #     # 基础路径
        #     BASE_PATH = '/data/wyt_dataset/gy_code/FastPoly/debug_output/45eda670b5a840acbb730aac15e63b19'
        #     ORIGINAL_PATH = f'{BASE_PATH}.json'
            
        #     # 如果原文件存在，先备份
        #     if os.path.exists(ORIGINAL_PATH):
        #         backup_timestamp = time.strftime("%Y%m%d_%H%M%S")
        #         backup_path = f'{BASE_PATH}_backup_{backup_timestamp}.json'
        #         shutil.copy2(ORIGINAL_PATH, backup_path)
        #         print(f"Backed up existing file to: {backup_path}")
            
        #     # 生成新的带时间戳的文件
        #     timestamp = time.strftime("%Y%m%d_%H%M%S")
        #     OUTPUT_PATH = f'{BASE_PATH}_{timestamp}.json'
        #     print(f"Writing detection data to {OUTPUT_PATH}")
        #     os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        #     data_info_viz = {x
        #         # basic infos
        #         'is_first_frame': curr_token in self.seq_first_token,
        #         'timestamp': item,
        #         'sample_token': curr_token,
        #         'seq_id': self.seq_id,
        #         'frame_id': self.frame_id,
        #         # high score dets
        #         'np_dets': data_info['np_dets'].tolist() if len(data_info['np_dets']) > 0 else [],
        #         'no_dets': data_info['no_dets'],
        #         'det_num': data_info['det_num'],
        #     }
        #     # 添加外观特征（如果存在）
        #     if 'np_dets_bbox2d' in data_info:
        #         data_info_viz['np_dets_bbox2d'] = data_info['np_dets_bbox2d'].tolist()
        #     if 'np_dets_bbox3d' in data_info:
        #         data_info_viz['np_dets_bbox3d'] = data_info['np_dets_bbox3d'].tolist()
        #     if 'visible_cam_id' in data_info:
        #         data_info_viz['visible_cam_id'] = data_info['visible_cam_id']
        #     # 添加低置信度检测（如果存在）
        #     if 'np_low_dets' in data_info:
        #         data_info_viz['np_low_dets'] = data_info['np_low_dets'].tolist()
        #         data_info_viz['np_low_dets_bbox2d'] = data_info['np_low_dets_bbox2d'].tolist()
        #         data_info_viz['low_visible_cam_id'] = data_info['low_visible_cam_id']
        #         data_info_viz['low_det_num'] = data_info['low_det_num']
        #     json.dump(data_info_viz, open(OUTPUT_PATH, "w"), indent=2)
        #     print(f"\n{'='*60}")
        #     print(f"Frame data saved to: {OUTPUT_PATH}")
        #     print(f"High-confidence detections: {data_info_viz['det_num']}")
        #     if 'low_det_num' in data_info_viz:
        #         print(f"Low-confidence detections: {data_info_viz['low_det_num']}")
        #     print(f"Use visualize_detections.py to visualize this frame.")
        #     print(f"{'='*60}\n")
        #     # 断点：生成 JSON 文件后停止执行
        #     pdb.set_trace()
        
        return data_info

    def __len__(self) -> int:
        return len(self.all_sample_token)
