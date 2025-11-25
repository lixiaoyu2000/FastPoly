#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化检测结果脚本
从 JSON 文件中读取检测数据，并在 NuScenes 图片上绘制检测框
支持高置信度和低置信度检测的可视化
# 使用默认输出目录
python visualize_detections.py \
    --json /data/wyt_dataset/gy_code/FastPoly/debug_output/45eda670b5a840acbb730aac15e63b19_20260102_170611.json \
    --nusc_root /data1/wyt_dataset1/nuscenes 
    --output /tmp12/vis_results
"""

import cv2
import os
import numpy as np
import json
import argparse
from nuscenes.nuscenes import NuScenes
from data.script.NUSC_CONSTANT import USED_ALLCAM


def visualize_low_score_detections(nusc, sample_token, data_info, save_dir='/tmp/visualization'):
    """
    可视化特定token的低分2D检测框（np_low_dets_bbox2d）
    
    Args:
        nusc: NuScenes实例
        sample_token: 当前样本token
        data_info: 包含检测信息的字典，必须包含 'np_low_dets_bbox2d' 和 'low_visible_cam_id'
        save_dir: 保存可视化结果的目录
    
    Returns:
        list: 保存的图片文件路径列表
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 检查是否有低分检测框
    if 'np_low_dets_bbox2d' not in data_info or 'low_det_num' not in data_info or data_info['low_det_num'] == 0:
        print(f"Token {sample_token}: 没有低分检测框可可视化")
        return []
    
    # 获取样本数据
    sample = nusc.get('sample', sample_token)
    
    # 获取低分检测框信息
    low_det_bbox2d = np.array(data_info['np_low_dets_bbox2d'])  # [low_det_num, 6, 4]
    low_visible_cam_ids = data_info.get('low_visible_cam_id', [])
    
    # 获取检测分数和类别（如果存在）
    if 'np_low_dets' in data_info and len(data_info['np_low_dets']) > 0:
        low_det_scores = np.array(data_info['np_low_dets'])[:, -2]  # 检测分数
        low_det_classes = np.array(data_info['np_low_dets'])[:, -1]  # 检测类别
    else:
        low_det_scores = np.zeros(len(low_det_bbox2d))
        low_det_classes = np.zeros(len(low_det_bbox2d))
    
    print(f"Token {sample_token}: 可视化 {len(low_det_bbox2d)} 个低分检测框")
    
    saved_paths = []
    
    # 为每个相机通道可视化
    for cam_idx, cam_channel in enumerate(USED_ALLCAM):
        # 获取相机数据
        cam_token = sample['data'][cam_channel]
        cam_data = nusc.get('sample_data', cam_token)
        
        # 获取图像路径
        img_path = os.path.join(nusc.dataroot, cam_data['filename'])
        
        # 读取图像
        if not os.path.exists(img_path):
            print(f"警告: 图像文件不存在: {img_path}")
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图像: {img_path}")
            continue
        
        img_height, img_width = img.shape[:2]
        
        # 在当前相机通道上绘制低分检测框
        drawn_count = 0
        for det_idx in range(len(low_det_bbox2d)):
            # 检查该检测是否在当前相机可见
            if low_visible_cam_ids and det_idx < len(low_visible_cam_ids):
                if cam_idx not in low_visible_cam_ids[det_idx]:
                    continue  # 该检测在当前相机不可见，跳过
            
            # 获取当前检测在当前相机上的2D框
            bbox = low_det_bbox2d[det_idx, cam_idx]
            
            # 检查框是否有效（坐标不为-1）
            if bbox[0] == -1 or bbox[1] == -1 or bbox[2] == -1 or bbox[3] == -1:
                continue
            
            # 确保框坐标在图像范围内
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(int(x1), img_width - 1))
            y1 = max(0, min(int(y1), img_height - 1))
            x2 = max(0, min(int(x2), img_width - 1))
            y2 = max(0, min(int(y2), img_height - 1))
            
            # 确保框有效
            if x2 <= x1 or y2 <= y1:
                continue
            
            # 获取检测信息
            score = low_det_scores[det_idx] if det_idx < len(low_det_scores) else 0.0
            class_id = int(low_det_classes[det_idx]) if det_idx < len(low_det_classes) else -1
            
            # 使用橙色表示低置信度检测
            color = (0, 165, 255)  # BGR格式：橙色
            
            # 绘制检测框（使用较细的线条表示低置信度）
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # 添加标签（类别和分数）
            label = f"Low{det_idx} Cls{class_id}: {score:.2f}"
            font_scale = 0.5
            thickness = 1
            
            # 计算文本大小
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # 绘制文本背景
            cv2.rectangle(
                img,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # 绘制文本
            cv2.putText(
                img,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness
            )
            
            drawn_count += 1
        
        # 保存可视化结果
        save_path = os.path.join(save_dir, f"{sample_token}_{cam_channel}_low.jpg")
        cv2.imwrite(save_path, img)
        saved_paths.append(save_path)
        print(f"已保存低分检测可视化结果: {save_path} (绘制了 {drawn_count} 个检测框)")
    
    return saved_paths


def visualize_high_score_detections(nusc, sample_token, data_info, save_dir='/tmp/visualization'):
    """
    可视化特定token的高分2D检测框
    Args:
        nusc: NuScenes实例
        sample_token: 当前样本token
        data_info: 包含检测信息的字典
        save_dir: 保存可视化结果的目录
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    # 获取样本数据
    sample = nusc.get('sample', sample_token)
    # 检查是否有高分检测框
    if 'np_dets_bbox2d' not in data_info or len(data_info['np_dets']) == 0:
        print(f"Token {sample_token}: 没有高分检测框可可视化")
        return
    # 获取检测框信息
    det_bbox2d = data_info['np_dets_bbox2d']  # [det_num, 6, 4]
    det_scores = data_info['np_dets'][:, -2]  # 检测分数
    det_classes = data_info['np_dets'][:, -1]  # 检测类别
    print(f"Token {sample_token}: 可视化 {len(det_bbox2d)} 个高分检测框")
    # 为每个相机通道可视化
    for cam_idx, cam_channel in enumerate(USED_ALLCAM):
        # 获取相机数据
        cam_token = sample['data'][cam_channel]
        cam_data = nusc.get('sample_data', cam_token)
        # 获取图像路径
        img_path = os.path.join(nusc.dataroot, cam_data['filename'])
        # 读取图像
        if not os.path.exists(img_path):
            print(f"警告: 图像文件不存在: {img_path}")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图像: {img_path}")
            continue
        img_height, img_width = img.shape[:2]
        # 在当前相机通道上绘制检测框
        for det_idx in range(len(det_bbox2d)):
            # 获取当前检测在当前相机上的2D框
            bbox = det_bbox2d[det_idx, cam_idx]
            # 检查框是否有效（坐标不为-1）
            if bbox[0] == -1 or bbox[1] == -1 or bbox[2] == -1 or bbox[3] == -1:
                continue
            # 确保框坐标在图像范围内
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(int(x1), img_width - 1))
            y1 = max(0, min(int(y1), img_height - 1))
            x2 = max(0, min(int(x2), img_width - 1))
            y2 = max(0, min(int(y2), img_height - 1))
            # 确保框有效
            if x2 <= x1 or y2 <= y1:
                continue
            # 获取检测信息
            score = det_scores[det_idx]
            class_id = int(det_classes[det_idx])
            # 根据类别选择颜色（可自定义）
            colors = [
                (0, 255, 0),    # 类别0: 绿色
                (255, 0, 0),    # 类别1: 蓝色
                (0, 0, 255),    # 类别2: 红色
                (255, 255, 0),  # 类别3: 青色
                (255, 0, 255),  # 类别4: 紫色
                (0, 255, 255),  # 类别5: 黄色
            ]
            color = colors[class_id % len(colors)]
            # 绘制检测框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            # 添加标签（类别和分数）
            label = f"Cls{class_id}: {score:.2f}"
            font_scale = 0.5
            thickness = 1
            # 计算文本大小
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            # 绘制文本背景
            cv2.rectangle(
                img, 
                (x1, y1 - text_height - baseline - 5), 
                (x1 + text_width, y1), 
                color, 
                -1
            )
            # 绘制文本
            cv2.putText(
                img, 
                label, 
                (x1, y1 - baseline - 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (255, 255, 255), 
                thickness
            )
        # 保存可视化结果
        save_path = os.path.join(save_dir, f"{sample_token}_{cam_channel}.jpg")
        cv2.imwrite(save_path, img)
        print(f"已保存可视化结果: {save_path}")


def visualize_from_json(json_path, nusc_root_path, save_dir='/tmp/visualization', show_low_dets=True):
    """
    从保存的JSON文件可视化检测结果
    
    Args:
        json_path: JSON文件路径
        nusc_root_path: NuScenes数据集根目录
        save_dir: 保存可视化结果的目录
        show_low_dets: 是否可视化低置信度检测（默认：True）
    
    Returns:
        dict: 包含保存的图片文件路径的字典
            {
                'high_score': [高置信度检测图片路径列表],
                'low_score': [低置信度检测图片路径列表]
            }
    """
    # 加载JSON数据
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 初始化NuScenes
    nusc = NuScenes(version='v1.0-trainval', dataroot=nusc_root_path, verbose=False)
    sample_token = data['sample_token']
    
    # 转换数据类型（JSON中的列表需要转回numpy数组）
    if 'np_dets_bbox2d' in data:
        data['np_dets_bbox2d'] = np.array(data['np_dets_bbox2d'])
    if 'np_dets' in data:
        data['np_dets'] = np.array(data['np_dets'])
    if 'np_low_dets_bbox2d' in data:
        data['np_low_dets_bbox2d'] = np.array(data['np_low_dets_bbox2d'])
    if 'np_low_dets' in data:
        data['np_low_dets'] = np.array(data['np_low_dets'])
    
    result_paths = {'high_score': [], 'low_score': []}
    
    # 可视化高置信度检测
    if 'np_dets_bbox2d' in data and len(data.get('np_dets', [])) > 0:
        visualize_high_score_detections(nusc, sample_token, data, save_dir)
        # 收集高置信度检测图片路径
        for cam_channel in USED_ALLCAM:
            high_path = os.path.join(save_dir, f"{sample_token}_{cam_channel}.jpg")
            if os.path.exists(high_path):
                result_paths['high_score'].append(high_path)
    
    # 可视化低置信度检测
    if show_low_dets:
        low_paths = visualize_low_score_detections(nusc, sample_token, data, save_dir)
        result_paths['low_score'] = low_paths
    
    return result_paths


def main():
    parser = argparse.ArgumentParser(description='Visualize detection results from JSON file')
    parser.add_argument('--json', type=str, required=True,
                        help='Path to JSON file containing detection data')
    parser.add_argument('--nusc_root', type=str, required=True,
                        help='Root path of NuScenes dataset')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: same as JSON file directory)')
    parser.add_argument('--show_low_dets', action='store_true', default=True,
                        help='Show low-confidence detections (default: True)')
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output is None:
        json_dir = os.path.dirname(os.path.abspath(args.json))
        args.output = os.path.join(json_dir, 'visualization')
    
    # 可视化并获取图片路径
    result_paths = visualize_from_json(args.json, args.nusc_root, args.output, args.show_low_dets)
    
    # 输出结果
    print(f"\n{'='*60}")
    print(f"可视化完成！")
    print(f"高置信度检测图片: {len(result_paths['high_score'])} 张")
    print(f"低置信度检测图片: {len(result_paths['low_score'])} 张")
    print(f"保存目录: {args.output}")
    print(f"{'='*60}")
    
    # 返回图片路径（用于脚本调用）
    return result_paths


if __name__ == "__main__":
    result_paths = main()
    # 打印所有图片路径
    print("\n所有保存的图片路径:")
    print("高置信度检测:")
    for path in result_paths['high_score']:
        print(f"  {path}")
    print("低置信度检测:")
    for path in result_paths['low_score']:
        print(f"  {path}")
