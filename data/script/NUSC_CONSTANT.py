"""
Public constant accessible to all files
"""

import numpy as np

# metrics with two return values
METRIC = ['iou_3d', 'giou_3d', 'giou_2d', 'a_giou_3d', 'a_iou_3d']
FAST_METRIC, FAST_NORM_METRIC = ['giou_3d', 'giou_bev'], ['a_giou_3d', 'a_giou_bev']
APP_METRIC = ['iou_2d', 'giou_2d']
ALL_METRIC = ['iou_bev', 'iou_3d', 'giou_bev', 'giou_3d', 'd_eucl', 'a_giou_3d', 'a_giou_bev', 'a_iou_bev']

# category name(str) <-> category label(int)
CLASS_SEG_TO_STR_CLASS = {'bicycle': 0, 'bus': 1, 'car': 2, 'motorcycle': 3, 'pedestrian': 4, 'trailer': 5, 'truck': 6}
CLASS_STR_TO_SEG_CLASS = {0: 'bicycle', 1: 'bus', 2: 'car', 3: 'motorcycle', 4: 'pedestrian', 5: 'trailer', 6: 'truck'}

# math
PI, TWO_PI = np.pi, 2 * np.pi

# init EKFP for different non-linear motion model
CTRA_INIT_EFKP = {
    # [x, y, z, w, l, h, v, a, theta, omega]
    'bus': [10, 10, 10, 10, 10, 10, 10, 10, 1000, 10],
    'car': [4, 4, 4, 4, 4, 4, 1000, 4, 1, 0.1],
    'trailer': [10, 10, 10, 10, 10, 10, 10, 10, 1000, 10],
    'truck': [10, 10, 10, 10, 10, 10, 10, 10, 1000, 10],
    'pedestrian': [10, 10, 10, 10, 10, 10, 10, 10, 1000, 10]
}
CTRV_INIT_EFKP = {
    # [x, y, z, w, l, h, v, a, theta, omega]
    'bus': [10, 10, 10, 10, 10, 10, 10, 1000, 10],
    'car': [4, 4, 4, 4, 4, 4, 1000, 1, 0.1],
    'trailer': [10, 10, 10, 10, 10, 10, 10, 1000, 10],
    'truck': [10, 10, 10, 10, 10, 10, 10, 1000, 10],
    'pedestrian': [10, 10, 10, 10, 10, 10, 10, 1000, 10]
}
BIC_INIT_EKFP = {
    # [x, y, z, w, l, h, v, a, theta, sigma]
    'bicycle': [10, 10, 10, 10, 10, 10, 10000, 10, 10, 10],
    'motorcycle': [4, 4, 4, 4, 4, 4, 100, 4, 4, 1],
}

FULL_CONSTANT_MODEL = ['CA', 'CV', 'CTRA', 'CTRV']

FINETUNE_Q = {1: 0.49, 0: 0.81, 2: 0.49, 3: 0.25, 4: 0.36, 5: 0.04, 6: 0.49}
FINETUNE_R = {1: 1, 0: 1, 2: 1, 3: 1, 4: 1, 5: 0.001, 6: 1}

# score estimatation method
SCORE_PREDICT = ['Minus', 'Normal', 'Prob']
SCORE_UPDATE = ['Multi', 'Parallel', 'Normal', 'Prob']