"""
3d Box Class inherit from nuscenes.utils.data_classes.box
"""
import pdb

import numpy as np
from typing import List, Tuple
from pyquaternion import Quaternion
from data.script.NUSC_CONSTANT import *
from nuscenes.utils.data_classes import Box


class NuscBox(Box):
    def __init__(self, center: List[float], size: List[float], rotation: List[float], label: int = np.nan,
                 score: float = np.nan, velocity: Tuple = (np.nan, np.nan, np.nan), name: str = None,
                 token: str = None, init_geo: bool = True):
        """
        following notes are from nuscenes.utils.data_classes.box
        :param center: Center of box given as x, y, z.
        :param size: Size of box in width, length, height.
        :param rotation: Box orientation.
        :param label: Integer label, optional.
        :param score: Classification score, optional.
        :param velocity: Box velocity in x, y, z direction.
        :param name: Box name, optional. Can be used e.g. for denote category name.
        :param token: Unique string identifier from DB.
        """
        super().__init__(center, size, self.abs_orientation_axisZ(Quaternion(rotation)), 
                         label, score, velocity, name, token)
        assert self.orientation.axis[-1] >= 0

        self.yaw = self.orientation.radians
        self.name_label = CLASS_SEG_TO_STR_CLASS[name]
        self.tracking_id, self.corners_, self.bottom_corners_ = None, None, None
        self.volume, self.area, self.norm_corners_ = None, None, None

        if init_geo:
            self.corners_ = self.corners()
            self.bottom_corners_ = self.corners_[:, [2, 3, 7, 6]][:2].T   # [4, 2]
            self.volume, self.area, self.norm_corners_ = self.box_volum(), self.box_bottom_area(), self.norm_corners()

    @staticmethod
    def abs_orientation_axisZ(orientation: Quaternion) -> Quaternion:
        # Double Cover, align with subsequent motion model
        return -orientation if orientation.axis[-1] < 0 else orientation

    def box_volum(self) -> float:
        return self.wlh[0] * self.wlh[1] * self.wlh[2]

    def box_bottom_area(self) -> float:
        return self.wlh[0] * self.wlh[1]

    def norm_corners(self) -> np.ndarray:
        """
        get normlized box corners in the global frame, implenment for Fast-Giou
        :return: normlized box corners
        """
        top, left = np.min(self.bottom_corners_[:, 0]), np.max(self.bottom_corners_[:, 1])
        bottom, right = np.max(self.bottom_corners_[:, 0]), np.min(self.bottom_corners_[:, 1])
        return np.array([top, right, bottom, left])

    def reset_box_infos(self):
        """
        the geometry infos need reset if self.wlh or self.center are changed
        """
        self.corners_ = self.corners()
        self.bottom_corners_ = self.corners_[:, [2, 3, 7, 6]][:2].T  # [4, 2]
        self.volume, self.area, self.norm_corners_ = self.box_volum(), self.box_bottom_area(), self.norm_corners()

    def __repr__(self):
        repr_str = super().__repr__() + ', tracking id: {}'
        return repr_str.format(self.tracking_id)
