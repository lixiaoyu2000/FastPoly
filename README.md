# Fast-Poly
This is the Official Repo For the Paper "Fast-Poly: A Fast Polyhedral Framework For 3D Multi-Object Tracking"

> [**Fast-Poly: A Fast Polyhedral Framework For 3D Multi-Object Tracking**](https://arxiv.org/abs/2403.13443),  
> Xiaoyu Li<sup>\*</sup>, Dedong Liu<sup>\*</sup>, Yitao Wu<sup>\*</sup>, Xian Wu<sup>\*</sup>, Jinghan Gao, Lijun Zhao         
> *arXiv technical report ([arXiv 2403.13443](https://arxiv.org/abs/2403.13443))*,  

### [paper](https://arxiv.org/abs/2403.13443) | [youtube](https://www.youtube.com/watch?v=nFmeL_PjOyA&ab_channel=LIXIAOYU) | [bilibili](https://www.bilibili.com/video/BV1iz421Z7qj/?vd_source=b170cf0cb90cd4c536ec11f67c9f6522)

## Note (Thank you for your patience)
- **The code will be uploaded after the first round of review.**

## Quick Overview

We propose Fast-Poly, a fast and effective 3D MOT method based on the Tracking-By-Detection framework. Building upon our previous work Poly-MOT, Fast-Poly integrates three core principles (Alignment, Densification, and Parallelization) to solve the real-time dilemma of filter-based methods while improving accuracy. **Fast-Poly achieves new state-of-the-art performance with 75.8\% AMOTA among all methods and can run at 34.2 FPS on the nuScenes test leaderboard.** On the Waymo dataset, Fast-Poly exhibits competitive accuracy with 63.6% MOTA and impressive inference speed (35.5 FPS).


## News

- 2024-03-20. Warm-up :fire:! The official repo and [paper](https://arxiv.org/abs/2403.13443) of Fast-Poly have been released. We will release the code soon. Welcome to follow.
- 2024-03-18. Our method ranks first among all methods on the nuScenes tracking benchmark :fire:.

## Main Results

### [nuScenes](https://www.nuscenes.org/tracking?externalData=all&mapData=all&modalities=Any)

#### 3D Multi-object tracking on nuScenes test set

 Method       | Detector      | AMOTA    | MOTA     | FPS      |   
--------------|---------------|----------|----------|----------|
 Fast-Poly    | LargeKernel3D | 75.8     | 62.8     | 34.2     |
 Poly-MOT     | LargeKernel3D | 75.4     | 62.1     | 3        |         
 
#### 3D Multi-object tracking on nuScenes val set

 Method        | Detector        | AMOTA    | MOTA     | FPS      |   
---------------|-----------------|----------|----------|----------|
 Fast-Poly     | Centerpoint     | 73.7     | 63.2     | 28.9     |  
 Poly-MOT      | Centerpoint     | 73.1     | 61.9     | 5.6      |  
 Fast-Poly     | LargeKernel3D   | 76.0     | 65.8     | 34.2     |  
 Poly-MOT      | LargeKernel3D   | 75.2     | 54.1     | 8.6      |

### [Waymo](https://waymo.com/open/challenges/2020/3d-tracking/)

### 3D Multi-object tracking on Waymo test set

 Method        | Detector        | MOTA     | FPS      |  
---------------|-----------------|----------|----------|
 Fast-Poly     | CasA            | 63.6     | 35.5     |  
 CasTrack      | CasA            | 62.6     | --       |

### 3D Multi-object tracking on Waymo val set

 Method        | Detector        | MOTA     | FPS      |   
---------------|-----------------|----------|----------|
 Fast-Poly     | CasA            | 62.3     | 35.5     |  
 CasTrack      | CasA            | 61.3     | --       |


## Contact

Welcome to follow our previous work "[Poly-MOT: A Polyhedral Framework For 3D Multi-Object Tracking](https://github.com/lixiaoyu2000/Poly-MOT/tree/main)".


## Citation
If you find this project useful in your research, please consider citing by :smile_cat::
```
@misc{li2024fastpoly,
      title={Fast-Poly: A Fast Polyhedral Framework For 3D Multi-Object Tracking}, 
      author={Xiaoyu Li and Dedong Liu and Lijun Zhao and Yitao Wu and Xian Wu and Jinghan Gao},
      year={2024},
      eprint={2403.13443},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
```
@inproceedings{li2023poly,
  title={Poly-mot: A polyhedral framework for 3d multi-object tracking},
  author={Li, Xiaoyu and Xie, Tao and Liu, Dedong and Gao, Jinghan and Dai, Kun and Jiang, Zhiqiang and Zhao, Lijun and Wang, Ke},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={9391--9398},
  year={2023},
  organization={IEEE}
}
```
