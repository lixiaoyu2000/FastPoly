# Fast-Poly
This is the Official Repo For the Paper "Fast-Poly: A Fast Polyhedral Framework For 3D Multi-Object Tracking"

> [**Fast-Poly: A Fast Polyhedral Framework For 3D Multi-Object Tracking**](pending),  
> Xiaoyu Li<sup>\*</sup>, Dedong Liu<sup>\*</sup>, Yitao Wu<sup>\*</sup>, Xian Wu<sup>\*</sup>, Jinghan Gao, Lijun Zhao         
> *arXiv technical report ([pending](pending))*,  

### [paper](pending) | [video](https://www.youtube.com/watch?v=nFmeL_PjOyA&ab_channel=LIXIAOYU)

## Note (Thank you for your patience)
- **Please note that the paper will be uploaded at 8pm ET on March 20th because the article is being processed by Arxiv.** 
- **The code will be uploaded after the first round of review.**
- Quantitative visualization is available now at [URL](https://www.youtube.com/watch?v=nFmeL_PjOyA&ab_channel=LIXIAOYU).

## Quick Overview

We propose Fast-Poly, a fast and effective 3D MOT method based on the Tracking-By-Detection framework. Building upon our previous work Poly-MOT, Fast-Poly integrates three core principles (Alignment, Densification, and Parallelization) to solve the real-time dilemma of filter-based methods while improving accuracy. **Fast-Poly achieves new state-of-the-art performance with 75.8\% AMOTA among all methods and can run at 34.2 FPS on the nuScenes test leaderboard.** On the Waymo dataset, Fast-Poly exhibits competitive accuracy with 63.6% MOTA and impressive inference speed (35.5 FPS).


## News

- 2024-03-20. Warm-up :fire:! The official repo and [paper](pending) of Fast-Poly have been released. We will release the code soon. Welcome to follow.
- 2024-03-18. Our method ranks first among all methods on the nuScenes tracking benchmark :fire:.

## Main Results

### nuScenes

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

### Waymo

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

Welcome to follow our previous work '[Poly-MOT: A Polyhedral Framework For 3D Multi-Object Tracking](https://github.com/lixiaoyu2000/Poly-MOT/tree/main)'.


## Citation
If you find this project useful in your research, please consider citing by :smile_cat::
```
**pending**
```
```
@misc{li2023polymot,
      title={Poly-MOT: A Polyhedral Framework For 3D Multi-Object Tracking}, 
      author={Xiaoyu Li and Tao Xie and Dedong Liu and Jinghan Gao and Kun Dai and Zhiqiang Jiang and Lijun Zhao and Ke Wang},
      year={2023},
      eprint={2307.16675},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
