import yaml, time, os, json, copy, multiprocessing
from dataloader.nusc_loader import NuScenesloader
from tracking.nusc_tracker import Tracker
from data.script.NUSC_CONSTANT import *
from utils.io import load_file
from typing import List
from tqdm import tqdm
import pdb
import hydra  
import time
from omegaconf import DictConfig, OmegaConf


def pre(shared_variable, nusc_loader,tra_done):
    for i in range(len(nusc_loader)):
        if i != 0:
            tra_done.wait()
            tra_done.clear()
        shared_variable.put(nusc_loader[i])

def tra(shared_variable, nusc_loader,tra_done,args):
    # PolyMOT modal is completely dependent on the detector modal
    # args = parser()
    result = {
        "results": {},
        "meta": {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }
    }
    # tracking and output file
    with open(args.end_token_path, 'r') as f:
        is_end = json.load(f)
    nusc_tracker = Tracker(config=nusc_loader.config)
    n = len(nusc_loader)
    while n > 0:
        # predict all valid trajectories
        nusc_tracker.tras_predict()
                
        frame_data = shared_variable.get()
        nusc_tracker.det_infos, nusc_tracker.frame_id, nusc_tracker.seq_id = frame_data, frame_data['frame_id'], frame_data['seq_id']
        sample_token = frame_data['sample_token']
        # track each sequence
        nusc_tracker.tracking(frame_data)
        tra_done.set()
        nusc_tracker.frame_id += 1
        """
        only for debug
        {
            'np_track_res': np.array, [num, 17] add 'tracking_id', 'seq_id', 'frame_id'
            'box_track_res': np.array[NuscBox], [num,]
            'no_val_track_result': bool
        }
        """
        # output process
        sample_results = []
        if 'no_val_track_result' not in frame_data:
            for predict_box in frame_data['box_track_res']:
                box_result = {
                    "sample_token": sample_token,
                    "translation": [float(predict_box.center[0]), float(predict_box.center[1]),
                                    float(predict_box.center[2])],
                    "size": [float(predict_box.wlh[0]), float(predict_box.wlh[1]), float(predict_box.wlh[2])],
                    "rotation": [float(predict_box.orientation[0]), float(predict_box.orientation[1]),
                                 float(predict_box.orientation[2]), float(predict_box.orientation[3])],
                    "velocity": [float(predict_box.velocity[0]), float(predict_box.velocity[1])],
                    "tracking_id": str(predict_box.tracking_id),
                    "tracking_name": predict_box.name,
                    "tracking_score": predict_box.score,
                }
                sample_results.append(box_result.copy())

        # add to the output file
        if sample_token in result["results"]:
            result["results"][sample_token] = result["results"][sample_token] + sample_results
        else:
            result["results"][sample_token] = sample_results

        n -= 1
        if sample_token in is_end: nusc_tracker.reset()

    # sort track result by the tracking score
    for sample_token in result["results"].keys():
        confs = sorted(
            [
                (-d["tracking_score"], ind)
                for ind, d in enumerate(result["results"][sample_token])
            ]
        )
        result["results"][sample_token] = [
            result["results"][sample_token][ind]
            for _, ind in confs[: min(500, len(confs))]
        ]

    # write file
    json.dump(result, open(args.result_path + "/results.json", "w"))

def main(result_path, token, process, nusc_loader):
    # PolyMOT modal is completely dependent on the detector modal
    result = {
        "results": {},
        "meta": {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }
    }
    # tracking and output file
    nusc_tracker = Tracker(config=nusc_loader.config)
    for frame_data in tqdm(nusc_loader, desc='Running', total=len(nusc_loader) // process, position=token):
        if process > 1 and frame_data['seq_id'] % process != token:
            continue
        sample_token = frame_data['sample_token']
        # track each sequence
        nusc_tracker.tracking(frame_data)
        """
        only for debug
        {
            'np_track_res': np.array, [num, 17] add 'tracking_id', 'seq_id', 'frame_id'
            'box_track_res': np.array[NuscBox], [num,]
            'no_val_track_result': bool
        }
        """
        # output process
        sample_results = []
        if 'no_val_track_result' not in frame_data:
            for predict_box in frame_data['box_track_res']:
                box_result = {
                    "sample_token": sample_token,
                    "translation": [float(predict_box.center[0]), float(predict_box.center[1]),
                                    float(predict_box.center[2])],
                    "size": [float(predict_box.wlh[0]), float(predict_box.wlh[1]), float(predict_box.wlh[2])],
                    "rotation": [float(predict_box.orientation[0]), float(predict_box.orientation[1]),
                                 float(predict_box.orientation[2]), float(predict_box.orientation[3])],
                    "velocity": [float(predict_box.velocity[0]), float(predict_box.velocity[1])],
                    "tracking_id": str(predict_box.tracking_id),
                    "tracking_name": predict_box.name,
                    "tracking_score": predict_box.score,
                }
                sample_results.append(box_result.copy())

        # add to the output file
        if sample_token in result["results"]:
            result["results"][sample_token] = result["results"][sample_token] + sample_results
        else:
            result["results"][sample_token] = sample_results

    # sort track result by the tracking score
    for sample_token in result["results"].keys():
        confs = sorted(
            [
                (-d["tracking_score"], ind)
                for ind, d in enumerate(result["results"][sample_token])
            ]
        )
        result["results"][sample_token] = [
            result["results"][sample_token][ind]
            for _, ind in confs[: min(500, len(confs))]
        ]

    # write file
    if process > 1:
        json.dump(result, open(result_path + str(token) + ".json", "w"))
    else:
        json.dump(result, open(result_path + "/results.json", "w"))


def eval(result_path, eval_path, nusc_path):
    from nuscenes.eval.tracking.evaluate import TrackingEval
    from nuscenes.eval.common.config import config_factory as track_configs
    cfg = track_configs("tracking_nips_2019")
    nusc_eval = TrackingEval(
        config=cfg,
        result_path=result_path,
        eval_set="val",
        output_dir=eval_path,
        verbose=True,
        nusc_version="v1.0-trainval",
        nusc_dataroot=nusc_path,
    )
    print("result in " + result_path)
    metrics_summary = nusc_eval.main()



def run_nusc_fastpoly(config, result_path, eval_path,args):
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(eval_path, exist_ok=True)
    
    # save config file
    json.dump(config, open(eval_path + "/config.json", "w"))
    print('writing config in folder: ' + os.path.abspath(eval_path))

    # load dataloader
    nusc_loader = NuScenesloader(args.detection_path,
                                 args.first_token_path,
                                 config)
    print('writing result in folder: ' + os.path.abspath(result_path))

    if config["basic"]["Multiprocessing"]:
        # run on two processes
        shared_variable = multiprocessing.Queue()
        if config["basic"]["Multiprocessing"]:
             shared_variable = multiprocessing.Queue()
             tra_done = multiprocessing.Event()
        p1 = multiprocessing.Process(target=pre, args=(shared_variable, nusc_loader))
        p2 = multiprocessing.Process(target=tra, args=(shared_variable, nusc_loader))
        p1.start()
        p2.start()
        p1.join()
        p2.join()
    else:
        main(result_path, 0, 1, nusc_loader)

    print('result is written in folder: ' + os.path.abspath(result_path))

    # eval result
    eval(os.path.join(result_path, 'results.json'), eval_path, args.nusc_path)


@hydra.main(version_base=None, config_path="conf", config_name="nusc_config")  
def run(cfg): 
    config = OmegaConf.to_container(cfg, resolve=True)
    
    from types import SimpleNamespace
    args = SimpleNamespace(  process=cfg.get('process', 1),
        nusc_path=cfg.paths.get('nusc_path', '/data1/wyt_dataset1/nuscenes/'),
        detection_path=cfg.paths.detection_path,
        first_token_path=cfg.paths.first_token_path,
        end_token_path=cfg.paths.end_token_path,
        result_path=cfg.paths.get('result_path', 'result/tmux_0'),
        eval_path=cfg.paths.get('eval_path', 'eval_results/eval_result_0/'),
        all_eval_path=cfg.paths.get('all_eval_path', 'eval_results/linear_search_parameters/'
    ))
    
    # 使用 Hydra 的输出目录
    args.result_path = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, 'results')
    args.eval_path = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, 'eval')
   
    tra_done = multiprocessing.Event()
    try:
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        args.result_path = os.path.join(output_dir, 'results')
        args.eval_path = os.path.join(output_dir, 'eval')
        print(f"Output directory: {output_dir}")
    except Exception as e:
        print(f"Warning: Could not get Hydra output dir: {e}")
    #  run Fast-Poly
    run_nusc_fastpoly(config, args.result_path, args.eval_path,args)

    # multi inference, linear search parameters
    

if __name__ == "__main__":
   run()
   
   