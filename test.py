import yaml, argparse, time, os, json, copy, multiprocessing
from dataloader.nusc_loader import NuScenesloader
from tracking.nusc_tracker import Tracker
from data.script.NUSC_CONSTANT import *
from utils.io import load_file
from typing import List
from tqdm import tqdm
import pdb

parser = argparse.ArgumentParser()
# running configurations
parser.add_argument('--process', type=int, default=1)
# paths
localtime = ''.join(time.asctime(time.localtime(time.time())).split(' '))
parser.add_argument('--nusc_path', type=str, default='/data1/wyt_dataset1/nuscenes/')
parser.add_argument('--config_path', type=str, default='config/nusc_config.yaml')
parser.add_argument('--detection_path', type=str, default='data/detector/val/val_centerpoint_new.json')
parser.add_argument('--first_token_path', type=str, default='data/utils/first_token_table/trainval/nusc_first_token.json')
parser.add_argument('--end_token_path', type=str, default='data/utils/end_token_table/trainval/nusc_end_token.json')
parser.add_argument('--result_path', type=str, default='result/tmux_0')
parser.add_argument('--eval_path', type=str, default='eval_results/eval_result_0/')
parser.add_argument('--all_eval_path', type=str, default='eval_results/linear_search_parameters/')
args = parser.parse_args()

def pre(shared_variable, nusc_loader):
    for i in range(len(nusc_loader)):
        if i != 0:
            tra_done.wait()
            tra_done.clear()
        shared_variable.put(nusc_loader[i])

def tra(shared_variable, nusc_loader):
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

def linear_search_parameters(interval: List, step: float, parameter: str = 'first_thre', sort_metric: str = 'amota'):
    """
    linear search specific parameters based on the AMOTA in the nuScenes val set.
    :param interval: List, range of to be fine-tune parameters, [left, right]
    :param step: float, linear search step size
    :param parameter: str, the name of parameter, 'first_thre'
    :param sort_metric: str, the metric used to sort, default 'amota'
    :return: parameter under the best performance
    """
    assert interval[0] <= interval[1], "invalid range."
    # assert parameter in ['first_thre', 'SF_thre', 'SCALE', 'NMS_thre', 'tp_thre', 'decay_rate'], \
    #     "To-be fine-tune parameters invalid"

    all_results, all_configs, best_configs = {}, {}, {}
    left, right, step = int(interval[0] * 100), int(interval[1] * 100), int(step * 100)
    root_path = args.all_eval_path
    best_cfg_path = os.path.join(root_path, f'best_{parameter}.json')

    os.makedirs(args.all_eval_path, exist_ok=True)
    module_name, sub_cfg_name, para_name, raw_config_path = 'preprocessing', None, parameter, args.config_path

    # load raw config
    config = copy.deepcopy(yaml.load(open(raw_config_path, 'r'), Loader=yaml.Loader))

    # iterative experiments
    for iter_num, epoch in enumerate(range(left, right, step)):
        paras = (left + iter_num * step) / 100
        config_name, config_path = f'config{epoch}.json', f'{root_path}/configs/'
        eval_path, res_path = f'{root_path}/eval_results/eval{epoch}/', f'{root_path}/results/result{epoch}'
        metrics_summary_path = os.path.join(eval_path, 'metrics_summary.json')

        # replace specific config with progression value
        search_cfg = config[module_name][para_name] if sub_cfg_name is None else config[module_name][sub_cfg_name][para_name]
        for cls, _ in search_cfg.items():
            search_cfg[cls] = paras
        all_configs[epoch] = search_cfg.copy()

        # inference Poly-MOT with changed config, and save result
        if not os.path.exists(metrics_summary_path):
            run_nusc_fastpoly(config, res_path, eval_path)

        # record each epoch eval result
        all_results[epoch] = load_file(metrics_summary_path)['label_metrics'][sort_metric]

        # sort accuracy and save parameters
        for cls_label, cls_name in CLASS_STR_TO_SEG_CLASS.items():
            # list all configs and eval results
            cls_cfg = [cfg[cls_label] for _, cfg in all_configs.items()]
            cls_acc = [amotas[cls_name] for _, amotas in all_results.items()]

            # best config and amota
            cls_cfg = {
                'category': cls_name,
                'category_idx': cls_label,
                'amota': max(cls_acc),
                'exp_idx': cls_acc.index(max(cls_acc)),
                'config': cls_cfg[cls_acc.index(max(cls_acc))],
            }
            best_configs[cls_name] = cls_cfg

        best_configs['all'] = {
            'amota': sum([best_configs[cls]['amota'] for cls in CLASS_SEG_TO_STR_CLASS]) / 7
        }

        # write config under best performance
        json.dump(best_configs, open(best_cfg_path, "w"))
        print('writing best configs in folder: ' + os.path.abspath(best_cfg_path))

def run_nusc_fastpoly(config, result_path, eval_path):
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


if __name__ == "__main__":
    # single inference, load and save config
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.Loader)

    tra_done = multiprocessing.Event()
    # run Fast-Poly
    run_nusc_fastpoly(config, args.result_path, args.eval_path)

    # multi inference, linear search parameters
    # linear_search_parameters([1, 11], 1, 'voxel_mask_size')
