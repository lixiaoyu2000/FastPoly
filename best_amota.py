
import os
import json
from copy import deepcopy
import sys

MULTIRUN_ROOT = "experiments/max_age_sweep/20251119_225057"
class Logger(object):
    def __init__(self, filename=os.path.join(MULTIRUN_ROOT,'best_amota_log.txt')):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)   # 继续打印到屏幕
        self.log.write(message)        # 同时写入文件

    def flush(self):
        pass


# NuScenes 类别 id 对应顺序
NUSC_CLASS_ORDER = [
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "pedestrian",
    "trailer",
    "truck"
]

def load_config(run_dir):
    hydra_yaml = os.path.join(run_dir, ".hydra", "config.yaml")
    config_json = os.path.join(run_dir, "config.json")

    if os.path.exists(config_json):
        with open(config_json, "r") as f:
            return json.load(f)

    if os.path.exists(hydra_yaml):
        try:
            import yaml
            with open(hydra_yaml, "r") as f:
                return yaml.safe_load(f)
        except Exception:
            return None

    return None


def find_best_amota_per_class(multirun_root):
    best = {}

    for root, dirs, files in os.walk(multirun_root):

        if "metrics_summary.json" not in files:
            continue

        metric_path = os.path.join(root, "metrics_summary.json")

        with open(metric_path, "r") as f:
            metrics = json.load(f)

        class_amota = metrics["label_metrics"]["amota"]
        config = load_config(root)

        for cls, value in class_amota.items():
            if cls not in best or value > best[cls]["amota"]:
                best[cls] = {
                    "amota": value,
                    "run_dir": root,
                    "config": config
                }

    return best


def merge_configs(best_dict):
    """
    合并 config:
    - 公共部分保持一致
    - life_cycle.basic.max_age 按类别编号替换为最优值
    """

    # 复制基础 config（随便选一个即可）
    any_cls = next(iter(best_dict))
    base = deepcopy(best_dict[any_cls]["config"])

    merged_max_age = {}

    # 遍历类别编号 0~6
    for idx, cls in enumerate(NUSC_CLASS_ORDER):
        cfg = best_dict[cls]["config"]

        # 取该类别所在编号的值，例如 bus 是类别1 → max_age["1"]
        merged_max_age[idx] = cfg["life_cycle"]["basic"]["max_age"][str(idx)]

    # 更新回 base config
    base["life_cycle"]["basic"]["max_age"] = merged_max_age

    return base


def main():
    best = find_best_amota_per_class(MULTIRUN_ROOT)

    print("\n=== 分类别最佳 AMOTA ===")
    for cls, data in best.items():
        print(f"{cls:12s}  AMOTA={data['amota']:.4f}  |  run={data['run_dir']}")

    merged = merge_configs(best)

    print("\n=== 合并后的 config(编号格式) ===")
    print(json.dumps(merged["life_cycle"]["basic"]["max_age"], indent=2))

    # 平均值
    amota_avg = sum(d["amota"] for d in best.values()) / len(best)
    print(f"\n=== AMOTA 平均值: {amota_avg:.4f} ===")


if __name__ == "__main__":
    sys.stdout = Logger()
    main()
