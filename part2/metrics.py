import numpy as np
import re
import pandas as pd
import methods as mh

splits = ["train", "val", "test", "bolivia"]


def calculate_metrics(model, feature_space, run, split, output_data, target_data):
    output = output_data[f"{split}_x"]
    target = mh.apply_mask(target_data[f"{split}_mask"], target_data[f"{split}_y"])
    return {"model": model, "feature_space": feature_space, "run": run, "split": split}


def generate_metrics(experiment_name):
    model, feature_space, run = re.match(r"(.+)-(.+)\.(\d+)", experiment_name).groups()

    target_data = np.load(f"./data_cache/{feature_space}.npz")
    output_data = np.load(f"./output/{experiment_name}.npz")
    df = pd.DataFrame()
    for split in splits:
        row = calculate_metrics(model, feature_space, run, output_data, target_data)
        df = df.append(row)


def calc_total_iou(output, target):
    target = np.delete(target, -1)
    output = np.delete(output, -1)
    assert target.size == output.size
    print(target.shape, output.shape)
    length = target.size
    intersection = np.sum(output * target)
    union = np.sum(target) + np.sum(output) - intersection
    iou = (intersection + 0.0000001) / (union + 0.0000001)
    assert 0 < iou < 1

    correct = np.sum(output, where=(output == target))
    accuracy = correct / length
    assert 0 < accuracy < 1
    return iou
