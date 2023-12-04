#----------------------------------------------------------------------------
# Generate Metrics
# Script to create a sheet of a variety of metrics for assessing model perf 
# @aeroniemi / Alex Beavil 2023
#----------------------------------------------------------------------------

import numpy as np
import re
import pandas as pd
import methods as mh
import sklearn.metrics as skm
import statistics
import os
import glob
splits = ["test", "bolivia"]
# splits = ["train", "val", "test", "bolivia"]

def assign_metric_to_region(regions, i, image_split_list, value):
    image = image_split_list[i]
    region = re.search(r"^(.+)_", image).group(1)
    if not region in regions:
        regions[region] = []
    regions[region].append(value)



def calc_regional_iou(output, target, split):
    regions = {}
    with open(f"../downloaded_data/flood_{split}_data.csv", 'r') as file:
        image_split_list = file.readlines()
    for i, image in enumerate(zip(output, target)):
        value = calc_iou(*image)
        assign_metric_to_region(regions, i, image_split_list, value)
    return regions
def calc_regional_wet(output, target, split):
    regions = {}
    with open(f"../downloaded_data/flood_{split}_data.csv", 'r') as file:
        image_split_list = file.readlines()
    for i, image in enumerate(zip(output, target)):
        value = np.count_nonzero(image[1]==1)
        assign_metric_to_region(regions, i, image_split_list, value)
    return regions



def calculate_metrics(model, feature_space, run, split, output_data, target_data):
    output = output_data[f"{split}_y"]
    target = mh.apply_mask(target_data[f"{split}_mask"], fy=target_data[f"{split}_y"])
    print(output.shape, target.shape, split)
    # print(output.shape)
    # print(list(map(lambda x: x.shape, output)))
    output_1d = np.concatenate(output)
    target_1d = np.concatenate( target)
    return {
        "model": model,
        "feature_space": feature_space,
        "run": run,
        "split": split,
        "mean_iou_flooded": calc_mean_iou(output, target),
        "std_iou_flooded": calc_std_iou(output, target),
        "total_iou_flooded": calc_iou(output_1d, target_1d),
        "mean_accuracy": calc_mean_accuracy(output, target),
        "std_accuracy": calc_std_accuracy(output, target),
        "total_accuracy": skm.accuracy_score(target_1d, output_1d),
        "total_precision_flooded": skm.precision_score(
            target_1d, output_1d, pos_label=1
        ),
        "total_recall_flooded": skm.recall_score(target_1d, output_1d, pos_label=1),
        "total_recall_dry": skm.recall_score(target_1d, output_1d, pos_label=0),
        "total_f1_score": skm.f1_score(target_1d, output_1d),
        "regional_iou": calc_regional_iou(output, target, split),
        "regional_wet": calc_regional_wet(output, target, split)
    }
metrics_path = "../metrics_outputs/metrics.output"

def generate_metrics(file):
    print(file)
    model, feature_space, run = re.search(r"/(\w+)-(.+)\.(\d+)\.npz", file).groups()

    target_data = np.load(f"../data_cache/{feature_space}.npz", allow_pickle=True)
    output_data = np.load(file, allow_pickle=True)

    for split in splits:
        row = pd.DataFrame([calculate_metrics(
            model, feature_space, run, split, output_data, target_data
        )])
        print(f"Saving results for {split}")
        if not os.path.isfile(metrics_path):
            row.to_csv(metrics_path, sep="\t", index=False)
        else:
            source = pd.read_csv(metrics_path, sep="\t")
            res = pd.concat([source, row], axis=0)
            res.to_csv(metrics_path, sep="\t", index=False)


def calc_iou(output, target):
    output = np.ravel(output)
    target = np.ravel(target)
    intersection = np.sum(output * target)
    union = np.sum(target) + np.sum(output) - intersection
    iou = (intersection + 0.0000001) / (union + 0.0000001)
    assert 0 <= iou <= 1
    return iou


def calc_mean_iou(output, target):
    list_iou = list(map(lambda x: calc_iou(*x), zip(output, target)))
    return statistics.mean(list_iou)


def calc_std_iou(output, target):
    list_iou = list(map(lambda x: calc_iou(*x), zip(output, target)))
    return statistics.stdev(list_iou)


def calc_mean_accuracy(output, target):
    list_accuracy = list(map(lambda x: skm.accuracy_score(*x), zip(target, output)))
    return statistics.mean(list_accuracy)


def calc_std_accuracy(output, target):
    list_accuracy = list(map(lambda x: skm.accuracy_score(*x), zip(target, output)))
    return statistics.stdev(list_accuracy)


if __name__ == "__main__":
    for file in glob.glob("../model_outputs/*.npz"):
        print(f"Generating metrics for {file}")
        generate_metrics(file)
