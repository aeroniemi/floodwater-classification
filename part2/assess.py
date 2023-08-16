import methods as mh
import os
from feature_spaces import *
from alive_progress import alive_bar
import pandas as pd
import re

cache_path = f"./data_cache/{feature_space}.npz"
output_path = f"./output/"
if not os.path.isfile(cache_path):
    raise Exception("Feature space not found in cache")

x_features = Feature.getMany([feature_space])

with np.load(cache_path) as file, alive_bar(
    1, title="Loading dataset from cache"
) as bar:
    dataset = file
    bar()


def calculate_metrics(file_name):
    df = pd.DataFrame(
        columns=(
            "model",
            "feature_space",
            "run",
            "split",
            "mean_iou",
            "mean_iou_std",
            "total_iou",
            "mean_accuracy",
            "total_accuracy",
            "total_precision_flooded",
            "total_recall_flooded",
            "total_recall_dry",
            "total_f1_score",
        )
    )
    model, feature_space, run = re.match("(.*?)-(.+?)\.(\d+)\.npy", file_name)
    output_file = np.load(output_path + file_name)
    for split in ["train", "val", "bolivia"]:
        metrics = {
            "model": model,
            "feature_space": feature_space,
            "run": run,
            "split": split,
        }
        output = output_file[f"{split}_output"]
        target = dataset[f"{split}_y"]
        
