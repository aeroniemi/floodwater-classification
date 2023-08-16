# ===========================================================================
#                            Imports
# ===========================================================================
import methods as mh
import numpy as np
from alive_progress import alive_it, alive_bar
from feature_spaces import *
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import lightgbm
import optuna
import os
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)

# ===========================================================================
#                            User editable settings
# ===========================================================================
folder = "../src/downloaded-data/label/"
classes = [-1, 0, 1]
with alive_bar(4, title="Loading file lists") as bar:
    train_images = mh.load_file_list("flood_train_data.csv")
    bar()
    valid_images = mh.load_file_list("flood_valid_data.csv")
    bar()
    tests_images = mh.load_file_list("flood_test_data.csv")
    bar()
    boliv_images = mh.load_file_list("flood_bolivia_data.csv")
    bar()
    # boliv_images = mh.filterPaths(folder, "Bolivia*")

# classifier = SGDClassifier()
# classifier = KNeighborsClassifier()
# classifier = GaussianNB()
# classifier = LGBMClassifier()
# ===========================================================================
#                            Train Classifer
# ===========================================================================
feature_space = "SAR_HSV(O3)+cAWEI+cNDWI"
x_features = Feature.getMany([feature_space])
# print(*train_images)
# for image in (bar := alive_it(train_images, title="Training")):
#     bar.text(image)
#     mh.partial_fit(classifier, classes, image, x_features)

# mh.full_fit(classifier, train_images, x_features)

# ===========================================================================
#                            Predict using Classifier
# ===========================================================================
# for image in (bar := alive_it(predict_images, title="Predicting")):
#     bar.text(image)
#     img, label, meta = mh.predict(classifier, image, x_features)
#     mh.write_geotiff(f"./predicted_{image}.tif", img.reshape((512, 512)), meta)

# image = "Bolivia_290290"

# x_features = Feature.getMany(["compositeTest"])

# x_features = Feature.getMany(["DEM", "OPT_R", "AWEI"])


# x, y, meta = mh.generate_feature_stack(image, x_features, y_features)

# print(x[:, :5])

# ---------------------------------------------------------------------------
#                            autotune
# ---------------------------------------------------------------------------
cache_path = f"./data_cache/{feature_space}.npz"
if os.path.isfile(cache_path):
    print("Loading dataset from cache")
    with np.load(cache_path) as file:
        train_x = file["train_x"]
        train_y = file["train_y"]
else:
    print("Generating dataset")
    train_x, train_y, _, _ = mh.generate_dataset(train_images, x_features)

    print("Saving dataset to cache")
    np.savez(cache_path, train_x=train_x, train_y=train_y)
    # np.savez(
    #     cache_path,
    #     train_x=train_x,
    #     train_y=train_y,
    #     val_x=val_x,
    #     val_y=val_y,
    #     val_x_stack=val_x_stack,
    #     val_y_stack=val_y_stack,
    # )
    print("Save complete")
val_x, val_y, val_x_stack, val_y_stack = mh.generate_dataset(valid_images, x_features)

scaler = StandardScaler().fit(train_x)
train_x = scaler.transform(train_x)
val_x = scaler.transform(val_x)
print(list(map(lambda x: x.shape, val_x_stack)))
val_x_stack = list(map(scaler.transform, val_x_stack))


def SGDObjective(trial: optuna.Trial):
    trial.suggest_categorical(
        "loss", ["hinge", "log_loss", "modified_huber", "squared_hinge"]
    )
    trial.suggest_int("alpha", 1, 4, 1)
    # trial.suggest_categorical("average", [True, False])
    # trial.suggest_categorical("classifier", ["SVC", "RandomForest"])
    classifier = SGDClassifier(**trial.params)
    mh.full_fit(classifier, train_images, x_features)
    iou = mh.calc_mean_iou(classifier, valid_images, x_features)
    return iou


def NBObjective(trial: optuna.Trial):
    classifier = GaussianNB()
    mh.full_fit(classifier, train_images, x_features)
    iou = mh.calc_mean_iou(classifier, valid_images, x_features)
    return iou


def LDAObjective(trial: optuna.Trial):
    trial.suggest_float("shrinkage", 0, 10, 0.1)
    classifier = LinearDiscriminantAnalysis(**trial.params)
    mh.full_fit(classifier, train_images, x_features)
    mean_iou, _ = mh.calc_mean_iou(classifier, valid_images, x_features)
    total_iou, _ = mh.iou(classifier.predict(val_x), val_y)
    return mean_iou, total_iou


def QDAObjective(trial: optuna.Trial):
    trial.suggest_categorical(
        "reg_param", [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2, 4, 8, 10]
    )
    classifier = QuadraticDiscriminantAnalysis(reg_param=0)
    # classifier = LinearDiscriminantAnalysis(**trial.params)
    mh.full_fit(classifier, train_images, x_features)
    mean_iou, _ = mh.calc_mean_iou(classifier, valid_images, x_features)
    total_iou, _ = mh.iou(classifier.predict(val_x), val_y)
    return mean_iou, total_iou


def GBRFEval(y_true, y_pred):
    return ("iou", mh.iou(y_pred, y_true)[0], True)


def GBRFObjective(trial: optuna.Trial):
    if feature_space in ["OPT", "S2", "SAR_OPT", "SAR_S2", "SAR_HSV(O3)+cAWEI+cNDWI"]:
        leaf_choices = [32, 64, 128]
    elif feature_space in ["SAR", "cNDWI", "cAWEI"]:
        leaf_choices = [2, 4]
    # feature spaces with at most 3 features
    elif feature_space in [
        "SAR",
        "O3",
        "RGB",
        "HSV(RGB)",
        "HSV(O3)",
        "cNDWI",
        "cAWEI",
    ]:  # , 'cNDWI+NDVI']:
        leaf_choices = [4, 8]
    # 4 features
    elif feature_space in ["SAR_cNDWI", "SAR_cAWEI", "RGBN", "cAWEI+cNDWI"]:
        leaf_choices = [4, 8, 16]
    # 5 features
    elif feature_space in [
        "SAR_O3",
        "SAR_RGB",
        "SAR_HSV(RGB)",
        "SAR_HSV(O3)",
    ]:  # , 'SAR_cNDWI+NDVI']:
        leaf_choices = [8, 16, 32]
    # 6 or 7 features
    elif feature_space in ["SAR_RGBN", "SAR_cAWEI+cNDWI", "HSV(O3)+cAWEI+cNDWI"]:
        leaf_choices = [16, 32, 64]
    else:
        raise ValueError(f"Unknown search space {feature_space}")
    trial.set_user_attr("seed", 10)
    trial.set_user_attr("n_jobs", 4)
    trial.set_user_attr("force_row_wise", True)
    trial.set_user_attr("num_iterations", 50)
    trial.suggest_categorical("is_unbalance", [True, False])
    trial.suggest_categorical("objective", ["binary"])
    trial.suggest_categorical("boosting_type", ["gbdt"])
    trial.suggest_categorical("num_leaves", leaf_choices)
    trial.suggest_int("max_depth", -1, 1000)
    trial.suggest_float("learning_rate", 1e-7, 1e3)
    trial.suggest_int("n_estimators", 1, 1000)
    # trial.suggest_int("subsample_for_bin", 1, 1_000_000_000)
    trial.suggest_categorical("class_weight", ["balanced", None])
    trial.suggest_float("min_split_gain", 0.0, 1000.0)
    trial.suggest_float("min_child_weight", 0.0, 1e6)
    trial.suggest_int("min_child_samples", 1, 1_000_000)
    trial.suggest_float("reg_alpha", 0.0, 10.0)
    trial.suggest_float("reg_lambda", 0.0, 10.0)

    classifier = lightgbm.LGBMClassifier(**trial.params)
    with alive_bar(1, title="Fitting") as bar:
        bar.text("Fit Classifier")
        classifier.fit(
            train_x,
            train_y,
            callbacks=[lightgbm.early_stopping(stopping_rounds=5)],
            eval_metric=GBRFEval,
            eval_set=[(val_x, val_y)],
        )
        bar.text("Fit Complete")
        bar()
    # iou = mh.calc_mean_iou(classifier, valid_images, x_features)
    iour, accr = mh.calc_mean_iou_stack(classifier, val_x_stack, val_y_stack)
    print(f"IOU: {iour}, ACC: {accr}")
    return iour


def output(study: optuna.Study, frozentrial: optuna.trial.FrozenTrial):
    study.trials_dataframe(
        attrs=(
            "number",
            "value",
            "datetime_start",
            "datetime_complete",
            "duration",
            "params",
            "user_attrs",
            "system_attrs",
            "state",
        )
    ).to_csv(f"./{study.study_name}-trial-results.csv")


study = optuna.create_study(direction="maximize")
study.optimize(
    GBRFObjective,
    n_trials=50,
    gc_after_trial=True,
    callbacks=[mh.study_output],
)
