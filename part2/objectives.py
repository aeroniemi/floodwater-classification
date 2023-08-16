import optuna
import numpy as np
import methods as mh
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
import lightgbm

def SGD(trial: optuna.Trial):
    trial.suggest_categorical(
        "loss", ["hinge", "log_loss", "modified_huber", "squared_hinge"]
    )
    trial.suggest_int("alpha", 0, 4, 1)
    # trial.suggest_categorical("average", [True, False])
    # trial.suggest_categorical("classifier", ["SVC", "RandomForest"])
    classifier = SGDClassifier(**trial.params)
    mh.full_fit(classifier, train_images, x_features)
    iou = mh.calc_mean_iou(classifier, valid_images, x_features)
    return iou


def NaiveBayes(trial: optuna.Trial):
    classifier = GaussianNB()
    classifier.fit(train_x, train_y)
    output = mh.predict_to_file(classifier, trial, val_x)
    total_iou, _ = mh.iou(output, val_y)
    return total_iou


def LDA(trial: optuna.Trial):
    trial.suggest_float("shrinkage", 0, 10, 0.1)
    classifier = LinearDiscriminantAnalysis(**trial.params)
    mh.full_fit(classifier, train_images, x_features)
    output = classifier.predict(val_x_stack.ravel()).shape(val_x_stack)

    # mean_iou, _ = mh.calc_mean_iou(, valid_images, x_features)
    total_iou, _ = mh.iou(output.ravel(), val_y)
    return total_iou


def QDA(trial: optuna.Trial):
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


def GBRF(trial: optuna.Trial):
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
    trial.set_user_attr("n_jobs", 4)
    trial.set_user_attr("num_iterations", 50)
    trial.set_user_attr("n_estimators", 200),
    trial.set_user_attr("subsample_for_bin", 262144)
    trial.suggest_categorical("is_unbalance", [True, False])
    trial.suggest_categorical("seed", [10, 55, 1921, 2132])
    trial.suggest_categorical("objective", ["binary"])
    trial.suggest_categorical("boosting_type", ["gbdt"])
    trial.suggest_categorical("num_leaves", leaf_choices)
    trial.suggest_int("max_depth", -1, 1000)
    trial.suggest_float("learning_rate", 1e-7, 1e3)
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
            # callbacks=[lightgbm.early_stopping(stopping_rounds=5)],
            # eval_metric=GBRFEval,
            # eval_set=[(val_x, val_y)],
        )
        bar.text("Fit Complete")
        bar()
    # iou = mh.calc_mean_iou(classifier, valid_images, x_features)
    total_iou, _ = mh.iou(classifier.predict(val_x), val_y)
    iour, accr = mh.calc_mean_iou_stack(classifier, val_x_stack, val_y_stack)
    print(f"Accuracy: {accr}")
    return iour, total_iou
