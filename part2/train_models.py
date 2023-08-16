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
import metrics as me
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.svm import LinearSVC
from datetime import datetime

# ---------------------------------------------------------------------------
#                            Model methods
# ---------------------------------------------------------------------------


def calc_training_metrics(output, target):
    total_iou = me.calc_total_iou(output, target)
    # total_accuracy = me.calc_total_accuracy(output, target)
    return total_iou  # , total_accuracy


class Model:
    def __init__(self):
        self.name = self.__class__.__name__
        return

    def __call__(self, trial: optuna.Trial or optuna.trial.FrozenTrial):
        self.trial = trial
        if isinstance(trial, optuna.trial.FrozenTrial):
            return self.run_frozen()
        else:
            return self.run_active()

    def run_active(self):
        self.suggest_params()
        self.build_model()
        self.fit_model()
        metrics = self.get_training_metrics()
        return metrics

    def run_frozen(self):
        self.build_model()
        self.fit_model()
        output = self.predict_all()
        self.output_to_file(output)

    def predict_all(self):
        output = {}
        for split in ["train", "val", "test", "bolivia"]:
            x = np.concatenate(data[f"{split}_x"])
            output[f"{split}_y"] = self.predict(x)
        return output

    def output_to_file(self, output):
        np.save(
            f"./output/{self.name}-{feature_space}.{self.trial.number}.npz",
            output,
        )
        return


class SklModel(Model):
    def fit_model(self):
        x = np.concatenate(data["train_x"])
        y = np.ravel(np.concatenate(data["train_y"]))
        self.model.fit(x, y)

    def get_training_metrics(self):  # totaliou
        x = np.concatenate(data["val_x"])
        output = self.model.predict(x)
        target = np.concatenate(data["val_y"])
        metrics = calc_training_metrics(output, target)
        return metrics

    def predict(self, x):
        return self.model.predict(x)


class SGD(SklModel):
    def suggest_params(self):
        self.trial.suggest_categorical(
            "loss", ["hinge", "log_loss", "modified_huber", "squared_hinge"]
        )
        self.trial.suggest_int("alpha", 0, 4, 1)
        self.trial.suggest_categorical("average", [True, False])
        self.trial.suggest_categorical("classifier", ["SVC", "RandomForest"])

    def build_model(self):
        self.model = SGDClassifier(**self.trial.params)


class NaiveBayes(SklModel):
    def suggest_params(self):
        return

    def build_model(self):
        self.model = GaussianNB()


class LDA(SklModel):
    def suggest_params(self):
        self.trial.suggest_float("shrinkage", 0, 1)
        self.trial.set_user_attr("solver", "lsqr")
        return

    def build_model(self):
        self.model = LinearDiscriminantAnalysis(**self.trial.params)


class QDA(SklModel):
    def suggest_params(self):
        self.trial.suggest_categorical(
            "reg_param", [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2, 4, 8, 10]
        )
        return

    def build_model(self):
        self.model = QuadraticDiscriminantAnalysis(**self.trial.params)


# ---------------------------------------------------------------------------
#                            Model running
# ---------------------------------------------------------------------------
model = NaiveBayes()
feature_space = "SAR"

data = mh.load_masked_dataset(feature_space)


study = optuna.create_study(
    study_name=f"{model.name}-{feature_space}",
    direction=optuna.study.StudyDirection.MAXIMIZE,
)
# study.set_metric_names(("total_iou"))
study.optimize(
    model,
    n_trials=1,
    gc_after_trial=True,
    callbacks=[mh.study_output],
)

model(study.best_trial)
