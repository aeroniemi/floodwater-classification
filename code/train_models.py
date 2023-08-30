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
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC, SVC
from datetime import datetime
import sys
model_classes = {}
output_images_path = "./output_images"
# ---------------------------------------------------------------------------
#                            Model methods
# ---------------------------------------------------------------------------


def calc_training_metrics(output, target):
    total_iou = me.calc_iou(output, target)
    # total_accuracy = me.calc_total_accuracy(output, target)
    return total_iou  # , total_accuracy


class Model:
    def __init__(self, data):
        self.name = self.__class__.__name__
        self.data = data
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
        # self.predict_to_geotiff(split="bolivia", i=9, image_name="Bolivia_314919")
        output = self.predict_all()
        self.output_to_file(output)

    def predict_all(self):
        output = {}
        for split in ["val", "test", "bolivia"]:
            res = []
            xa = self.data[f"{split}_x"]
            # print(xa.shape)
            for x in xa:
                # print(x.shape, split)
                res.append(self.predict(x))
            output[f"{split}_y"] = np.array(res, dtype=object)
        return output

    def output_to_file(self, output):
        # print(outpu/t.shape)
        np.savez(
            f"../model_outputs/{self.name}-{feature_space}.{self.trial.number}.npz",
            **output,
        )
        return

    def get_recommended_n_trials(self):
        return self.recommended_n_trials
    def predict_to_geotiff(self, split, i, image_name):
        x = mh.load_dataset(self.feature_space)[f"{split}_x"][i, :, :]
        predicted = self.predict(x)
        _,meta = mh.load_source(image_name, mh.IMAGE_DEM)
        mh.write_geotiff(f"{output_images_path}/{image_name}_{self.name}.tif", predicted, meta)




class SklModel(Model):
    def fit_model(self):
        x = np.concatenate(self.data["train_x"])
        y = np.ravel(np.concatenate(self.data["train_y"]))
        self.model.fit(x, y)

    def get_training_metrics(self):  # totaliou
        x = np.concatenate(self.data["val_x"])
        output = self.predict(x)
        target = np.ravel(np.concatenate(self.data["val_y"]))
        metrics = calc_training_metrics(output, target)
        return metrics

    def predict(self, x):
        return self.model.predict(x)


class SGD(SklModel):
    recommended_n_trials = 35

    def suggest_params(self):
        self.trial.suggest_categorical(
            "loss", ["log_loss", "modified_huber", "squared_hinge"]
        )
        self.trial.suggest_int("alpha", 0, 4, 1)
        self.trial.suggest_categorical("average", [True, False])
        # self.trial.suggest_categorical("classifier", ["SVC"])
        # self.trial.suggest_categorical("classifier", ["SVC", "RandomForest"])

    def build_model(self):
        self.model = SGDClassifier(**self.trial.params, **self.trial.user_attrs)


class NaiveBayes(SklModel):
    recommended_n_trials = 2

    def suggest_params(self):
        return

    def build_model(self):
        self.model = GaussianNB()


class LDA(SklModel):
    recommended_n_trials = 10

    def suggest_params(self):
        self.trial.suggest_float("shrinkage", 0, 1, step=0.1)
        self.trial.suggest_categorical("solver", ["lsqr"])
        return

    def build_model(self):
        self.model = LinearDiscriminantAnalysis(**self.trial.params)


class QDA(SklModel):
    recommended_n_trials = 8

    def suggest_params(self):
        self.trial.suggest_categorical(
            "reg_param", [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1]
        )
        return

    def build_model(self):
        self.model = QuadraticDiscriminantAnalysis(**self.trial.params)
class StackingClassifier(SklModel):
    recommended_n_trials = 8

    def suggest_params(self):
        self.trial.suggest_categorical(
            "reg_param", [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1]
        )
        return

    def build_model(self):
        self.model = QuadraticDiscriminantAnalysis(**self.trial.params)
class SVM(SGD):
    recommended_n_trials = 3

    def suggest_params(self):
        self.trial.set_user_attr("loss","hinge")
        self.trial.set_user_attr("n_jobs",-1)
        # self.trial.suggest_categorical("fit_intercept",[True, False])
        self.trial.set_user_attr("validation_fraction",0.2)
        self.trial.set_user_attr("early_stopping",True)
        self.trial.suggest_float("alpha", 0.00001, 3)
        # self.trial.suggest_categorical("learning_rate", ["optimal"])
        self.trial.suggest_categorical("class_weight", ["balanced"])
        # self.trial.suggest_categorical("class_weight", ["balanced", None, {1: 2, 0:1}])
class approxSVM(SVM):
    def build_model(self):
        super().build_model()
        self.map = Nystroem(random_state=1,n_components=100)
    def fit_model(self):
        x = np.concatenate(self.data["train_x"])
        y = np.ravel(np.concatenate(self.data["train_y"]))
        x_t = self.map.fit_transform(x)
        self.model.fit(x_t, y)
    def predict(self, x):
        x_t = self.map.transform(x)
        return self.model.predict(x_t)

class SVC(SklModel):
    recommended_n_trials=3
    def suggest_params(self):
        self.trial.set_user_attr("cache_size", 10000)
        self.trial.suggest_float("C", 0, 1)
        self.trial.suggest_categorical("kernel", ["rbf", "sigmoid", "poly"])
        if self.trial.params == "poly":
            self.trial.suggest_int("degree", 2, 6)
        self.trial.suggest_categorical("gamma", ["scale", "auto"])
        if self.trial.params in ["poly", "sigmoid"]:
            self.trial.suggest_float("coef0", 0, 5)
        
        self.trial.params["class_weight"] = {
            0: 1,
            1: self.trial.suggest_float("wet_class_weight", 1, 5)
        }
        self.trial.params["wet_class_weight"] = None
        return
    def build_model(self):
        self.model = SVC(**self.trial.params, **self.trial.user_attrs)

class GBDT(SklModel):
    recommended_n_trials = 5
    def my_suggest_params(self):
        self.trial.set_user_attr("n_jobs", -1)
        self.trial.suggest_int("max_depth", 2, 7)
        self.trial.suggest_int("min_data_in_leaf", 100, 2000)
        self.trial.suggest_int("num_leaves", 2, 2**self.trial.params["max_depth"])
    def suggest_params(self):
        return self.my_suggest_params()
        if feature_space in ["OPT", "S2", "SAR_OPT", "SAR_S2", "SAR_HSV(O3)+cAWEI+cNDWI", "DEM_SAR_HSV(O3)+cAWEI+cNDWI","LDEM_SAR_HSV(O3)+cAWEI+cNDWI"]:
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
        elif feature_space in ["SAR_cNDWI","DEM_SAR", "LDEM_SAR", "SAR_cAWEI", "RGBN", "cAWEI+cNDWI"]:
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
            leaf_choices = [16, 32, 64]
        self.trial.set_user_attr("n_jobs", -1)
        self.trial.set_user_attr("num_iterations", 50)
        self.trial.set_user_attr("n_estimators", 200),
        # self.trial.set_user_attr("subsample_for_bin", 262144)
        self.trial.suggest_categorical("is_unbalance", [True, False])
        self.trial.suggest_categorical("seed", [10, 55, 1921, 2132])
        self.trial.suggest_categorical("objective", ["binary"])
        self.trial.suggest_categorical("boosting_type", ["gbdt"])
        self.trial.suggest_categorical("num_leaves", leaf_choices)
        self.trial.suggest_int("max_depth", -1, 1000)
        self.trial.suggest_float("learning_rate", 1e-7, 1e3)
        self.trial.suggest_int("subsample_for_bin", 1, 1_000_000_000)
        self.trial.suggest_categorical("class_weight", ["balanced", None])
        self.trial.suggest_float("min_split_gain", 0.0, 1000.0)
        self.trial.suggest_float("min_child_weight", 0.0, 1e6)
        self.trial.suggest_int("min_child_samples", 1, 1_000_000)
        self.trial.suggest_float("reg_alpha", 0.0, 10.0)
        self.trial.suggest_float("reg_lambda", 0.0, 10.0)
        return

    def build_model(self):
        self.model = lightgbm.LGBMClassifier(**self.trial.params, **self.trial.user_attrs)


# ---------------------------------------------------------------------------
#                            Model running
# ---------------------------------------------------------------------------
# model = NaiveBayes()
# feature_space = "DP_SAR_HSV(O3)+cAWEI+cNDWI"
def train_model(model, feature_space):
    data = mh.load_masked_dataset(feature_space)
    model = model(data)
    model.feature_space = feature_space
    study = optuna.create_study(
        study_name=f"{model.name}-{feature_space}",
        direction=optuna.study.StudyDirection.MAXIMIZE
    )
    # study.set_metric_names(("total_iou"))
    study.optimize(
        model,
        n_trials=model.get_recommended_n_trials(),
        gc_after_trial=True,
        callbacks=[mh.study_output],
        timeout = 60*60,
    )

    # model(study.best_trial)
    model(study.best_trial)


def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])

def getModel(model_name):
    models = all_subclasses(Model)
    for model in models:
        if model.__name__ == model_name:
            return model

if __name__=="__main__":
    feature_space = sys.argv[1]
    model_name = sys.argv[2]
    model = getModel(model_name)
    print(model)
    train_model(model, feature_space)
else: 
    train_model(NaiveBayes(), "SAR_HSV(O3)+cAWEI+cNDWI")
