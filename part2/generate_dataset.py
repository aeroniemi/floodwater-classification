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
from sklearn.svm import LinearSVC
from datetime import datetime


feature_spaces = ["SAR", "SAR_HSV(O3)+cAWEI+cNDWI"]
with alive_bar(4, title="Loading file lists") as bar:
    train_images = mh.load_file_list("flood_train_data.csv")
    bar()
    valid_images = mh.load_file_list("flood_valid_data.csv")
    bar()
    tests_images = mh.load_file_list("flood_test_data.csv")
    bar()
    boliv_images = mh.load_file_list("flood_bolivia_data.csv")
    bar()


for feature_space in feature_spaces:
    with alive_bar(3, title="Generating dataset") as bar:
        bar.title = feature_space
        x_features = Feature.getMany([feature_space])
        bar()
        cache_path = f"./data_cache/{feature_space}.npz"
        # load
        train_x, train_y, train_mask = mh.generate_dataset(train_images, x_features)
        val_x, val_y, val_mask = mh.generate_dataset(valid_images, x_features)
        test_x, test_y, test_mask = mh.generate_dataset(tests_images, x_features)
        bolivia_x, bolivia_y, bolivia_mask = mh.generate_dataset(
            boliv_images, x_features
        )

        # scale
        scaler = StandardScaler()

        def apply_scaler(dataset):
            c = np.concatenate(dataset)
            c = scaler.transform(c)
            return np.reshape(c, dataset.shape)

        # train_mask = mh.generate_mask(train_x)
        scaler.fit(np.concatenate(mh.apply_mask(train_mask, fx=train_x)))
        train_x = apply_scaler(train_x)
        val_x = apply_scaler(val_x)
        test_x = apply_scaler(test_x)
        bolivia_x = apply_scaler(bolivia_x)
        bar()
        # save
        np.savez(
            cache_path,
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            test_x=test_x,
            test_y=test_y,
            bolivia_x=bolivia_x,
            bolivia_y=bolivia_y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            bolivia_mask=bolivia_mask,
        )
        bar()
