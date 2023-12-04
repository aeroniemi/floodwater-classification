#----------------------------------------------------------------------------
# Generate dataset
# Script to create datasets as groups of numpy arrays, to speed up model runs
# @aeroniemi / Alex Beavil 2023
#----------------------------------------------------------------------------

import methods as mh
import numpy as np
from alive_progress import alive_it, alive_bar
from feature_spaces import *
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
import sys
from sklearn.feature_extraction.image import extract_patches_2d

# set the feature spaces that you want to create the dataset for
# a slightly easier way to pre-generate multiple feature spaces
feature_spaces = [ 
    # "SAR",
    # "SAR_HSV(O3)",
    # "SAR_cAWEI+cNDWI",
    # "cAWEI",
    # "cAWEI+cNDWI",
    # "SAR_HSV(O3)+cAWEI+cNDWI",
    # "DEM_SAR",
    # "DEM_SAR_HSV(O3)+cAWEI+cNDWI",
    # "HSV(O3)",
    # "HSV(O3)+cAWEI+cNDWI"
    # "LDEM_SAR",
    # "SDEM+LDEM_SAR"
    # "ACU+SDEM+LDEM_SAR"
    "DP_SAR_HSV(O3)+cAWEI+cNDWI"
]

with alive_bar(4, title="Loading file lists") as bar:
    train_images = mh.load_file_list("flood_train_data_short.csv")
    bar()
    valid_images = mh.load_file_list("flood_val_data.csv")
    bar()
    tests_images = mh.load_file_list("flood_test_data.csv")
    bar()
    boliv_images = mh.load_file_list("flood_bolivia_data.csv")
    bar()

def generate_patches(x,y):
    patches_x = extract_patches_2d(x, patch_size=(3,3))
    patches_y = extract_patches_2d(y, patch_size=(3,3))
    return patches_x, patches_y


def generate(feature_space):
    with alive_bar(3, title="Generating dataset") as bar:
        bar.title = feature_space
        x_features = Feature.getMany([feature_space])
        bar()
        cache_path = f"../data_cache/{feature_space}.npz"
        # load
        train_x,train_ux, train_y, train_mask = mh.generate_dataset(train_images, x_features)
        val_x, val_ux, val_y, val_mask = mh.generate_dataset(valid_images, x_features)
        test_x, test_ux, test_y, test_mask = mh.generate_dataset(tests_images, x_features)
        bolivia_x, bolivia_ux, bolivia_y, bolivia_mask = mh.generate_dataset(
            boliv_images, x_features
        )

        # scale
        scaler = StandardScaler()

        def apply_scaler(dataset, unscaled):
            c = np.concatenate(dataset)
            c = scaler.transform(c)
            x = np.reshape(c, dataset.shape)
            if unscaled.size == 0:
                return x
            return np.dstack((x, unscaled))

        # train_mask = mh.generate_mask(train_x)
        scaler.fit(np.concatenate(mh.apply_mask(train_mask, fx=train_x)))
        train_x = apply_scaler(train_x, train_ux)
        val_x = apply_scaler(val_x, val_ux)
        test_x = apply_scaler(test_x, test_ux)
        bolivia_x = apply_scaler(bolivia_x, bolivia_ux)

        # print("scaled", bolivia_x.shape)
        # train_x, train_y = generate_patches(train_x, train_y)
        # val_x, train_y = generate_patches(val_x, val_y)
        # test_x, train_y = generate_patches(test_x, test_y)
        # bolivia_x, train_y = generate_patches(bolivia_x, bolivia_y)
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
        np.savetxt("./check.csv", train_x[0,:,:])
        bar()

if __name__=="__main__":
    feature_space = sys.argv[1]
    generate(feature_space)
else: 
    for feature_space in feature_spaces:
        generate(feature_space)
