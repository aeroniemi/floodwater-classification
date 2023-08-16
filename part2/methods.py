# ===========================================================================
#                            Imports
# ===========================================================================
import glob
import os
import re
import signal
import sys
import functools

from skimage.transform import resize
import numpy as np
from colored import fore, stylize
from feature_spaces import *
from osgeo import gdal
from skimage.io import imread
from sklearn.base import ClassifierMixin
from alive_progress import alive_it, alive_bar
from statistics import mean
from sklearn import config_context
import lightgbm
import optuna

# ===========================================================================
#                            CLI Colours
# ===========================================================================
primary: str = f"{fore('cyan')}"
error: str = f"{fore('red')}"

# ===========================================================================
#                            User editable settings
# ===========================================================================
sources_x = [
    ("DEM", "./dem/", "dem"),
    ("SAR", "./S1Hand/", "S1Hand"),
    ("Optical", "./S2Hand/", "S2Hand"),
]
sources_y = [
    ("Label", "./label/", "LabelHand"),
]
data_path = "../src/downloaded-data/"
output_path = "./output"


# ===========================================================================
#                            Functions
# ===========================================================================
def load_file_list(path: str):
    with open(path, "r") as file:
        data = file.read()
        return [i for i in data.split("\n") if i]


def read_geotiff(path: str):
    """
    Load GEOTIFF
    """
    ds = gdal.Open(path)
    return ds.ReadAsArray(), ds


def write_geotiff(path: str, arr, in_ds: gdal.Dataset):
    """
    Write GEOTIFF
    """
    try:  # used to check if the file is writable - raise error if not
        file = open(path, "w")
        file.close()
    except OSError:
        print(stylize(f"Cannot access {path}", error))
        raise SystemError

    if arr.dtype == np.float32:
        arr_type = gdal.GDT_Float32
    else:
        arr_type = gdal.GDT_Int32

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(path, arr.shape[1], arr.shape[0], 1, arr_type)
    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())
    band = out_ds.GetRasterBand(1)
    band.WriteArray(arr)
    band.FlushCache()
    band.ComputeStatistics(False)


def load_image(image_name: str):
    dem, _ = load_source(image_name, IMAGE_DEM)
    s1, _ = load_source(image_name, IMAGE_S1)
    s2, _ = load_source(image_name, IMAGE_S2)
    lab, meta = load_source(image_name, IMAGE_LAB)
    return dem, s1, s2, lab, meta


def load_source(image_name, source):
    file_path = os.path.join(data_path, source[0], f"{image_name}_{source[1]}.tif")
    if not os.path.isfile(file_path):
        print(f"File not found:", file_path)
        raise FileNotFoundError()
    try:
        assert len(image_name) > 0
    except:
        print("Invalid image name to load_source")
        raise ValueError()
    img, meta = read_geotiff(file_path)
    if img.ndim > 2:  ## if multiband put the bands at the end rather than the beginning
        img = np.transpose(img, (1, 2, 0))
    else:
        img = np.reshape(
            img, img.shape + (1,)
        )  # if single band then make it into a 2d array of length 1
    # check image shape is valid
    try:
        assert img.shape[0] == 512
        assert img.shape[1] == 512
        assert img.shape[2] >= 1
    except:  # some of the DEM isn't the right size because reasons (normally 511*512 or similar), we can just pretend that isn't a thing
        # print(
        #     f"Invalid shape: {image_name}_{source[1]}.tif:",
        #     img.shape,
        # )
        img = resize(img, (512, 512), mode="reflect")
        # print(img.shape)
    return img, meta


def generate_feature_stack(
    image_name: str, x_features: list, y_features: list = Feature.getMany(["QC"])
):
    dem, s1, s2, lab, meta = load_image(image_name)
    feature_stack_x = []
    feature_stack_y = []

    def extract(feature):
        if type(feature) == np.ndarray:  # it's a band
            # print(feature.shape)
            feature_stack_x.append(feature.ravel())
        else:
            return [extract(f) for f in feature]

    for feature in x_features:
        f = feature.access(dem, s1, s2, lab)
        extract(f)
    for feature in y_features:
        feature_stack_y.append(feature.access(dem, s1, s2, lab).ravel())
    # print(list(map(lambda x: x.shape, feature_stack_y)))
    return np.asarray(feature_stack_x).T, np.asarray(feature_stack_y).T, meta


def handleNaN(fx, fy):
    missing_fy = np.any(fy == -1, axis=1)
    missing_fx = np.any(np.isnan(fx), axis=1)
    mask = np.logical_or(missing_fx, missing_fy)
    validFx = np.delete(fx, mask, axis=0)
    validFy = np.delete(
        fy, mask, axis=0
    ).ravel()  # needs to be 1d array for classifier training
    return validFx, validFy, mask


def rebuildShape(fy, mask):
    res = fy.copy()
    for i in range(mask.size):
        if mask[i] == True:
            res = np.insert(res, i, -1)
    return res


def iou(target, output):
    target = np.delete(target, -1)
    output = np.delete(output, -1)
    assert target.size == output.size
    length = target.size
    intersection = np.sum(output * target)
    union = np.sum(target) + np.sum(output) - intersection
    iou = (intersection + 0.0000001) / (union + 0.0000001)
    assert 0 < iou < 1

    correct = np.sum(output, where=(output == target))
    accuracy = correct / length
    assert 0 < accuracy < 1
    return iou, accuracy


# def iou(real, predicted):
#     real = np.delete(real, -1)
#     predicted = np.delete(predicted, -1)
#     assert real.size == predicted.size
#     length = real.size
#     tp = np.sum((real == 1) & (predicted == 1)) + 1 / length
#     fp = np.sum((real == 0) & (predicted == 1)) + 1 / length
#     tn = np.sum((real == 0) & (predicted == 0)) + 1 / length
#     fn = np.sum((real == 1) & (predicted == 0)) + 1 / length
#     ioures = tp / (tp + fp + fn)
#     accuracyres = (tp + tn) / (tp + tn + fp + fn)
#     return ioures, accuracyres


def pathsToTitle(paths: list):
    return list(map(lambda x: re.findall("\\\\(.+)_LabelHand.tif", x)[0], paths))


def filterPaths(folder: str, filter: str):
    return pathsToTitle(glob.glob(f"{folder}{filter}.tif"))


def sigint_handler(signal, frame):
    print(stylize("User interrupted operations", error))
    sys.exit(10)


signal.signal(signal.SIGINT, sigint_handler)


def partial_fit(
    classifier: ClassifierMixin, classes: list, image: str, x_features: list
):
    feature_stack_x, feature_stack_y, meta = generate_feature_stack(image, x_features)
    feature_stack_x_filtered, feature_stack_y_filtered, mask = handleNaN(
        feature_stack_x, feature_stack_y
    )
    classifier.partial_fit(feature_stack_x_filtered, feature_stack_y_filtered, classes)


def full_fit(classifier: ClassifierMixin, images: list, x_features: list):
    feature_stack_all_x = []
    feature_stack_all_y = []
    for image in (bar := alive_it(images, title="Loading Training Data")):
        bar.text(image)
        feature_stack_x, feature_stack_y, meta = generate_feature_stack(
            image, x_features
        )
        feature_stack_x_filtered, feature_stack_y_filtered, _ = handleNaN(
            feature_stack_x, feature_stack_y
        )
        feature_stack_all_x.append(feature_stack_x_filtered)
        feature_stack_all_y.append(feature_stack_y_filtered)
    with alive_bar(3, title="Fitting") as bar:
        bar.text("Concat X")
        arr_x = np.concatenate(feature_stack_all_x)
        bar()
        bar.text("Concat y")
        arr_y = np.concatenate(feature_stack_all_y)
        bar()
        bar.text("Fit Classifier")
        classifier.fit(arr_x, arr_y)
        bar()
        bar.text("Complete")


def generate_dataset(images: list, x_features: list):
    feature_stack_all_x = []
    feature_stack_all_y = []
    for image in (bar := alive_it(images, title="Loading Data")):
        bar.text(image)
        feature_stack_x, feature_stack_y, meta = generate_feature_stack(
            image, x_features
        )
        feature_stack_x_filtered, feature_stack_y_filtered, _ = handleNaN(
            feature_stack_x, feature_stack_y
        )
        if feature_stack_y_filtered.size > 0:
            feature_stack_all_x.append(feature_stack_x_filtered)
            feature_stack_all_y.append(feature_stack_y_filtered)
    arr_x = np.concatenate(feature_stack_all_x)
    arr_y = np.concatenate(feature_stack_all_y)
    return arr_x, arr_y, feature_stack_all_x, feature_stack_all_y


def create_gbm_dataset(images: list, x_features: list):
    feature_stack_all_x = []
    feature_stack_all_y = []
    for image in (bar := alive_it(images, title="Loading Training Data")):
        bar.text(image)
        feature_stack_x, feature_stack_y, meta = generate_feature_stack(
            image, x_features
        )
        feature_stack_x_filtered, feature_stack_y_filtered, _ = handleNaN(
            feature_stack_x, feature_stack_y
        )
        feature_stack_all_x.append(feature_stack_x_filtered)
        feature_stack_all_y.append(feature_stack_y_filtered)
    arr_x = np.concatenate(feature_stack_all_x)
    arr_y = np.concatenate(feature_stack_all_y)
    dataset = lightgbm.Dataset(arr_x, label=arr_y)
    return dataset, x_features


def predict(classifier: ClassifierMixin, image: str, x_features: list):
    feature_stack_x, feature_stack_y, meta = generate_feature_stack(image, x_features)
    feature_stack_x_filtered, feature_stack_y_filtered, mask = handleNaN(
        feature_stack_x, feature_stack_y
    )
    if feature_stack_x_filtered.shape[0] == 0:
        raise Exception(f"No valid label, cannot predict: {image}")
    with config_context(assume_finite=True):
        predicted = classifier.predict(feature_stack_x_filtered)
    built = rebuildShape(predicted, mask)
    return built, feature_stack_y, meta


def calc_mean_iou(classifier, images, x_features):
    ious = []
    acc = []
    for image in (bar := alive_it(images, title="Predicting")):
        bar.text(image)
        try:
            img, label, meta = predict(classifier, image, x_features)
            iour, accuracyr = iou(img, label)
            ious.append(iour)
            acc.append(accuracyr)
        except Exception:
            pass
    return mean(ious), mean(acc)


def calc_mean_iou_stack(classifier, x, y):
    ious = []
    acc = []
    for i in (bar := alive_it(range(len(x)), title="Predicting")):
        try:
            img = classifier.predict(x[i])
            iour, accuracyr = iou(img, y[i])
            ious.append(iour)
            acc.append(accuracyr)
        except Exception:
            pass
    return mean(ious), mean(acc)
def study_output(study: optuna.Study, frozentrial: optuna.trial.FrozenTrial):
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
    ).to_csv(f"{output_path}/{study.study_name}.csv")
