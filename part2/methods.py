# ===========================================================================
#                            Imports
# ===========================================================================
import glob
import os
import re
import signal
import sys

import numpy as np
from colored import fore, stylize
from feature_spaces import *
from osgeo import gdal
from skimage.io import imread
from sklearn.base import ClassifierMixin

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

# ===========================================================================
#                            Functions
# ===========================================================================


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
    img, meta = read_geotiff(file_path)
    if img.ndim > 2:  ## if multiband put the bands at the end rather than the beginning
        img = np.transpose(img, (1, 2, 0))
    else:
        img = np.reshape(
            img, img.shape + (1,)
        )  # if single band then make it into a 2d array of length 1
    return img, meta


def generate_feature_stack(
    image_name: str, x_features: list, y_features: list = Feature.getMany(["QC"])
):
    dem, s1, s2, lab, meta = load_image(image_name)
    feature_stack_x = []
    feature_stack_y = []

    for feature in x_features:
        f = feature.access(dem, s1, s2, lab)
        if type(f) == tuple:
            for minifeature in f:
                feature_stack_x.append(minifeature.ravel())
        else:
            feature_stack_x.append(f.ravel())
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


def iou(real, predicted):
    real = np.delete(real, -1)
    predicted = np.delete(predicted, -1)
    length = real.size
    tp = np.sum((real == 1) & (predicted == 1)) / length
    fp = np.sum((real == 0) & (predicted == 1)) / length
    tn = np.sum((real == 0) & (predicted == 0)) / length
    fn = np.sum((real == 1) & (predicted == 0)) / length
    ioures = tp / (tp + fp + fn)
    return ioures


def pathsToTitle(paths: list):
    return list(map(lambda x: re.findall("\\\\(.+)_LabelHand.tif", x)[0], paths))


def filterPaths(folder: str, filter: str):
    return pathsToTitle(glob.glob(f"{folder}{filter}.tif"))


def sigint_handler(signal, frame):
    print(stylize("User interrupted operations", error))
    sys.exit(0)


signal.signal(signal.SIGINT, sigint_handler)


def partial_fit(
    classifier: ClassifierMixin, classes: list, image: str, x_features: list
):
    feature_stack_x, feature_stack_y, meta = generate_feature_stack(image, x_features)
    feature_stack_x_filtered, feature_stack_y_filtered, mask = handleNaN(
        feature_stack_x, feature_stack_y
    )
    classifier.partial_fit(feature_stack_x_filtered, feature_stack_y_filtered, classes)


def predict(classifier: ClassifierMixin, image: str, x_features: list):
    feature_stack_x, feature_stack_y, meta = generate_feature_stack(image, x_features)
    feature_stack_x_filtered, feature_stack_y_filtered, mask = handleNaN(
        feature_stack_x, feature_stack_y
    )
    predicted = classifier.predict(feature_stack_x_filtered)
    built = rebuildShape(predicted, mask)
    return built, feature_stack_y, meta
