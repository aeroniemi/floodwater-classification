# set the things up
from sklearn.ensemble import HistGradientBoostingClassifier
from skimage.io import imread, imshow, imsave
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
import os
from osgeo import gdal
data_path = "../src/downloaded-data/"

sources_x = [
    ("DEM", "./dem/", "dem"),
    ("SAR", "./S1Hand/", "S1Hand"),
    ("Optical", "./S2Hand/", "S2Hand")
]
sources_y = [
    ("Label", "./label/", "LabelHand"),
]

def read_geotiff(filename):
    ds = gdal.Open(filename)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    return arr, ds

def write_geotiff(filename, arr, in_ds):
    if arr.dtype == np.float32:
        arr_type = gdal.GDT_Float32
    else:
        arr_type = gdal.GDT_Int32

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(filename, arr.shape[1], arr.shape[0], 1, arr_type)
    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())
    band = out_ds.GetRasterBand(1)
    band.WriteArray(arr)
    band.FlushCache()
    band.ComputeStatistics(False)

def generate_feature_stack(image_name):
    # The ravel() function turns a nD image into a 1-D image.
    feature_stack_x = [
    ]
    feature_stack_y = [
    ]
    meta = None
    for source in sources_x:
        file_path = os.path.join(data_path, source[1], f"{image_name}_{source[2]}.tif")
        img, m = read_geotiff(file_path)
        meta = m
        if img.ndim > 2: ## if multiband, add each band seporately as a feature
            tr = np.transpose(img, (1, 2, 0))
           
            for band in range(tr.shape[-1]):
                # print(tr[...,band].shape)
                feature_stack_x.append(tr[...,band].ravel())
        else:
            feature_stack_x.append(img.ravel())
    
    for source in sources_y:
        file_path = os.path.join(data_path, source[1], f"{image_name}_{source[2]}.tif")
        img = imread(file_path)
        feature_stack_y.append(img.ravel())
    # print(feature_stack_x.shape)
    # return stack as numpy-array
    return np.asarray(feature_stack_x), np.asarray(feature_stack_y).ravel(), meta

def handleNaN(fx, fy):
    missing = fy==-1
    validFy = np.delete(fy, missing, axis=0)
    validFx = np.delete(fx, missing, axis=1)
    return validFx, validFy, missing

def rebuildShape(fy, mask):
    res = fy.copy()
    for i in range(len(mask)):
        if mask[i] == True:
            res = np.insert(res, i, -1)
    return res

feature_stack_x, feature_stack_y, meta = generate_feature_stack("Bolivia_290290")
# print(feature_stack_x.shape, feature_stack_y.shape)
feature_stack_x_filtered, feature_stack_y_filtered, mask = handleNaN(feature_stack_x, feature_stack_y)
# print(feature_stack_x.shape, feature_stack_x_filtered.shape)
classifier = HistGradientBoostingClassifier(max_depth=2, random_state=0)
classifier.fit(feature_stack_x_filtered.T, feature_stack_y_filtered)
res = classifier.predict(feature_stack_x_filtered.T)  # we subtract 1 to make background = 0
print("Got here", res.shape)
res2 = rebuildShape(res, mask)
print(res2.shape, res2.reshape((512, 512)).shape)

write_geotiff("./again2_290290.tif",res2.reshape((512, 512)),meta )
