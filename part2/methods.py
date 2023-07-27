import numpy as np
from skimage.io import imread, imshow, imsave
from osgeo import gdal
import os
sources_x = [
    ("DEM", "./dem/", "dem"),
    ("SAR", "./S1Hand/", "S1Hand"),
    ("Optical", "./S2Hand/", "S2Hand")
]
sources_y = [
    ("Label", "./label/", "LabelHand"),
]
data_path = "../src/downloaded-data/"

def read_geotiff(filename):
    ds = gdal.Open(filename)
    return ds.ReadAsArray(), ds

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
        print(source, img.shape)
        if img.ndim > 2: ## if multiband, add each band seporately as a feature
            tr = np.transpose(img, (1, 2, 0))
           
            for band in range(tr.shape[-1]):
                
                feature_stack_x.append(tr[...,band].ravel())
        else:
            feature_stack_x.append(img.ravel())
    
    for source in sources_y:
        file_path = os.path.join(data_path, source[1], f"{image_name}_{source[2]}.tif")
        img = imread(file_path)
        # img = rr.astype('float')
        # img[img == -1] = None
        feature_stack_y.append(img.ravel())
    # print(feature_stack_x.shape)
    # return stack as numpy-array
    return np.asarray(feature_stack_x), np.asarray(feature_stack_y).ravel(), meta

def handleNaN(fx, fy):
    missing_fy = np.array(fy==-1)
    missing_fx = np.any(np.isnan(fx), axis=0)
    overall_missing = np.logical_or(missing_fx, missing_fy)
    validFy = np.delete(fy, overall_missing, axis=0)
    validFx = np.delete(fx, overall_missing, axis=1)
    return validFx, validFy, overall_missing

def rebuildShape(fy, mask):
    res = fy.copy()
    for i in range(len(mask)):
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
    ioures = tp / (tp+fp+fn)
    return ioures