import glob
import re
from sklearn.linear_model import SGDClassifier
import methods as mh
import numpy as np
folder = "../src/downloaded-data/label/"

def pathsToTitle(paths):
    return list(map(lambda x: re.findall("\\\\(.+)_LabelHand.tif", x)[0], paths))
def filterPaths(filter):
    return pathsToTitle(glob.glob(f"{folder}{filter}.tif"))

train_images = filterPaths("Bolivia*")
predict_images = filterPaths("Bolivia*")
classes = [0, 1]
def partial_fit(classifier, image):
    feature_stack_x, feature_stack_y, meta = mh.generate_feature_stack(image)
    feature_stack_x_filtered, feature_stack_y_filtered, mask = mh.handleNaN(feature_stack_x, feature_stack_y)
    print(feature_stack_x_filtered.T.shape)
    np.savetxt(f"{image}.csv",feature_stack_x_filtered.T, delimiter=',')
    classifier.partial_fit(feature_stack_x_filtered.T, feature_stack_y_filtered, classes)

def predict(classifier, image):
    feature_stack_x, feature_stack_y, meta = mh.generate_feature_stack(image)
    feature_stack_x_filtered, feature_stack_y_filtered, mask = mh.handleNaN(feature_stack_x, feature_stack_y)
    predicted = classifier.predict(feature_stack_x_filtered.T) 
    built = mh.rebuildShape(predicted, mask)
    return built, meta

## fit
classifier = SGDClassifier()
for image in train_images:
    partial_fit(classifier, image)
    # break

## predict
for image in predict_images:
    img, meta = predict(classifier, image)
    mh.write_geotiff(f"./predicted_{image}.tif",img.reshape((512, 512)),meta)
