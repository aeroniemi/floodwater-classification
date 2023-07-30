# ===========================================================================
#                            Imports
# ===========================================================================
import methods as mh
import numpy as np
from alive_progress import alive_it
from feature_spaces import *
from sklearn.linear_model import SGDClassifier

# ===========================================================================
#                            User editable settings
# ===========================================================================
folder = "../src/downloaded-data/label/"
train_images = mh.filterPaths(folder, "Bolivia*")
predict_images = mh.filterPaths(folder, "Bolivia*")
classes = [0, 1]
classifier = SGDClassifier()

# ===========================================================================
#                            Train Classifer
# ===========================================================================

# for image in (bar := alive_it(train_images, title="Training")):
#     bar.text(image)
#     mh.partial_fit(classifier, classes, image)

# ===========================================================================
#                            Predict using Classifier
# ===========================================================================
# for image in (bar := alive_it(predict_images, title="Predicting")):
#     bar.text(image)
#     img, label, meta = mh.predict(classifier, image)
#     mh.write_geotiff(f"./predicted_{image}.tif", img.reshape((512, 512)), meta)

image = "Bolivia_290290"

# x_features = Feature.getMany(["compositeTest"])
x_features = Feature.getMany(["DEM", "OPT_R", "compositeTest"])
# x_features = Feature.getMany(["DEM", "OPT_R", "AWEI"])
y_features = Feature.getMany(["QC"])

x, y, meta = mh.generate_feature_stack(image, x_features, y_features)

print(x[:, :5])
