# set the things up
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report

import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
import os


import glob
import numpy.ma as ma





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

# write_geotiff("./again2_290290.tif",res2.reshape((512, 512)),meta )
print(classification_report(feature_stack_y, res2))
print(iou(feature_stack_y, res2))
