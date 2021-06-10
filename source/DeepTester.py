import random
import autokeras as ak
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.saving.save import load_model
from DataHelper import *
import cv2
import colored

imgData = loadImages("IMGS/",100,0.5)
labelData = getLabels(100)

_, xtest = divideData(imgData,0.7)

img = xtest[int(random.uniform(0,30))]

loaded = load_model("shapeRegression.h5",custom_objects=ak.CUSTOM_OBJECTS)
prediction = loaded.predict(np.array([img]))
cv2.imshow('img',img)
print(str(prediction[0][0]))
cv2.waitKey(0)