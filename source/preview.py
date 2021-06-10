from DataHelper import *
print(spamColor)

import autokeras as ak
import numpy as np
import tensorflow as tf
import json
import random
from PIL import Image

file = open('../operations/parameters.json')
jdata = json.load(file)

maxDataSize = jdata['maxDataSize']
ratio = jdata['trainTestRatio']
scale = jdata['imgSizeRatio']
epochs = jdata['trainEpochs']
trials = jdata['maxTrials']

labelData = getLabels(maxDataSize)
imgData = loadImages("../data/IMGS/",len(labelData),scale)

_, xtest = divideData(imgData,ratio)
_, ytest = divideData(labelData,ratio)

print(messageColor)
print("Loading Model")
print(spamColor)
model = tf.keras.models.load_model("../saved_model.h5", custom_objects=ak.CUSTOM_OBJECTS)


print(messageColor)
print("Displaying Preview")
print(spamColor)

origdata = xtest[int(random.uniform(0,30))]
data = np.array([origdata])

prediction = model.predict(data,verbose=0) # an array of predictions for an array of data, each prediction is an array of values

print(messageColor)
try:
    img = Image.fromarray(origdata.astype('uint8')).show()
except:
    print("Input:",data[0])
print("\nPrediction:",prediction[0]) # the first value of the first prediction
waitForExit()