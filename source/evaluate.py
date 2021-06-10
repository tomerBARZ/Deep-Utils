from DataHelper import *
print(spamColor)

import autokeras as ak
import numpy as np
import tensorflow as tf
import json

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
print("Evaluating Best Model")
print(spamColor)

eval_loss = model.evaluate(xtest, ytest)
print(messageColor)
print("Evaluation Loss:",eval_loss[0]," Evaluation Accuracy:",eval_loss[1])
waitForExit()

# print("\nShowing Results----")
# img = xtest[-1]
# data = np.array([img])

# prediction = classifier.predict(data,verbose=0) # an array of predictions for an array of data, each prediction is an array of values
# print("\nPrediction:",prediction[0,0]) # the first value of the first prediction
# cv2.imshow('IMG',img)
# cv2.waitKey(0)