from DataHelper import *
print(spamColor)

import autokeras as ak
import numpy as np
import tensorflow as tf
import argparse
import json

file = open('../operations/parameters.json')
jdata = json.load(file)

maxDataSize = jdata['maxDataSize']
ratio = jdata['trainTestRatio']
scale = jdata['imgSizeRatio']
epochs = jdata['trainEpochs']
trials = jdata['maxTrials']
ypos = jdata['yPosition']

xData, yData = getData(maxDataSize,scale,ypos)

xtrain, xtest = divideData(xData,ratio)
ytrain, ytest = divideData(yData,ratio)

classifier = ak.ImageRegressor(max_trials=trials)

print(messageColor)
print("\nStarted Training")
print(spamColor)
classifier.fit(xtrain,ytrain,epochs=epochs,validation_data=(xtest,ytest))
print(messageColor)
print("Training Complete")
waitForExit()


model = classifier.export_model()
model.save("../saved_model.h5")