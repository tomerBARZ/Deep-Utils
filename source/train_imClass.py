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

labelData = getLabels(maxDataSize)
imgData = loadImages("../data/IMGS/",len(labelData),scale)

xtrain, xtest = divideData(imgData,ratio)
ytrain, ytest = divideData(labelData,ratio)

classifier = ak.ImageClassifier(max_trials=trials)

print(messageColor)
print("\nStarted Training")
print(spamColor)
classifier.fit(xtrain,ytrain,epochs=epochs,validation_data=(xtest,ytest))
print(messageColor)
print("Training Complete")
waitForExit()


model = classifier.export_model()
model.save("../saved_model.h5")