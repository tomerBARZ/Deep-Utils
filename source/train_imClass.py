from DataHelper import *
print(spamColor)

import autokeras as ak
import numpy as np
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description='Start an Image Classifier.')
parser.add_argument('maxDataSize',help='The max number of rows to load from the data csv')
parser.add_argument('trainTestRatio',help='The ratio between training set size and testing set size')
parser.add_argument('imgSizeRatio',help='The percent scale of the input images')
parser.add_argument('trainEpochs',help='The amount of epochs to train on')
parser.add_argument('maxTrials',help='The max number of models to try')
args = parser.parse_args()

maxDataSize = int(args.maxDataSize)
ratio = float(args.trainTestRatio)
scale = float(args.imgSizeRatio)
epochs = int(args.trainEpochs)

labelData = getLabels(maxDataSize)
imgData = loadImages("../data/IMGS/",len(labelData),scale)

xtrain, xtest = divideData(imgData,ratio)
ytrain, ytest = divideData(labelData,1 - ratio)

classifier = ak.ImageClassifier(max_trials=1)

print(msgColor)
print("\nTraining Models----")
print(spamColor)
trainResults = classifier.fit(xtrain,ytrain,epochs=50,validation_data=(xtest,ytest))
print(msgColor)
print("Training Loss:",trainResults)
print(spamColor)

# print("\nEvaluating Best Model----")
# eval_loss = classifier.evaluate(xtest, ytest,verbose=0)
# print("Evaluation Loss:",eval_loss)


# print("\nShowing Results----")
# img = xtest[-1]
# data = np.array([img])

# prediction = classifier.predict(data,verbose=0) # an array of predictions for an array of data, each prediction is an array of values
# print("\nPrediction:",prediction[0,0]) # the first value of the first prediction
# cv2.imshow('IMG',img)
# cv2.waitKey(0)

model = classifier.export_model()
model.save("model_autokeras.h5")