import autokeras as ak
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
from DataHelper import *
import cv2

datasize = 3000

imgData = loadImages("IMGS/",datasize,0.5)
labelData = getLabels(datasize)

xtrain, xtest = divideData(imgData,0.7)
ytrain, ytest = divideData(labelData,0.7)

classifier = ak.ImageRegressor(max_trials=1)

# print("\nTraining Models----")
# classifier.fit(xtrain,ytrain,epochs=50,validation_data=(xtest,ytest))

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