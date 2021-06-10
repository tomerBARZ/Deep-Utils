from DataHelper import *
print(spamColor)

import autokeras as ak
import tensorflow as tf

print(messageColor)
print("Loading Model")
print(spamColor)
model = tf.keras.models.load_model("../saved_model.h5", custom_objects=ak.CUSTOM_OBJECTS)

print(messageColor)
print("Summarizing Model\n")
print(spamColor)

model.summary()

waitForExit()