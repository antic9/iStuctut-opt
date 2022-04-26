# coding:UTF-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from decimal import *
import math
import tensorflow as tf

#回帰法によるAccuracy検証

predict_func = tf.keras.models.load_model("models/prediction_model")

temp = [[-1.52866867, -1.86774023,  1.93501212,  0.14645577, -1.53911974, -1.33124485,
  -1.13056826, -1.19553216, -0.67861839,  0.84064899]]
temp = np.array(temp)
print(len(temp))

fitness = predict_func.predict(temp)
print(fitness)