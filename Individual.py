# coding:UTF-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from decimal import *
import math
import tensorflow as tf


#parameters
DIMENSION = 10

#Minist
MIN_VALUE  = -2 #-2
MAX_VALUE  = 2 #2

MID_VALUE = MAX_VALUE - (( MAX_VALUE - MIN_VALUE) /2)

predict_func = tf.keras.models.load_model("models/prediction_model")
        
#with tf.Session() as sess:
#    init= tf.global_variables_initializer()
#    sess.run(init)
    
        
class Indivi:

    def __init__(self, gene = None, max_value = MAX_VALUE, min_value = MIN_VALUE):
    
        
        if gene is None:
            self.gene = np.random.uniform(-0.2,0.2,(1,DIMENSION))
            
        else:
            self.gene = gene
        
        self.fitness_entropy = self.Evaluate()
        self.F  = 1.0
        self.CR = 0.5

    def Evaluate(self):
        temp = (self.gene)
        fitness = predict_func.predict(temp)
        # print(fitness)
         
        return float(fitness)


    def Finish(self, best_gene):
        print(best_gene)
        predict_func = tf.keras.models.load_model("models/prediction_model")
        out =  predict_func.predict(best_gene)
        print(out)

