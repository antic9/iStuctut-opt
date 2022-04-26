# coding:UTF-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from decimal import *
import math
import tensorflow as tf
import dataset_CNN
import test


#parameters
DIMENSION = 10

#Minist
MIN_VALUE  = -2 #-2
MAX_VALUE  = 2 #2

MID_VALUE = MAX_VALUE - (( MAX_VALUE - MIN_VALUE) /2)

accinit = [ 0.6009,0.5933, 0.5841,0.5266,0.4620,0.6214,0.6263, 0.6288, 0.5965, 0.5955,0.4552]
        
class Indivi:

    def __init__(self, gene = None, max_value = MAX_VALUE, min_value = MIN_VALUE):
    
        
        if gene is None:
            self.gene = np.float32(np.random.uniform(-6.2,6.1,(1,DIMENSION)))
            # self.gene = np.random.randn(1,DIMENSION)

            # for i in range(DIMENSION):
            #     # numpy.random.normal(loc=0.0, scale=1.0, size=None)は、平均loc、標準偏差scaleの正規分布に従う乱数を返す。
            #     scale = math.sqrt(math.fabs(sigma[i]))
            #     scale = math.exp(scale)
            #     print(scale)
            #     self.gene[i] = np.random.normal(loc=mu[i], scale=scale)
            
        else:
            self.gene = gene
        self.fitness_entropy = self.Evaluate()
        self.F  = 1.0
        self.CR = 0.5

    def Evaluate(self):
        temp = (self.gene)
        # # Regression Model
        # fitness = predict_func.predict(temp)
        fitness = self.correct_fitness()
        # if fitness>1:
        #     fitness = self.correct_fitness()
        return float(fitness)



    def correct_fitness(self):
        # try:
        #     struc = test.decodez(self.gene)
        # except:
        #     print(type(self.gene))
        #     print(self.gene)
        # print(type(self.gene[0][0]))
        # print(self.gene[0])
        struc = test.decodez(self.gene[0])
        # print(struc[0])
        try:
            actual_fitness = dataset_CNN.set_acc(struc[0])
        except:
            actual_fitness = 0
        return actual_fitness

    def stringtolist(self, string):
        struc_list = string.split("', '")
        return struc_list
        

    def Finish(self, best_gene):
        print(best_gene)
        predict_func = tf.keras.models.load_model("models/prediction_model")
        out =  predict_func.predict(best_gene)
        print(out)

