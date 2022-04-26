# -*- coding: utf-8 -*-
import csv
import sys

import IDE_crossentropy
import Individual_entorpy
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#parameters

#メモ:試行回数を引数から呼び出して、その回数実行するように仕様変更、最後に配列から平均、中央値、標準偏差をだす
GENERATION = 1000
POPULATION = 500
COUNTER = 10
DIMENSION = 256
batch_size = 200
fitness_value = "entropy"
Method = "IDE" 

if __name__ == '__main__':
    args = sys.argv
    tf.get_logger().setLevel('WARNING')

    #define parameters
    count = 0.0
    Avarage_fitness = []
    Bestaccuracy = []
    Bestindividual = []
    Bestgene = []
    UpdatedIndividual = []
    ps2=0.1
    euclidgraph = []
    euclidavelist = []
    r_glist=[]
    pslist = []
    un_update_counter = 0


    de = IDE_crossentropy.IDE_NN(POPULATION)

    indiv = Individual_entorpy.Indivi()

    #main program
    for i in range(GENERATION):
        euclidlist =[]
        print("Generation="+str(i))

        count += 1.0
        
        de.Mutation(count,ps2, UpdatedIndividual)
        
        de.Best_Individual()
        Avarage_fitness.append(de.Average_Fitness())
        de.Best_plot(Bestindividual,Bestgene,Bestaccuracy)
               
        if (UpdatedIndividual[i] == 0):
            un_update_counter += 1
        if (un_update_counter >= COUNTER ):
            print("!---reset---!")
            un_update_counter = 0
    name_str = "_p"+str(POPULATION)+"_g"+str(GENERATION)+"_b"+str(batch_size)+"_MAGIC_"+fitness_value
    if(Method=="jDE"):
        with open('Result_data/jde_Accuracy'+name_str+'.csv', 'a+') as f:
            w = csv.writer(f,lineterminator = "\n")
            w = w.writerow(Bestaccuracy)
        with open('Result_data/jde_crossentropy'+name_str+'.csv', 'a+',newline='') as f:
            w = csv.writer(f,lineterminator = "\n")
            w = w.writerow(Bestindividual)
    elif(Method == "JADE"):
        with open('Result_data/jade_Accuracy'+name_str+'.csv', 'a') as f:
            w = csv.writer(f)
            w = w.writerow(Bestaccuracy)
        with open('Result_data/jade_crossentropy'+name_str+'.csv', 'a',newline='') as f:
            w = csv.writer(f)
            w = w.writerow(Bestindividual)
    elif(Method == "IDE"):
        with open('Result_data/ide_Accuracy_A'+name_str+'.csv', 'a') as f:
            w = csv.writer(f)
            w = w.writerow(Bestaccuracy)
        with open('Result_data/ide_crossentropy_A'+name_str+'.csv', 'a',newline='') as f:
            w = csv.writer(f)
            w = w.writerow(Bestindividual)
    indiv.Finish(Bestgene[i])
    print(Method)
    print(de)
    
        
        
        
        
"""
    #↓output↓
    f = open('Results/%s/Best/bestindividual_%s.csv' % (args[1],args[2]), 'w')
    for n in Bestindividual:
        f.write(str(n) + "\n")
    f.close()

    f = open('Results/%s/Update/Update_times_%s.csv' % (args[1],args[2]), 'w')
    for n in UpdatedIndividual:
        f.write(str(n) + "\n")
    f.close()

    f = open('Results/%s/Ps/Ps_%s.csv' % (args[1],args[2]), 'w')
    for n in pslist:
        f.write(str(n) + "\n")
    f.close()
    
    f = open('Results/%s/Rg/Rg_%s.csv' % (args[1],args[2]), 'w')
    for n in r_glist:
        f.write(str(n) + "\n")
    f.close()
"""

