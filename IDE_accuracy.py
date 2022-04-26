# coding:UTF-8
from cgi import test
from pickle import POP
import Individual_accuracy
import numpy as np
import operator
import random
import main_accuracy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import test



#parameters
POPULATION = 0
MAX_VALUE = Individual_accuracy.MAX_VALUE
MIN_VALUE = Individual_accuracy.MIN_VALUE
F_l        =  0.1
F_u        =  0.9
#CR         =  0.5
gamma1     =  0.1
gamma2     =  0.1
eps = 1e-3
Population = []
SPACE = MAX_VALUE - MIN_VALUE
Accuracy = []
init_pop = [
[[-1.806650,-1.259611,-4.593678,0.754508,4.408788,-3.590508,-2.050025,2.901726,2.378924,-2.508429]],
[[0.061264,0.386933,2.146899,2.363063,5.547554,-1.536651,1.085178,0.206526,2.153552,-3.391303]],
[[-1.326169,-2.099437,-2.052811,1.336117,5.353471,-2.584306,0.401562,1.751854,1.576210,-4.710070]],
[[0.788188,0.797396,2.527108,2.834859,4.534751,-2.305614,-1.530185,0.971076,2.908106,-1.141360]],
[[0.455912,0.725066,-1.376439,1.688611,4.770318,-3.050532,-2.236627,2.753801,2.577978,-2.241990]],
[[0.425857,-1.456150,-4.063758,-2.671803,4.280477,-4.728182,-1.453768,1.962271,1.386933,-2.742712]],
[[2.341053,0.143539,3.126164,-0.715449,5.434090,-2.306279,1.716402,-0.725642,1.255214,-3.599971]],
[[0.873721,-2.208282,-1.436165,-2.005476,5.096578,-3.539072,1.081310,0.768448,0.582161,-4.862061]],
[[3.307106,0.596477,3.455099,-0.640334,4.357800,-3.347328,-0.894252,0.040314,1.949867,-1.372638]],
[[2.865660,0.613872,-0.700821,-1.836170,4.569812,-4.161679,-1.582316,1.790975,1.528178,-2.433009]],
[[-2.031138,-5.106332,-2.811800,1.690259,2.828764,-2.658628,-4.081154,1.289007,0.867967,-0.046589]],
[[-0.620160,-3.455148,3.509162,3.197917,4.430429,-0.558651,-0.151168,-1.023969,0.930445,-1.924477]],
[[-1.658613,-5.951454,-0.333047,2.233943,3.919630,-1.691484,-1.710761,0.298392,0.202562,-2.367162]],
[[0.395078,-2.691622,3.096292,3.365607,3.074644,-1.636450,-3.036153,0.326070,1.982468,0.666761]],
[[0.228044,-2.635235,-0.079955,2.421180,3.154449,-2.357329,-4.530226,1.605123,1.285220,0.196676]],
[[-0.587113,-1.243987,-6.958839,1.844726,3.063962,-2.912841,0.936722,1.813673,1.742184,-0.988119]],
[[2.082819,0.481542,-0.309254,3.625787,4.108460,-0.793499,3.892551,-0.635789,2.057443,-1.454617]],
[[-0.303794,-1.972725,-4.499140,2.381344,4.037942,-2.068296,2.987254,0.750763,1.139922,-3.030222]],
[[2.615124,1.190509,0.314697,3.805323,2.956744,-1.573814,1.358191,0.226309,2.791255,0.808269]],
[[4.910395,-2.606102,-2.323764,0.098067,0.736356,-2.412240,-0.028068,-0.795255,-0.579536,2.488297 ]],
[[-1.081721,-4.099480,0.188324,-0.817342,0.954812,0.290141,3.438121,-1.762266,0.261769,-1.510853 ]]
]

class IDE_NN:
    def __init__(self, popu,u_CR=0.5,u_F=0.5,p2=0.05,c=0.1):
        global POPULATION 
        POPULATION = popu
        print(POPULATION)
        for pop in range(POPULATION):
            Population.append(Individual_accuracy.Indivi(gene = init_pop[pop]))
            # Population.append(Individual_accuracy.Indivi())

            self.u_CR=u_CR
            self.u_F=u_F
            self.p2=p2
            self.c=c
            self.POPULATION=POPULATION
            self.A=[]
            self.S_CR=[]
            self.S_F=[]

    def Mutation(self,count,ps2,UpdatedIndividual):
        fitness_entropy = []
        update = 0
        c = count
        float (c)
        gcount = c / main_accuracy.GENERATION
        float (gcount)
        #Set Ps Pd and dr3 for mutant vector
        ps = 0.1 + 0.9*10 ** (5 * (gcount-1))
        if(count==1):
            print("output first population")
            for a in range(len(Population)):
                struc = test.decodez(Population[a].gene[0])
                with open('struc_first.txt', 'a+') as f:
                    f.writelines(str(struc[0])+str(Population[a].fitness_entropy))
                    f.writelines("\n")
        
        #Mutation
        for i in range(POPULATION):
           
            ps2 = ps
           
            #print "pop",i,"ps2=",ps2
            pd = ps2 * 0.1
            #print "Population",i,"ps2=",ps2
            Srange = int(POPULATION*ps2)
                        
            Irange = POPULATION - Srange
            p = np.arange(POPULATION);p
            
            np.random.shuffle(p)
            if(count < 10 ): item = i
            else:
                item = p[0]
                
            #Classify two sets and Create Xbetter
            As = []
            for l in range (len(p)):
                As.append(Population[p[l]].fitness_entropy)
            As.sort()
            Ss = []
            for m in range (0,Srange):
                Ss.append(As[m])
            Is = []
            for n in range (Srange,len(As)):
                Is.append(As[n])
            Ss_gene = []
            for o in range (POPULATION):
                if Population[p[o]].fitness_entropy in Ss:
                    xbetter = Population[p[o]].fitness_entropy
                    Ss_gene.append(Population[p[o]].gene)
            Xb = random.choice(Ss_gene)

            dr3 = np.empty((1,Individual_accuracy.DIMENSION))
            for k in range (Individual_accuracy.DIMENSION):
                if np.random.rand() < pd:
                    #TODO
                    dr3[0][k] =  MIN_VALUE + np.random.rand()*( MAX_VALUE - MIN_VALUE)
                    # dr3[0,k] =  np.random.randn()

                else:
                    dr3[0][k] = Population[p[3]].gene[0][k]

            fNP = float(i)/float(len(Population))
            F = np.random.normal(fNP,0.1)
            if Population[i].fitness_entropy in Ss:
                vector_xx = Population[item].gene + F  * (Population[p[1]].gene - Population[item].gene) + F * (Population[p[2]].gene - dr3)
                vector_xnew = vector_xx
            elif Population[i].fitness_entropy in Is:
                vector_xx = Population[item].gene + F * (Xb - Population[item].gene) + F * (Population[p[2]].gene - dr3)
                vector_xnew = vector_xx
            else:
                print("No match")
                break

            #CrossOver Operation
            jrand = np.random.randint(Individual_accuracy.DIMENSION)
            cNP = float(i)/float(len(Population))
            tCR=np.random.normal(cNP,0.1)
            for j in range(Individual_accuracy.DIMENSION):
                if ( np.random.rand() > tCR and not j==jrand):
                    vector_xnew[0][j] = Population[i].gene[0][j]
                if (vector_xnew[0][j] < MIN_VALUE or vector_xnew[0][j] > MAX_VALUE ):
                    #TODO
                    vector_xnew[0j] = MIN_VALUE + np.random.rand()*( MAX_VALUE - MIN_VALUE)
                    # vector_xnew[0,j] = np.random.randn()
            
            #Select Operation
            try:
                new_Individual = Individual_accuracy.Indivi(vector_xnew)
                fitness = Individual_accuracy.Indivi.Evaluate(Population[i])
                if (new_Individual.fitness_entropy > fitness and new_Individual.fitness_entropy < 1):
                    Population[i] = new_Individual
                    update += 1
                fitness_entropy.append(Population[i].fitness_entropy)
            except:
                fitness_entropy.append(Population[i].fitness_entropy)
                
            #entropy
            # fitness = Individual_accuracy.Indivi.Evaluate(Population[i],batch_xs,batch_ys)
            # if (new_Individual_accuracy.fitness_entropy < fitness):
            #Accuracy
            
            # if(new_Individual.fitness_entropy < 1):
            #     decode()

            

        print("UpdatedIndividualy=",update)
        UpdatedIndividual.append(update)
        if(count==1000):
            print("output last population")
            for a in range(len(Population)):
                struc = test.decodez(Population[a].gene[0].tolist())
                with open('struc_last.txt', 'a+') as f:
                    f.writelines(str(struc[0])+str(Population[a].fitness_entropy))
                    f.writelines("\n")
    # def correct_fitness():
    #     deocode
        
        
    def euclid(self,euclidlist,euclidgraph,euclidavelist):
        varlist = []
        euclidlist2 = []
        for i in range(POPULATION):
            varlist.append(Population[i].gene)
        varar = np.array(varlist)
        gravity=np.mean(varar, axis=0)

        for j in range(POPULATION):
            euclidtemp =0.0
            for k in range (Individual_accuracy.DIMENSION):
                euclidtemp += (varar[j,0,k] - gravity[0,k])**2
            euclid = np.sqrt(euclidtemp)
            euclidlist2.append(euclid)
        euclidlist3 = np.array(euclidlist2)
        #print("eurange=",len(euclidlist3))
        euclidlist.append(euclidlist3)
        euclidgraph.append(euclidlist3)
        euclidave = np.average(euclidlist3)
        
        for j in range(POPULATION):
            temp = (euclidlist3-euclidave)**2
        temp = temp/len(Population)
        euclidave = np.average(temp)
        euclidavelist.append(euclidave)
        
        

    
    def Average_Fitness(self):
        x =0.0
        y =0.0
        for i in range(POPULATION):
            x += Population[i].fitness_entropy
        string = "Entropy = " + str(x/POPULATION) + " Accuracy = " + str(y/POPULATION)
        print(string)
        return string

    def Best_Individual(self):
        Population.sort(key = operator.attrgetter('fitness_entropy'),reverse = False)
        # print( "Best Individual_accuracy")
        # print( Population[0].fitness_entropy , Population[0].gene)
        struc = test.decodez(Population[0].gene[0].tolist())
        # print(len(struc))
        with open('struc_bestindividual.txt', 'a+') as f:
            f.writelines(str(struc[0])+str(Population[0].fitness_entropy))
            f.writelines("\n")


    def Best_plot(self,Bestindividual,Bestgene,Bestaccuracy):
        Population.sort(key = operator.attrgetter('fitness_entropy'),reverse = False)
        Bestindividual.append(Population[0].fitness_entropy)
        Bestgene.append(Population[0].gene)
        Bestaccuracy.append(Individual_accuracy.Indivi.Evaluate(Population[0]))