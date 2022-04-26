# coding:UTF-8
import Individual
import numpy as np
import operator
import random
import main
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf


#parameters
POPULATION = 0
MAX_VALUE = Individual.MAX_VALUE
MIN_VALUE = Individual.MIN_VALUE
F_l        =  0.1
F_u        =  0.9
#CR         =  0.5
gamma1     =  0.1
gamma2     =  0.1
eps = 1e-3
Population = []
SPACE = MAX_VALUE - MIN_VALUE
class IDE_NN:
    def __init__(self, popu,u_CR=0.5,u_F=0.5,p2=0.05,c=0.1):
        global POPULATION 
        POPULATION = popu
        tf.get_logger().setLevel('WARNING')
        for pop in range(POPULATION):
            Population.append(Individual.Indivi())
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
        gcount = c / main.GENERATION
        float (gcount)
        #Set Ps Pd and dr3 for mutant vector
        ps = 0.1 + 0.9*10 ** (5 * (gcount-1))
        
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

            dr3 = np.empty((1,Individual.DIMENSION))
            for k in range (Individual.DIMENSION):
                if np.random.rand() < pd:
                    dr3[0,k] =  MIN_VALUE + np.random.rand()*( MAX_VALUE - MIN_VALUE)
                else:
                    dr3[0,k] = Population[p[3]].gene[0,k]

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
            jrand = np.random.randint(Individual.DIMENSION)
            cNP = float(i)/float(len(Population))
            tCR=np.random.normal(cNP,0.1)
            for j in range(Individual.DIMENSION):
                if ( np.random.rand() > tCR and not j==jrand):
                    vector_xnew[0,j] = Population[i].gene[0,j]
                if (vector_xnew[0,j] < MIN_VALUE or vector_xnew[0,j] > MAX_VALUE):
                    vector_xnew[0,j] = MIN_VALUE + np.random.rand()*( MAX_VALUE - MIN_VALUE)
            
            #Select Operation
            new_Individual = Individual.Indivi(vector_xnew)
            #entropy
            # fitness = Individual.Indivi.Evaluate(Population[i],batch_xs,batch_ys)
            # if (new_Individual.fitness_entropy < fitness):
            #Accuracy
            fitness = Individual.Indivi.Evaluate(Population[i])

            if (new_Individual.fitness_entropy < fitness and new_Individual.fitness_entropy >0):
                Population[i] = new_Individual
                update += 1
            fitness_entropy.append(Population[i].fitness_entropy)

        print("UpdatedIndividual=",update)
        UpdatedIndividual.append(update)
        
        
    def euclid(self,euclidlist,euclidgraph,euclidavelist):
        varlist = []
        euclidlist2 = []
        for i in range(POPULATION):
            varlist.append(Population[i].gene)
        varar = np.array(varlist)
        gravity=np.mean(varar, axis=0)

        for j in range(POPULATION):
            euclidtemp =0.0
            for k in range (Individual.DIMENSION):
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
        Population.sort(key = operator.attrgetter('fitness_entropy'),reverse = True)
        print( "Best Individual")
        print( Population[0].fitness_entropy , Population[0].gene)

    def Best_plot(self,Bestindividual,Bestgene,Bestaccuracy):
        Population.sort(key = operator.attrgetter('fitness_entropy'),reverse = True)
        Bestindividual.append(Population[0].fitness_entropy)
        Bestgene.append(Population[0].gene)
        Bestaccuracy.append(Individual.Indivi.Evaluate(Population[0]))