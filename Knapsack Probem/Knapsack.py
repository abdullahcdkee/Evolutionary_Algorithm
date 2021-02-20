#!/usr/bin/env python
# coding: utf-8

# In[64]:


import numpy as np
import random
import matplotlib.pyplot as plt


# In[294]:


class EA:
    def __init__(self, value, weight, size):
        self.value = value
        self.weight = weight
        self.knapsack_capacity = 878
        self.num_items = 20
        self.population_size = size
        self.mutation_rate = 0.5
        self.population = np.random.randint(2, size = (self.population_size, self.num_items)).astype(int) #Initial random population

    def fitness_func(self):
        self.fitness = np.empty(len(self.population))
        
        for i in range(len(self.population)):
            sum_val = np.sum(self.population[i] * self.value)
            sum_weight = np.sum(self.population[i] * self.weight)
            
            if sum_weight <= self.knapsack_capacity:
                self.fitness[i] = sum_val
            else:
                self.fitness[i] = 0
    
    def selection(self, num, scheme, survivor): #num is the size of individuals to select and scheme defines the type of selection scheme
        self.selected_candidates = []
        
        if scheme == "Random":
            
            for i in range(num):
                random_num = random.randrange(len(self.population))
                self.selected_candidates.append(self.population[random_num])
        

        elif scheme == "FPS":

            norm_fitness = []
            sum_fitness = sum(self.fitness)
            for i in self.fitness:
                norm_fitness.append(i/sum_fitness)

            cum_prob = []
            a = 0
            
            for k in norm_fitness:
                a += k
                cum_prob.append(a)
            
            np.array(cum_prob)
            
            for i in range(num):
                random_num = random.uniform(0,1)
                
                for j in range(len(cum_prob)-1):
                    
                    if (random_num > cum_prob[j] and random_num <= cum_prob[j+1]):
                        self.selected_candidates.append(self.population[j+1])
                        break
                    elif (random_num <= cum_prob[0]):
                        self.selected_candidates.append(self.population[0])
                        break
        

        elif scheme == "Truncation":
            fitness_copy = np.copy(self.fitness)

            numbered_fitness_copy = list(enumerate(fitness_copy))
            numbered_fitness_copy.sort(key=lambda x:x[1])
            numbered_fitness_copy = numbered_fitness_copy[::-1]

            for i in range(num):
                index = numbered_fitness_copy[i][0]
                self.selected_candidates.append(self.population[index])
            

        elif scheme == "RBS":
            
            fitness_copy = np.copy(self.fitness)
            numbered_fitness_copy = list(enumerate(fitness_copy))
            numbered_fitness_copy.sort(key=lambda x:x[1])

            rank_dic = dict()
            rank_lst = []
            for i in range(1,len(self.fitness)+1):
                rank_dic[i] = numbered_fitness_copy[i-1]
                rank_lst.append(i)
            # print(rank_dic)

            norm_rank_dic = dict()
            norm_lst = []  
            for j in range(1,len(self.fitness)+1):
                norm_rank_dic[rank_lst[j-1]/sum(rank_lst)] = rank_dic[j]
                norm_lst.append(rank_lst[j-1]/sum(rank_lst))
            
            cum_lst = []
            a = 0
            for p in norm_lst:
                a+=p
                cum_lst.append(a)
            
            for k in range(num):
                random_num2 = random.uniform(0,1)
                
                for m in range (len(cum_lst)-1):
                    
                    if (random_num2 > cum_lst[m] and random_num2 <= cum_lst[m+1]):
                        fitness = norm_rank_dic[norm_lst[m+1]][1]
                        index = norm_rank_dic[norm_lst[m+1]][0]
                        self.selected_candidates.append(self.population[index])
                        break
                    elif random_num2 <= cum_lst[0]:
                        fitness = norm_rank_dic[norm_lst[0]][1]
                        index = norm_rank_dic[norm_lst[0]][0]
                        self.selected_candidates.append(self.population[index])
                        break
        

        elif scheme == 'BT':
            for i in range(num):
                cand_dic = dict()
                
                for j in range(2):
                    rand = random.randrange(len(self.population))
                    cand_dic[rand] = self.fitness[rand]
                
                max_index = max(cand_dic, key=cand_dic.get)
                self.selected_candidates.append(self.population[max_index])
       

        else:
            print("Please enter a valid scheme")
        
        
        if survivor == True:
            self.population = np.array(self.selected_candidates)
        else:
            self.selected_candidates = np.array(self.selected_candidates)
   

    def crossover(self):
        selected_candidates_copy = np.copy(self.selected_candidates)
        selected_candidates_copy = selected_candidates_copy.tolist()
        temp_offsprings = []
        
        #One point crossover
        for i in range(0,len(selected_candidates_copy)-1,2):
            p = selected_candidates_copy[i]
            q = selected_candidates_copy[i+1]
            crossover_point = int(random.randint(1,len(p)-2)/2)
            p1 = p[:crossover_point]
            p2 = p[crossover_point:]
            q1 = q[:crossover_point]
            q2 = q[crossover_point:]
            random.shuffle(q1)
            random.shuffle(q2)
            offspring1 = np.array(p1+q2)
            offspring2 = np.array(q1+p2)
            temp_offsprings.append(offspring1)
            temp_offsprings.append(offspring2)
        
        self.offsprings = np.array(temp_offsprings)
   

    def mutation(self):
        for i in range(len(self.offsprings)):
            random_num = random.uniform(0,1)

            if random_num < self.mutation_rate:
                random_idx = random.randint(0, len(self.offsprings)-1)
                random_idx2 = random.randint(0, len(self.offsprings)-1)          
                counter = 0

                while (self.offsprings[i][random_idx] == self.offsprings[i][random_idx2] and counter < 100):  
                    random_idx = random.randint(0, len(self.offsprings)-1)
                    random_idx2 = random.randint(0, len(self.offsprings)-1)
                    counter += 1

                self.offsprings[i][random_idx], self.offsprings[i][random_idx2] =  self.offsprings[i][random_idx2], self.offsprings[i][random_idx]
        
        self.population = np.concatenate((self.population,self.offsprings))




def solve_knapsack(generations, iterations, parent_size, population_size, value, weight):

    total_BFS = []
    total_ASF = []

    for i in range(generations):
        total_BFS.append([])
        total_ASF.append([])

    for i in range(iterations):
        knapsack_prob = EA(value, weight, population_size)

        for j in range(generations):
            knapsack_prob.fitness_func()

            max_generation_fitness = np.amax(knapsack_prob.fitness)

            if len(total_BFS[j])==0:
                total_BFS[j].append(max_generation_fitness)
            elif max_generation_fitness >= total_BFS[j][-1]:
                total_BFS[j].append(max_generation_fitness)
            else:
                total_BFS[j].append(total_BFS[j][-1])

            avg_generation_fitness = (np.average(knapsack_prob.fitness) + sum(total_ASF[j])) / (len(total_ASF[j])+1)
            total_ASF[j].append(avg_generation_fitness)

            knapsack_prob.selection(parent_size, "BT", False) 
            knapsack_prob.crossover()
            knapsack_prob.mutation()
            knapsack_prob.fitness_func()
            knapsack_prob.selection(population_size, "Truncation", True) #survivor selection hence last argument is True


    generations_list = []
    for i in range(1,generations+1):
        generations_list.append(i)


        BFS = []
    ASF = []
    gen_file = open("gen_BFS_ASF.txt","w")
    for i in range(0,generations):
        BFS.append(np.average(total_BFS[i]))
        ASF.append(np.average(total_ASF[i]))
        gen_file.write(str(generations_list[i]))
        gen_file.write(" | ")
        gen_file.write(str(np.average(total_BFS[i])))
        gen_file.write(" | ")
        gen_file.write(str(np.average(total_ASF[i])))
        gen_file.write("\n")

    print("BFS Final Value:", BFS[-1])

    BFS_iter_gen = []
    ASF_iter_gen = []
    BFS_iter = open("BFS_iteration_gen.txt","w")
    ASF_iter = open("ASF_iteration_gen.txt","w") 
    for i in range(0,generations):
      temp_BFS = []
      temp_ASF = []
      BFS_iter.write(str(generations_list[i]))
      ASF_iter.write(str(generations_list[i]))
      for j in range(0,iterations):
        temp_BFS.append(total_BFS[i][j])
        temp_ASF.append(total_ASF[i][j])
        BFS_iter.write(" | ")
        ASF_iter.write(" | ")
        BFS_iter.write(str(total_BFS[i][j]))
        ASF_iter.write(str(total_ASF[i][j]))
      BFS_iter_gen.append(temp_BFS)
      ASF_iter_gen.append(temp_ASF)
      BFS_iter.write(" | ")
      BFS_iter.write(str(np.average(temp_BFS)))
      BFS_iter.write("\n")
      ASF_iter.write(" | ")
      ASF_iter.write(str(np.average(temp_ASF)))
      ASF_iter.write("\n")
    BFS_iter.close()
    ASF_iter.close()

    plt.close('all')
    plt.title('Fitness vs Generation (FPS and Truncation)')
    plt.plot(generations_list, BFS, label="BFS")
    plt.ylabel('Fitness')
    plt.xlabel('Generations')
    plt.plot(generations_list, ASF, label="ASF")
    plt.legend(framealpha=1, frameon=True);
    plt.savefig('FPSandT.png')
    plt.show()



value = np.loadtxt('knapsack_dataset.txt', skiprows = 1, usecols = 0)
weight = np.loadtxt('knapsack_dataset.txt', skiprows = 1, usecols = 1)
generations = 100
iterations = 50
parent_size = 15
population_size = 40


solve_knapsack(generations, iterations, parent_size, population_size, value, weight)

