import numpy as np
import random
import matplotlib.pyplot as plt
import math


class EA:
    def __init__(self, nodelist, size):
        self.nodelist = nodelist
        self.population_size = size
        self.mutation_rate = 0.5
        self.population = [] 

        for i in range(population_size):        #Initial random population
            individual = list(range(1, 195))
            random.shuffle(individual) 
            self.population.append(individual)

    def calculate_distance(self, node1, node2):
        distance = (math.sqrt( (node1[0]-node2[0])**2 + (node1[1]-node2[1])**2 ))
        return distance

    def fitness_func(self): #By default fitness is being maximized, for minimization the value's reciprocal is stored
        self.fitness = []

        for i in range(len(self.population)):
            individual_fitness = 0

            for j in range(len(self.population[i])-1):
                idx1, idx2 = self.population[i][j], self.population[i][j+1]
                node1 = self.nodelist[idx1-1]   #subtract 1 to index nodelist
                node2 = self.nodelist[idx2-1]
                individual_fitness += self.calculate_distance(node1, node2)
            
            individual_fitness += self.calculate_distance(nodelist[0], nodelist[-1])
            self.fitness.append(1/individual_fitness)

        maximum = 0
        idx = 0

        for i in range(0,len(self.fitness)):
            if self.fitness[i] > maximum:
                maximum = self.fitness[i]
                idx = i
        self.max_solution = self.population[idx]
    
    
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
        
        #Two point crossover
        for i in range(0,len(selected_candidates_copy)-1,2):
            parent1 = selected_candidates_copy[i]
            parent2 = selected_candidates_copy[i+1]

            child1 = self.create_child(parent1, parent2)
            child2 = self.create_child(parent2, parent1)
            
            offspring1 = np.array(child1)
            offspring2 = np.array(child2)
            temp_offsprings.append(offspring1)
            
        self.offsprings = np.array(temp_offsprings)

   
    def create_child(self,parent1, parent2):
        crossover_point1 = int(random.randint(0,len(parent1)))
        crossover_point2 = int(random.randint(0,len(parent1)))

        while crossover_point2 == crossover_point1:
            crossover_point2 = int(random.randint(0,len(parent1)))

        start = min(crossover_point1, crossover_point2)
        end = max(crossover_point1, crossover_point2)
        insertion = parent1[start:end]
        stored = []
        child = [None]*len(parent1)

        j = 0
        for i in range(len(parent1  )):

            if i>=start and i<end:
                child[i] = insertion[i-start]
            else:
                while (parent2[j] in insertion or parent2[j] in stored or parent2[j] in child) and j<len(parent2):
                    j += 1
                child[i] = parent2[j]
                stored.append(parent2[j])

        return child

    def mutation(self):
        for i in range(len(self.offsprings)):
            random_num = random.uniform(0,1)

            if random_num < self.mutation_rate:
                random_idx = random.randint(0, len(self.population[0])-1)
                random_idx2 = random.randint(0, len(self.population[0])-1)          
                counter = 0
                self.offsprings[i][random_idx], self.offsprings[i][random_idx2] =  self.offsprings[i][random_idx2], self.offsprings[i][random_idx]
        
        self.population = np.concatenate((self.population,self.offsprings))



def solve_tsp(generations, iterations, parent_size, population_size, nodelist):

    total_BFS = []
    total_ASF = []

    for i in range(generations):
        total_BFS.append([])
        total_ASF.append([])

    for i in range(iterations):
        tsp_prob = EA(nodelist, population_size)
        print('iteration:', i)
        for j in range(generations):

            tsp_prob.fitness_func()

            max_generation_fitness = np.amax(tsp_prob.fitness)

            if len(total_BFS[j])==0:
                total_BFS[j].append(max_generation_fitness)
            elif max_generation_fitness >= total_BFS[j][-1]:
                total_BFS[j].append(max_generation_fitness)
            else:
                total_BFS[j].append(total_BFS[j][-1])
            print('generation:',j,'  BFS:', int(1/total_BFS[j][0]))

            avg_generation_fitness = (np.average(tsp_prob.fitness) + sum(total_ASF[j])) / (len(total_ASF[j])+1)
            total_ASF[j].append(avg_generation_fitness)

            tsp_prob.selection(parent_size, "BT", False) 
            tsp_prob.crossover()
            tsp_prob.mutation()
            tsp_prob.fitness_func()
            tsp_prob.selection(population_size, "Truncation", True) #survivor selection hence last argument is True

    print(tsp_prob.max_solution)
    empty = []
    for i in tsp_prob.max_solution:
        if i not in empty:
            empty.append(i)
        else:
            print('DUPLICATION')

    generations_list = []
    for i in range(1,generations+1):
        generations_list.append(i)


        BFS_gen = []
    ASF_gen = []
    gen_file = open("gen_BFS_ASF.txt","w")
    for i in range(0,generations):
        BFS_gen.append(1/np.average(total_BFS[i]))
        ASF_gen.append(1/np.average(total_ASF[i]))
        gen_file.write(str(generations_list[i]))
        gen_file.write(" | ")
        gen_file.write(str(1/np.average(total_BFS[i])))
        gen_file.write(" | ")
        gen_file.write(str(1/np.average(total_ASF[i])))
        gen_file.write("\n")

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
        temp_BFS.append(1/total_BFS[i][j])
        temp_ASF.append(1/total_ASF[i][j])
        BFS_iter.write(" | ")
        ASF_iter.write(" | ")
        BFS_iter.write(str(1/total_BFS[i][j]))
        ASF_iter.write(str(1/total_ASF[i][j]))
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
    plt.title('Fitness vs Generation (Truncation and FPS)')
    plt.plot(generations_list, BFS_gen, label="BFS")
    plt.ylabel('Fitness')
    plt.xlabel('Generations')
    plt.plot(generations_list, ASF_gen, label="ASF")
    plt.legend(framealpha=1, frameon=True);
    plt.savefig('truncationandFPS.png')
    plt.show()


infile = open('qa194.tsp', 'r')

# Read instance header
Name = infile.readline().strip().split()[2] # NAME
infile.readline()
infile.readline()
infile.readline()
Dimension = infile.readline().strip().split()[2] # DIMENSION
EdgeWeightType = infile.readline().strip().split()[2] # EDGE_WEIGHT_TYPE
infile.readline()

nodelist = []
N = int(Dimension)
for i in range(0, N):
    x,y = infile.readline().strip().split()[1:]
    nodelist.append([float(x), float(y)])



# Close input file
infile.close()

generations = 20
iterations = 1
parent_size = 350
population_size = 500

solve_tsp(generations, iterations, parent_size, population_size, nodelist)

