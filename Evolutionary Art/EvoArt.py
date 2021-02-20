from PIL import Image, ImageDraw, ImagePath
import random
import numpy as np
from matplotlib import pyplot as plt
import aggdraw
# import cv2
import math
import os


class Polygon:
	def __init__(self, vertex_low = 3, vertex_high = 3, coordinate_range = [200,200]):
		self.coordinate_range = coordinate_range

		if vertex_low == vertex_high:
			self.vertices_number = vertex_low
		elif vertex_high > vertex_low:
			self.vertices_number = int(random.randint(vertex_low, vertex_high))
		else:
			self.vertices_number = vertex_low

		self.generate_polygon()

	def generate_coordinate(self):
		rand_coordinate1 = int(random.uniform(0, self.coordinate_range[0]))
		rand_coordinate2 = int(random.uniform(0, self.coordinate_range[1]))

		return rand_coordinate1, rand_coordinate2

	def generate_color(self):
		self.color = tuple([random.randint(0, 255) for i in range(4)])

	def generate_polygon(self):
		self.vertices = []

		for i in range(self.vertices_number):
			self.vertices.append(self.generate_coordinate())

		self.generate_color()
		self. poly_representation = [self.vertices, self.color] #represents one polygon
		return self.poly_representation

	def get_coordinates(self, rand_value):

		length = self.poly_representation[0]

		coordinates = []
		for i in range(rand_value):
			coordinates.append(self.poly_representation[0][i])

		return coordinates

	def set_coordinates(self, coordinates):

		for i in range(len(self.poly_representation[0])):
			self.poly_representation[0][i] = coordinates[i]


class EvoArt:
	def __init__(self, population_size, polygon_number, original_image):
		self.max_coordinate = min(original_image.size[0], original_image.size[1])
		self.coordinate_range = [original_image.size[0], original_image.size[1]]
		self.original_image = original_image.convert('RGBA')
		self.original_image_array = np.array(original_image)
		self.size = original_image.size
		self.candidate_array = np.zeros((self.size[0], self.size[1], 4))
		self.population_size = population_size
		self.polygon_number = polygon_number
		self.mutation_rate = 0.5

		new_image = ImagePath.Path(((0,0),(0,self.coordinate_range[1]),(self.coordinate_range[0],self.coordinate_range[1]),(self.coordinate_range[0],0))).getbbox()  #hardcoded for MonaLisa pic
		size = list(map(int, map(math.ceil, new_image[2:]))) 
		self.test_img = Image.new("RGBA", size)  
		self.background = Image.new("RGBA", size, (255,0,0,0))
		self.random_initialization(population_size, polygon_number)


	def add_polygon(self, list_of_polygons):
		list_of_polygons.append(Polygon())

	def draw_solution(self, list_of_polygons):

		self.background = Image.new("RGBA", self.size, (255,0,0,0))
		for poly in list_of_polygons:
			self.test_img = Image.new("RGBA", self.size)  
			self.draw_img = ImageDraw.Draw(self.test_img)
			self.draw_img.polygon(poly.vertices, fill = poly.color)
			self.background = Image.alpha_composite(self.background, self.test_img)

		candidate_array = np.array(self.background)
		del self.draw_img

		return candidate_array


	# def draw_polygon(self):

	#   self.test_img = Image.new("RGBA", self.size)  
	#   self.draw_poly = ImageDraw.Draw(self.test_img)
	#   if (len(self.test_solution) > 0):
	#       poly = self.test_solution[-1]
	#       self.draw_poly.polygon(poly.vertices, fill = poly.color)

	#   self.background.paste(self.test_img, mask = self.test_img)
	#   del self.draw_poly


	def save_solution(self, iteration):
		plt.imshow(self.background)
		plt.savefig("C:/Users/abdul/Desktop/Semester 8/Computational Intelligence/Assignment 1/Mona Lisa/Solution/Sol/"+str(iteration+1)+".png")
		plt.clf()


	def fitness_func(self):
		self.fitness = []

		for individual in self.population:
			candidate_array = self.draw_solution(individual)
			pix_difference = (abs(self.original_image_array - candidate_array)).sum()
			norm_diff = pix_difference/(self.size[0]*self.size[1])

			self.fitness.append(norm_diff)


	def random_initialization(self, population_size,polygons, vertex_low = 3, vertex_high = 3):
		self.population = []

		for i in range(population_size):
			polygons_list = []
			for j in range(polygons):
				polygons_list.append(Polygon(vertex_low, vertex_high))
			self.population.append(polygons_list)

		self.fitness_func()


	def selection(self, num, scheme, survivor): #num is the size of individuals to select and scheme defines the type of selection schem
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

		start = min(crossover_point1, crossover_point2)
		end = max(crossover_point1, crossover_point2)
		insertion = parent1[start:end]
		child = [None]*len(parent1)

		j = 0
		for i in range(len(parent1 )):

			if i>=start and i<end:
				child[i] = insertion[i-start]
			else:
				child[i] = parent2[j]

		return child


	def mutate_child(self):
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


orig_img = Image.open("mona.png")

parent_size = 2
population_size = 5
iterations = 1
generations = 50

total_BFS = []

for i in range(generations):
	total_BFS.append([])

for i in range(iterations):
	EvoLisa = EvoArt(population_size, 50, orig_img)

	for j in range(generations):
		EvoLisa.fitness_func()

		max_generation_fitness = np.amax(EvoLisa.fitness)
		
		if len(total_BFS[j])==0:
			total_BFS[j].append(max_generation_fitness)
		elif max_generation_fitness >= total_BFS[j][-1]:
			total_BFS[j].append(max_generation_fitness)
		else:
			total_BFS[j].append(total_BFS[j][-1])

		EvoLisa.selection(parent_size, 'BT', False)
		EvoLisa.crossover()
		EvoLisa.mutate_child()
		EvoLisa.fitness_func()
		EvoLisa.selection(population_size, 'Truncation', True)
		EvoLisa.save_solution(j)
