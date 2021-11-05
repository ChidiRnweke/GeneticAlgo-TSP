import Reporter
import numpy as np
from random import sample
# Modify the class name to match your student number.
class r0123456:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)

	# The evolutionary algorithm's main loop
	def optimize(self, filename):
		# Read distance matrix from file.		
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

		#Parameters
		populationSize = 30
		maxIterations = 50

		#Initialize the population
		population = initialize(distanceMatrix.shape[0], populationSize)

		#Main loop TODO add a stopping condition beside a max number of iterations
		iteration = 0 
		while( iteration < maxIterations ):
			meanObjective = 0.0
			bestObjective = 0.0
			bestSolution = np.array([1,2,3,4,5])

			# Your code here.

			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution 
			#    with city numbering starting from 0
			populationEvaluation = evaluatePopulation(distanceMatrix, population)
			meanObjective = populationEvaluation[0]
			bestObjective = populationEvaluation[1]
			bestSolution = populationEvaluation[2].path
			timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
			print(f"Time left: {timeLeft}")
			if timeLeft < 0:
				break
			iteration += 1

		# Your code here.
		return 0

#Class representing individuals
class Individual():

	#Initializes a new path individual with a given size. If an array is given it uses this array instead of randomizing.
	#The class has a path variable representing the chosen path as a numpy array.
	def __init__(self, size: int, path: np.array = None):
		if path is None:
			self.path = np.arange(size)
			np.random.shuffle(self.path)
		else:
			self.path = path

#Create the initial population 
def initialize(individualSize: int, populationSize: int) -> list:
	population = []
	for _ in range(populationSize):
		individual = Individual(individualSize)
		population.append(individual)
	return np.array(population)

def mutate(individual: Individual) -> None:
	indices = sample(range(len(individual.path)),2)
	individual.path[indices[0]], individual.path[indices[1]] = individual.path[indices[1]], individual.path[indices[0]].copy()
	return

def recombination(parent1: np.array, parent2: np.array) -> None:
	#PMX still needs random splits, right now it's deterministic
    splitp1 = np.array_split(parent1,3)
    splitp2 = np.array_split(parent2,3)
    for key,val in zip(splitp1[1],splitp2[1]):
        splitp1[0][splitp1[0]==val] = key
        splitp1[2][splitp1[2]==val] = key
        splitp1[0][splitp1[0]==val] = key
        splitp1[2][splitp1[2]==val] = key

        splitp2[0][splitp2[0]==key] = val
        splitp2[2][splitp2[2]==key] = val
        splitp2[0][splitp2[0]==key] = val
        splitp2[2][splitp2[2]==key] = val
    o1 = np.concatenate((splitp1[0], splitp2[1], splitp1[2]))
    o2 = np.concatenate((splitp2[0], splitp1[1], splitp2[2]))
    return Individual(o1), Individual(o2)

#Calculates the fitness of one individual
def fitness(TSP: np.array, individual: Individual) -> int:
	totalDistance = 0
	#For every two following cities add the distance between them to the sum.
	for i in range(individual.path.shape[0]):
		departingCity = individual.path[i-1]
		arrivingCity = individual.path[i]
		#Assumed the rows represent departing cities and the column ariving cities
		totalDistance += TSP[departingCity, arrivingCity]
	return totalDistance

#Calculates the mean fitness of the population and the best fitting individual (Needed for the Reporter class)
def evaluatePopulation(TSP, population):
	bestFit = np.inf
	sumFit = 0
	bestIndividual = None
	for individual in population:
		fit = fitness(TSP, individual)
		sumFit += fit
		if fit < bestFit:
			bestFit = fit
			bestIndividual = individual
	return (sumFit/population.shape[0], bestFit, bestIndividual)


if __name__ == "__main__":
	algorithm = r0123456()
	algorithm.optimize("tour29.csv")