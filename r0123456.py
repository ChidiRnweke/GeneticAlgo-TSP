import Reporter
import numpy as np
from random import sample
from numba import njit, types

# Modify the class name to match your student number.


class r0786701:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    def elimination(self, population, numberOfSelections, kTournment, distanceMatrix):
        populationSize = len(population)
        newPopulation = np.zeros((numberOfSelections, population.shape[1]))
        for idx in range(numberOfSelections):
            randomIndices = sample(range(populationSize), kTournment)
            bestFit = 1e9
            bestIndice = randomIndices[0]
            for indice in randomIndices:
                fit = fitness(distanceMatrix, population[indice])
                if fit < bestFit:
                    bestFit = fit
                    bestIndice = indice
            newPopulation[idx] = population[bestIndice].copy()
        return newPopulation

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()
        # Parameters
        populationSize = 500
        maxIterations = 3000
        kTournment = 3
        numberOfOffspring = 500
        sameSolutionIterations = 20
        mu = 0.15
        # fitness.map = {}

        population = initialize(distanceMatrix, populationSize)

        iteration = 0
        meanObjective = 1.0
        bestObjective = 0.0

        prevSolution = 1e9
        tolerance = 0.001
        sameSolutionCount = 0

        while iteration < maxIterations and sameSolutionCount < sameSolutionIterations:
            meanObjective = 0.0
            bestObjective = 0.0
            bestSolution = np.array([1, 2, 3, 4, 5])

            # Your code here.
            offspring = np.zeros((numberOfOffspring, distanceMatrix.shape[0]))
            for i in range(0, numberOfOffspring, 2):
                parent1 = selection(population, kTournment, distanceMatrix)
                parent2 = selection(population, kTournment, distanceMatrix)
                offspring1, offspring2 = recombination(distanceMatrix, parent1, parent2)
                offspring[i] = offspring1.copy()
                offspring[i + 1] = offspring2.copy()
            population = np.vstack((population, offspring))

            population = self.elimination(
                population, populationSize, kTournment, distanceMatrix
            )
            for individual in population:
                individual = k_opt(individual, distanceMatrix, 1)
                probability = np.random.uniform(0, 1)
                if probability < mu:
                    mutate(individual)

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            populationEvaluation = evaluatePopulation(distanceMatrix, population)
            meanObjective = populationEvaluation[0]
            bestObjective = populationEvaluation[1]
            bestSolution = populationEvaluation[2]
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)

            # checking if the objectscore reduces or not
            difference = prevSolution - bestObjective
            if difference < tolerance:
                sameSolutionCount += 1
            else:
                sameSolutionCount = 0
            prevSolution = bestObjective

            print(f"Time left: {timeLeft}")
            if timeLeft < 0:
                break
            iteration += 1

        # Your code here.
        return 0


# Class representing individuals


class Individual:
    def __init__(self, TSP: np.array, size: int = 0, path: np.array = None):
        """[Initializes a new path individual with a given size. If an array is given it uses this array instead of randomizing.
            The class has a path variable representing the chosen path as a numpy array.]

        Args:
            TSP (np.array): [The cost matrix]
            size (int, optional): [Amount of cities]. Defaults to 0.
            path (np.array, optional): [Initialise with a given path]. Defaults to None.
        """
        if path is None:
            self.path = np.arange(size)
            np.random.shuffle(self.path)
        else:
            self.path = path

        self.fitness = fitness(TSP, self.path)


@njit(nogil=True)
def k_opt(candidate: np.array, problem: np.array, k: int) -> np.array:
    """[Creates the full neighbour sructure for and candidate and selects the best one]

    Args:
        candidate (Individual): [The given candidate]
        

    Returns:
        Individual: [The best candidate in the neighbourhood]
    """
    for _ in range(k):
        best_path = candidate
        best_fit = fitness(problem, candidate)
        initial = np.copy(candidate)
        # neighbourhood = np.empty(initial.shape[1], dtype=object)
        for i in range(initial.size):
            for j in range(i + 1, initial.size):
                neighbour = initial.copy()
                neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
                fit = fitness(TSP=problem, path=neighbour)
                best_path = neighbour if fit < best_fit else best_path
    return best_path


# Create the initial population


def initialize(TSP, populationSize: int) -> np.ndarray:
    rng = np.random.default_rng()

    population = np.arange(TSP.shape[1])
    population = np.broadcast_to(population, (populationSize, TSP.shape[1]))
    population = rng.permuted(population, axis=1)
    out = []
    for row in population:
        row = k_opt(row, TSP, 2)
        out.append(row)
    return np.array(out)


@njit()
def mutate(individual: np.array) -> None:
    indices = np.random.randint(low=0, high=len(individual), size=2)
    individual[indices[0]], individual[indices[1]] = (
        individual[indices[1]],
        individual[indices[0]],
    )


def recombination(TSP, par1: np.array, par2: np.array) -> tuple:
    # PMX
    parent1 = np.copy(par1)
    parent2 = np.copy(par2)
    index1 = sample(range(1, int(parent1.shape[0] / 2)), 1)
    index2 = sample(range(index1[0] + 2, parent1.shape[0] - 1), 1)
    indices = np.array([index1[0], index2[0]])
    splitp1 = np.array_split(parent1, indices)
    splitp2 = np.array_split(parent2, indices)
    o1 = np.concatenate((splitp1[0], splitp2[1], splitp1[2]))
    o2 = np.concatenate((splitp2[0], splitp1[1], splitp2[2]))

    while np.unique(o1).size != o1.size:
        for key, val in zip(splitp1[1], splitp2[1]):
            splitp1[0][splitp1[0] == val] = key
            splitp1[2][splitp1[2] == val] = key
        o1 = np.concatenate((splitp1[0], splitp2[1], splitp1[2]))
    while np.unique(o2).size != o2.size:
        for key, val in zip(splitp1[1], splitp2[1]):
            splitp2[0][splitp2[0] == key] = val
            splitp2[2][splitp2[2] == key] = val
        o2 = np.concatenate((splitp2[0], splitp1[1], splitp2[2]))
    return o1, o2


def selection(population: np.array, k: int, TSP):
    rng = np.random.default_rng()
    selected = rng.choice(population, k, axis=0)
    highest = np.argmin([fitness(TSP=TSP, path=ind) for ind in selected])
    return selected[highest]


@njit()
# Calculates the fitness of one individual
def fitness(TSP: np.array, path: np.array) -> float:
    """[Calculates the fitness of an individual]

    Args:
        TSP (np.array): [The given problem]
        path (np.array): [The path of the individual]

    Returns:
        float: [The fitness value]
    """
    totalDistance = 0
    # string_rep = str(path)
    # if string_rep in fitness.map.keys():
    #   return fitness.map[string_rep]
    # else:
    for i in range(path.shape[0]):
        departingCity = int(path[i - 1])
        arrivingCity = int(path[i])
        totalDistance += TSP[departingCity, arrivingCity]
        if totalDistance == np.inf:
            return 1e99999999999
    # fitness.map[string_rep] = totalDistance
    return totalDistance


# Calculates the mean fitness of the population and the best fitting individual (Needed for the Reporter class)
@njit(locals={"meanfit": types.float64})
def evaluatePopulation(TSP, population):
    bestFit = 1e99999999999
    bestIndividual = None
    fitnesses = np.array([fitness(TSP=TSP, path=ind) for ind in population])
    meanfit = np.mean(fitnesses)
    bestidx = np.argmin(fitnesses)
    bestFit = fitnesses[bestidx]
    bestIndividual = population[bestidx]
    return (meanfit, bestFit, bestIndividual)


if __name__ == "__main__":
    algorithm = r0786701()

    algorithm.optimize("tour29.csv")

