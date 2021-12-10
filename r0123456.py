import Reporter
import numpy as np
from random import sample

# Modify the class name to match your student number.


class r0786701:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    def elimination(self, population, numberOfSelections, kTournment, distanceMatrix):
        populationSize = len(population)
        newPopulation = []
        for idx in range(numberOfSelections):
            randomIndices = sample(range(populationSize), kTournment)
            bestFit = 1e9
            bestIndice = randomIndices[0]
            for indice in randomIndices:
                fit = population[indice].fitness
                if fit < bestFit:
                    bestFit = fit
                    bestIndice = indice
            newPopulation.append(population[bestIndice])
        return np.array(newPopulation)

    def applyMutation(self, population):
        newPopulation = []
        for idx in range(int(len(population) / 2)):
            path1 = population[idx].path
            path2 = population[int(len(population) / 2) + idx].path
            newPath1, newPath2 = self.mutatePaths(path1, path2)
            population[idx].path = newPath1
            population[int(len(population) / 2) + idx].path = newPath2
            newPopulation.append(population[idx])
            newPopulation.append(population[int(len(population) / 2) + idx])
        return np.array(newPopulation)

    def mutatePaths(self, path1, path2):
        new_path1 = np.zeros(path1.shape, dtype=path1.dtype) - 1
        new_path1[: int(path1.shape[0] / 2)] = path1[: int(path1.shape[0] / 2)]

        new_path2 = np.zeros(path2.shape, dtype=path1.dtype) - 1
        new_path2[: int(path2.shape[0] / 2)] = path2[: int(path2.shape[0] / 2)]

        for val in range(int(path1.shape[0] / 2), path1.shape[0]):
            if path2[val] not in new_path1:
                new_path1[val] = path2[val]
            if path1[val] not in new_path2:
                new_path2[val] = path1[val]

        for i in range(path1.shape[0]):
            if i not in new_path1:
                new_path1[np.where(new_path1 == -1)[0][0]] = i
            if i not in new_path2:
                new_path2[np.where(new_path2 == -1)[0][0]] = i
        return new_path1, new_path2

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.genfromtxt(
            file, delimiter=",", missing_values="inf", filling_values=1000000000000
        )
        file.close()
        # Parameters
        populationSize = 5000
        maxIterations = 3000
        kTournment = 3
        numberOfOffspring = 5000
        sameSolutionIterations = 20
        mu = 0.15

        # Initialize the population
        population = initialize(distanceMatrix, populationSize)

        # Main loop TODO add a stopping condition beside a max number of iterations
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
            offspring = []
            for count in range(numberOfOffspring):
                parent1 = selection(population, kTournment)
                parent2 = selection(population, kTournment)
                offspring1, offspring2 = recombination(distanceMatrix, parent1, parent2)
                offspring.append(offspring1)
                offspring.append(offspring2)
            population = np.append(population, offspring)

            population = self.elimination(
                population, populationSize, kTournment, distanceMatrix
            )
            for individual in population:
                # individual = k_opt(individual, distanceMatrix, 1)
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
            bestSolution = populationEvaluation[2].path
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


def k_opt(candidate: Individual, problem: np.array, k: int) -> Individual:
    """[Creates the full neighbour sructure for and candidate and selects the best one]

    Args:
        candidate (Individual): [The given candidate]
        

    Returns:
        Individual: [The best candidate in the neighbourhood]
    """
    for _ in range(k):
        best_path = candidate.path
        best_fit = candidate.fitness
        initial = np.copy(candidate.path)
        # neighbourhood = np.empty(initial.shape[1], dtype=object)
        for i in range(initial.size):
            for j in range(i + 1, initial.size):
                neighbour = initial.copy()
                neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
                fit = fitness(TSP=problem, path=neighbour)
                best_path = neighbour if fit < best_fit else best_path
        candidate = Individual(TSP=problem, path=best_path)
    return Individual(TSP=problem, path=best_path)


# Create the initial population
def initialize(TSP, populationSize: int) -> np.ndarray:
    population = []
    for _ in range(populationSize):
        individual = Individual(TSP, TSP.shape[0])
        # individual = k_opt(individual, TSP, 1)
        population.append(individual)
    return np.array(population)


def mutate(individual: Individual) -> None:
    indices = sample(range(len(individual.path)), 2)
    individual.path[indices[0]], individual.path[indices[1]] = (
        individual.path[indices[1]],
        individual.path[indices[0]].copy(),
    )


def recombination(TSP, par1: Individual, par2: Individual) -> tuple:
    # PMX
    parent1 = np.copy(par1.path)
    parent2 = np.copy(par2.path)
    index1 = sample(range(1, int(parent1.size / 2)), 1)
    index2 = sample(range(index1[0] + 2, parent1.size - 1), 1)
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
    return Individual(TSP, path=o1), Individual(TSP, path=o2)


def selection(population: np.array, k: int):
    selected = np.random.choice(population, k)
    highest = np.argmin([ind.fitness for ind in selected])
    return selected[highest]


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
    # For every two following cities add the distance between them to the sum.
    for i in range(path.shape[0]):
        departingCity = path[i - 1]
        arrivingCity = path[i]
        # Assumed the rows represent departing cities and the column ariving cities
        totalDistance += TSP[departingCity, arrivingCity]
        if totalDistance == np.inf:
            return float("inf")
    return totalDistance


# Calculates the mean fitness of the population and the best fitting individual (Needed for the Reporter class)
def evaluatePopulation(TSP, population):
    bestFit = float("inf")
    sumFit = 0
    bestIndividual = None
    for individual in population:
        fit = fitness(TSP, individual.path)
        sumFit += fit
        if fit < bestFit:
            bestFit = fit
            bestIndividual = individual
    return (sumFit / population.shape[0], bestFit, bestIndividual)


if __name__ == "__main__":
    algorithm = r0786701()

    algorithm.optimize("tour250.csv")

