import Reporter
import numpy as np
from numba import njit, types, prange
import cProfile, pstats
from concurrent import futures

# Modify the class name to match your student number.


class r0786701:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()
        # Parameters
        populationSize = 200
        maxIterations = 1e9
        kTournment = 3
        numberOfOffspring = 200
        sameSolutionIterations = 1000
        mu = 0.3
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

            population = recombination(
                population, kTournment, distanceMatrix, numberOfOffspring
            )

            for individual in population:
                probability = np.random.uniform(0, 1)
                if probability < mu:
                    individual = inversion_mutation(individual)

            population = elimination(
                population, populationSize, kTournment, distanceMatrix
            )

            populationEvaluation = evaluatePopulation(distanceMatrix, population)
            meanObjective = populationEvaluation[0]
            bestObjective = populationEvaluation[1]
            bestSolution = populationEvaluation[2]
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
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


@njit()
def k_opt(candidate: np.array, problem: np.array, k: int) -> np.array:
    """[Creates the full neighbour sructure for and candidate and selects the best one]

    Args:
        candidate (Individual): [The given candidate]
        

    Returns:
        Individual: [The best candidate in the neighbourhood]
    """
    size = candidate.size
    for _ in range(k):
        best_path = candidate
        best_fit = fitness(problem, candidate)
        for i in range(size):
            for j in range(i + 1, size):
                neighbour = candidate.copy()
                neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
                fit = fitness(TSP=problem, path=neighbour)
                best_path = neighbour if fit < best_fit else best_path
    return best_path


# Create the initial population


@njit(nogil=True)
def recombination(pop: np.array, kTournment: int, distanceMatrix: np.array, n: int):
    offspring = np.zeros((n, distanceMatrix.shape[0]))

    for i in range(0, n, 2):
        parent1 = selection(pop, kTournment, distanceMatrix)
        parent2 = selection(pop, kTournment, distanceMatrix)
        offspring1, offspring2 = OX(parent1, parent2)
        # offspring1 = k_opt(offspring1, distanceMatrix, 1)
        # offspring2 = k_opt(offspring2, distanceMatrix, 1)
        offspring[i] = offspring1.copy()
        offspring[i + 1] = offspring2.copy()
    return np.vstack((pop, offspring))


def initialize(TSP, populationSize: int) -> np.ndarray:
    rng = np.random.default_rng()

    population = np.arange(TSP.shape[1])
    population = np.broadcast_to(population, (populationSize, TSP.shape[1]))
    population = rng.permuted(population, axis=1)
    population[0] = greedy(TSP)
    out = []
    for row in population:
        # row = k_opt(row, TSP, 1)
        out.append(row)
    return np.array(out)


def swap_mutation(individual: np.array) -> None:
    indices = np.random.randint(low=0, high=len(individual), size=2)
    individual[indices[0]], individual[indices[1]] = (
        individual[indices[1]],
        individual[indices[0]],
    )
    return individual


def inversion_mutation(individual: np.array) -> None:
    cut1 = np.random.randint(low=1, high=int(individual.shape[0] / 2))
    cut2 = np.random.randint(low=cut1 + 2, high=individual.shape[0] - 1)
    individual[cut1:cut2] = np.flip(individual[cut1:cut2])
    return individual


def greedy(distanceMatrix):
    solution = np.empty(distanceMatrix.shape[0])
    dm = np.where(distanceMatrix != 0, distanceMatrix, np.inf)
    minimum = np.unravel_index(dm.argmin(), dm.shape)
    solution[0] = minimum[0]
    solution[1] = minimum[1]
    dm[:, minimum] = np.inf
    minimum = minimum[1]
    for index in range(2, distanceMatrix.shape[0]):
        minimum = np.argmin(dm[minimum, :])
        solution[index] = minimum
        dm[:, minimum] = np.inf
    return solution


@njit()
def OX(parent1: np.array, parent2: np.array):
    o1 = np.empty_like(parent1)
    o2 = np.empty_like(parent1)
    cut1 = np.random.randint(low=1, high=int(parent1.shape[0] / 2))
    cut2 = np.random.randint(low=cut1 + 2, high=parent1.shape[0] - 1)
    order = np.concatenate(
        (np.arange(cut2, len(parent1)), np.arange(cut1), np.arange(cut1, cut2))
    )
    to_check = order[: cut1 - cut2]
    o1[cut1:cut2] = parent1[cut1:cut2]
    o2[cut1:cut2] = parent2[cut1:cut2]
    j = 0
    for i in to_check:
        for j in order:
            if parent2[j] not in o1:
                o1[i] = parent2[j]
                break

    j = 0
    for i in to_check:
        for j in order:
            if parent1[j] not in o2:
                o2[i] = parent1[j]
                break
    return o1, o2


# @njit(nogil=True)
def CX(parent1: np.array, parent2: np.array):
    initialp1 = parent1.copy()
    initialp2 = parent2.copy()
    to_check = np.arange(1, len(parent1))
    o1 = np.empty_like(parent1)
    o2 = np.empty_like(parent1)

    value = parent1[0]
    o1[0] = parent1[0]
    to_check = np.delete(to_check, 0)
    while to_check.size > 0:
        pos = int(np.argwhere(parent2 == value))
        next = parent1[pos]
        if next not in o1:
            o1[pos] = next
            value = next.copy()
            to_check = np.delete(to_check, np.argwhere(to_check == next))
        else:
            parent1, parent2 = parent2, parent1
            value = to_check[0]
            continue

    parent1 = initialp2.copy()
    parent2 = initialp1.copy()
    to_check = np.arange(1, len(parent1))

    value = parent1[0]
    o2[0] = parent1[0]
    to_check = np.delete(to_check, 0)
    while to_check.size > 0:
        pos = int(np.argwhere(parent2 == value))
        next = parent1[pos]
        if next not in o2:
            o2[pos] = next
            value = next.copy()
            to_check = np.delete(to_check, np.argwhere(to_check == next))
        else:
            parent1, parent2 = parent2, parent1
            value = to_check[0]
            continue
    parent1, parent2 = initialp1, initialp2

    return o1, o2


@njit()
def PMX(parent1: np.array, parent2: np.array) -> tuple:
    """[Partially mapped crossover: take two parents, produce 2 random indices to split both.
        These indices form a mapping for which elements outside of the split need to be changed to.]

    Args:
        par1 (np.array): [First parent]
        par2 (np.array): [Second parent]

    Returns:
        tuple of numpy arrays: [Contains both offspring 1 and offspring 2]
    """

    index1 = np.random.randint(low=1, high=int(parent1.shape[0] / 2))
    index2 = np.random.randint(low=index1 + 2, high=parent1.shape[0] - 1)
    indices = np.array([index1, index2])
    splitp1 = np.array_split(parent1, indices)
    splitp2 = np.array_split(parent2, indices)
    o1 = np.concatenate((splitp1[0], splitp2[1], splitp1[2]))
    o2 = np.concatenate((splitp2[0], splitp1[1], splitp2[2]))
    mapping = set(zip(splitp1[1], splitp2[1]))

    while np.unique(o1).size != o1.size:
        for key, val in mapping:
            splitp1[0][splitp1[0] == val] = key
            splitp1[2][splitp1[2] == val] = key
        o1 = np.concatenate((splitp1[0], splitp2[1], splitp1[2]))
    while np.unique(o2).size != o2.size:
        for key, val in mapping:
            splitp2[0][splitp2[0] == key] = val
            splitp2[2][splitp2[2] == key] = val
        o2 = np.concatenate((splitp2[0], splitp1[1], splitp2[2]))
    return o1, o2


@njit()
def selection(population: np.array, k: int, TSP):
    indices = np.random.choice(np.arange(population.shape[0]), k)
    selected = population[indices]
    highest = population[0]
    bestfit = 1e9
    for row in selected:
        fit = fitness(TSP, row)
        if fit > bestfit:
            bestfit = fit
            highest = row
    return highest


@njit()
def elimination(population, numberOfSelections, kTournment, distanceMatrix):
    populationSize = len(population)
    newPopulation = np.zeros((numberOfSelections, population.shape[1]))
    start = np.arange(populationSize)
    for idx in range(numberOfSelections):
        randomIndices = np.random.choice(start, size=kTournment)
        bestFit = 1e9
        bestIndice = randomIndices[0]
        for indice in randomIndices:
            fit = fitness(distanceMatrix, population[indice])
            if fit < bestFit:
                bestFit = fit
                bestIndice = indice
        newPopulation[idx] = population[bestIndice]
    return newPopulation


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
    for i in prange(path.shape[0]):
        departingCity = int(path[i - 1])
        arrivingCity = int(path[i])
        totalDistance += TSP[departingCity, arrivingCity]
        if totalDistance == np.inf:
            return 1e99999999999
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
    profiler = cProfile.Profile()
    with open("profile.txt", "w") as f:

        profiler.enable()
        algorithm = r0786701()
        algorithm.optimize("tour250.csv")
        profiler.disable()
        stats = pstats.Stats(profiler, stream=f).sort_stats(pstats.SortKey.CUMULATIVE)
        stats.strip_dirs()
        stats.print_stats()

