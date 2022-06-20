from poplib import POP3
import Reporter
import numpy as np
from numba import njit, types
from dask import delayed


class r0786701:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        TSP = np.loadtxt(file, delimiter=",")
        file.close()
        # Parameters
        POPSIZE = TSP.shape[0] * 2
        MAXN = 1000
        NELITES = round(0.01 * POPSIZE)
        K = 5
        prob = 0.5
        OFFSPRINGSIZE = -(-POPSIZE // 4)  # Ceiling division
        TERMINATION = 250

        population, fitnesses = initialize(TSP, POPSIZE)
        elites = population[:NELITES, :]
        eliteFit = fitnesses[:NELITES]

        iteration = 0
        meanObjective = 1.0
        bestObjective = 0.0

        prevSolution = 1e9
        tolerance = 0
        sameSolutionCount = 0

        while iteration < MAXN and sameSolutionCount < TERMINATION:
            meanObjective = 0.0
            bestObjective = 0.0
            bestSolution = np.array([1, 2, 3, 4, 5])

            results = []
            for _ in range(4):
                pop_part = delayed(recombination)(
                    population, prob, 10, TSP, OFFSPRINGSIZE
                )
                pop_part = delayed(rand_opt)(pop_part, TSP)
                results.append(pop_part)
            pop_lazy = delayed(np.vstack)(results)
            population = pop_lazy.compute(scheduler="threads", num_workers=4)

            population, fitnesses = elimination(population, POPSIZE - NELITES, K, TSP)

            population, fitnesses, elites, eliteFit = elitism(
                population, TSP, elites, eliteFit, fitnesses, NELITES
            )

            populationEvaluation = evaluatePopulation(TSP, population)
            meanObjective = populationEvaluation[0]
            bestObjective = populationEvaluation[1]
            bestSolution = populationEvaluation[2]

            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            difference = prevSolution - bestObjective
            if difference <= tolerance:
                sameSolutionCount += 1
                prob = np.minimum(0.8, prob + 0.01)
            else:
                sameSolutionCount = 0
                prob = np.maximum(0.3, prob - 0.02)
            prevSolution = bestObjective

            print(f"Time left: {timeLeft}")
            if timeLeft < 0:
                break
            iteration += 1

        return 0


@njit(nogil=True, cache=True)
def rand_opt(partition: np.array, problem: np.array, max_depth: int = 20) -> np.array:
    """[k_opt without constructing the full neighbourhood.
        The max_depth parametercontrols how many edges can be changed at most.
        The actual result is sampled at random.]

    Args:
        candidate (np.array): [Candidate solution]
        problem (np.array): [Distance Matrix]
        max_depth (int): [Controls the maximum amount of edges that can be checked]

    Returns:
        np.array: [Best solution sampled from the neighbourhood]
    """
    for c in range(partition.shape[0]):
        candidate = partition[c]
        bestPath = candidate
        bestFit = fitness(problem, candidate)
        depth = np.random.choice(np.arange(candidate.shape[0]), max_depth)
        for i in depth:
            depth_2 = np.random.choice(np.arange(candidate.shape[0]), max_depth)
            for j in depth_2:
                neighbour = candidate.copy()
                neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
                fit = fitness(TSP=problem, path=neighbour)
                if fit < bestFit:
                    bestFit = fit
                    bestPath = neighbour.copy()
        partition[c] = bestPath.copy()
    return partition


@njit(nogil=True, cache=True)
def recombination(pop: np.array, prob, K: int, TSP: np.array, n: int,) -> np.ndarray:
    """[Carries out selection, crossover and mutation steps n / 2 times.]

    Args:
        pop (np.array): [population]
        K (int): [tournament size]
        TSP (np.array): [Distance Matrix]
        n (int): [Amount of offspring to be generated]

    Returns:
        np.ndarray: [returns the merged population]
    """

    offspring = np.zeros((n, TSP.shape[0]), dtype=np.int32)
    for i in range(0, n, 2):
        parent1 = selection(pop, 1, TSP)
        parent2 = selection(pop, 1, TSP)
        offspring1, offspring2 = OX(parent1, parent2)
        offspring[i] = offspring1.copy()
        offspring[i + 1] = offspring2.copy()
    merged = np.vstack((pop, offspring))
    for individual in merged:
        probability = np.random.uniform(0, 1)
        if probability < prob:
            inversionMutation(individual)
    return merged


def initialize(TSP: np.array, POPSIZE: int) -> np.ndarray:
    """[Randomly initialises the population with the greedy solution at position 0.
        Potentially carries out a local search operator after initialisation.]

    Args:
        TSP ([type]): [Distance matrix]
        POPSIZE (int): [Size of the population]

    Returns:
        np.ndarray: [Initialised population]
    """
    rng = np.random.default_rng()
    population = np.arange(TSP.shape[1], dtype=np.int32)
    population = np.broadcast_to(population, (POPSIZE, TSP.shape[1]))
    population = rng.permuted(population, axis=1)
    population[: TSP.shape[0], :] = greedy(TSP)
    population = rand_opt(population, TSP)
    fitnesses = np.empty(POPSIZE)
    for i in range(POPSIZE):
        fitnesses[i] = fitness(TSP, population[i, :])
    indices = fitnesses.argsort()
    population = population[indices]
    fitnesses = fitnesses[indices]
    return population, fitnesses


@njit(cache=True)
def inversionMutation(individual: np.array) -> None:
    """[Inverts part of an individual in place]

    Args:
        individual (np.array): [individual]
    """
    cut1 = np.random.randint(low=0, high=int(individual.shape[0] / 2))
    cut2 = np.random.randint(low=cut1 + 1, high=individual.shape[0] - 1)
    individual[cut1:cut2] = np.flip(individual[cut1:cut2])


def greedy(TSP: np.array) -> np.array:
    rng = np.random.default_rng()
    solution = np.empty(TSP.shape[0], dtype=np.int32)
    start_pop = np.empty_like(TSP, dtype=np.int32)
    dm = np.where(TSP != 0, TSP, np.inf)
    for i in range(TSP.shape[0]):
        checked = set()
        dm = np.where(TSP != 0, TSP, np.inf)
        minimum = i
        solution[0] = i
        checked.add(minimum)
        dm[:, minimum] = np.inf
        for index in range(1, TSP.shape[0]):
            minimum = np.argmin(dm[minimum, :])
            if minimum not in checked:
                checked.add(minimum)
                solution[index] = minimum
                dm[:, minimum] = np.inf
            else:
                start_pop[i] = rng.permuted(np.arange(TSP.shape[0]))
                break
        start_pop[i] = solution.copy()
    return start_pop


@njit(cache=True)
def OX(parent1: np.array, parent2: np.array):
    """[Ordered crossover]

    Args:
        [np.array] parent1
        [np.array] parent2

    Returns:
        [tuple of numpy arrays]: [2 children]
    """
    o1 = np.ones_like(parent1) * -1
    o2 = np.ones_like(parent1) * -1
    cut1 = np.random.randint(low=1, high=int(parent1.shape[0] / 2))
    cut2 = np.random.randint(low=cut1 + 2, high=parent1.shape[0] - 1)
    order = np.concatenate(
        (np.arange(cut2, len(parent1)), np.arange(cut1), np.arange(cut1, cut2))
    )

    to_check = set(order[: cut1 - cut2])
    o1[cut1:cut2] = parent1[cut1:cut2]
    o2[cut1:cut2] = parent2[cut1:cut2]
    set_1 = set(parent1[cut1:cut2])
    set_2 = set(parent2[cut1:cut2])

    j = 0
    for i in to_check:
        for j in order:
            if parent2[j] not in set_1:
                o1[i] = parent2[j]
                set_1.add(parent2[j])
                break

    j = 0
    for i in to_check:
        for j in order:
            if parent1[j] not in set_2:
                o2[i] = parent1[j]
                set_2.add(parent1[j])
                break
    return o1, o2


@njit(cache=True)
def elitism(population, TSP, elites, eliteFit, fitnesses, numelites):
    population = np.vstack((elites, population))
    fitnesses = np.hstack((eliteFit, fitnesses))
    indices = fitnesses.argsort()
    population = population[indices]
    fitnesses = fitnesses[indices]
    if np.array_equal(population[:numelites, :], elites):
        pass
    else:
        population[:numelites, :] = rand_opt(population[:numelites, :], TSP)
        elites = population[:numelites, :].copy()
        eliteFit = fitnesses[:numelites].copy()
    return population, fitnesses, elites, eliteFit


@njit(cache=True)
def selection(population: np.array, k: int, TSP) -> np.array:
    """[K-tournament selection (with replacement) for recombination]

    Args:
        population (np.array): [Population]
        k (int): [Amount of values to compare]
        TSP ([type]): [Distance Matrix]

    Returns:
        [np.array]: [best individual]
    """
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


@njit(cache=True)
def elimination(
    population: np.array, numberOfSelections: int, K: int, TSP: np.array,
):
    """[Samples K samples (without replacement) from the population and picks the best.
        Repeats this numberOfSelectons times.]

    Args:
        population (np.array): [Array]
        numberOfSelections (int): [The amount of individuals you want left]
        K (int): [Amount of samples per tournament]
        TSP (np.array): [Problem definition]

    Returns:
        [type]: [description]
    """
    initialSize = population.shape[0]
    fitnesses = np.empty(initialSize)
    newFitnesses = np.empty(numberOfSelections)
    for i in range(initialSize):
        fitnesses[i] = fitness(TSP, population[i, :])
    newPopulation = np.zeros((numberOfSelections, population.shape[1]), dtype=np.int32)
    for i in range(numberOfSelections):
        sample = np.random.choice(initialSize, K, replace=False)
        bestIndividual = sample[np.argmin(fitnesses[sample])]
        newPopulation[i] = population[bestIndividual]
        newFitnesses[i] = fitnesses[bestIndividual]
    return newPopulation, newFitnesses[:numberOfSelections]


@njit(cache=True)
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
    for i in range(path.shape[0]):
        departingCity = path[i - 1]
        arrivingCity = path[i]

        if TSP[departingCity, arrivingCity] == np.inf:
            totalDistance += 1e99
        else:
            totalDistance += TSP[departingCity, arrivingCity]
    return totalDistance


@njit(locals={"meanfit": types.float64}, cache=True)
def evaluatePopulation(TSP, population):
    bestFit = 1e99999999999
    bestIndividual = np.empty(population.shape[1], dtype=np.int32)
    fitnesses = np.array([fitness(TSP=TSP, path=ind) for ind in population])
    meanfit = np.mean(fitnesses)
    bestidx = np.argmin(fitnesses)
    bestFit = fitnesses[bestidx]
    bestIndividual = population[bestidx]
    return (meanfit, bestFit, bestIndividual)


if __name__ == "__main__":

    algorithm = r0786701()
    algorithm.optimize("./tour500.csv")
