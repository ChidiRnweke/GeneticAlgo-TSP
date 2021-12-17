import Reporter
import numpy as np
from numba import njit, types
from dask import delayed


# Modify the class name to match your student number.


class r0786701:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    def optimize(self, filename):

        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        populationSize = 250
        maxIterations = 5000
        kTournment = 5
        numberOfOffspringPT = 62
        sameSolutionIterations = 200

        population = initialize(distanceMatrix, populationSize)
        eliteFit = fitness(distanceMatrix, population[0])
        elite = population[0]

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

            results = []
            for _ in range(4):
                pop_part = delayed(recombination)(
                    population, 10, distanceMatrix, numberOfOffspringPT
                )
                pop_part = delayed(rand_opt)(pop_part, distanceMatrix, 25)
                results.append(pop_part)
            pop_lazy = delayed(np.vstack)(results)
            population = pop_lazy.compute(scheduler="threads", num_workers=4)

            # population = rand_opt(population, distanceMatrix)

            population = elimination(
                population, populationSize, kTournment, distanceMatrix
            )
            population[0] = elite

            populationEvaluation = evaluatePopulation(distanceMatrix, population)
            meanObjective = populationEvaluation[0]
            bestObjective = populationEvaluation[1]
            bestSolution = populationEvaluation[2]

            if bestObjective < eliteFit:
                elite = bestSolution
                eliteFit = bestObjective

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

        return 0


@njit(cache=True)
def k_opt(population: np.array, problem: np.array, k: int) -> np.array:
    """[Creates the full neighbour sructure for each candidate and selects the best one]

    Args:
        candidate (Individual): [The given candidate]
        

    Returns:
        Individual: [The best candidate in the neighbourhood]
    """
    size = population.shape[1]
    pop_size = population.shape[0]
    for c in range(pop_size):
        candidate = population[c]
        for _ in range(k):
            best_path = candidate
            best_fit = fitness(problem, candidate)
            for i in range(size):
                for j in range(i + 1, size):
                    neighbour = candidate.copy()
                    neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
                    fit = fitness(TSP=problem, path=neighbour)
                    best_path = neighbour.copy() if fit < best_fit else best_path
    population[c] = best_path.copy()


@njit(nogil=True, cache=True)
def rand_opt(partition: np.array, problem: np.array, max_depth: int = 10) -> np.array:
    """[k_opt without constructing the full neighbourhood. The max_depth parameter
        controls how many edges can be changed at most. The actual result is sampled
        at random.]

    Args:
        candidate (np.array): [Candidate solution]
        problem (np.array): [Distance Matrix]
        max_depth (int): [Controls the maximum amount of edges that can be checked]

    Returns:
        np.array: [Best solution sampled from the neighbourhood]
    """
    for c in range(partition.shape[0]):
        candidate = partition[c]
        best_path = candidate
        best_fit = fitness(problem, candidate)
        depth = np.random.randint(0, max_depth)
        for i in range(depth):
            depth = np.random.randint(0, max_depth)
            for j in range(i + 1, depth):
                neighbour = candidate.copy()
                neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
                fit = fitness(TSP=problem, path=neighbour)
                best_path = neighbour.copy() if fit < best_fit else best_path
        partition[c] = best_path.copy()
    return partition


@njit(nogil=True, cache=True)
def recombination(
    pop: np.array, kTournment: int, distanceMatrix: np.array, n: int
) -> np.ndarray:
    """[Carries out selection, crossover and mutation steps n / 2 times.]

    Args:
        pop (np.array): [population]
        kTournment (int): [tournament size]
        distanceMatrix (np.array): [Distance Matrix]
        n (int): [Amount of offspring to be generated]

    Returns:
        np.ndarray: [returns the merged population]
    """
    offspring = np.zeros((n, distanceMatrix.shape[0]), dtype=np.int32)
    for i in range(0, n, 2):
        parent1 = selection(pop, kTournment, distanceMatrix)
        parent2 = selection(pop, kTournment, distanceMatrix)
        offspring1, offspring2 = OX(parent1, parent2)
        offspring[i] = offspring1.copy()
        offspring[i + 1] = offspring2.copy()
    merged = np.vstack((pop, offspring))
    for individual in merged:
        probability = np.random.uniform(0, 1)
        if probability < 0.4:
            individual = inversionMutation(individual)
    return merged


def initialize(TSP: np.array, populationSize: int) -> np.ndarray:
    """[Randomly initialises the population with the greedy solution at position 0.
        Potentially carries out a local search operator after initialisation.]

    Args:
        TSP ([type]): [Distance matrix]
        populationSize (int): [Size of the population]

    Returns:
        np.ndarray: [Initialised population]
    """
    rng = np.random.default_rng()
    population = np.arange(TSP.shape[1], dtype=np.int32)
    population = np.broadcast_to(population, (populationSize, TSP.shape[1]))
    population = rng.permuted(population, axis=1)
    population[0] = greedy(TSP)
    # k_opt(population, TSP, 2)
    return population


@njit(cache=True)
def swapMutation(individual: np.array) -> None:
    """[Swaps two random edges in place.]

    Args:
        individual (np.array): [Individual]
    """
    indices = np.random.randint(low=0, high=len(individual), size=2)
    individual[indices[0]], individual[indices[1]] = (
        individual[indices[1]],
        individual[indices[0]],
    )


@njit(cache=True)
def inversionMutation(individual: np.array) -> None:
    """[Inverts part of an individual in place]

    Args:
        individual (np.array): [individual]
    """
    cut1 = np.random.randint(low=0, high=int(individual.shape[0] / 2))
    cut2 = np.random.randint(low=cut1 + 1, high=individual.shape[0])
    individual[cut1:cut2] = np.flip(individual[cut1:cut2])


@njit(cache=True)
def scrambleMutation(individual: np.array) -> None:
    """[Scrambles part of an individual in place]

    Args:
        individual (np.array): [1 individual]
    """
    cut1 = np.random.randint(low=0, high=int(individual.shape[0] / 2))
    cut2 = np.random.randint(low=cut1 + 1, high=individual.shape[0])
    np.random.shuffle(individual[cut1:cut2])


def greedy(distanceMatrix: np.array) -> np.array:
    """[Finds the greedy heuristic of the problem by always picking the shortest distance]

    Args:
        distanceMatrix (np.array): [Distance Matrix]

    Returns:
        np.array: [Greedy solution]
    """
    solution = np.empty(distanceMatrix.shape[0], dtype=np.int32)
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


@njit(cache=True)
def OX(parent1: np.array, parent2: np.array):
    """[Ordered crossover]

    Args:
        [np.array] parent1 
        [np.array] parent2

    Returns:
        [tuple of numpy arrays]: [2 children]
    """
    o1 = np.empty_like(parent1)
    o2 = np.empty_like(parent1)
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


# @njit(nogil=True)
def CX(parent1: np.array, parent2: np.array) -> np.array:
    """[Cycl crosover, not used]

    Args:
        parent1 (np.array): [description]
        parent2 (np.array): [description]

    Returns:
        [type]: [description]
    """
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


@njit(cache=True)
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
    splitp1 = list(np.array_split(parent1, indices))
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
    population: np.array,
    numberOfSelections: int,
    kTournment: int,
    distanceMatrix: np.array,
):
    """[Samples K samples (without replacement) from the population and picks the best.
        Repeats this numberOfSelectons times.]

    Args:
        population (np.array): [Array]
        numberOfSelections (int): [The amount of individuals you want left]
        kTournment (int): [Amount of samples per tournament]
        distanceMatrix (np.array): [Problem definition]

    Returns:
        [type]: [description]
    """
    newPopulation = np.zeros((numberOfSelections, population.shape[1]), dtype=np.int32)
    for i in range(numberOfSelections):
        sample = population[
            np.random.choice(population.shape[0], kTournment, replace=False)
        ]
        bestFit = 1e9
        bestIndivual = sample[0]
        for indiviual in sample:
            fit = fitness(distanceMatrix, indiviual)
            if fit < bestFit:
                bestFit = fit
                bestIndivual = indiviual.copy()
        newPopulation[i] = bestIndivual.copy()
    return newPopulation


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


# Calculates the mean fitness of the population and the best fitting individual (Needed for the Reporter class)
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
    algorithm.optimize("tour250.csv")

