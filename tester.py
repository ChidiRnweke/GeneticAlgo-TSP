import numpy as np
import dask.config
import dask.distributed
import pandas as pd

a = np.array((1, 2, 3, 4, 5, 6, 7, 8, 9))
b = np.array((9, 3, 7, 8, 2, 6, 5, 1, 4))

c = np.array((3, 4, 6, 8, 1, 2, 9, 7, 5))
d = np.array((2, 7, 1, 9, 5, 3, 6, 4, 8))


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


def OX(parent1: np.array, parent2: np.array):
    o1 = np.empty_like(parent1) * -1
    o2 = np.empty_like(parent1) * -1
    cut1 = np.random.randint(low=0, high=int(parent1.shape[0] / 2))
    cut2 = np.random.randint(low=cut1 + 2, high=parent1.shape[0])
    order = np.concatenate(
        (np.arange(cut2, len(parent1)), np.arange(cut1), np.arange(cut1, cut2))
    )
    to_check = order[: cut1 - cut2]
    o1[cut1:cut2] = parent1[cut1:cut2]
    o2[cut1:cut2] = parent2[cut1:cut2]
    set_1 = set(o1)
    set_2 = set(o2)

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


def OX2(par1: np.array, par2: np.array):
    parent1 = np.copy(par1)
    parent2 = np.copy(par2)
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
    o1 = np.concatenate((splitp1[0], splitp2[1], splitp1[2]), dtype=np.int32)
    o2 = np.concatenate((splitp2[0], splitp1[1], splitp2[2]), dtype=np.int32)
    mapping = set(zip(splitp1[1], splitp2[1]))

    while np.unique(o1).size != o1.size:
        for key, val in mapping:
            splitp1[0][splitp1[0] == val] = key
            splitp1[2][splitp1[2] == val] = key
        o1 = np.concatenate((splitp1[0], splitp2[1], splitp1[2]), dtype=np.int32)
    while np.unique(o2).size != o2.size:
        for key, val in mapping:
            splitp2[0][splitp2[0] == key] = val
            splitp2[2][splitp2[2] == key] = val
        o2 = np.concatenate((splitp2[0], splitp1[1], splitp2[2]), dtype=np.int32)
    return o1, o2


def inversion_mutation(individual: np.array) -> None:
    cut1 = np.random.randint(low=1, high=int(individual.shape[0] / 2))
    cut2 = np.random.randint(low=cut1 + 2, high=individual.shape[0] - 1)
    individual[cut1:cut2] = np.flip(individual[cut1:cut2])


b = pd.read_csv(
    "r0786701.csv",
    usecols=["# Iteration", "Elapsed time", "Mean value", "Best value", "Cycle"],
    skiprows=1,
    skipinitialspace=True,
)
b = b.drop("Cycle", axis=1)
print(b)
