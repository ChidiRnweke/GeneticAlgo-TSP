import numpy as np

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


def OX(par1: np.array, par2: np.array):
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
    for i in to_check:
        j = 0
        for j in order:
            if parent2[j] not in o1:
                o1[i] = parent2[j]
                break
    for i in to_check:
        j = 0
        for j in order:
            if parent1[j] not in o2:
                o2[i] = parent1[j]
                break
    return o1, o2


def PMX(parent1: np.array, parent2: np.array) -> tuple:
    # PMX
    # parent1 = np.copy(par1)
    # parent2 = np.copy(par2)
    index1 = np.random.randint(low=1, high=int(parent1.shape[0] / 2))
    index2 = np.random.randint(low=index1 + 2, high=parent1.shape[0] - 1)
    o1 = np.concatenate((parent1[:index1], parent2[index1:index2], parent1[index2:]))
    o2 = np.concatenate((parent2[:index1], parent1[index1:index2], parent2[index2:]))

    mask = np.ones_like(parent1, dtype=bool)
    mask[np.arange(index1, index2)] = False

    while np.unique(o1).size != o1.size:
        for key, val in zip(parent1[index1:index2], parent2[index1:index2]):
            o1[mask] = np.where(o1[mask] == val, key, o1[mask])

    while np.unique(o2).size != o2.size:
        for key, val in zip(parent1[index1:index2], parent2[index1:index2]):
            o2[mask] = np.where(o2[mask] == key, val, o2[mask])
    print(o1, o2)
    return o1, o2


def inversion_mutation(individual: np.array) -> None:
    cut1 = np.random.randint(low=1, high=int(individual.shape[0] / 2))
    cut2 = np.random.randint(low=cut1 + 2, high=individual.shape[0] - 1)
    individual[cut1:cut2] = np.flip(individual[cut1:cut2])
    return individual


print(CX(a, b))

