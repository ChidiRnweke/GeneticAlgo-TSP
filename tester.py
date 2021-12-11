import numpy as np

a = np.array((1, 2, 3, 4, 5, 6, 7, 8))
b = np.array((2, 4, 6, 8, 7, 5, 3, 1))


def CX(par1: np.array, par2: np.array):
    parent1 = np.copy(par1)
    parent2 = np.copy(par2)
    to_check = np.arange(1, len(parent1))
    o1 = np.empty_like(parent1)
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
            value = to_check[0]
            continue
    return o1


def OX(par1: np.array, par2: np.array):
    parent1 = np.copy(par1)
    parent2 = np.copy(par2)
    o1 = np.empty_like(parent1)
    o2 = np.empty_like(parent1)
    cut1 = int(len(parent1) / 3)
    cut2 = int(len(parent1) * 2 / 3)
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


OX(a, b)

