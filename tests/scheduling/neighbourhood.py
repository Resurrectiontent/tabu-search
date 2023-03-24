import random

import numpy as np
from sampo.scheduler.genetic.operators import copy_chromosome

from sampo.scheduler.genetic.converter import ChromosomeType


# TODO: modify sampo resource mutation to neighbourhood function
def mut_uniform_int(ind: ChromosomeType, low: np.ndarray, up: np.ndarray, type_of_worker: int,
                    probability_mutate_resources: float, contractor_count: int, rand: random.Random) -> ChromosomeType:
    """
    Mutation function for resources
    It changes selected numbers of workers in random work in certain interval for this work

    :param contractor_count:
    :param ind:
    :param low: lower bound specified by `WorkUnit`
    :param up: upper bound specified by `WorkUnit`
    :param type_of_worker:
    :param probability_mutate_resources:
    :param rand:
    :return: mutate individual
    """
    ind = copy_chromosome(ind)

    # select random number from interval from min to max from uniform distribution
    size = len(ind[1][type_of_worker])

    if type_of_worker == len(ind[1]) - 1:
        # print('Contractor mutation!')
        for i in range(size):
            if rand.random() < probability_mutate_resources:
                ind[1][type_of_worker][i] = rand.randint(0, contractor_count - 1)
        return ind

    # change in this interval in random number from interval
    for i, xl, xu in zip(range(size), low, up):
        if rand.random() < probability_mutate_resources:
            # borders
            contractor = ind[1][-1][i]
            border = ind[2][contractor][type_of_worker]
            ind[1][type_of_worker][i] = rand.randint(xl, min(xu, border))

    return ind