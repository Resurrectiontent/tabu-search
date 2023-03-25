from itertools import chain

import random

import numpy as np
from numpy.random import Generator, default_rng

from sampo.scheduler.genetic.operators import copy_chromosome

from sampo.scheduler.genetic.converter import ChromosomeType



def variable_partitioning_neighbourhood(ind: ChromosomeType, levels: int | None = None,
                                        max_level_mutations: int | None = None,
                                        rng: Generator = default_rng()):
    """
    Generates neighbourhood via variable partitioning. DOI:10.1007/s10489-011-0321-0
    :param ind: Initial chromosome
    :param levels: Max number of partitions to alter. Defaults to 5 or `ind[1].shape[0]`, if it is lower.
    :param max_level_mutations: Max number of variables to change in each altered partitions.
        Defaults to 5 or `ind[1].shape[1]`, if it is lower.
    :param rng: numpy random number generator
    :return:
    """
    def np_sorted(a):
        a.sort()
        return a

    x = ind[1]
    assert len(x.shape) == 2

    partitions_len, variables_len = x.shape
    levels = levels or min(5, partitions_len)
    max_level_mutations = max_level_mutations or min(5, variables_len)
    assert partitions_len >= levels
    assert variables_len >= max_level_mutations

    result = []
    for partitions_number in range(1, levels + 1):
        partitions = np_sorted(rng.choice(partitions_len, partitions_number, replace=False))
        for variables_number in range(1, max_level_mutations + 1):
            instance_positive = x.copy()
            instance_negative = x.copy()

            partition_variables = [[[partition] * variables_number,
                                    np_sorted(rng.choice(variables_len, variables_number, replace=False))]
                                   for partition in partitions]

            index_str = ','.join([f'{p[0]}[{",".join(map(str, v))}]' for p, v in partition_variables])
            indices = tuple(list(chain(*x)) for x in zip(*partition_variables))
            instance_positive[indices] = instance_positive[indices] + 1
            instance_negative[indices] = instance_negative[indices] - 1

            result.append((instance_positive, '+' + index_str))
            result.append((instance_negative, '-' + index_str))

    # TODO: return whole chromosome
    return result

def check_chromosome():
    ...

# sampo resource mutation
# TODO: erase
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