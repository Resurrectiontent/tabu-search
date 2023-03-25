import numpy as np
from itertools import chain
from numpy.random import Generator, default_rng
from numpy.typing import NDArray
from sampo.scheduler.genetic.operators import copy_chromosome

from sampo.scheduler.genetic.converter import ChromosomeType


def variable_partitioning_resource_neighbourhood(ind: ChromosomeType,
                                                 worker_reqs: NDArray,
                                                 levels: int | None = None,
                                                 max_level_mutations: int | None = None,
                                                 rng: Generator = default_rng()):
    """
    Generates neighbourhood via variable partitioning. DOI:10.1007/s10489-011-0321-0
    This implementation takes each resource as a separate partition.
    :param ind: Initial chromosome
    :param worker_reqs: Worker reqs defined by `WorkUnit` in format:
    `[[[res1_work1_lower, res1_work2_lower, ...], [res2_work1_lower, ...], ...], [[res1_work1_upper, ...], ...]]`
    :param levels: Max number of partitions to alter. Defaults to 5 or `ind[1].shape[0]`, if it is lower.
    Denoted as L in the paper.
    :param max_level_mutations: Max number of variables to change in each altered partitions.
    Defaults to 5 or `ind[1].shape[1]`, if it is lower. Denoted as μ in the paper.
    :param rng: numpy random number generator.
    :return:
    """
    # sorting is necessary for proper solution indexing
    def np_sorted(a):
        a.sort()
        return a

    # check that the required number of mutations doesn't exceed mutation possibilities defined by chromosome size
    partitions_len, variables_len = ind[1].shape
    levels = levels or min(5, partitions_len)
    max_level_mutations = max_level_mutations or min(5, variables_len)
    assert partitions_len >= levels
    assert variables_len >= max_level_mutations

    result = []
    # generate trial solutions
    # alter l partitions. l = 1,...L
    for partitions_number in range(1, levels + 1):
        # indices of partitions to alter
        partitions = np_sorted(rng.choice(partitions_len, partitions_number, replace=False))
        # in each selected partition alter m variables. m = 1,...μ
        for variables_number in range(1, max_level_mutations + 1):
            instance_positive = copy_chromosome(ind)
            instance_negative = copy_chromosome(ind)

            # for every selected partition select particular variables to alter
            partition_variables = [[[partition] * variables_number,
                                    np_sorted(rng.choice(variables_len, variables_number, replace=False))]
                                   for partition in partitions]

            index_str = ','.join([f'{p[0]}[{",".join(map(str, v))}]' for p, v in partition_variables])
            indices = tuple(list(chain(*x)) for x in zip(*partition_variables))

            # alter all selected variables
            instance_positive[1][indices] = instance_positive[1][indices] + 1
            instance_negative[1][indices] = instance_negative[1][indices] - 1

            # check validity and save
            if check_increased_resources(instance_positive[1], instance_positive[2], worker_reqs[1]):
                result.append((instance_positive, '+' + index_str))
            if check_decreased_resources(instance_negative[1], worker_reqs[0]):
                result.append((instance_negative, '-' + index_str))

    return result


# TODO: consider making one-stringers from check functions
def check_decreased_resources(res: NDArray, res_low_borders: NDArray) \
        -> bool:
    """
    Checks that the somehow decreased resource part of chromosome satisfy all the constraints:
      1. no works are assigned to contractor with negative id
      2. all allocated resource are not less than minimal worker reqs and zero
    :param res: resource part of chromosome
    `[[res1_work1, res1_work2,..], [res2_work2, ...], ...[work1_contractor_id, ...]]`
    :param res_low_borders: lower borders of worker reqs
    `[[res1_work1_limit, res1_work2_limit, ...], [res2_work1_limit, ...], ...]`
    :return: True, if all constraints are satisfied, otherwise, False.
    """
    # no works are assigned to contractor with id "-1"
    if (res[-1] < 0).any():
        return False

    # all allocated resource are not less than minimal worker reqs
    # (implies non-negativity check)
    if (res[:-1] < res_low_borders).any():
        return False

    return True


def check_increased_resources(res: NDArray, contractor_borders: NDArray, res_up_borders: NDArray) \
        -> bool:
    """
    Checks that the somehow increased resource part of chromosome satisfy all the constraints:
      1. all contractors indices are in contractor pools,
      2. all assigned contractors' resources are in contractors' limits,
      3. all allocated resources do not exceed upper borders from worker reqs.
    :param res: resource part of chromosome
    `[[res1_work1, res1_work2,..], [res2_work2, ...], ...[work1_contractor_id, ...]]`
    :param contractor_borders: contractor worker pools part of chromosome
    `[[contractor1_res1, contractor1_res2, ...], [contractor2_res1, ...], ...]`
    :param res_up_borders: upper borders of worker reqs
    `[[res1_work1_limit, res1_work2_limit, ...], [res2_work1_limit, ...], ...]`
    :return: True, if all constraints are satisfied, otherwise, False.
    """
    # all contractors indices are in contractor pools
    if (res[-1] > contractor_borders.shape[0]).any():
        return False

    # TODO: consider using all(<generator>) instead of the cycle
    for ct_id, ct_border in enumerate(contractor_borders):
        ct_works, = np.where(res[-1] == ct_id)
        # all assigned contractor's resources are in contractor's limits
        if (res[:-1, ct_works].max(axis=1) > ct_border).any():
            return False

    # all allocated resources do not exceed upper borders from worker reqs
    if (res[:-1] > res_up_borders).any():
        return False

    return True
