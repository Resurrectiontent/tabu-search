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

    # check that the required number of mutations doesn't exceed mutation possibilities defined by chromosome size
    partitions_len, variables_len = ind[1].shape
    levels = levels or min(5, partitions_len)
    max_level_mutations = max_level_mutations or min(5, variables_len)
    assert partitions_len >= levels
    assert variables_len >= max_level_mutations
    # TODO: For accurate resource generation, make a map of work resource indices that are eligible for increment
    #  with respect to worker reqs + same for all contractors
    # TODO: Same for decrement

    result = []
    # generate trial solutions
    # alter l partitions. l = 1,...L
    for partitions_number in range(1, levels + 1):
        # in each selected partition alter m variables. m = 1,...μ
        for variables_number in range(1, max_level_mutations + 1):
            # indices of partitions to alter
            # sorting is necessary for proper solution indexing
            partitions = np_sorted(rng.choice(partitions_len, partitions_number, replace=False))

            # alter all selected variables
            increased_resource = generate_increased_resource(ind, partitions, variables_number, worker_reqs[1], rng)
            decreased_resource = generate_decreased_resource(ind, partitions, variables_number, worker_reqs[0], rng)

            # check validity and save
            if increased_resource \
                    and check_increased_resources(increased_resource[0][1], increased_resource[0][2], worker_reqs[1]):
                result.append(increased_resource)
            # no need to validate decreased resource, because it was generated worker reqs aware
            # and it can't exceed contractor capacity
            if decreased_resource:
                result.append(decreased_resource)

    return result


# TODO: extract repeated code from two functions below
def generate_increased_resource(init_res: ChromosomeType, partitions: NDArray, variables_number: int,
                                res_upper_bounds: NDArray, rng: Generator) -> tuple[ChromosomeType, str] | None:
    """
    Tries to generate new resource part of chromosome by increasing resources of `init_res` in worker req bounds
    for resources that are enumerated in `partitions`. Doesn't check that contractor capacity can satisfy new resource.
    :param init_res: Initial chromosome.
    :param partitions: Resource indices to alter.
    :param variables_number: Number of works for which we should alter given resources.
    :param res_upper_bounds: Upper border of worker reqs.
    :param rng: Numpy random number generator.
    :return: None, if can't alter resources under given conditions, otherwise, new chromosome and its altering index.
    """

    def choice_for_partition(partition):
        idx = np.where(init_res[1][-1] < init_res[2].shape[0] - 1)[0] \
            if partition == init_res[1].shape[0] - 1 \
            else np.where(init_res[1][partition] < res_upper_bounds[partition])[0]
        if idx.size < variables_number:
            raise IndexError()
        return np_sorted(rng.choice(idx, variables_number, replace=False))

    try:
        # for every selected partition select particular variables to alter
        partition_variables = [[[partition] * variables_number, choice_for_partition(partition)]
                               for partition in partitions]
    except IndexError:
        return None

    index_str = '+' + ','.join([f'{p[0]}[{",".join(map(str, v))}]' for p, v in partition_variables])
    indices = tuple(list(chain(*x)) for x in zip(*partition_variables))
    new_res = copy_chromosome(init_res)
    new_res[1][indices] = new_res[1][indices] + 1

    return new_res, index_str


def generate_decreased_resource(init_res: ChromosomeType, partitions: NDArray, variables_number: int,
                                res_low_bounds: NDArray, rng: Generator) -> tuple[ChromosomeType, str] | None:
    """
    Tries to generate new resource part of chromosome by decreasing resources of `init_res` in worker req bounds
    for resources that are enumerated in `partitions`. Doesn't check that contractor capacity can satisfy new resource.
    :param init_res: Initial chromosome.
    :param partitions: Resource indices to alter.
    :param variables_number: Number of works for which we should alter given resources.
    :param res_low_bounds: Lower border of worker reqs.
    :param rng: Numpy random number generator.
    :return: None, if can't alter resources under given conditions, otherwise, new chromosome and its altering index.
    """
    def choice_for_partition(partition):
        idx = np.where(init_res[1][-1] > 0)[0] \
            if partition == init_res[1].shape[0] - 1 \
            else np.where(init_res[1][partition] > res_low_bounds[partition])[0]
        if idx.size < variables_number:
            raise IndexError()
        return np_sorted(rng.choice(idx, variables_number, replace=False))

    try:
        # for every selected partition select particular variables to alter
        partition_variables = [[[partition] * variables_number, choice_for_partition(partition)]
                               for partition in partitions]
    except IndexError:
        return None

    index_str = '-' + ','.join([f'{p[0]}[{",".join(map(str, v))}]' for p, v in partition_variables])
    indices = tuple(list(chain(*x)) for x in zip(*partition_variables))
    new_res = copy_chromosome(init_res)
    new_res[1][indices] = new_res[1][indices] - 1

    return new_res, index_str


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
      1. (deactivated) all contractors indices are in contractor pools,
      2. (deactivated) all allocated resources do not exceed upper borders from worker reqs,
      3. all assigned contractors' resources are in contractors' limits.
    :param res: resource part of chromosome
    `[[res1_work1, res1_work2,..], [res2_work2, ...], ...[work1_contractor_id, ...]]`
    :param contractor_borders: contractor worker pools part of chromosome
    `[[contractor1_res1, contractor1_res2, ...], [contractor2_res1, ...], ...]`
    :param res_up_borders: upper borders of worker reqs
    `[[res1_work1_limit, res1_work2_limit, ...], [res2_work1_limit, ...], ...]`
    :return: True, if all constraints are satisfied, otherwise, False.
    """
    # all contractors indices are in contractor pools
    # if (res[-1] > contractor_borders.shape[0]).any():
    #     return False

    # all allocated resources do not exceed upper borders from worker reqs
    # if (res[:-1] > res_up_borders).any():
    #     return False
    # return True

    # all assigned contractor's resources are in contractor's limits
    return all((res[:-1, np.where(res[-1] == ct_id)[0]].max(axis=1, initial=0) <= ct_border).all()
               for ct_id, ct_border in enumerate(contractor_borders))


def np_sorted(a: NDArray):
    a.sort()
    return a
