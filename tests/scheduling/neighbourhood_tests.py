from operator import itemgetter

from time import time

import numpy as np
import pytest
from matplotlib import pyplot as plt

from scheduling.order_neighbourhood import variable_partitioning_order_neighbourhood
from scheduling.resource_neighbourhood import variable_partitioning_resource_neighbourhood
from tabusearch import TabuSearch
from tabusearch.solution.base import Solution
from tests.scheduling.fixtures import setup_schedule, setup_toolbox, setup_wg, setup_contractors, setup_worker_pool


def test_resource_neighbourhood(setup_schedule, setup_toolbox):
    init_schedule = setup_toolbox.schedule_to_chromosome(schedule=setup_schedule)

    t1 = time()
    neighbourhood = variable_partitioning_resource_neighbourhood(init_schedule, setup_toolbox.get_worker_reqs())#, 15, 30)
    elapsed = time() - t1

    print(f'variable_partitioning_resource_neighbourhood produced {len(neighbourhood)} neighbours in {elapsed:.4f} sec.')
    # TODO: think on more accurate assertions
    assert len(neighbourhood) > 1
    assert all(map(setup_toolbox.validate, map(itemgetter(0), neighbourhood)))


def test_order_neighbourhood(setup_schedule, setup_toolbox):
    init_schedule = setup_toolbox.schedule_to_chromosome(schedule=setup_schedule)

    t1 = time()
    neighbourhood = variable_partitioning_order_neighbourhood(init_schedule, setup_toolbox.is_order_correct)
    elapsed = time() - t1

    print(f'variable_partitioning_order_neighbourhood produced {len(neighbourhood)} neighbours in {elapsed:.4f} sec.')
    # TODO: think on more accurate assertions
    assert len(neighbourhood) > 10
    assert all(map(setup_toolbox.validate, map(itemgetter(0), neighbourhood)))