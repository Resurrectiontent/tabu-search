import numpy as np
import pytest
from matplotlib import pyplot as plt

from scheduling.resource_neighbourhood import variable_partitioning_resource_neighbourhood
from tabusearch import TabuSearch
from tabusearch.solution.base import Solution
from tests.scheduling.fixtures import setup_schedule, setup_toolbox, setup_wg, setup_contractors, setup_worker_pool


def test_resource_neighbourhood(setup_schedule, setup_toolbox):
    init_schedule = setup_toolbox.schedule_to_chromosome(schedule=setup_schedule)

    neighbourhood = variable_partitioning_resource_neighbourhood(init_schedule, setup_toolbox.get_worker_reqs(), 15, 30)
    # TODO: think on more accurate assertions
    assert len(neighbourhood) > 10