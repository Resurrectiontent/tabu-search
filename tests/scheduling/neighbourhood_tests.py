import numpy as np
import pytest
from matplotlib import pyplot as plt

from scheduling.neighbourhood import variable_partitioning_neighbourhood
from tabusearch import TabuSearch
from tabusearch.solution.base import Solution
from tests.scheduling.fixtures import setup_schedule, setup_toolbox, setup_wg, setup_contractors, setup_worker_pool

def test_resource_neighbourhood(setup_schedule, setup_toolbox):
    init_schedule = setup_toolbox.schedule_to_chromosome(schedule=setup_schedule)

    neighbourhood = variable_partitioning_neighbourhood(init_schedule)
    pass