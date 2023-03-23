import numpy as np
import pytest
from matplotlib import pyplot as plt

from tabusearch import TabuSearch
from tabusearch.solution.base import Solution
from tests.scheduling.fixtures import setup_schedule, setup_toolbox, setup_wg, setup_contractors, setup_worker_pool


def test_scheduling(setup_schedule, setup_toolbox):
    init_schedule = setup_toolbox.schedule_to_chromosome(schedule=setup_schedule)

    optimiser = TabuSearch(quality=setup_toolbox.evaluate,
                           max_iter=1000,
                           tabu_time=10,)
    s = optimiser.optimize(init_schedule)
    h: list[Solution] = optimiser._history
    print('\n', s.position)
    print(s.quality)
    plt.plot(np.linspace(0, 1, len(h)), np.array([float(x.quality) for x in h]))
    plt.show()