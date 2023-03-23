from collections import defaultdict
from typing import Callable

import numpy as np
import pytest
from deap.base import Toolbox

from sampo import generator
from sampo.scheduler.base import Scheduler
from sampo.scheduler.genetic.operators import init_toolbox, TimeAndResourcesFitness
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.schemas.contractor import Contractor, WorkerContractorPool
from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.schedule import Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.utilities.collections import reverse_dictionary


@pytest.fixture(scope='session')
def setup_wg():
    wg = generator.SimpleSynthetic().advanced_work_graph(works_count_top_border=100,
                                                         uniq_works=30,
                                                         uniq_resources=10)
    return wg


@pytest.fixture(scope='session')
def setup_contractors(setup_wg):
    contractors = [generator.get_contractor_by_wg(setup_wg)]
    return contractors


@pytest.fixture(scope='session')
def setup_worker_pool(setup_contractors) -> WorkerContractorPool:
    worker_pool = defaultdict(dict)
    for contractor in setup_contractors:
        for worker in contractor.workers.values():
            worker_pool[worker.name][worker.contractor_id] = worker
    return worker_pool


@pytest.fixture(scope='session', params=[HEFTScheduler])
def setup_schedule(request, setup_wg, setup_contractors) -> Schedule:
    scheduler: Callable[[], Scheduler] = request.param

    schedule = scheduler().schedule(setup_wg, setup_contractors)
    return schedule


@pytest.fixture(scope='session')
def setup_toolbox(setup_wg, setup_contractors, setup_worker_pool) -> Toolbox:
    """
    returns a Toolbox that supports following methods:
      evaluate
      validate
      schedule_to_chromosome
      chromosome_to_schedule
    """
    return create_toolbox(setup_wg,
                          setup_contractors,
                          setup_worker_pool)


def create_toolbox(wg: WorkGraph,
                   contractors: list[Contractor],
                   worker_pool: WorkerContractorPool,
                   spec: ScheduleSpec = ScheduleSpec(),
                   work_estimator: WorkTimeEstimator = None) -> Toolbox:
    nodes = [node for node in wg.nodes if not node.is_inseparable_son()]

    index2node: dict[int, GraphNode] = {index: node for index, node in enumerate(nodes)}
    work_id2index: dict[str, int] = {node.id: index for index, node in index2node.items()}
    worker_name2index = {worker_name: index for index, worker_name in enumerate(worker_pool)}
    index2contractor = {ind: contractor.id for ind, contractor in enumerate(contractors)}
    index2contractor_obj = {ind: contractor for ind, contractor in enumerate(contractors)}
    contractor2index = reverse_dictionary(index2contractor)
    worker_pool_indices = {worker_name2index[worker_name]: {
        contractor2index[contractor_id]: worker for contractor_id, worker in workers_of_type.items()
    } for worker_name, workers_of_type in worker_pool.items()}
    node_indices = list(range(len(nodes)))

    contractors_capacity = np.zeros((len(contractors), len(worker_pool)))
    for w_ind, cont2worker in worker_pool_indices.items():
        for c_ind, worker in cont2worker.items():
            contractors_capacity[c_ind][w_ind] = worker.count

    resources_border = np.zeros((2, len(worker_pool), len(index2node)))
    resources_min_border = np.zeros((len(worker_pool)))
    for work_index, node in index2node.items():
        for req in node.work_unit.worker_reqs:
            worker_index = worker_name2index[req.kind]
            resources_border[0, worker_index, work_index] = req.min_count
            resources_border[1, worker_index, work_index] = req.max_count
            resources_min_border[worker_index] = max(resources_min_border[worker_index], req.min_count)

    contractor_borders = np.zeros((len(contractor2index), len(worker_name2index)), dtype=int)
    for ind, contractor in enumerate(contractors):
        for ind_worker, worker in enumerate(contractor.workers.values()):
            contractor_borders[ind, ind_worker] = worker.count

    # construct inseparable_child -> inseparable_parent mapping
    inseparable_parents = {}
    for node in nodes:
        for child in node.get_inseparable_chain_with_self():
            inseparable_parents[child] = node

    # here we aggregate information about relationships from the whole inseparable chain
    children = {work_id2index[node.id]: [work_id2index[inseparable_parents[child].id]
                                         for inseparable in node.get_inseparable_chain_with_self()
                                         for child in inseparable.children]
                for node in nodes}

    parents = {work_id2index[node.id]: [] for node in nodes}
    for node, node_children in children.items():
        for child in node_children:
            parents[child].append(node)

    return init_toolbox(wg=wg,
                        work_id2index=work_id2index,
                        worker_name2index=worker_name2index,
                        contractor2index=contractor2index,
                        contractor_borders=contractor_borders,
                        fitness_constructor=TimeAndResourcesFitness,
                        node_indices=node_indices,
                        parents=parents,
                        worker_pool=worker_pool,
                        index2node=index2node,
                        index2contractor_obj=index2contractor_obj,
                        worker_pool_indices=worker_pool_indices,
                        spec=spec,
                        work_estimator=work_estimator,
                        # don't fill parameters that won't be used during tabu search
                        contractors=None,
                        index2contractor=[],
                        index2node_list=None,
                        init_chromosomes=None,
                        mutate_order=None,
                        mutate_resources=None,
                        rand=None,
                        selection_size=None)
