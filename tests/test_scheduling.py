from random import Random

import numpy as np
from deap.base import Toolbox
import pytest
from sampo import generator
from sampo.scheduler.heft.base import HEFTScheduler

from sampo.utilities.resource_cost import schedule_cost
from sampo.scheduler.genetic.converter import ChromosomeType, convert_schedule_to_chromosome
from sampo.scheduler.genetic.operators import init_toolbox, TimeFitness
from sampo.schemas.contractor import Contractor, WorkerContractorPool
from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.schedule import Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.utilities.collections import reverse_dictionary



def setup_wg():
    wg = generator.SimpleSynthetic().advanced_work_graph(works_count_top_border=100, uniq_works=30,
                                                         uniq_resources=10)
    return wg


@pytest.fixture(scope='module')
def setup_contractors(setup_wg):
    contractors = [generator.get_contractor_by_wg(setup_wg)]
    return contractors


@pytest.fixture(scope='module', params=[HEFTScheduler])
def setup_schedule(request, setup_wg, setup_contractors):
    scheduler = request.param

    schedule = scheduler().schedule(setup_wg, setup_contractors)
    return schedule


# TODO: remove data that is unnecessary for tabu searcg
#   from toolbox initialization

@pytest.fixture(scope='module')
def setup_toolbox(setup_wg, setup_contractors, setup_worker_pool, setup_schedule) -> Toolbox:

    return create_toolbox(setup_wg,
                          setup_contractors,
                          setup_worker_pool,
                          0, .0, .0,
                          {'schedule': setup_schedule})

def create_toolbox(wg: WorkGraph,
                   contractors: list[Contractor],
                   worker_pool: WorkerContractorPool,
                   selection_size: int,
                   mutate_order: float,
                   mutate_resources: float,
                   init_schedules: dict[str, Schedule],
                   rand: Random = Random(),
                   spec: ScheduleSpec = ScheduleSpec(),
                   work_estimator: WorkTimeEstimator = None) -> Toolbox:
    nodes = [node for node in wg.nodes if not node.is_inseparable_son()]

    index2node: dict[int, GraphNode] = {index: node for index, node in enumerate(nodes)}
    work_id2index: dict[str, int] = {node.id: index for index, node in index2node.items()}
    worker_name2index = {worker_name: index for index, worker_name in enumerate(worker_pool)}
    index2contractor = {ind: contractor.id for ind, contractor in enumerate(contractors)}
    index2contractor_obj = {ind: contractor for ind, contractor in enumerate(contractors)}
    contractor2index = reverse_dictionary(index2contractor)
    index2node_list = [(index, node) for index, node in enumerate(nodes)]
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

    init_chromosomes: dict[str, ChromosomeType] = \
        {name: convert_schedule_to_chromosome(wg, work_id2index, worker_name2index,
                                              contractor2index, contractor_borders, schedule)
         for name, schedule in init_schedules.items()}

    return init_toolbox(wg,
                        contractors,
                        worker_pool,
                        index2node,
                        work_id2index,
                        worker_name2index,
                        index2contractor,
                        index2contractor_obj,
                        init_chromosomes,
                        mutate_order,
                        mutate_resources,
                        selection_size,
                        rand,
                        spec,
                        worker_pool_indices,
                        contractor2index,
                        contractor_borders,
                        node_indices,
                        index2node_list,
                        parents,
                        TimeFitness,
                        Time(0),
                        work_estimator)
