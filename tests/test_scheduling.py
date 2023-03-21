import pytest
from sampo import generator
from sampo.scheduler.heft.base import HEFTScheduler


@pytest.fixture(scope='module')
def setup_wg():
    wg = generator.SimpleSynthetic().advanced_work_graph(works_count_top_border=100, uniq_works=30,
                                                         uniq_resources=10)
    contractors = [generator.get_contractor_by_wg(wg)]

    return {'wg': wg,
            'contractors': contractors}


@pytest.fixture(scope='module', params=[HEFTScheduler])
def setup_schedule(setup_wg, request):
    scheduler = request.param
    wg = setup_wg['wg']
    contractors = setup_wg['contractors']

    schedule = scheduler().schedule(wg, contractors)
    return {'schedule': schedule}