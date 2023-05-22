from tests.scheduling.fixtures import setup_wg, setup_contractors, setup_schedule_genetic


def test_real_wg_loading(setup_wg, setup_contractors, setup_schedule_genetic):
    name, wg = setup_wg
    l = list(setup_contractors[0].workers.keys())
    t = type(l[0])
    print(t)
    pass
