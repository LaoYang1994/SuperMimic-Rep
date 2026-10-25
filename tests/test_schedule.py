"""Collection schedule: one iteration = one update, 1/UTD simulator steps per iteration on average."""

from supermimic.training.schedule import sim_steps_this_iter


def test_utd_04_alternates_2_and_3():
    acc, total, seen = 0.0, 0, set()
    for _ in range(1000):
        n, acc = sim_steps_this_iter(acc, 0.4)
        total += n
        seen.add(n)
    assert total == 2500
    assert seen == {2, 3}


def test_utd_1_is_one_step():
    acc = 0.0
    for _ in range(100):
        n, acc = sim_steps_this_iter(acc, 1.0)
        assert n == 1
