"""Log-interval aggregation: means, except *_max / *_min tails (CPU)."""

import math

from supermimic.training.metrics import accumulate, interval_values


def test_mean_max_min_and_lr():
    agg: dict[str, float] = {}
    for loss, a_max, r_min in [(1.0, 2.0, -0.1), (3.0, 5e3, -1e17), (2.0, 1.0, -0.2)]:
        accumulate(agg, "policy/critic_loss", loss)
        accumulate(agg, "policy/imag_action_abs_max", a_max)
        accumulate(agg, "policy/imag_reward_min", r_min)
    agg["lr"] = 1e-4
    out = interval_values(agg, 3)
    assert out == {"policy/critic_loss": 2.0, "policy/imag_action_abs_max": 5e3, "policy/imag_reward_min": -1e17, "lr": 1e-4}


def test_nan_propagates_in_tails():
    for order in ([1.0, math.nan, 2.0], [math.nan, 1.0, 2.0], [1.0, 2.0, math.nan]):
        agg: dict[str, float] = {}
        for v in order:
            accumulate(agg, "policy/critic_loss_max", v)
            accumulate(agg, "policy/imag_reward_min", v)
        out = interval_values(agg, 3)
        assert math.isnan(out["policy/critic_loss_max"]) and math.isnan(out["policy/imag_reward_min"])
