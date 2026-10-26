"""Aggregation of per-iteration metrics over one log interval.

Most keys are averaged. Keys ending in ``_max`` / ``_min`` are tails (largest |action|, smallest reward, largest
critic minibatch loss): averaging would dilute one runaway iteration by the interval length, so they keep the
interval max / min instead. NaN propagates in both.
"""

from __future__ import annotations

import math


def _extreme(a: float, b: float, pick) -> float:
    if math.isnan(a) or math.isnan(b):
        return math.nan
    return pick(a, b)


def accumulate(agg: dict[str, float], key: str, value: float) -> None:
    """Add one iteration's ``value`` of ``key`` to the interval aggregate ``agg`` (in place)."""
    if key not in agg:
        agg[key] = value
    elif key.endswith("_max"):
        agg[key] = _extreme(agg[key], value, max)
    elif key.endswith("_min"):
        agg[key] = _extreme(agg[key], value, min)
    else:
        agg[key] += value


def interval_values(agg: dict[str, float], n: int) -> dict[str, float]:
    """Values to log for an interval of ``n`` iterations: sums become means; ``*_max`` / ``*_min`` and ``lr`` as is."""
    return {k: v if (k == "lr" or k.endswith("_max") or k.endswith("_min")) else v / n for k, v in agg.items()}
