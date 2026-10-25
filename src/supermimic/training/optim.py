"""Optimizer helpers."""

from __future__ import annotations

import copy

import torch

# Implementation choices, not hyperparameters: a checkpoint must not switch them back.
_IMPL_FLAGS = ("fused", "foreach", "capturable", "differentiable")


def load_optimizer_state(opt: torch.optim.Optimizer, state: dict) -> None:
    """``opt.load_state_dict(state)``, keeping ``opt``'s implementation flags.

    ``Optimizer.load_state_dict`` replaces every param-group entry with the saved one, so loading a checkpoint
    written by a non-fused Adam silently turns a ``fused=True`` optimizer back into the default implementation.
    """
    state = copy.copy(state)
    state["param_groups"] = [
        {**saved, **{k: cur[k] for k in _IMPL_FLAGS if k in cur}} for saved, cur in zip(state["param_groups"], opt.param_groups, strict=True)
    ]
    opt.load_state_dict(state)
