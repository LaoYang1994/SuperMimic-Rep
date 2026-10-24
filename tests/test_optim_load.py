"""Loading a checkpoint keeps the optimizer implementation (fused) but restores state and hyperparameters (CPU)."""

import torch

from supermimic.training.optim import load_optimizer_state


def _stepped(fused: bool | None, lr: float):
    torch.manual_seed(0)
    m = torch.nn.Linear(4, 2)
    opt = torch.optim.Adam(m.parameters(), lr=lr, fused=fused)
    m(torch.randn(8, 4)).sum().backward()
    opt.step()
    return m, opt


def test_fused_survives_loading_a_default_adam_checkpoint():
    _, old = _stepped(None, lr=3e-4)
    saved = old.state_dict()
    m, opt = _stepped(True, lr=1e-2)
    load_optimizer_state(opt, saved)
    g = opt.param_groups[0]
    assert g["fused"] is True
    assert g["lr"] == 3e-4  # hyperparameters still come from the checkpoint
    for p, q in zip(opt.param_groups[0]["params"], old.param_groups[0]["params"]):
        assert torch.equal(opt.state[p]["exp_avg"], old.state[q]["exp_avg"])
    m(torch.randn(8, 4)).sum().backward()
    opt.step()  # fused step runs on the loaded state


def test_plain_load_state_dict_drops_fused():
    # Documents the torch behaviour load_optimizer_state works around.
    _, old = _stepped(None, lr=3e-4)
    _, opt = _stepped(True, lr=1e-2)
    opt.load_state_dict(old.state_dict())
    assert not opt.param_groups[0]["fused"]
