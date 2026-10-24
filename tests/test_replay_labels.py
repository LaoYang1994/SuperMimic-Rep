"""Replay windows label each transition with the termination of its own result state (CPU)."""

import torch

from supermimic.data.replay_buffer import ReplayBuffer

S, A, X = 4, 2, 3


def _write_episode(buf: ReplayBuffer, env: int, n: int, terminated: bool) -> None:
    """Rows as the recorder writes them: n transition rows (s_t, done = label of s_{t+1}) and a terminal row
    holding s_n with no labels. State feature 0 = step index, feature 1 = 1 iff that state is a termination."""
    ids = torch.tensor([env])

    def state(t):
        s = torch.zeros(1, S)
        s[0, 0] = t
        s[0, 1] = float(terminated and t == n)
        return s

    zeros_a = torch.zeros(1, A)
    for t in range(n):
        done = torch.tensor([float(terminated and t == n - 1)])
        buf.add(ids, s=state(t), a=zeros_a, a_exec=zeros_a, xi=torch.zeros(1, X), r=torch.ones(1), done=done,
                frame=torch.tensor([t]), clip=torch.zeros(1, dtype=torch.long), terminal=torch.tensor([False]),
                ep_start=torch.tensor([t == 0]))  # fmt: skip
    buf.add(ids, s=state(n), a=zeros_a, a_exec=zeros_a, xi=torch.zeros(1, X), r=torch.zeros(1), done=torch.zeros(1),
            frame=torch.tensor([n]), clip=torch.zeros(1, dtype=torch.long), terminal=torch.tensor([True]),
            ep_start=torch.tensor([False]))  # fmt: skip
    buf.end_episode(ids)


def test_window_done_is_label_of_each_transition_result():
    torch.manual_seed(0)
    buf = ReplayBuffer(num_envs=2, capacity=32, state_dim=S, act_dim=A, xi_dim=X, device="cpu")
    _write_episode(buf, env=0, n=6, terminated=True)  # ends in a real termination
    _write_episode(buf, env=1, n=6, terminated=False)  # ends in a time-limit truncation
    w = buf.sample_windows(4000, 4)
    expected = w.states[1:, :, 1]  # termination flag of the state each transition reaches
    assert torch.equal(w.done, expected)
    # Only the transition that reaches the terminated state is positive: never the step before it, never a truncation.
    pos = w.done.bool()
    assert pos.any()
    assert torch.all(w.states[1:, :, 0][pos] == 6)
