import numpy as np
import pytest

from brlo.firstmile.heuristics import check, dist_matrix, solve_classical, total_length
from brlo.firstmile.instances import clustered, uniform


@pytest.mark.parametrize("gen", [uniform, clustered])
def test_classical_valid_and_ls_improves(gen):
    b = gen(5, 30, seed=1)
    for i in range(5):
        D = dist_matrix(b.coords[i])
        cw = solve_classical(b.coords[i], b.demand[i], b.capacity, "cw")
        ls = solve_classical(b.coords[i], b.demand[i], b.capacity, "cw+ls")
        check(cw, b.demand[i], b.capacity)
        check(ls, b.demand[i], b.capacity)
        assert total_length(ls, D) <= total_length(cw, D) + 1e-9


torch = pytest.importorskip("torch")


def test_pomo_rollout_feasible_and_differentiable():
    from brlo.firstmile.neural import AttentionPolicy, reinforce_loss, rollout, solve
    torch.manual_seed(0)
    b = uniform(4, 10, seed=0)
    model = AttentionPolicy(d=32, heads=4, layers=2, ff=64)
    c = torch.tensor(b.coords, dtype=torch.float32)
    q = torch.tensor(b.demand, dtype=torch.float32) / b.capacity
    r = rollout(model, c, q)
    assert r.length.shape == (4, 10) and torch.isfinite(r.logp).all()
    loss = reinforce_loss(r)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    L, routes = solve(model, b.coords, b.demand, b.capacity, augment=True)
    for i in range(4):
        check(routes[i], b.demand[i], b.capacity)
        assert abs(total_length(routes[i], dist_matrix(b.coords[i])) - L[i]) < 1e-4  # x8 augmentation is an isometry
