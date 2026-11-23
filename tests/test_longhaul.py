import pytest

from brlo.data.instances import iberia_base, random_instance
from brlo.longhaul.alns import ALNS
from brlo.longhaul.greedy import greedy_plan
from brlo.longhaul.legacy import legacy_plan
from brlo.longhaul.milp import solve_two_phase
from brlo.longhaul.solution import evaluate


@pytest.fixture(scope="module")
def inst():
    return iberia_base()


@pytest.fixture(scope="module")
def two_phase(inst):
    return solve_two_phase(inst, time_limit=60)


def test_greedy_feasible(inst):
    e = evaluate(inst, greedy_plan(inst))
    assert e.feasible, e.violations


def test_two_phase_certified(inst, two_phase):
    e = evaluate(inst, two_phase.plan)
    assert e.feasible, e.violations
    assert two_phase.bound <= e.cost + 1e-3            # LB <= UB
    assert abs(e.cost - two_phase.objective) < 1.0      # evaluator agrees with MILP objective


def test_milp_beats_greedy(inst, two_phase):
    assert evaluate(inst, two_phase.plan).cost < evaluate(inst, greedy_plan(inst)).cost


def test_legacy_underreports_and_overloads(inst):
    L = legacy_plan(inst, 1)
    e = evaluate(inst, L.plan)
    assert L.reported_cost < e.cost                     # ignores return leg, labour, tolls
    assert any("overload" in v for v in e.violations)  # MMA used as payload


def test_alns_improves_greedy():
    inst = random_instance(8, 1, seed=3)
    g = greedy_plan(inst)
    r = ALNS(inst, seed=0, eval_time=2).run(g, iterations=30, time_limit=60)
    e = evaluate(inst, r.plan)
    assert e.feasible, e.violations
    assert e.cost <= evaluate(inst, g).cost + 1e-6
