"""BRLO - Battery Reverse Logistics Optimization.

A three-level decision-support framework for the reverse logistics of
end-of-life (EOL) traction and portable batteries:

* ``brlo.strategic``  - two-stage stochastic, multi-period network design
  (which recyclers to contract / which plants to open, under demand scenarios).
* ``brlo.longhaul``   - exact set-partitioning MILP and an ALNS matheuristic for
  dispatching consolidated waste from storage depots to treatment facilities
  (heterogeneous fleet, multi-commodity, split deliveries, homologated
  containers, load-dependent fuel/CO2, EU Regulation 561/2006 driving times).
* ``brlo.firstmile``  - collection routing from workshops/dealers to depots
  (CVRP) with classical heuristics and a PyTorch attention model trained with
  reinforcement learning (POMO-style shared baseline).

Every solution can be checked by an independent evaluator
(``brlo.longhaul.evaluate``) so that every optimiser is audited against the
same cost and feasibility model.
"""

__version__ = "0.1.0"
