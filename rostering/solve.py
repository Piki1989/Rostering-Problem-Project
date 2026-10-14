"""Lexicographic (multi-phase) solution of the rostering model.

Phase 1   minimise weighted understaffing                -> fixed
Phase 1b  pandemic: minimise cross-bubble contacts        -> fixed
Phase 2   minimise the number of hired candidates         -> fixed
Phase 3   minimise Z >= every normalised deviation (Chebyshev / min-max compromise)

The constraints and objectives are exactly those of the original solve_3_phase(); the additions
(status checks, warm start, logging) only affect how the solver is driven.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from .config import ObjectiveTargets, SINGLE_PHASE_TARGETS, SolverConfig
from .model import UNDERWORK_TARGET, RosteringModel

logger = logging.getLogger(__name__)

SOLVED = (cp_model.OPTIMAL, cp_model.FEASIBLE)


class PhaseFailed(RuntimeError):
    pass


@dataclass
class PhaseResult:
    name: str
    status: str
    objective: float | None
    bound: float | None
    wall_time: float


@dataclass
class SolveOutcome:
    solver: cp_model.CpSolver
    status: int
    phases: list[PhaseResult] = field(default_factory=list)


def _new_solver(cfg: SolverConfig) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = cfg.workers()
    if cfg.random_seed is not None:
        solver.parameters.random_seed = cfg.random_seed
    if cfg.log_search_progress:
        solver.parameters.log_search_progress = True
    return solver


def _hint_from_solution(model: cp_model.CpModel, solver: cp_model.CpSolver) -> None:
    """Use the solution of the previous phase as the starting point of the next one.

    It stays feasible because every phase only adds `objective == value found` constraints.
    Hints never change the model's constraints or objective.
    """
    solution = list(solver.ResponseProto().solution)
    model.ClearHints()
    for index, value in enumerate(solution):
        model.AddHint(model.GetIntVarFromProtoIndex(index), value)


def _run_phase(name, model, solver, cfg: SolverConfig, time_limit, outcome: SolveOutcome) -> int:
    solver.parameters.max_time_in_seconds = time_limit
    start = time.time()
    status = solver.Solve(model)
    solved = status in SOLVED
    result = PhaseResult(
        name=name,
        status=solver.StatusName(status),
        objective=solver.ObjectiveValue() if solved else None,
        bound=solver.BestObjectiveBound() if solved else None,
        wall_time=round(time.time() - start, 2),
    )
    outcome.phases.append(result)
    logger.info("Phase %-12s %-10s objective=%s bound=%s (%.1fs of %ss)",
                name, result.status, result.objective, result.bound, result.wall_time, time_limit)
    if not solved:
        raise PhaseFailed(f"Phase '{name}' ended with status {result.status} - no solution to continue from")
    if cfg.use_hints:
        _hint_from_solution(model, solver)
    return status


def _coeff(scale: int, max_value: int) -> int:
    """Normalisation coefficient SCALE // max_value; 0 (term disabled) when the metric cannot occur."""
    return scale // max_value if max_value > 0 else 0


def _add_chebyshev(m: RosteringModel, targets: ObjectiveTargets) -> cp_model.IntVar:
    """Z >= normalised deviation of every criterion from its target. Returns Z."""
    model, p = m.model, m.p
    scale = targets.scale
    D = m.ctx.D_all

    total_cost = sum(m.doctor_cost.values())
    total_underwork = sum(m.underwork.values())
    total_weekly_underwork = sum(m.weekly_underwork[d][w] for d in m.weekly_underwork for w in p.weeks)
    total_pref = sum(m.pref_terms) if m.pref_terms else 0

    max_pref = len(m.pref_terms)
    max_night_spread = len(p.night_shifts)
    max_weekend_spread = len(p.weekend_shifts)
    max_hard_spread = len(p.hard_shifts)
    max_underwork = sum(int(UNDERWORK_TARGET * p.adjusted_max_hours[d]) for d in p.regular_doctors)
    max_weekly_underwork = sum(int(UNDERWORK_TARGET * p.weekly_limits[d][w]) for d in D for w in p.weeks)

    Z = model.NewIntVar(0, scale * 10, "chebyshev_Z")
    model.Add(Z >= 0)

    # Budget: cost above B is penalised. The upper bound of over_budget also makes
    # cost <= B * (1 + max_over_budget_pct) a hard constraint (original behaviour).
    budget = targets.cost
    max_over = int(budget * targets.max_over_budget_pct)
    over_budget = model.NewIntVar(0, max_over, "over_budget")
    model.Add(over_budget >= total_cost - budget)
    model.Add(over_budget >= 0)
    if max_over > 0:
        model.Add(Z >= over_budget * (scale // max_over))

    model.Add(Z >= (m.spread - targets.spread) * _coeff(scale, max_night_spread))
    model.Add(Z >= (m.weekend_spread - targets.weekend_spread) * _coeff(scale, max_weekend_spread))
    model.Add(Z >= (m.hard_spread - targets.hard_spread) * _coeff(scale, max_hard_spread))
    model.Add(Z >= (total_underwork - targets.underwork) * _coeff(scale, max_underwork))
    model.Add(Z >= (total_weekly_underwork - targets.weekly_underwork) * _coeff(scale, max_weekly_underwork))
    model.Add(Z >= (total_pref - targets.pref) * _coeff(scale, max_pref))
    return Z


def solve_lexicographic(
    m: RosteringModel,
    cfg: SolverConfig | None = None,
    targets: ObjectiveTargets | None = None,
    *,
    pandemic_mode: bool = False,
) -> SolveOutcome:
    cfg = cfg or SolverConfig()
    targets = targets or ObjectiveTargets()
    model = m.model
    solver = _new_solver(cfg)
    outcome = SolveOutcome(solver=solver, status=cp_model.UNKNOWN)

    limit1, limit2, limit3 = cfg.time_limit_phase1, cfg.time_limit_phase2, cfg.time_limit_phase3
    if pandemic_mode:
        solver.parameters.linearization_level = 2

    if cfg.feasibility_probe:
        probe = cp_model.CpSolver()
        probe.parameters.max_time_in_seconds = 30
        probe.parameters.num_search_workers = 1
        model.Minimize(0)
        logger.info("Feasibility probe: %s", probe.StatusName(probe.Solve(model)))

    # Phase 1 - weighted understaffing.
    weighted_slack = cp_model.LinearExpr.Sum([m.slacks[s] * m.slack_weights[s] for s in m.slacks])
    raw_slack = cp_model.LinearExpr.Sum(list(m.slacks.values()))
    model.Minimize(weighted_slack)
    _run_phase("understaffing", model, solver, cfg, limit1, outcome)
    best_weighted_slack = solver.Value(weighted_slack)
    logger.info("Understaffing: weighted %d, uncovered positions %d", best_weighted_slack, solver.Value(raw_slack))
    model.Add(weighted_slack == best_weighted_slack)

    # Phase 1b - pandemic: cross-bubble contacts.
    pb = m.pandemic_bundle
    if pb is not None and pb["crossbubble_violations"]:
        violations = cp_model.LinearExpr.Sum(pb["crossbubble_violations"])
        model.Minimize(violations)
        _run_phase("isolation", model, solver, cfg, cfg.time_limit_isolation, outcome)
        best_violations = solver.Value(violations)
        logger.info("Cross-bubble contacts: %d", best_violations)
        model.Add(violations == best_violations)

    # Phase 2 - number of hires.
    total_hire = cp_model.LinearExpr.Sum(list(m.hire.values()))
    model.Minimize(total_hire)
    _run_phase("hiring", model, solver, cfg, limit2, outcome)
    best_hire = solver.Value(total_hire)
    model.Add(total_hire == best_hire)

    # Phase 3 - Chebyshev compromise of cost, fairness, underwork and preferences.
    Z = _add_chebyshev(m, targets)
    model.Minimize(Z)
    try:
        outcome.status = _run_phase("compromise", model, solver, cfg, limit3, outcome)
    except PhaseFailed as exc:
        if outcome.phases[-1].status == "INFEASIBLE":
            raise PhaseFailed(
                f"{exc}. Most likely the cost exceeds the hard cap budget * (1 + max_over_budget_pct) = "
                f"{targets.cost * (1 + targets.max_over_budget_pct):,.0f} (ObjectiveTargets)."
            ) from exc
        raise
    return outcome


def solve_single_phase_comparison(
    m: RosteringModel,
    cfg: SolverConfig | None = None,
    targets: ObjectiveTargets = SINGLE_PHASE_TARGETS,
    *,
    time_limit: float = 300,
) -> SolveOutcome:
    """Experiment: one weighted objective 1e6 * understaffing + 1e4 * hires + Z instead of the phases."""
    cfg = cfg or SolverConfig()
    model = m.model
    Z = _add_chebyshev(m, targets)
    model.Minimize(sum(m.slacks.values()) * 1_000_000 + sum(m.hire.values()) * 10_000 + Z)

    solver = cp_model.CpSolver()     # the original experiment used default solver parameters
    solver.parameters.max_time_in_seconds = time_limit
    outcome = SolveOutcome(solver=solver, status=cp_model.UNKNOWN)
    start = time.time()
    outcome.status = solver.Solve(model)
    outcome.phases.append(PhaseResult("single", solver.StatusName(outcome.status),
                                      solver.ObjectiveValue() if outcome.status in SOLVED else None,
                                      solver.BestObjectiveBound() if outcome.status in SOLVED else None,
                                      round(time.time() - start, 2)))
    return outcome
