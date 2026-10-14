"""Every equation of the rostering model as a separate function.

Contract of an equation function: ``fn(m: RosteringModel) -> None`` (constraints) or
``-> linear expression / variable`` (objectives). It reads only

* ``m.model``  - the CpModel to add constraints and variables to,
* ``m.x[d, s]`` - BoolVar, doctor d works shift s; ``m.hire[c]`` - BoolVar, candidate c is hired,
* ``m.ctx``    - index sets (D, C, D_all, S, id_to_role, ...),
* ``m.p``      - parameters derived from the input files (see rostering.model.Params),
* the outputs of the equations listed in ``requires``,

and stores its own outputs in the ``RosteringModel`` fields named in ``outputs``. Each function
can be replaced by another implementation and checked alone with
``python -m rostering.verify <ID> --impl file.py:function`` (see docs/EQUATIONS.md).

The mathematical form of each equation is in its docstring and in docs/MODEL.md.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Callable

from ortools.sat.python import cp_model

from .pandemic import CLINIC_POOL
from .model import (
    MAX_CONSECUTIVE_NIGHTS,
    MAX_WORKING_DAYS_PER_WEEK,
    MIN_REST_HOURS,
    PANDEMIC_SLACK_WEIGHT_OTHER,
    PANDEMIC_SLACK_WEIGHTS,
    SPECIALIST_ROLES,
    UNDERWORK_TARGET,
    RosteringModel,
    weekly_target,
)

logger = logging.getLogger(__name__)

# =============================================================================================
# Hard constraints
# =============================================================================================

def h1_skills(m: RosteringModel) -> None:
    """x[d,s] = 0 if the skill required by shift s is not among the skills of doctor d."""
    for d in m.ctx.D_all:
        for s in m.ctx.S:
            if m.p.req_skill[s] not in m.p.skill_map[d]:
                m.model.Add(m.x[(d, s)] == 0)


def h2_one_shift_per_day(m: RosteringModel) -> None:
    """sum_{s in S_t} x[d,s] <= 1 for every doctor d and day t."""
    for d in m.ctx.D_all:
        for date in m.p.date_list:
            m.model.Add(sum(m.x[(d, s)] for s in m.p.days_to_shifts[date]) <= 1)


def h3_min_rest(m: RosteringModel) -> None:
    """x[d,s1] + x[d,s2] <= 1 for s1 != s2 with end(s1) < start(s2) < end(s1) + 11 h."""
    p, S = m.p, m.ctx.S
    rest_pairs = [
        (s1, s2) for s1 in S for s2 in S
        if s1 != s2 and p.abs_start[s2] > p.abs_end[s1] and p.abs_start[s2] - p.abs_end[s1] < MIN_REST_HOURS
    ]
    for d in m.ctx.D_all:
        for s1, s2 in rest_pairs:
            m.model.Add(m.x[(d, s1)] + m.x[(d, s2)] <= 1)


def h4_weekly_hours(m: RosteringModel) -> None:
    """sum_{s in S_w} hours[s] * x[d,s] <= L[d,w] (weekly limit scaled by available days)."""
    p = m.p
    shifts_by_week = {w: [s for s in m.ctx.S if p.shift_week[s] == w] for w in p.weeks}
    for d in m.ctx.D_all:
        for w in p.weeks:
            m.model.Add(sum(m.x[(d, s)] * p.hours[s] for s in shifts_by_week[w]) <= p.weekly_limits[d][w])


def h5_mentor(m: RosteringModel) -> None:
    """x[d,s] <= sum_{k in specialists} x[k,s] for every doctor d needing a mentor and shift s."""
    for s in m.ctx.S:
        for d in m.p.needs_mentor:
            m.model.Add(m.x[(d, s)] <= sum(m.x[(spec, s)] for spec in m.p.specialists))


def h6_consecutive_nights(m: RosteringModel) -> None:
    """At most 2 night shifts in any 3 consecutive days: sum_{s in N_t, N_t+1, N_t+2} x[d,s] <= 2."""
    p = m.p
    for d in m.ctx.D_all:
        for i in range(len(p.date_list) - 2):
            window = [s for day in p.date_list[i:i + 3] for s in p.night_shifts_by_date[day]]
            if window:
                m.model.Add(sum(m.x[(d, s)] for s in window) <= MAX_CONSECUTIVE_NIGHTS)


def h7_day_off_after_night(m: RosteringModel) -> None:
    """sum_{s in N_t} x[d,s] + sum_{s in S_t+1} x[d,s] <= 1: no shift on the day after a night shift."""
    p = m.p
    for d in m.ctx.D_all:
        for i in range(len(p.date_list) - 1):
            nights = p.night_shifts_by_date[p.date_list[i]]
            next_day = p.days_to_shifts[p.date_list[i + 1]]
            if nights and next_day:
                m.model.Add(sum(m.x[(d, s)] for s in nights) + sum(m.x[(d, s)] for s in next_day) <= 1)


def h8_working_days(m: RosteringModel) -> None:
    """At most 6 working days in every block of 7 days counted from the first day of the horizon.

    work[d,t] <= sum_{s in S_t} x[d,s] <= |S_t| * work[d,t];  sum_{t in block} work[d,t] <= 6.
    (Described in the original code as "minimum 35-hour weekly rest".)
    """
    p = m.p
    for d in m.ctx.D_all:
        for week_start in range(0, len(p.date_list), 7):
            works_vars = []
            for date in p.date_list[week_start:week_start + 7]:
                day_shifts = p.days_to_shifts[date]
                works = m.model.NewBoolVar(f"works_{d}_{date}")
                works_vars.append(works)
                m.model.Add(sum(m.x[(d, s)] for s in day_shifts) >= works)
                m.model.Add(sum(m.x[(d, s)] for s in day_shifts) <= len(day_shifts) * works)
            m.model.Add(sum(works_vars) <= MAX_WORKING_DAYS_PER_WEEK)


def h9_twentyfour_eligibility(m: RosteringModel) -> None:
    """x[d,s] = 0 for every 24 h shift s if doctor d may not work 24 h shifts."""
    for d in m.ctx.D_all:
        if m.p.twentyfour_allowed[d] == 0:
            for s in m.p.twentyfour_shifts:
                m.model.Add(m.x[(d, s)] == 0)


def h9a_pandemic_no_24h(m: RosteringModel) -> None:
    """Pandemic mode: x[d,s] = 0 for every doctor and every 24 h shift."""
    for d in m.ctx.D_all:
        for s in m.p.twentyfour_shifts:
            m.model.Add(m.x[(d, s)] == 0)


def h10a_unavailable_days(m: RosteringModel) -> None:
    """x[d,s] = 0 for every shift s on a day the doctor is unavailable (leave, absence)."""
    for d, date in m.p.unavailable_days:
        for s in m.p.days_to_shifts[date]:
            m.model.Add(m.x[d, s] == 0)


def h10b_unavailable_shifts(m: RosteringModel) -> None:
    """x[d,s] = 0 for every single shift the doctor is unavailable for."""
    for d, s in m.p.unavailable_shifts:
        m.model.Add(m.x[d, s] == 0)


def h11_candidate_hired(m: RosteringModel) -> None:
    """x[c,s] <= hire[c]: a candidate works only if hired."""
    for c in m.ctx.C:
        for s in m.ctx.S:
            m.model.Add(m.x[(c, s)] <= m.hire[c])


def h12_night_eligibility(m: RosteringModel) -> None:
    """x[d,s] = 0 for every night shift s if doctor d may not work at night."""
    for d in m.ctx.D_all:
        if m.p.night_allowed[d] == 0:
            for s in m.p.night_shifts:
                m.model.Add(m.x[(d, s)] == 0)


def h13_no_hiring(m: RosteringModel) -> None:
    """Baseline scenario (hiring not allowed): hire[c] = 0 for every candidate."""
    for c in m.ctx.C:
        m.model.Add(m.hire[c] == 0)


def hp1_bubble_isolation(m: RosteringModel) -> None:
    """Pandemic isolation of bubbles on contact shifts (departments in config.contact_depts).

    For every contact shift s and every pair d1 < d2 (order of D_all) assigned to different
    bubbles, both outside the CLINIC pool: x[d1,s] + x[d2,s] <= 1 + v[d1,d2,s], v boolean.
    Output: m.pandemic_bundle = {"crossbubble_violations": [v, ...]}.
    """
    D_all, bubbles = m.ctx.D_all, m.bubble_assignment
    contact_depts = set(m.pandemic_config.contact_depts)
    contact_shifts = [s for s in m.ctx.S if m.p.dept[s] in contact_depts]
    logger.info("Pandemic constraints: contact departments %s, contact shifts %d / %d",
                contact_depts, len(contact_shifts), len(m.ctx.S))

    missing = [d for d in D_all if d not in bubbles]
    if missing:
        warnings.warn(f"[PandemicMode] {len(missing)} doctors without bubble assignment, "
                      f"skipped in pandemic constraints: {missing}")

    violations = []
    for s in contact_shifts:
        for i, d1 in enumerate(D_all):
            if d1 not in bubbles or bubbles[d1] == CLINIC_POOL:
                continue
            for d2 in D_all[i + 1:]:
                if d2 not in bubbles or bubbles[d2] == CLINIC_POOL:
                    continue
                if bubbles[d1] != bubbles[d2]:
                    viol = m.model.NewBoolVar(f"crossbubble_{d1}_{d2}_{s}")
                    m.model.Add(m.x[(d1, s)] + m.x[(d2, s)] <= 1 + viol)
                    violations.append(viol)
    logger.info("Added %d pandemic constraints", len(violations))
    m.pandemic_bundle = {"crossbubble_violations": violations}


# =============================================================================================
# Staffing
# =============================================================================================

def c1_min_staffing(m: RosteringModel) -> None:
    """sum_{d in D_all} x[d,s] + u[s] >= min_staff[s], u[s] in [0, min_staff[s]] (understaffing).

    Output: m.slacks[s] = u[s].
    """
    for s in m.ctx.S:
        slack = m.model.NewIntVar(0, m.p.min_staff[s], f"slack_{s}")
        m.slacks[s] = slack
        m.model.Add(sum(m.x[(d, s)] for d in m.ctx.D_all) + slack >= m.p.min_staff[s])


def c2_max_staffing(m: RosteringModel) -> None:
    """sum_{d in D} x[d,s] + o[s] <= regular_staff[s], o[s] in [0, regular_staff[s]].

    o[s] is not penalised anywhere, so this is the hard cap sum_{d in D} x[d,s] <= regular_staff[s]
    (employed doctors only). Output: m.slacks_o[s] = o[s].
    """
    for s in m.ctx.S:
        slack_o = m.model.NewIntVar(0, m.p.regular_staff[s], f"slack_{s}")   # same name as in the original
        m.slacks_o[s] = slack_o
        m.model.Add(sum(m.x[(d, s)] for d in m.ctx.D) + slack_o <= m.p.regular_staff[s])


# =============================================================================================
# Hours and cost
# =============================================================================================

def d1_worked_hours(m: RosteringModel) -> None:
    """H[d] = sum_s hours[s] * x[d,s], H[d] in [0, M[d]] (monthly limit). Output: m.worked_hours[d]."""
    for d in m.ctx.D_all:
        h_var = m.model.NewIntVar(0, m.p.adjusted_max_hours[d], f"worked_hours_{d}")
        m.model.Add(h_var == sum(m.x[(d, s)] * m.p.hours[s] for s in m.ctx.S))
        m.worked_hours[d] = h_var


def d2a_employee_cost(m: RosteringModel) -> None:
    """cost[d] = rate[d] * H[d] for employed doctors, cost[d] in [0, 10^7]. Output: m.doctor_cost[d]."""
    for d in m.ctx.D:
        cost = m.model.NewIntVar(0, 10_000_000, f"cost_{d}")
        m.model.Add(cost == m.worked_hours[d] * m.p.hourly_cost_map[d])
        m.doctor_cost[d] = cost


def d2b_candidate_cost(m: RosteringModel) -> None:
    """E[c] = H[c] * hire[c] (E in [0, 2000]), cost[c] = rate[c] * E[c]. Output: m.doctor_cost[c]."""
    for c in m.ctx.C:
        effective_hours = m.model.NewIntVar(0, 2000, f"effective_hours_{c}")
        m.model.AddMultiplicationEquality(effective_hours, [m.worked_hours[c], m.hire[c]])
        cost = m.model.NewIntVar(0, 10_000_000, f"cost_{c}")
        m.model.Add(cost == effective_hours * m.p.hourly_cost_map[c])
        m.doctor_cost[c] = cost


def d3_weekly_hours(m: RosteringModel) -> None:
    """WH[d,w] = sum_{s in S_w} hours[s] * x[d,s], WH in [0, L[d,w]]. Output: m.weekly_hours[d][w]."""
    p = m.p
    shifts_by_week = {w: [s for s in m.ctx.S if p.shift_week[s] == w] for w in p.weeks}
    for d in m.ctx.D_all:
        m.weekly_hours[d] = {}
        for w in p.weeks:
            wh = m.model.NewIntVar(0, p.weekly_limits[d][w], f"weekly_hours_{d}_{w}")
            m.model.Add(wh == sum(m.x[(d, s)] * p.hours[s] for s in shifts_by_week[w]))
            m.weekly_hours[d][w] = wh


# =============================================================================================
# Soft criteria
# =============================================================================================

def s1_preferences(m: RosteringModel) -> None:
    """P = sum_{like} (-x[d,s]) + sum_{dislike} x[d,s]. Output: m.pref_terms (list of terms of P)."""
    for d, s, preference in m.p.preferences:
        if preference == "like":
            m.pref_terms.append(-1 * m.x[(d, s)])
        if preference == "dislike":
            m.pref_terms.append(1 * m.x[(d, s)])


def _spread(model: cp_model.CpModel, counts: list, name: str) -> cp_model.IntVar:
    hi = model.NewIntVar(0, 100, f"max_{name}")
    lo = model.NewIntVar(0, 100, f"min_{name}")
    spread = model.NewIntVar(0, 100, f"{name}_spread")
    model.AddMaxEquality(hi, counts)
    model.AddMinEquality(lo, counts)
    model.Add(spread == hi - lo)
    return spread


def _is_specialist(m: RosteringModel, d) -> bool:
    return m.ctx.id_to_role[d] in SPECIALIST_ROLES


def f1_night_spread(m: RosteringModel) -> None:
    """n[d] = sum_{s in N} x[d,s] for specialists (D_all) allowed at night; spread_N = max n - min n.

    Output: m.spread.
    """
    model, p = m.model, m.p
    night_count = {}
    for d, allowed in p.night_allowed.items():
        if _is_specialist(m, d) and allowed == 1:
            count = model.NewIntVar(0, 100, f"night_count_{d}")
            model.Add(count == sum(m.x[(d, s)] for s in p.night_shifts))
            night_count[d] = count
    max_nights = model.NewIntVar(0, 100, "max_nights")
    min_nights = model.NewIntVar(0, 100, "min_nights")
    model.AddMaxEquality(max_nights, list(night_count.values()))
    model.AddMinEquality(min_nights, list(night_count.values()))
    m.spread = model.NewIntVar(0, 100, "night_spread")
    model.Add(m.spread == max_nights - min_nights)


def f2_weekend_spread(m: RosteringModel) -> None:
    """we[d] = sum_{s in WE} x[d,s] for employed specialists; spread_WE = max - min. Output: m.weekend_spread."""
    counts = []
    for d in m.ctx.D:
        if _is_specialist(m, d):
            wc = m.model.NewIntVar(0, 100, f"weekend_count_{d}")
            m.model.Add(wc == sum(m.x[(d, s)] for s in m.p.weekend_shifts))
            counts.append(wc)
    m.weekend_spread = _spread(m.model, counts, "weekend")


def f3_hard_spread(m: RosteringModel) -> None:
    """hd[d] = sum_{s in HS} x[d,s] for employed specialists; spread_H = max - min. Output: m.hard_spread."""
    counts = []
    for d in m.ctx.D:
        if _is_specialist(m, d):
            hc = m.model.NewIntVar(0, 100, f"hard_count_{d}")
            m.model.Add(hc == sum(m.x[(d, s)] for s in m.p.hard_shifts))
            counts.append(hc)
    m.hard_spread = _spread(m.model, counts, "hard")


def f4_workload_ratio(m: RosteringModel) -> None:
    """r[d] * M[d] in [1000 H[d] - 50, 1000 H[d] + 50] for opt-out doctors, r in [0, 2000];
    spread_R = max r - min r (0 without opt-out doctors). Not used by any objective. Output: m.ratio_spread.
    """
    model, p = m.model, m.p
    ratios = []
    for d in p.opt_out_doctors:
        ratio = model.NewIntVar(0, 2000, f"workload_ratio_{d}")
        model.Add(ratio * p.adjusted_max_hours[d] <= m.worked_hours[d] * 1000 + 50)
        model.Add(ratio * p.adjusted_max_hours[d] >= m.worked_hours[d] * 1000 - 50)
        ratios.append(ratio)

    max_ratio = model.NewIntVar(0, 2000, "max_ratio")
    min_ratio = model.NewIntVar(0, 2000, "min_ratio")
    m.ratio_spread = model.NewIntVar(0, 2000, "ratio_spread")
    if ratios:
        model.AddMaxEquality(max_ratio, ratios)
        model.AddMinEquality(min_ratio, ratios)
        model.Add(m.ratio_spread == max_ratio - min_ratio)
    else:
        model.Add(max_ratio == 0)
        model.Add(min_ratio == 0)
        model.Add(m.ratio_spread == 0)


def u1_monthly_underwork(m: RosteringModel) -> None:
    """UW[d] >= floor(0.95 M[d]) - H[d], UW[d] >= 0, UW in [0, M[d]] for regular doctors. Output: m.underwork[d]."""
    p = m.p
    for d in p.regular_doctors:
        uw = m.model.NewIntVar(0, p.adjusted_max_hours[d], f"underwork_{d}")
        target = int(UNDERWORK_TARGET * p.adjusted_max_hours[d])
        h = m.hire[d] if d in m.hire else 1
        m.model.Add(uw >= target * h - m.worked_hours[d])
        m.model.Add(uw >= 0)
        m.underwork[d] = uw


def u2_weekly_underwork(m: RosteringModel) -> None:
    """WU[d,w] >= target[d,w] - WH[d,w], WU >= 0, WU in [0, L[d,w]] for every doctor and week.

    target[d,w] = floor(0.95 L[d,w]) for "regular" doctors (p.weekly_regular_lookup),
    floor(0.95 * 48 * avail[d,w] / 7) otherwise (rostering.model.weekly_target). Output: m.weekly_underwork[d][w].
    """
    p = m.p
    for d in m.ctx.D_all:
        for w in p.weeks:
            if weekly_target(m, d, w) > p.weekly_limits[d][w]:
                logger.debug("Weekly target above the limit: doctor=%s week=%s target=%s limit=%s",
                             d, w, weekly_target(m, d, w), p.weekly_limits[d][w])
    for d in m.ctx.D_all:
        m.weekly_underwork[d] = {}
        for w in p.weeks:
            uw = m.model.NewIntVar(0, p.weekly_limits[d][w], f"weekly_underwork_{d}_{w}")
            m.model.Add(uw >= weekly_target(m, d, w) - m.weekly_hours[d][w])
            m.model.Add(uw >= 0)
            m.weekly_underwork[d][w] = uw


# =============================================================================================
# Objectives of the lexicographic phases
# =============================================================================================

def slack_weights(m: RosteringModel) -> dict:
    """w[s] = 1; in pandemic mode ICU 100, WARD 50, CLINIC 1, other departments 10."""
    if m.pandemic_config is None:
        return {s: 1 for s in m.ctx.S}
    return {s: PANDEMIC_SLACK_WEIGHTS.get(m.p.dept[s], PANDEMIC_SLACK_WEIGHT_OTHER) for s in m.ctx.S}


def o1_understaffing(m: RosteringModel):
    """Phase 1 objective: sum_s w[s] * u[s]."""
    m.slack_weights = slack_weights(m)
    return cp_model.LinearExpr.Sum([m.slacks[s] * m.slack_weights[s] for s in m.slacks])


def o1b_isolation(m: RosteringModel):
    """Phase 1b objective (pandemic): sum of cross-bubble contacts v."""
    return cp_model.LinearExpr.Sum(m.pandemic_bundle["crossbubble_violations"])


def o2_hiring(m: RosteringModel):
    """Phase 2 objective: number of hired candidates, sum_c hire[c]."""
    return cp_model.LinearExpr.Sum(list(m.hire.values()))


def _coeff(scale: int, max_value: int) -> int:
    """Normalisation SCALE // max_value; 0 (term disabled) when the criterion cannot occur."""
    return scale // max_value if max_value > 0 else 0


def o3_compromise(m: RosteringModel):
    """Phase 3: Chebyshev compromise. Returns Z to minimise, subject to (targets = m.targets)

    Z >= 0;  ob >= sum cost - B, ob >= 0, ob in [0, floor(p B)];  Z >= ob * (SCALE // floor(p B))
    Z >= (spread_N - 4) * (SCALE // |N|);  Z >= (spread_WE - 3) * (SCALE // |WE|)
    Z >= (spread_H - 2) * (SCALE // |HS|); Z >= (sum UW) * (SCALE // sum_R floor(0.95 M))
    Z >= (sum WU) * (SCALE // sum_{d,w} floor(0.95 L));  Z >= (P + 300) * (SCALE // |preferences|)
    Z in [0, 10 SCALE]. A coefficient with a zero denominator is 0.
    """
    model, p, targets = m.model, m.p, m.targets
    scale = targets.scale

    total_cost = sum(m.doctor_cost.values())
    total_underwork = sum(m.underwork.values())
    total_weekly_underwork = sum(m.weekly_underwork[d][w] for d in m.weekly_underwork for w in p.weeks)
    total_pref = sum(m.pref_terms) if m.pref_terms else 0

    max_pref = len(m.pref_terms)
    max_night_spread = len(p.night_shifts)
    max_weekend_spread = len(p.weekend_shifts)
    max_hard_spread = len(p.hard_shifts)
    max_underwork = sum(int(UNDERWORK_TARGET * p.adjusted_max_hours[d]) for d in p.regular_doctors)
    max_weekly_underwork = sum(int(UNDERWORK_TARGET * p.weekly_limits[d][w]) for d in m.ctx.D_all for w in p.weeks)

    Z = model.NewIntVar(0, scale * 10, "chebyshev_Z")
    model.Add(Z >= 0)

    # The upper bound of over_budget makes cost <= B * (1 + p) a hard constraint (original behaviour).
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


# =============================================================================================
# Registry
# =============================================================================================

def _always(m: RosteringModel) -> bool:
    return True


def _pandemic(m: RosteringModel) -> bool:
    return m.pandemic_config is not None


def _pandemic_bubbles(m: RosteringModel) -> bool:
    return m.pandemic_config is not None and m.bubble_assignment is not None


def _no_hiring(m: RosteringModel) -> bool:
    return not m.allow_hire


def _has_isolation(m: RosteringModel) -> bool:
    return m.pandemic_bundle is not None and bool(m.pandemic_bundle["crossbubble_violations"])


@dataclass(frozen=True)
class Equation:
    id: str
    name: str
    fn: Callable
    kind: str                                   # "constraint" or "objective"
    requires: tuple[str, ...] = ()              # equations whose outputs fn reads
    outputs: tuple[str, ...] = ()               # RosteringModel fields written by fn
    applies: Callable[[RosteringModel], bool] = _always

    @property
    def summary(self) -> str:
        return (self.fn.__doc__ or "").strip().splitlines()[0]


# Constraints in the order in which build_model() adds them.
CONSTRAINTS: list[Equation] = [
    Equation("H1", "skills", h1_skills, "constraint"),
    Equation("H2", "one shift per day", h2_one_shift_per_day, "constraint"),
    Equation("H3", "minimum rest of 11 h", h3_min_rest, "constraint"),
    Equation("H4", "weekly hour limit", h4_weekly_hours, "constraint"),
    Equation("H5", "mentor present", h5_mentor, "constraint"),
    Equation("H6", "max 2 nights in 3 days", h6_consecutive_nights, "constraint"),
    Equation("H7", "day off after a night shift", h7_day_off_after_night, "constraint"),
    Equation("H8", "max 6 working days per 7-day block", h8_working_days, "constraint"),
    Equation("H9", "24 h shift eligibility", h9_twentyfour_eligibility, "constraint"),
    Equation("H9a", "pandemic: no 24 h shifts", h9a_pandemic_no_24h, "constraint", applies=_pandemic),
    Equation("H10a", "unavailable days", h10a_unavailable_days, "constraint"),
    Equation("H10b", "unavailable shifts", h10b_unavailable_shifts, "constraint"),
    Equation("H11", "candidate works only if hired", h11_candidate_hired, "constraint"),
    Equation("H12", "night shift eligibility", h12_night_eligibility, "constraint"),
    Equation("HP1", "pandemic: bubble isolation", hp1_bubble_isolation, "constraint",
             outputs=("pandemic_bundle",), applies=_pandemic_bubbles),
    Equation("C1", "minimum staffing with understaffing slack", c1_min_staffing, "constraint", outputs=("slacks",)),
    Equation("C2", "maximum staffing (employed doctors)", c2_max_staffing, "constraint", outputs=("slacks_o",)),
    Equation("D1", "worked hours", d1_worked_hours, "constraint", outputs=("worked_hours",)),
    Equation("D2a", "cost of employed doctors", d2a_employee_cost, "constraint",
             requires=("D1",), outputs=("doctor_cost",)),
    Equation("D2b", "cost of candidates", d2b_candidate_cost, "constraint",
             requires=("D1",), outputs=("doctor_cost",)),
    Equation("D3", "weekly hours", d3_weekly_hours, "constraint", outputs=("weekly_hours",)),
    Equation("S1", "preferences", s1_preferences, "constraint", outputs=("pref_terms",)),
    Equation("F1", "night shift fairness", f1_night_spread, "constraint", outputs=("spread",)),
    Equation("F2", "weekend fairness", f2_weekend_spread, "constraint", outputs=("weekend_spread",)),
    Equation("F3", "hard shift fairness", f3_hard_spread, "constraint", outputs=("hard_spread",)),
    Equation("F4", "workload ratio of opt-out doctors", f4_workload_ratio, "constraint",
             requires=("D1",), outputs=("ratio_spread",)),
    Equation("U1", "monthly underwork", u1_monthly_underwork, "constraint",
             requires=("D1",), outputs=("underwork",)),
    Equation("U2", "weekly underwork", u2_weekly_underwork, "constraint",
             requires=("D3",), outputs=("weekly_underwork",)),
    Equation("H13", "no hiring in the baseline", h13_no_hiring, "constraint", applies=_no_hiring),
]

OBJECTIVES: list[Equation] = [
    Equation("O1", "phase 1: weighted understaffing", o1_understaffing, "objective", requires=("C1",)),
    Equation("O1b", "phase 1b: cross-bubble contacts", o1b_isolation, "objective", requires=("HP1",),
             applies=_has_isolation),
    Equation("O2", "phase 2: hires", o2_hiring, "objective"),
    Equation("O3", "phase 3: Chebyshev compromise", o3_compromise, "objective",
             requires=("D1", "D2a", "D2b", "D3", "S1", "F1", "F2", "F3", "U1", "U2")),
]

EQUATIONS: dict[str, Equation] = {eq.id: eq for eq in CONSTRAINTS + OBJECTIVES}


def add_constraints(m: RosteringModel, implementations: dict[str, Callable] | None = None) -> None:
    """Add all applicable constraints; ``implementations`` replaces chosen equations (id -> function)."""
    implementations = implementations or {}
    for eq in CONSTRAINTS:
        if eq.applies(m):
            implementations.get(eq.id, eq.fn)(m)
