"""Every equation checked alone against the independent oracle (rostering/oracles.py), plus
deliberately wrong implementations that the check must reject."""
from __future__ import annotations

import pytest
from ortools.sat.python import cp_model

from rostering.equations import EQUATIONS
from rostering.verify import check_equation


@pytest.mark.parametrize("eq_id", list(EQUATIONS))
def test_reference_equation_matches_oracle(eq_id):
    result = check_equation(eq_id, samples=30)
    assert result.passed, "\n".join([result.summary()] + result.failures[:5])


# ---------------------------------------------------------------------------------------------
# Wrong implementations - each differs from the specification in one detail.
# ---------------------------------------------------------------------------------------------

def h2_two_shifts_per_day(m):
    for d in m.ctx.D_all:
        for date in m.p.date_list:
            m.model.Add(sum(m.x[(d, s)] for s in m.p.days_to_shifts[date]) <= 2)


def h3_thirteen_hours_rest(m):
    # The data has no gaps of 10-11 h between shifts, so any threshold in (9, 12] gives the same
    # model as 11 h; 13 h does not (gaps of 12 h exist).
    for d in m.ctx.D_all:
        for s1 in m.ctx.S:
            for s2 in m.ctx.S:
                gap = m.p.abs_start[s2] - m.p.abs_end[s1]
                if s1 != s2 and 0 < gap < 13:
                    m.model.Add(m.x[(d, s1)] + m.x[(d, s2)] <= 1)


def h5_mentor_from_all_doctors(m):
    for s in m.ctx.S:
        for d in m.p.needs_mentor:
            m.model.Add(m.x[(d, s)] <= sum(m.x[(k, s)] for k in m.ctx.D_all if k != d))


def h8_calendar_weeks(m):
    for d in m.ctx.D_all:
        for w in m.p.weeks:
            days = [t for t in m.p.date_list if m.p.shift_week[m.p.days_to_shifts[t][0]] == w]
            works = []
            for t in days:
                b = m.model.NewBoolVar("")
                m.model.AddMaxEquality(b, [m.x[(d, s)] for s in m.p.days_to_shifts[t]])
                works.append(b)
            m.model.Add(sum(works) <= 6)


def c1_employed_only(m):
    for s in m.ctx.S:
        u = m.model.NewIntVar(0, m.p.min_staff[s], "")
        m.slacks[s] = u
        m.model.Add(sum(m.x[(d, s)] for d in m.ctx.D) + u >= m.p.min_staff[s])


def d1_without_limit(m):
    for d in m.ctx.D_all:
        h = m.model.NewIntVar(0, 10_000, "")
        m.model.Add(h == sum(m.x[(d, s)] * m.p.hours[s] for s in m.ctx.S))
        m.worked_hours[d] = h


def f1_employed_only(m):
    counts = []
    for d in m.ctx.D:
        if m.ctx.id_to_role[d] in ("specialist", "icu_specialist") and m.p.night_allowed[d] == 1:
            c = m.model.NewIntVar(0, 100, "")
            m.model.Add(c == sum(m.x[(d, s)] for s in m.p.night_shifts))
            counts.append(c)
    hi, lo, m.spread = (m.model.NewIntVar(0, 100, "") for _ in range(3))
    m.model.AddMaxEquality(hi, counts)
    m.model.AddMinEquality(lo, counts)
    m.model.Add(m.spread == hi - lo)


def u2_true_regular_doctors(m):
    """Uses the doctor ids of regular doctors - the "fixed" version of known issue 1."""
    for d in m.ctx.D_all:
        m.weekly_underwork[d] = {}
        for w in m.p.weeks:
            if d in m.p.regular_doctors:
                target = int(0.95 * m.p.weekly_limits[d][w])
            else:
                target = int(0.95 * (48 * m.p.weekly_available_days[d][w] / 7))
            uw = m.model.NewIntVar(0, m.p.weekly_limits[d][w], "")
            m.model.Add(uw >= target - m.weekly_hours[d][w])
            m.weekly_underwork[d][w] = uw


def o1_unweighted(m):
    return cp_model.LinearExpr.Sum(list(m.slacks.values()))


def o3_without_budget(m):
    from rostering.equations import o3_compromise
    saved = m.targets
    m.targets = type(saved)(**{**saved.__dict__, "max_over_budget_pct": 10.0})
    try:
        return o3_compromise(m)
    finally:
        m.targets = saved


WRONG = {
    "H2": h2_two_shifts_per_day,
    "H3": h3_thirteen_hours_rest,
    "H5": h5_mentor_from_all_doctors,
    "H8": h8_calendar_weeks,
    "C1": c1_employed_only,
    "D1": d1_without_limit,
    "F1": f1_employed_only,
    "U2": u2_true_regular_doctors,
    "O1": o1_unweighted,
    "O3": o3_without_budget,
}


@pytest.mark.parametrize("eq_id", list(WRONG))
def test_wrong_implementation_is_rejected(eq_id):
    result = check_equation(eq_id, WRONG[eq_id], samples=30)
    assert not result.passed, f"{eq_id}: the wrong implementation was not detected"
