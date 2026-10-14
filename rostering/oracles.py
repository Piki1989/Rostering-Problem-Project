"""Independent pure-Python evaluation of every equation on a concrete schedule.

Written from the specification (docs/MODEL.md), not from rostering/equations.py: for a fixed
schedule ``a`` (which doctor works which shift, which candidates are hired) each oracle says what
the equation must do -

* restriction  -> list of violations (empty: the schedule satisfies the equation),
* value        -> {output name: value} the defined variables must take, or None (infeasible),
* minimum      -> {output name: smallest value} of slack-like variables, or None (infeasible),
* objective    -> optimal value of the phase objective for this schedule, or None (infeasible).

rostering.verify compares these answers with what CP-SAT does with an equation function, so an
implementation (for example one generated from a prompt) can be checked without looking at its
variable names. ``check_schedule`` uses the restrictions to validate a complete schedule.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .model import RosteringModel

SPECIALIST_ROLES = ("specialist", "icu_specialist")
CLINIC_POOL = -1
PANDEMIC_WEIGHTS = {"ICU": 100, "WARD": 50, "CLINIC": 1}


@dataclass
class Assignment:
    works: set = field(default_factory=set)      # {(doctor, shift)} with x = 1
    hired: set = field(default_factory=set)      # {candidate} with hire = 1
    label: str = ""

    def x(self, d, s) -> int:
        return 1 if (d, s) in self.works else 0

    def y(self, c) -> int:
        return 1 if c in self.hired else 0

    def shifts_of(self) -> dict:
        out = defaultdict(list)
        for d, s in self.works:
            out[d].append(s)
        return out


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------

def _hours(m: RosteringModel, a: Assignment) -> dict:
    hours = {d: 0 for d in m.ctx.D_all}
    for d, s in a.works:
        hours[d] += m.p.hours[s]
    return hours


def _weekly_hours(m: RosteringModel, a: Assignment) -> dict:
    wh = {(d, w): 0 for d in m.ctx.D_all for w in m.p.weeks}
    for d, s in a.works:
        wh[(d, m.p.shift_week[s])] += m.p.hours[s]
    return wh


def _staff(m: RosteringModel, a: Assignment, doctors) -> dict:
    count = {s: 0 for s in m.ctx.S}
    members = set(doctors)
    for d, s in a.works:
        if d in members:
            count[s] += 1
    return count


def _d1(m, a):
    hours = _hours(m, a)
    if any(hours[d] > m.p.adjusted_max_hours[d] for d in hours):
        return None
    return hours


def _d3(m, a):
    wh = _weekly_hours(m, a)
    if any(wh[(d, w)] > m.p.weekly_limits[d][w] for d, w in wh):
        return None
    return wh


def _weekly_target(m, d, w) -> int:
    if d in m.p.weekly_regular_lookup:       # see docs/MODEL.md, known issue 1
        return int(0.95 * m.p.weekly_limits[d][w])
    return int(0.95 * (48 * m.p.weekly_available_days[d][w] / 7))


def _spread(values: list) -> int:
    return max(values) - min(values) if values else 0


def _specialist(m, d) -> bool:
    return m.ctx.id_to_role[d] in SPECIALIST_ROLES


# ---------------------------------------------------------------------------------------------
# Restrictions
# ---------------------------------------------------------------------------------------------

def h1(m, a):
    return [f"{d} works {s} without skill {m.p.req_skill[s]}"
            for d, s in a.works if m.p.req_skill[s] not in m.p.skill_map[d]]


def h2(m, a):
    per_day = defaultdict(int)
    for d, s in a.works:
        per_day[(d, m.p.shift_date[s])] += 1
    return [f"{d} works {n} shifts on {t}" for (d, t), n in per_day.items() if n > 1]


def h3(m, a):
    out = []
    for d, shifts in a.shifts_of().items():
        for s1 in shifts:
            for s2 in shifts:
                gap = m.p.abs_start[s2] - m.p.abs_end[s1]
                if s1 != s2 and 0 < gap < 11:
                    out.append(f"{d}: only {gap} h between {s1} and {s2}")
    return out


def h4(m, a):
    wh = _weekly_hours(m, a)
    return [f"{d} works {h} h in week {w} > {m.p.weekly_limits[d][w]}"
            for (d, w), h in wh.items() if h > m.p.weekly_limits[d][w]]


def h5(m, a):
    specialists = set(m.p.specialists)
    staffed_by_specialist = {s for d, s in a.works if d in specialists}
    return [f"{d} works {s} without a specialist"
            for d, s in a.works if d in set(m.p.needs_mentor) and s not in staffed_by_specialist]


def h6(m, a):
    nights = defaultdict(int)
    night_set = set(m.p.night_shifts)
    for d, s in a.works:
        if s in night_set:
            nights[(d, m.p.shift_date[s])] += 1
    dates = m.p.date_list
    out = []
    for d in m.ctx.D_all:
        for i in range(len(dates) - 2):
            n = sum(nights[(d, t)] for t in dates[i:i + 3])
            if n > 2:
                out.append(f"{d}: {n} nights from {dates[i]}")
    return out


def h7(m, a):
    dates = m.p.date_list
    night_set = set(m.p.night_shifts)
    nights, any_shift = defaultdict(int), defaultdict(int)
    for d, s in a.works:
        nights[(d, m.p.shift_date[s])] += s in night_set
        any_shift[(d, m.p.shift_date[s])] += 1
    out = []
    for d in m.ctx.D_all:
        for t, t_next in zip(dates, dates[1:]):
            if not m.p.night_shifts_by_date[t] or not m.p.days_to_shifts[t_next]:
                continue                       # no constraint is generated for this pair of days
            if nights[(d, t)] + any_shift[(d, t_next)] > 1:
                out.append(f"{d}: {nights[(d, t)]} night shift(s) on {t} + {any_shift[(d, t_next)]} shift(s) on {t_next} > 1")
    return out


def h8(m, a):
    worked_days = {(d, m.p.shift_date[s]) for d, s in a.works}
    dates = m.p.date_list
    out = []
    for d in m.ctx.D_all:
        for start in range(0, len(dates), 7):
            days = sum((d, t) in worked_days for t in dates[start:start + 7])
            if days > 6:
                out.append(f"{d}: {days} working days from {dates[start]}")
    return out


def h9(m, a):
    tf = set(m.p.twentyfour_shifts)
    return [f"{d} not allowed 24 h shift {s}" for d, s in a.works if s in tf and m.p.twentyfour_allowed[d] == 0]


def h9a(m, a):
    tf = set(m.p.twentyfour_shifts)
    return [f"{d} works 24 h shift {s} in pandemic mode" for d, s in a.works if s in tf]


def h10a(m, a):
    off = set(m.p.unavailable_days)
    return [f"{d} works {s} on an unavailable day" for d, s in a.works if (d, m.p.shift_date[s]) in off]


def h10b(m, a):
    off = set(m.p.unavailable_shifts)
    return [f"{d} works unavailable shift {s}" for d, s in a.works if (d, s) in off]


def h11(m, a):
    candidates = set(m.ctx.C)
    return [f"candidate {d} works {s} but is not hired" for d, s in a.works if d in candidates and d not in a.hired]


def h12(m, a):
    nights = set(m.p.night_shifts)
    return [f"{d} not allowed night shift {s}" for d, s in a.works if s in nights and m.p.night_allowed[d] == 0]


def h13(m, a):
    return [f"candidate {c} hired in the baseline" for c in a.hired]


def c2(m, a):
    staff = _staff(m, a, m.ctx.D)
    return [f"shift {s}: {n} employed doctors > {m.p.regular_staff[s]}"
            for s, n in staff.items() if n > m.p.regular_staff[s]]


# ---------------------------------------------------------------------------------------------
# Minimum values of slack-like variables
# ---------------------------------------------------------------------------------------------

def _crossbubble(m, a) -> int:
    bubbles = m.bubble_assignment
    contact = set(m.pandemic_config.contact_depts)
    on_shift = defaultdict(list)
    for d, s in a.works:
        if m.p.dept[s] in contact and d in bubbles and bubbles[d] != CLINIC_POOL:
            on_shift[s].append(bubbles[d])
    count = 0
    for members in on_shift.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                count += members[i] != members[j]
    return count


def hp1(m, a):
    return {"crossbubble_total": _crossbubble(m, a)}


def c1(m, a):
    staff = _staff(m, a, m.ctx.D_all)
    return {f"u[{s}]": max(0, m.p.min_staff[s] - staff[s]) for s in m.ctx.S}


def f4(m, a):
    hours = _d1(m, a)
    if hours is None:
        return None
    lows, highs = [], []
    for d in m.p.opt_out_doctors:
        M, H = m.p.adjusted_max_hours[d], hours[d]
        if M == 0:
            if not -50 <= 1000 * H <= 50:
                return None
            lo, hi = 0, 2000
        else:
            lo = max(0, -((-(1000 * H - 50)) // M))      # ceil
            hi = min(2000, (1000 * H + 50) // M)
        if lo > hi:
            return None
        lows.append(lo)
        highs.append(hi)
    return {"ratio_spread": max(0, max(lows) - min(highs)) if lows else 0}


def u1(m, a):
    hours = _d1(m, a)
    if hours is None:
        return None
    out = {}
    for d in m.p.regular_doctors:
        h = a.y(d) if d in set(m.ctx.C) else 1
        out[f"UW[{d}]"] = max(0, int(0.95 * m.p.adjusted_max_hours[d]) * h - hours[d])
    return out


def u2(m, a):
    wh = _d3(m, a)
    if wh is None:
        return None
    out = {}
    for d in m.ctx.D_all:
        for w in m.p.weeks:
            need = max(0, _weekly_target(m, d, w) - wh[(d, w)])
            if need > m.p.weekly_limits[d][w]:
                return None
            out[f"WU[{d},{w}]"] = need
    return out


# ---------------------------------------------------------------------------------------------
# Exact values of defined variables
# ---------------------------------------------------------------------------------------------

def d1(m, a):
    hours = _d1(m, a)
    return None if hours is None else {f"H[{d}]": h for d, h in hours.items()}


def d2a(m, a):
    hours = _d1(m, a)
    if hours is None:
        return None
    cost = {d: hours[d] * m.p.hourly_cost_map[d] for d in m.ctx.D}
    if any(v > 10_000_000 for v in cost.values()):
        return None
    return {f"cost[{d}]": v for d, v in cost.items()}


def d2b(m, a):
    hours = _d1(m, a)
    if hours is None:
        return None
    out = {}
    for c in m.ctx.C:
        effective = hours[c] * a.y(c)
        if effective > 2000 or effective * m.p.hourly_cost_map[c] > 10_000_000:
            return None
        out[f"cost[{c}]"] = effective * m.p.hourly_cost_map[c]
    return out


def d3(m, a):
    wh = _d3(m, a)
    return None if wh is None else {f"WH[{d},{w}]": h for (d, w), h in wh.items()}


def s1(m, a):
    value = 0
    for d, s, kind in m.p.preferences:
        if kind == "like":
            value -= a.x(d, s)
        elif kind == "dislike":
            value += a.x(d, s)
    return {"P": value}


def _counts(m, a, doctors, shifts) -> list:
    shift_set = set(shifts)
    count = {d: 0 for d in doctors}
    for d, s in a.works:
        if d in count and s in shift_set:
            count[d] += 1
    return list(count.values())


def f1(m, a):
    doctors = [d for d, allowed in m.p.night_allowed.items() if allowed == 1 and _specialist(m, d)]
    return {"spread_N": _spread(_counts(m, a, doctors, m.p.night_shifts))}


def f2(m, a):
    doctors = [d for d in m.ctx.D if _specialist(m, d)]
    return {"spread_WE": _spread(_counts(m, a, doctors, m.p.weekend_shifts))}


def f3(m, a):
    doctors = [d for d in m.ctx.D if _specialist(m, d)]
    return {"spread_H": _spread(_counts(m, a, doctors, m.p.hard_shifts))}


# ---------------------------------------------------------------------------------------------
# Phase objectives
# ---------------------------------------------------------------------------------------------

def o1(m, a):
    staff = _staff(m, a, m.ctx.D_all)
    total = 0
    for s in m.ctx.S:
        weight = 1 if m.pandemic_config is None else PANDEMIC_WEIGHTS.get(m.p.dept[s], 10)
        total += weight * max(0, m.p.min_staff[s] - staff[s])
    return total


def o1b(m, a):
    return _crossbubble(m, a)


def o2(m, a):
    return len(a.hired)


def o3(m, a):
    t = m.targets
    parts = [d2a(m, a), d2b(m, a), d3(m, a), u1(m, a), u2(m, a)]
    if any(part is None for part in parts):
        return None
    cost_e, cost_c, _, uw, wu = parts
    total_cost = sum(cost_e.values()) + sum(cost_c.values())

    def coeff(max_value):
        return t.scale // max_value if max_value > 0 else 0

    max_over = int(t.cost * t.max_over_budget_pct)
    over = max(0, total_cost - t.cost)
    if over > max_over:
        return None
    terms = [0]
    if max_over > 0:
        terms.append(over * (t.scale // max_over))
    p = m.p
    terms.append((f1(m, a)["spread_N"] - t.spread) * coeff(len(p.night_shifts)))
    terms.append((f2(m, a)["spread_WE"] - t.weekend_spread) * coeff(len(p.weekend_shifts)))
    terms.append((f3(m, a)["spread_H"] - t.hard_spread) * coeff(len(p.hard_shifts)))
    terms.append((sum(uw.values()) - t.underwork)
                 * coeff(sum(int(0.95 * p.adjusted_max_hours[d]) for d in p.regular_doctors)))
    terms.append((sum(wu.values()) - t.weekly_underwork)
                 * coeff(sum(int(0.95 * p.weekly_limits[d][w]) for d in m.ctx.D_all for w in p.weeks)))
    terms.append((s1(m, a)["P"] - t.pref) * coeff(len(p.preferences)))
    z = max(terms)
    return z if z <= 10 * t.scale else None


RESTRICTIONS = {
    "H1": h1, "H2": h2, "H3": h3, "H4": h4, "H5": h5, "H6": h6, "H7": h7, "H8": h8, "H9": h9,
    "H9a": h9a, "H10a": h10a, "H10b": h10b, "H11": h11, "H12": h12, "H13": h13, "C2": c2,
}


def check_schedule(m: RosteringModel, a: Assignment, *, pandemic: bool = False, allow_hire: bool = True) -> dict:
    """Violations of every hard constraint by a complete schedule (empty lists: valid schedule)."""
    out = {}
    for eq_id, oracle in RESTRICTIONS.items():
        if eq_id == "H9a" and not pandemic:
            continue
        if eq_id == "H13" and allow_hire:
            continue
        out[eq_id] = oracle(m, a)
    return out
