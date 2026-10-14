"""CP-SAT model of the monthly doctor rostering problem.

The equations are documented in docs/MODEL.md. The order in which variables and constraints are
created is part of the contract: tests/test_model_equivalence.py checks that the model is identical
(constraint by constraint) to the original implementation.
"""
from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field

import pandas as pd
from ortools.sat.python import cp_model

from .data import ProblemData
from .pandemic import PandemicConfig, add_pandemic_constraints

logger = logging.getLogger(__name__)

SPECIALIST_ROLES = ["specialist", "icu_specialist"]
DEFAULT_CANDIDATE_MAX_HOURS = 48
MIN_REST_HOURS = 11
MAX_CONSECUTIVE_NIGHTS = 2          # in any window of 3 consecutive days
MAX_WORKING_DAYS_PER_WEEK = 6       # per 7-day block counted from the first day of the horizon
UNDERWORK_TARGET = 0.95             # share of the hour limit regarded as full employment
NON_REGULAR_WEEKLY_BASE_HOURS = 48  # weekly target base for opt-out doctors
PANDEMIC_SLACK_WEIGHTS = {"ICU": 100, "WARD": 50, "CLINIC": 1}
PANDEMIC_SLACK_WEIGHT_OTHER = 10


# ---------------------------------------------------------------------------------------------
# Sets and parameters
# ---------------------------------------------------------------------------------------------

@dataclass
class Context:
    """Index sets and id mappings."""
    D: list            # employed doctors
    C: list            # candidates that may be hired
    D_all: list        # D + C
    S: list            # shifts
    doctor_idx: dict
    shift_idx: dict
    id_to_name: dict
    id_to_role: dict

    def __getitem__(self, key):         # dict-style access kept for callers of the old API
        return getattr(self, key)


def prepare_context(doctors, candidates, shifts, unavail_day=None, unavail_shift=None, pref=None) -> Context:
    D = list(doctors["id"])
    C = list(candidates["id"])
    people = pd.concat([doctors, candidates])
    return Context(
        D=D,
        C=C,
        D_all=D + C,
        S=list(shifts["id"]),
        doctor_idx={doc_id: idx for idx, doc_id in enumerate(D)},
        shift_idx={shift_id: idx for idx, shift_id in enumerate(shifts["id"])},
        id_to_name={row["id"]: row["name"] for _, row in people.iterrows()},
        id_to_role={row["id"]: row["role"] for _, row in people.iterrows()},
    )


@dataclass
class Params:
    """Everything derived from the input data that the equations use."""
    date_list: list
    shift_date: dict
    weekend: set
    abs_start: dict
    abs_end: dict
    hours: dict
    max_hours: dict
    specialists: list
    needs_mentor: list
    opt_out_doctors: list
    regular_doctors: list
    weekly_regular_lookup: set
    night_allowed: dict
    twentyfour_allowed: dict
    twentyfour_shifts: list
    night_shifts: list
    night_shifts_by_date: dict
    days_to_shifts: dict
    code_to_id: dict
    full_day_unavail: dict
    adjusted_max_hours: dict
    hourly_cost_map: dict
    weeks: list
    shift_week: dict
    weekly_available_days: dict
    weekly_limits: dict
    monthly_limit_hours: dict
    skill_map: dict
    req_skill: dict
    weekend_shifts: list
    hard_shifts: list


def _with_candidate_defaults(values: dict, candidates: pd.DataFrame, column: str, default) -> dict:
    for _, cand in candidates.iterrows():
        cid = cand["id"]
        if cid not in values:
            values[cid] = cand[column] if column in cand else default
    return values


def compute_params(ctx: Context, doctors, candidates, shifts, unavail_day) -> Params:
    D_all, C, S = ctx.D_all, ctx.C, ctx.S
    shift_rows = list(shifts.iterrows())

    shift_date = {sh["id"]: sh["date"] for _, sh in shift_rows}
    date_list = sorted(shifts["date"].unique())
    date_index = {d: i for i, d in enumerate(date_list)}
    weekend = {d for d in date_list if datetime.datetime.strptime(d, "%Y-%m-%d").weekday() >= 5}

    # Shift times measured in hours from the beginning of the planning horizon.
    shift_start = {sh["id"]: sh["start_hour"] for _, sh in shift_rows}
    shift_end = {sh["id"]: sh["end_hour"] for _, sh in shift_rows}
    abs_start = {s: date_index[shift_date[s]] * 24 + shift_start[s] for s in S}
    abs_end = {s: date_index[shift_date[s]] * 24 + shift_end[s] for s in S}
    hours = {sh["id"]: sh["hours"] for _, sh in shift_rows}

    max_hours = {doc["id"]: doc["max_hours"] for _, doc in doctors.iterrows()}
    _with_candidate_defaults(max_hours, candidates, "max_hours", DEFAULT_CANDIDATE_MAX_HOURS)

    specialists = list(doctors[doctors["role"].isin(SPECIALIST_ROLES)]["id"])
    needs_mentor = list(doctors[doctors["needs_mentor"] == 1]["id"]) + list(candidates[candidates["needs_mentor"] == 1]["id"])
    opt_out_doctors = list(doctors[doctors["opt_out"] == 1]["id"])
    regular_series = doctors[doctors["opt_out"] == 0]["id"]
    regular_doctors = list(regular_series)
    # Candidates without a valid opt_out flag count as regular doctors.
    extra_regular = [
        int(cand["id"]) for _, cand in candidates.iterrows()
        if "opt_out" not in cand or cand["opt_out"] not in (0, 1)
    ]
    regular_doctors += extra_regular
    # ORIGINAL BEHAVIOUR, kept on purpose (see docs/MODEL.md, "Known issues"): the weekly underwork
    # target tested `d in regular_doctors` on a pandas Series, which checks the row *index*, not the
    # doctor id. With extra regular candidates the original used a pd.Index, which checks the values.
    weekly_regular_lookup = set(regular_doctors) if extra_regular else set(regular_series.index)

    night_allowed = _with_candidate_defaults(
        {doc["id"]: doc["night_allowed"] for _, doc in doctors.iterrows()}, candidates, "night_allowed", 1)
    twentyfour_allowed = _with_candidate_defaults(
        {doc["id"]: doc["twentyfour_allowed"] for _, doc in doctors.iterrows()}, candidates, "twentyfour_allowed", 1)

    shift_hours = dict(zip(shifts["id"], shifts["hours"]))
    shift_code = dict(zip(shifts["id"], shifts["code"]))
    twentyfour_shifts = [s for s in S if shift_hours[s] == 24]
    night_shifts = [s for s in S if "_N_" in shift_code[s] or shift_hours[s] == 24]
    night_shifts_by_date = {day: [s for s in night_shifts if shift_date[s] == day] for day in date_list}
    days_to_shifts = {day: [s for s in S if shift_date[s] == day] for day in date_list}
    code_to_id = {row["code"]: row["id"] for _, row in shift_rows}

    full_day_unavail = {d: set(unavail_day[unavail_day["doctor_id"] == d]["date"].tolist()) for d in D_all}
    for c in C:
        full_day_unavail[c] = set()
    available_days = {d: len([day for day in date_list if day not in full_day_unavail[d]]) for d in D_all}

    # Monthly hour limit scaled by the number of days the doctor is available.
    adjusted_max_hours = {
        d: 0 if available_days[d] == 0 else int(max_hours[d] * available_days[d] / 7) for d in D_all
    }

    people = pd.concat([doctors, candidates])
    hourly_cost_map = {row["id"]: int(row["hourly_cost"]) for _, row in people.iterrows()}

    # Weeks are counted from the Monday of the week containing the first day of the horizon.
    date_objects = [pd.to_datetime(d) for d in date_list]
    start_week_monday = date_objects[0] - pd.Timedelta(days=date_objects[0].weekday())
    date_to_week = {d.strftime("%Y-%m-%d"): (d - start_week_monday).days // 7 for d in date_objects}
    shift_week_series = shifts["date"].map(date_to_week)
    shift_week = dict(zip(shifts["id"], shift_week_series))
    weeks = sorted(shift_week_series.unique())
    week_to_dates = {w: sorted(shifts.loc[shift_week_series == w, "date"].unique()) for w in weeks}

    weekly_available_days = {
        d: {w: sum(1 for day in week_to_dates[w] if day not in full_day_unavail[d]) for w in weeks}
        for d in D_all
    }
    weekly_limits = {
        d: {w: 0 if weekly_available_days[d][w] == 0 else int(max_hours[d] * weekly_available_days[d][w] / 7)
            for w in weeks}
        for d in D_all
    }
    monthly_limit_hours = {d: sum(weekly_limits[d].values()) for d in D_all}

    skill_map = {row["id"]: row["skill_list"] for _, row in people.iterrows()}
    req_skill = {row["id"]: row["required_skill"] for _, row in shift_rows}

    return Params(
        date_list=date_list,
        shift_date=shift_date,
        weekend=weekend,
        abs_start=abs_start,
        abs_end=abs_end,
        hours=hours,
        max_hours=max_hours,
        specialists=specialists,
        needs_mentor=needs_mentor,
        opt_out_doctors=opt_out_doctors,
        regular_doctors=regular_doctors,
        weekly_regular_lookup=weekly_regular_lookup,
        night_allowed=night_allowed,
        twentyfour_allowed=twentyfour_allowed,
        twentyfour_shifts=twentyfour_shifts,
        night_shifts=night_shifts,
        night_shifts_by_date=night_shifts_by_date,
        days_to_shifts=days_to_shifts,
        code_to_id=code_to_id,
        full_day_unavail=full_day_unavail,
        adjusted_max_hours=adjusted_max_hours,
        hourly_cost_map=hourly_cost_map,
        weeks=weeks,
        shift_week=shift_week,
        weekly_available_days=weekly_available_days,
        weekly_limits=weekly_limits,
        monthly_limit_hours=monthly_limit_hours,
        skill_map=skill_map,
        req_skill=req_skill,
        weekend_shifts=[s for s in S if shift_date[s] in weekend],
        hard_shifts=shifts[shifts["is_hard_shift"] == 1]["id"].to_list(),
    )


# ---------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------

@dataclass
class RosteringModel:
    model: cp_model.CpModel
    ctx: Context
    p: Params
    data: ProblemData
    x: dict                                   # x[d, s] = 1 if doctor d works shift s
    hire: dict                                # hire[c] = 1 if candidate c is hired
    slacks: dict = field(default_factory=dict)          # understaffing of shift s
    slacks_o: dict = field(default_factory=dict)        # overstaffing room of shift s
    slack_weights: dict = field(default_factory=dict)
    worked_hours: dict = field(default_factory=dict)
    doctor_cost: dict = field(default_factory=dict)
    weekly_hours: dict = field(default_factory=dict)
    pref_terms: list = field(default_factory=list)
    spread: cp_model.IntVar | None = None               # night shifts max - min (specialists)
    weekend_spread: cp_model.IntVar | None = None
    hard_spread: cp_model.IntVar | None = None
    ratio_spread: cp_model.IntVar | None = None
    underwork: dict = field(default_factory=dict)
    weekly_underwork: dict = field(default_factory=dict)
    pandemic_bundle: dict | None = None


def _add_hard_constraints(m: RosteringModel, pandemic_config) -> None:
    model, x, ctx, p = m.model, m.x, m.ctx, m.p
    D_all, S = ctx.D_all, ctx.S

    # H1. Only shifts matching the doctor's skills.
    for d in D_all:
        for s in S:
            if p.req_skill[s] not in p.skill_map[d]:
                model.Add(x[(d, s)] == 0)

    # H2. At most one shift per day.
    for d in D_all:
        for date in p.date_list:
            model.Add(sum(x[(d, s)] for s in p.days_to_shifts[date]) <= 1)

    # H3. At least MIN_REST_HOURS between the end of one shift and the start of a later one.
    rest_pairs = [
        (s1, s2) for s1 in S for s2 in S
        if s1 != s2 and p.abs_start[s2] > p.abs_end[s1] and p.abs_start[s2] - p.abs_end[s1] < MIN_REST_HOURS
    ]
    for d in D_all:
        for s1, s2 in rest_pairs:
            model.Add(x[(d, s1)] + x[(d, s2)] <= 1)

    # H4. Weekly hour limit (scaled by the days available in that week).
    shifts_by_week = {w: [s for s in S if p.shift_week[s] == w] for w in p.weeks}
    for d in D_all:
        for w in p.weeks:
            model.Add(sum(x[(d, s)] * p.hours[s] for s in shifts_by_week[w]) <= p.weekly_limits[d][w])

    # H5. A doctor who needs a mentor only works together with a specialist.
    for s in S:
        for d in p.needs_mentor:
            model.Add(x[(d, s)] <= sum(x[(spec, s)] for spec in p.specialists))

    # H6. At most MAX_CONSECUTIVE_NIGHTS night shifts in any 3 consecutive days.
    for d in D_all:
        for i in range(len(p.date_list) - 2):
            window = [s for day in p.date_list[i:i + 3] for s in p.night_shifts_by_date[day]]
            if window:
                model.Add(sum(x[(d, s)] for s in window) <= MAX_CONSECUTIVE_NIGHTS)

    # H7. No shift on the day after a night shift.
    for d in D_all:
        for i in range(len(p.date_list) - 1):
            nights = p.night_shifts_by_date[p.date_list[i]]
            next_day = p.days_to_shifts[p.date_list[i + 1]]
            if nights and next_day:
                model.Add(sum(x[(d, s)] for s in nights) + sum(x[(d, s)] for s in next_day) <= 1)

    # H8. At most MAX_WORKING_DAYS_PER_WEEK working days in every 7-day block of the horizon
    #     (described in the original code as "minimum 35-hour weekly rest").
    for d in D_all:
        for week_start in range(0, len(p.date_list), 7):
            works_vars = []
            for date in p.date_list[week_start:week_start + 7]:
                day_shifts = p.days_to_shifts[date]
                works = model.NewBoolVar(f"works_{d}_{date}")
                works_vars.append(works)
                model.Add(sum(x[(d, s)] for s in day_shifts) >= works)
                model.Add(sum(x[(d, s)] for s in day_shifts) <= len(day_shifts) * works)
            model.Add(sum(works_vars) <= MAX_WORKING_DAYS_PER_WEEK)

    # H9. 24-hour shifts only for eligible doctors.
    for d in D_all:
        if p.twentyfour_allowed[d] == 0:
            for s in p.twentyfour_shifts:
                model.Add(x[(d, s)] == 0)

    # H9a. Pandemic policy: no 24-hour shifts at all.
    if pandemic_config is not None:
        for d in D_all:
            for s in p.twentyfour_shifts:
                model.Add(x[(d, s)] == 0)


def _add_unavailability(m: RosteringModel, unavail_day, unavail_shift) -> None:
    model, x, p = m.model, m.x, m.p
    # H10. Leave and other absences: whole days and single shifts.
    for d, date in zip(unavail_day["doctor_id"], unavail_day["date"]):
        for s in p.days_to_shifts[date]:
            model.Add(x[d, s] == 0)
    for d, code in zip(unavail_shift["doctor_id"], unavail_shift["code"]):
        model.Add(x[d, p.code_to_id[code]] == 0)


def _add_eligibility(m: RosteringModel) -> None:
    model, x, ctx, p = m.model, m.x, m.ctx, m.p
    # H11. A candidate works only if hired.
    for c in ctx.C:
        for s in ctx.S:
            model.Add(x[(c, s)] <= m.hire[c])

    # H12. Night shifts only for doctors allowed to work at night.
    for d in ctx.D_all:
        if p.night_allowed[d] == 0:
            for s in p.night_shifts:
                model.Add(x[(d, s)] == 0)


def _add_coverage(m: RosteringModel, shifts: pd.DataFrame, pandemic: bool) -> None:
    """Staffing levels with slack: understaffing is minimised in phase 1 instead of being infeasible."""
    model, x, ctx = m.model, m.x, m.ctx
    for s, min_staff, regular_staff in zip(shifts["id"], shifts["min_staff"], shifts["regular_staff"]):
        slack = model.NewIntVar(0, min_staff, f"slack_{s}")
        m.slacks[s] = slack
        model.Add(sum(x[(d, s)] for d in ctx.D_all) + slack >= min_staff)

        slack_o = model.NewIntVar(0, regular_staff, f"slack_{s}")   # same name as in the original model
        m.slacks_o[s] = slack_o
        model.Add(sum(x[(d, s)] for d in ctx.D) + slack_o <= regular_staff)

    # In pandemic mode ICU/WARD coverage is far more important than outpatient clinics.
    for s, dept in zip(shifts["id"], shifts["dept"]):
        m.slack_weights[s] = PANDEMIC_SLACK_WEIGHTS.get(dept, PANDEMIC_SLACK_WEIGHT_OTHER) if pandemic else 1


def _add_hours_and_cost(m: RosteringModel, doctors, candidates) -> None:
    model, x, ctx, p = m.model, m.x, m.ctx, m.p

    for d in ctx.D_all:
        h_var = model.NewIntVar(0, p.adjusted_max_hours[d], f"worked_hours_{d}")
        model.Add(h_var == sum(x[(d, s)] * p.hours[s] for s in ctx.S))
        m.worked_hours[d] = h_var

    for d in ctx.D:
        cost = model.NewIntVar(0, 10_000_000, f"cost_{d}")
        model.Add(cost == m.worked_hours[d] * doctors.loc[ctx.doctor_idx[d], "hourly_cost"])
        m.doctor_cost[d] = cost

    for c in ctx.C:
        hourly = int(candidates.loc[candidates["id"] == c, "hourly_cost"].iloc[0])
        effective_hours = model.NewIntVar(0, 2000, f"effective_hours_{c}")    # worked_hours * hire
        model.AddMultiplicationEquality(effective_hours, [m.worked_hours[c], m.hire[c]])
        cost = model.NewIntVar(0, 10_000_000, f"cost_{c}")
        model.Add(cost == effective_hours * hourly)
        m.doctor_cost[c] = cost

    shifts_by_week = {w: [s for s in ctx.S if p.shift_week[s] == w] for w in p.weeks}
    for d in ctx.D_all:
        m.weekly_hours[d] = {}
        for w in p.weeks:
            wh = model.NewIntVar(0, p.weekly_limits[d][w], f"weekly_hours_{d}_{w}")
            model.Add(wh == sum(x[(d, s)] * p.hours[s] for s in shifts_by_week[w]))
            m.weekly_hours[d][w] = wh


def _add_preferences(m: RosteringModel, pref) -> None:
    """S1. Preferences: -1 for every satisfied 'like', +1 for every violated 'dislike'."""
    for d, code, preference in zip(pref["doctor_id"], pref["code"], pref["preference"]):
        s = m.p.code_to_id[code]
        if preference == "like":
            m.pref_terms.append(-1 * m.x[(d, s)])
        if preference == "dislike":
            m.pref_terms.append(1 * m.x[(d, s)])


def _spread(model, counts: list, name: str) -> cp_model.IntVar:
    hi = model.NewIntVar(0, 100, f"max_{name}")
    lo = model.NewIntVar(0, 100, f"min_{name}")
    spread = model.NewIntVar(0, 100, f"{name}_spread")
    model.AddMaxEquality(hi, counts)
    model.AddMinEquality(lo, counts)
    model.Add(spread == hi - lo)
    return spread


def _add_fairness(m: RosteringModel) -> None:
    """S2. Fair distribution of night, weekend and hard shifts among specialists (max - min)."""
    model, x, ctx, p = m.model, m.x, m.ctx, m.p
    is_specialist = lambda d: ctx.id_to_role[d] in SPECIALIST_ROLES  # noqa: E731

    night_count = {}
    for d, allowed in p.night_allowed.items():
        if is_specialist(d) and allowed == 1:
            count = model.NewIntVar(0, 100, f"night_count_{d}")
            model.Add(count == sum(x[(d, s)] for s in p.night_shifts))
            night_count[d] = count
    max_nights = model.NewIntVar(0, 100, "max_nights")
    min_nights = model.NewIntVar(0, 100, "min_nights")
    model.AddMaxEquality(max_nights, list(night_count.values()))
    model.AddMinEquality(min_nights, list(night_count.values()))
    m.spread = model.NewIntVar(0, 100, "night_spread")
    model.Add(m.spread == max_nights - min_nights)

    # Weekend and hard shifts: employed doctors only (candidates are not included).
    weekend_count, hard_count = {}, {}
    for d in ctx.D:
        if is_specialist(d):
            wc = model.NewIntVar(0, 100, f"weekend_count_{d}")
            model.Add(wc == sum(x[(d, s)] for s in p.weekend_shifts))
            weekend_count[d] = wc
    for d in ctx.D:
        if is_specialist(d):
            hc = model.NewIntVar(0, 100, f"hard_count_{d}")
            model.Add(hc == sum(x[(d, s)] for s in p.hard_shifts))
            hard_count[d] = hc
    m.weekend_spread = _spread(model, list(weekend_count.values()), "weekend")
    m.hard_spread = _spread(model, list(hard_count.values()), "hard")


def _add_workload_ratio(m: RosteringModel) -> None:
    """S3. Workload / hour limit (per mille) of opt-out doctors and its spread.

    Built as in the original model, but the spread is not used by any objective.
    """
    model, p = m.model, m.p
    workload_ratio = {}
    for d in p.opt_out_doctors:
        ratio = model.NewIntVar(0, 2000, f"workload_ratio_{d}")
        model.Add(ratio * p.adjusted_max_hours[d] <= m.worked_hours[d] * 1000 + 50)
        model.Add(ratio * p.adjusted_max_hours[d] >= m.worked_hours[d] * 1000 - 50)
        workload_ratio[d] = ratio

    max_ratio = model.NewIntVar(0, 2000, "max_ratio")
    min_ratio = model.NewIntVar(0, 2000, "min_ratio")
    m.ratio_spread = model.NewIntVar(0, 2000, "ratio_spread")
    if workload_ratio:
        model.AddMaxEquality(max_ratio, list(workload_ratio.values()))
        model.AddMinEquality(min_ratio, list(workload_ratio.values()))
        model.Add(m.ratio_spread == max_ratio - min_ratio)
    else:
        model.Add(max_ratio == 0)
        model.Add(min_ratio == 0)
        model.Add(m.ratio_spread == 0)


def weekly_target(m: RosteringModel, d, w) -> int:
    """Weekly hours regarded as full employment."""
    if d in m.p.weekly_regular_lookup:
        return int(UNDERWORK_TARGET * m.p.weekly_limits[d][w])
    return int(UNDERWORK_TARGET * (NON_REGULAR_WEEKLY_BASE_HOURS * m.p.weekly_available_days[d][w] / 7))


def _add_underwork(m: RosteringModel) -> None:
    """S4. Hours missing to UNDERWORK_TARGET of the limit: monthly (regular doctors) and weekly (all)."""
    model, p = m.model, m.p
    for d in p.regular_doctors:
        uw = model.NewIntVar(0, p.adjusted_max_hours[d], f"underwork_{d}")
        target = int(UNDERWORK_TARGET * p.adjusted_max_hours[d])
        h = m.hire[d] if d in m.hire else 1
        model.Add(uw >= target * h - m.worked_hours[d])
        model.Add(uw >= 0)
        m.underwork[d] = uw

    for d in m.ctx.D_all:
        for w in p.weeks:
            if weekly_target(m, d, w) > p.weekly_limits[d][w]:
                logger.debug("Weekly target above the limit: doctor=%s week=%s target=%s limit=%s",
                             d, w, weekly_target(m, d, w), p.weekly_limits[d][w])

    for d in m.ctx.D_all:
        m.weekly_underwork[d] = {}
        for w in p.weeks:
            uw = model.NewIntVar(0, p.weekly_limits[d][w], f"weekly_underwork_{d}_{w}")
            model.Add(uw >= weekly_target(m, d, w) - m.weekly_hours[d][w])
            model.Add(uw >= 0)
            m.weekly_underwork[d][w] = uw


def build_model(
    ctx: Context,
    data: ProblemData,
    *,
    allow_hire: bool,
    pandemic_config: PandemicConfig | None = None,
    bubble_assignment: dict | None = None,
) -> RosteringModel:
    model = cp_model.CpModel()
    p = compute_params(ctx, data.doctors, data.candidates, data.shifts, data.unavail_day)

    x = {(d, s): model.NewBoolVar(f"x_{d}_{s}") for d in ctx.D_all for s in ctx.S}
    hire = {c: model.NewBoolVar(f"hire_{c}") for c in ctx.C}
    m = RosteringModel(model=model, ctx=ctx, p=p, data=data, x=x, hire=hire)

    _add_hard_constraints(m, pandemic_config)
    _add_unavailability(m, data.unavail_day, data.unavail_shift)
    _add_eligibility(m)

    if pandemic_config is not None and bubble_assignment is not None:
        m.pandemic_bundle = add_pandemic_constraints(
            model=model, x=x, D_all=ctx.D_all, S=ctx.S, shifts=data.shifts, shift_idx=ctx.shift_idx,
            config=pandemic_config, bubble_assignment=bubble_assignment, id_to_role=ctx.id_to_role,
        )

    _add_coverage(m, data.shifts, pandemic=pandemic_config is not None)
    _add_hours_and_cost(m, data.doctors, data.candidates)
    _add_preferences(m, data.pref)
    _add_fairness(m)
    _add_workload_ratio(m)
    _add_underwork(m)

    if not allow_hire:
        for c in ctx.C:
            model.Add(hire[c] == 0)
    return m
