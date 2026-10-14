"""Sets, parameters and the container of the CP-SAT rostering model.

The equations themselves are in rostering/equations.py (one function per equation) and are
documented in docs/MODEL.md and docs/EQUATIONS.md. tests/test_model_equivalence.py checks that
the assembled model is identical (constraint by constraint) to the original implementation.
"""
from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field

import pandas as pd
from ortools.sat.python import cp_model

from .config import ObjectiveTargets
from .data import ProblemData
from .pandemic import PandemicConfig

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
    dept: dict                 # shift -> department
    min_staff: dict
    regular_staff: dict
    unavailable_days: list     # (doctor, date)
    unavailable_shifts: list   # (doctor, shift)
    preferences: list          # (doctor, shift, "like" | "dislike")


def _with_candidate_defaults(values: dict, candidates: pd.DataFrame, column: str, default) -> dict:
    for _, cand in candidates.iterrows():
        cid = cand["id"]
        if cid not in values:
            values[cid] = cand[column] if column in cand else default
    return values


def compute_params(ctx: Context, data: ProblemData) -> Params:
    doctors, candidates, shifts, unavail_day = data.doctors, data.candidates, data.shifts, data.unavail_day
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
        dept=dict(zip(shifts["id"], shifts["dept"])),
        min_staff=dict(zip(shifts["id"], shifts["min_staff"])),
        regular_staff=dict(zip(shifts["id"], shifts["regular_staff"])),
        unavailable_days=list(zip(unavail_day["doctor_id"], unavail_day["date"])),
        unavailable_shifts=[(d, code_to_id[code]) for d, code in
                            zip(data.unavail_shift["doctor_id"], data.unavail_shift["code"])],
        preferences=[(d, code_to_id[code], kind) for d, code, kind in
                     zip(data.pref["doctor_id"], data.pref["code"], data.pref["preference"])],
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
    allow_hire: bool = False
    pandemic_config: PandemicConfig | None = None
    bubble_assignment: dict | None = None     # doctor -> bubble id (pandemic mode)
    targets: ObjectiveTargets = field(default_factory=ObjectiveTargets)
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


def weekly_target(m: RosteringModel, d, w) -> int:
    """Weekly hours regarded as full employment."""
    if d in m.p.weekly_regular_lookup:
        return int(UNDERWORK_TARGET * m.p.weekly_limits[d][w])
    return int(UNDERWORK_TARGET * (NON_REGULAR_WEEKLY_BASE_HOURS * m.p.weekly_available_days[d][w] / 7))


def build_model(
    ctx: Context,
    data: ProblemData,
    *,
    allow_hire: bool,
    pandemic_config: PandemicConfig | None = None,
    bubble_assignment: dict | None = None,
    implementations: dict | None = None,
) -> RosteringModel:
    """Decision variables plus every applicable equation of rostering.equations.CONSTRAINTS.

    ``implementations`` (equation id -> function) replaces chosen equations, e.g. by an
    implementation generated from a prompt.
    """
    from .equations import add_constraints

    model = cp_model.CpModel()
    p = compute_params(ctx, data)
    x = {(d, s): model.NewBoolVar(f"x_{d}_{s}") for d in ctx.D_all for s in ctx.S}
    hire = {c: model.NewBoolVar(f"hire_{c}") for c in ctx.C}
    m = RosteringModel(model=model, ctx=ctx, p=p, data=data, x=x, hire=hire, allow_hire=allow_hire,
                       pandemic_config=pandemic_config, bubble_assignment=bubble_assignment)
    add_constraints(m, implementations)
    return m
