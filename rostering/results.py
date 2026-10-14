"""Turning a solved model into schedules, per-doctor statistics and satisfaction scores."""
from __future__ import annotations

import math

import pandas as pd
from ortools.sat.python import cp_model

from .model import RosteringModel

SOLVED = (cp_model.OPTIMAL, cp_model.FEASIBLE)

# Bonuses of the raw satisfaction score S_raw.
BONUS_FREE_WEEKEND = 8
BONUS_FREE_BLOCK = 6
BONUS_COMFORT_WEEK = 5
OVERWORK_PENALTY = 15

# Per-role standards of the satisfaction score S_pct (compute_satisfaction_v2).
STANDARDS_BY_ROLE = {
    "icu_specialist": {"night": 8, "twentyfour": 3, "weekend": 2, "hard": 3},
    "specialist": {"night": 6, "twentyfour": 2, "weekend": 2, "hard": 3},
    "resident": {"night": 3, "twentyfour": 1, "weekend": 2, "hard": 1},
    "outpatient": {"night": 0, "twentyfour": 0, "weekend": 1, "hard": 0},
}
DEFAULT_STANDARDS = {"night": 5, "twentyfour": 2, "weekend": 2, "hard": 3}


# ---------------------------------------------------------------------------------------------
# Satisfaction helpers
# ---------------------------------------------------------------------------------------------

def count_free_weekends(schedule: pd.DataFrame, doctor_name) -> int:
    """ISO weeks (among the weeks the doctor works) without any Saturday/Sunday shift."""
    df = schedule[schedule["Doctor"] == doctor_name].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Week"] = df["Date"].dt.isocalendar().week
    df["Weekday"] = df["Date"].dt.weekday

    worked_weekends = df[df["Weekday"].isin([5, 6])].groupby("Week").size()
    return len([w for w in df["Week"].unique() if w not in worked_weekends.index])


def count_free_day_blocks(schedule: pd.DataFrame, doctor_name) -> int:
    """Number of runs of at least 2 consecutive days off within the schedule period."""
    worked_days = set(schedule[schedule["Doctor"] == doctor_name]["Date"].dt.date)
    blocks = current = 0
    for day in pd.date_range(schedule["Date"].min(), schedule["Date"].max()).date:
        if day not in worked_days:
            current += 1
        else:
            if current >= 2:
                blocks += 1
            current = 0
    if current >= 2:
        blocks += 1
    return blocks


def count_comfort_weeks(weekly_hours: dict, weekly_limits: dict, d) -> int:
    """Weeks with 70-90 % of the weekly hour limit."""
    good = 0
    for w, hours in weekly_hours[d].items():
        limit = weekly_limits[d].get(w, 0)
        if limit == 0:
            continue
        if 0.7 * limit <= hours <= 0.9 * limit:
            good += 1
    return good


def compute_satisfaction_v2(row) -> float:
    """Satisfaction 0-100 relative to role standards scaled by the doctor's utilisation."""
    util = max(0.0, row["TotalHours"] / row["MonthlyLimitHours"])
    base = 82 - 12 * util
    std = STANDARDS_BY_ROLE.get(row.get("Role", "unknown"), DEFAULT_STANDARDS)
    scale = max(0.2, math.sqrt(util))

    def scaled_limit(std_value):
        return 0 if std_value == 0 else max(1, int(round(std_value * scale)))

    def excess(value, limit):
        return max(0, value - limit)

    penalty = (
        7 * (excess(row["TwentyFourCount"], scaled_limit(std["twentyfour"])) ** 2)
        + 5 * excess(row["NightCount"], scaled_limit(std["night"]))
        + 4 * excess(row["WeekendCount"], scaled_limit(std["weekend"]))
        + 3 * excess(row["HardCount"], scaled_limit(std["hard"]))
        + 8 * row["dislike_violated"]
    )
    bonus = (
        6 * math.sqrt(max(0, row["like_satisfied"]))
        + 5 * min(row["ComfortWeeks"], 4)
        + 6 * min(row["FreeWeekends"], 4)
        + 4 * min(row["FreeBlocks"], 6)
    )
    return round(max(0, min(100, base + bonus - penalty)), 1)


# ---------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------

def build_solver_stats(solver: cp_model.CpSolver, status: int, model: cp_model.CpModel) -> pd.DataFrame:
    proto = model.Proto()
    stats = {
        "objective_value": solver.ObjectiveValue() if status in SOLVED else None,
        "status": solver.StatusName(status),
        "conflicts": solver.NumConflicts(),
        "branches": solver.NumBranches(),
        "wall_time": solver.WallTime(),
        "num_variables": len(proto.variables),
        "num_constraints": len(proto.constraints),
        "best_bound": None,
        "gap_abs": None,
        "gap_pct": None,
    }
    obj = stats["objective_value"]
    if obj is not None:
        bound = solver.BestObjectiveBound()
        stats["best_bound"] = bound
        stats["gap_abs"] = abs(obj - bound)
        stats["gap_pct"] = round(100 * abs(obj - bound) / abs(obj), 2) if abs(obj) > 1e-6 else 0.0
    return pd.DataFrame([stats])


def _schedule(m: RosteringModel, shifts: pd.DataFrame, value) -> pd.DataFrame:
    ctx = m.ctx
    rows = []
    shifts_sorted = shifts.sort_values(by=["date", "start_hour"])
    for date in m.p.date_list:
        for _, sh in shifts_sorted[shifts_sorted["date"] == date].iterrows():
            s = sh["id"]
            base = {"Date": date, "ShiftCode": sh["code"], "Dept": sh["dept"], "StartHour": sh["start_hour"],
                    "EndHour": sh["end_hour"], "Hours": sh["hours"]}
            assigned = [d for d in ctx.D_all
                        if value(m.x[(d, s)]) == 1 and (d in ctx.D or value(m.hire.get(d, 0)) == 1)]
            for d in assigned:
                rows.append({**base, "Doctor": ctx.id_to_name[d], "Role": ctx.id_to_role[d], "Missing": 0})
            for _ in range(value(m.slacks[s])):
                rows.append({**base, "Doctor": None, "Role": None, "Missing": 1})
    schedule = pd.DataFrame(rows)
    schedule["Date"] = pd.to_datetime(schedule["Date"])
    return schedule


def _doctor_stats(m: RosteringModel, shifts: pd.DataFrame, value) -> pd.DataFrame:
    ctx, p, x = m.ctx, m.p, m.x
    dept = dict(zip(shifts["id"], shifts["dept"]))
    rows = []
    for d in ctx.D_all:
        rows.append({
            "Doctor": ctx.id_to_name[d],
            "Role": ctx.id_to_role[d],
            "TotalHours": sum(value(x[(d, s)]) * p.hours[s] for s in ctx.S),
            "MaxHours": p.adjusted_max_hours[d],
            "NightShifts": sum(value(x[(d, s)]) for s in p.night_shifts),
            "TwentyFourCount": sum(value(x[(d, s)]) for s in p.twentyfour_shifts),
            "WardCount": sum(value(x[(d, s)]) for s in ctx.S if dept[s] == "WARD"),
            "ICUCount": sum(value(x[(d, s)]) for s in ctx.S if dept[s] == "ICU"),
            "ClinicCount": sum(value(x[(d, s)]) for s in ctx.S if dept[s] == "CLINIC"),
        })
    return pd.DataFrame(rows)


def extract_results(m: RosteringModel, solver: cp_model.CpSolver, status: int) -> dict:
    ctx, p, x = m.ctx, m.p, m.x
    value = solver.Value
    C, D_all = ctx.C, ctx.D_all
    name_to_id = {v: k for k, v in ctx.id_to_name.items()}

    raw_hire = {c: value(m.hire[c]) for c in C}
    raw_x = {(d, s): value(x[(d, s)]) for d in D_all for s in ctx.S}
    hired_candidates = [c for c in C if raw_hire[c] == 1]
    hired_df = pd.DataFrame([
        {"Doctor": ctx.id_to_name[c], "Role": ctx.id_to_role[c], "Hours": value(m.worked_hours[c])}
        for c in hired_candidates
    ])

    shifts, pref = m.data.shifts, m.data.pref
    schedule_df = _schedule(m, shifts, value)
    stats_df = _doctor_stats(m, shifts, value)

    stats_df["MonthlyLimitHours"] = stats_df["Doctor"].apply(lambda nm: p.monthly_limit_hours[name_to_id[nm]])
    stats_df["IsCandidate"] = stats_df["Doctor"].isin([ctx.id_to_name[c] for c in C])
    stats_df["Hired"] = stats_df["Doctor"].isin([ctx.id_to_name[c] for c in hired_candidates])
    stats_df["UtilizationPct"] = (stats_df["TotalHours"] / stats_df["MaxHours"].replace(0, 1) * 100).round(1)

    # Preferences.
    stats_df["like_satisfied"] = 0
    stats_df["dislike_violated"] = 0
    for d, code, preference in zip(pref["doctor_id"], pref["code"], pref["preference"]):
        if value(x[(d, p.code_to_id[code])]) == 1:
            col = "like_satisfied" if preference == "like" else "dislike_violated"
            stats_df.loc[stats_df["Doctor"] == ctx.id_to_name[d], col] += 1

    # Demanding shifts.
    def count_assigned(d, shift_ids):
        return sum(value(x[(d, s)]) for s in shift_ids)

    for column, shift_ids in (("NightCount", p.night_shifts), ("WeekendCount", p.weekend_shifts),
                              ("HardCount", p.hard_shifts), ("TwentyFourCount", p.twentyfour_shifts)):
        shift_set = set(shift_ids)
        stats_df[column] = stats_df["Doctor"].apply(lambda nm, ss=shift_set: count_assigned(name_to_id[nm], ss))

    weekly_hours_val = {d: {w: value(m.weekly_hours[d][w]) for w in p.weeks} for d in D_all}
    opt_out = set(p.opt_out_doctors)

    def overwork_penalty(d) -> int:
        """Opt-out doctors working >= 90 % of the limit in every active week."""
        if d not in opt_out:
            return 0
        active_weeks = [w for w in p.weeks if p.weekly_limits[d][w] > 0]
        if not active_weeks:
            return 0
        all_over_90 = all(weekly_hours_val[d][w] >= 0.9 * p.weekly_limits[d][w] for w in active_weeks)
        return OVERWORK_PENALTY if all_over_90 else 0

    stats_df["OverworkPenalty"] = stats_df["Doctor"].apply(lambda nm: overwork_penalty(name_to_id[nm]))

    stats_df["S_likes"] = 10 * stats_df["like_satisfied"]
    stats_df["S_dislikes"] = -8 * stats_df["dislike_violated"]
    stats_df["S_night"] = -5 * stats_df["NightCount"]
    stats_df["S_24h"] = -7 * stats_df["TwentyFourCount"]
    stats_df["S_hard"] = -4 * stats_df["HardCount"]
    stats_df["S_weekend"] = -3 * stats_df["WeekendCount"]
    stats_df["S_overwork"] = -1 * stats_df["OverworkPenalty"]

    sched_real = schedule_df.dropna(subset=["Doctor"]).copy()
    free_weekends = stats_df["Doctor"].apply(lambda nm: count_free_weekends(sched_real, nm))
    free_blocks = stats_df["Doctor"].apply(lambda nm: count_free_day_blocks(sched_real, nm))
    comfort_weeks = stats_df["Doctor"].apply(
        lambda nm: count_comfort_weeks(weekly_hours_val, p.weekly_limits, name_to_id[nm]))

    stats_df["S_free_weekends"] = BONUS_FREE_WEEKEND * free_weekends
    stats_df["S_free_blocks"] = BONUS_FREE_BLOCK * free_blocks
    stats_df["S_comfort"] = BONUS_COMFORT_WEEK * comfort_weeks
    stats_df["S_raw"] = stats_df[
        ["S_likes", "S_dislikes", "S_night", "S_24h", "S_hard", "S_weekend", "S_overwork",
         "S_free_weekends", "S_free_blocks", "S_comfort"]
    ].sum(axis=1)

    stats_df["ComfortWeeks"] = comfort_weeks
    stats_df["FreeWeekends"] = free_weekends
    stats_df["FreeBlocks"] = free_blocks
    stats_df["S_pct"] = stats_df.apply(compute_satisfaction_v2, axis=1)

    stats_eval_df = stats_df[(~stats_df["IsCandidate"]) | (stats_df["Hired"])]

    return {
        "schedule_df": schedule_df,
        "stats_df": stats_df,
        "hired_df": hired_df,
        "solver_stats_df": build_solver_stats(solver, status, m.model),

        # metrics
        "objective": solver.ObjectiveValue() if status in SOLVED else None,
        "slack": sum(value(sl) for sl in m.slacks.values()),
        "total_cost": sum(value(m.doctor_cost[d]) for d in m.doctor_cost),
        "spread": value(m.spread),
        "weekend_spread": value(m.weekend_spread),
        "hard_spread": value(m.hard_spread),

        # raw debug
        "raw_hire": raw_hire,
        "raw_x": raw_x,
        "avg_satisfaction": round(stats_eval_df["S_pct"].mean(), 2),
        "min_satisfaction": round(stats_eval_df["S_pct"].min(), 2),
        "max_satisfaction": round(stats_eval_df["S_pct"].max(), 2),

        # pandemic mode
        "pandemic_bundle": m.pandemic_bundle,
    }
