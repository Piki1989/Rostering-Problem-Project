from ortools.sat.python import cp_model
from pathlib import Path
import pandas as pd
import datetime
import multiprocessing
from datetime import datetime as dt
from compute_global_metrics import compute_hospital_efficiency_v2
from pandemic_mode import (
    PandemicConfig,
    prepare_pandemic_bubbles,
    validate_bubbles,
    add_pandemic_constraints,
    get_pandemic_results,
    assign_hired_candidates_to_bubbles
)

cores = multiprocessing.cpu_count()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# === HELPERS ===

def build_doctor_stats(
    D, S, id_to_name, id_to_role, x, hours,
    night_shifts, twentyfour_shifts, shifts, shift_idx, max_hours, solver
):
    """Builds a DataFrame containing workload and assignment statistics for each physician."""
    rows = []
    for d in D:
        rows.append({
            "Doctor": id_to_name[d],
            "Role": id_to_role[d],
            "TotalHours": sum(solver.Value(x[(d, s)]) * hours[s] for s in S),
            "MaxHours": max_hours[d],
            "NightShifts": sum(solver.Value(x[(d, s)]) for s in night_shifts),
            "TwentyFourCount": sum(solver.Value(x[(d, s)]) for s in twentyfour_shifts),
            "WardCount": sum(
                solver.Value(x[(d, s)])
                for s in S if shifts.loc[shift_idx[s], "dept"] == "WARD"
            ),
            "ICUCount": sum(
                solver.Value(x[(d, s)])
                for s in S if shifts.loc[shift_idx[s], "dept"] == "ICU"
            ),
            "ClinicCount": sum(
                solver.Value(x[(d, s)])
                for s in S if shifts.loc[shift_idx[s], "dept"] == "CLINIC"
            ),
        })
    return pd.DataFrame(rows)


def add_preference_stats(stats_df, pref, x, solver, id_to_name, code_to_id):
    """Adds preference satisfaction statistics to the physician statistics DataFrame."""
    stats_df["like_satisfied"] = 0
    stats_df["dislike_violated"] = 0

    for _, row in pref.iterrows():
        d = row["doctor_id"]
        s = code_to_id[row["code"]]
        pref_type = row["preference"]

        if solver.Value(x[(d, s)]) == 1:
            col = "like_satisfied" if pref_type == "like" else "dislike_violated"
            stats_df.loc[stats_df["Doctor"] == id_to_name[d], col] += 1

    return stats_df


def build_solver_stats(solver, status, model):
    """Returns a DataFrame with global solver statistics and optimization metrics."""
    solver_stats = {
        "objective_value": solver.ObjectiveValue()
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
        "status": solver.StatusName(status),
        "conflicts": solver.NumConflicts(),
        "branches": solver.NumBranches(),
        "wall_time": solver.WallTime(),
        "num_variables": len(model.Proto().variables),
        "num_constraints": len(model.Proto().constraints),
    }
    try:
        obj = solver_stats["objective_value"]
        bb = solver.BestObjectiveBound()
        solver_stats["best_bound"] = bb

        solver_stats["gap_abs"] = abs(obj - bb)

        if obj is not None and abs(obj) > 1e-6:
            solver_stats["gap_pct"] = round(100 * abs(obj - bb) / abs(obj), 2)
        else:
            solver_stats["gap_pct"] = 0.0
    except Exception as e:
        print(f"[build_solver_stats] Nie udało się obliczyć gap: {e}")
        solver_stats["best_bound"] = None
        solver_stats["gap_abs"] = None
        solver_stats["gap_pct"] = None

    return pd.DataFrame([solver_stats])

# Prepare sets, mappings, and auxiliary dictionaries
def prepare_context(doctors, candidates, shifts, unavail_day, unavail_shift, pref):
    D = list(doctors["id"])
    C = list(candidates["id"])
    D_all = D + C
    S = list(shifts["id"])

    doctor_idx = {doc_id: idx for idx, doc_id in enumerate(D)}
    shift_idx = {shift_id: idx for idx, shift_id in enumerate(S)}

    id_to_name = {
        row["id"]: row["name"]
        for _, row in pd.concat([doctors, candidates]).iterrows()
    }
    id_to_role = {
        row["id"]: row["role"]
        for _, row in pd.concat([doctors, candidates]).iterrows()
    }

    return {
        "D": D,
        "C": C,
        "D_all": D_all,
        "S": S,
        "doctor_idx": doctor_idx,
        "shift_idx": shift_idx,
        "id_to_name": id_to_name,
        "id_to_role": id_to_role,
    }


def build_model(
    context,
    doctors,
    candidates,
    shifts,
    unavail_day,
    unavail_shift,
    pref,
    allow_hire,

    pandemic_config=None,
    bubble_assignment=None,
):
    model = cp_model.CpModel()

    D = context["D"]
    C = context["C"]
    D_all = context["D_all"]
    S = context["S"]
    shift_idx = context["shift_idx"]
    doctor_idx = context["doctor_idx"]
    id_to_role = context["id_to_role"]

    """ VARIABLES """
    x = {}
    for d in D_all:
        for s in S:
            x[(d, s)] = model.NewBoolVar(f"x_{d}_{s}")

    hire = {}
    for c in C:
        hire[c] = model.NewBoolVar(f"hire_{c}")

    """ LISTS AND DICTS """
    shift_date = {shift["id"]: shift["date"] for _, shift in shifts.iterrows()}

    date_list = sorted(shifts["date"].unique())
    date_index = {d: i for i, d in enumerate(date_list)}

    weekend = {
        d for d in date_list
        if datetime.datetime.strptime(d, "%Y-%m-%d").weekday() >= 5
    }

    shift_start = {shift["id"]: shift["start_hour"] for _, shift in shifts.iterrows()}
    shift_end = {shift["id"]: shift["end_hour"] for _, shift in shifts.iterrows()}

    # Absolute shift timing measured from the beginning of the planning horizon
    abs_start = {s: date_index[shift_date[s]] * 24 + shift_start[s] for s in S}
    abs_end = {s: date_index[shift_date[s]] * 24 + shift_end[s] for s in S}

    hours = {shift["id"]: shift["hours"] for _, shift in shifts.iterrows()}
    max_hours = {doctor["id"]: doctor["max_hours"] for _, doctor in doctors.iterrows()}


    for _, cand in candidates.iterrows():
        cid = cand["id"]
        if cid not in max_hours:
            max_hours[cid] = cand["max_hours"] if "max_hours" in cand else 48

    specialists = doctors[doctors["role"].isin(["specialist", "icu_specialist"])]["id"]

    needs_mentor = (
            list(doctors[doctors["needs_mentor"] == 1]["id"]) +
            list(candidates[candidates["needs_mentor"] == 1]["id"])
    )
    opt_out_doctors = doctors[doctors["opt_out"] == 1]["id"]
    regular_doctors = doctors[doctors["opt_out"] == 0]["id"]

    # for _, cand in candidates.iterrows():
    #     if "opt_out" in cand and cand["opt_out"] in (0, 1):
    #         pass
    #     else:
    #         cand_id = cand["id"]
    #         regular_doctors = regular_doctors.append(pd.Index([cand_id]))
    extra_regular_ids = [
        int(cand["id"])
        for _, cand in candidates.iterrows()
        if "opt_out" not in cand or cand["opt_out"] not in (0, 1)
    ]
    if extra_regular_ids:
        regular_doctors = pd.Index(list(regular_doctors) + extra_regular_ids)


    night_allowed = {doctor["id"]: doctor["night_allowed"] for _, doctor in doctors.iterrows()}
    for _, cand in candidates.iterrows():
        cid = cand["id"]
        if cid not in night_allowed:
            night_allowed[cid] = cand["night_allowed"] if "night_allowed" in cand else 1

    twentyfour_allowed = {doctor["id"]: doctor["twentyfour_allowed"] for _, doctor in doctors.iterrows()}
    for _, cand in candidates.iterrows():
        cid = cand["id"]
        if cid not in twentyfour_allowed:
            twentyfour_allowed[cid] = cand["twentyfour_allowed"] if "twentyfour_allowed" in cand else 1

    twentyfour_shifts = [s for s in S if shifts.loc[shift_idx[s], "hours"] == 24]

    night_shifts = [s for s in S if
                    "_N_" in shifts.loc[shift_idx[s], "code"] or shifts.loc[shift_idx[s], "hours"] == 24]
    night_shifts_by_date = {day: [s for s in night_shifts if shift_date[s] == day] for day in date_list}

    day_shifts = {day: [s for s in S if shift_date[s] == day and hours[s] < 24] for day in date_list}
    day_24h = {day: [s for s in S if shift_date[s] == day and hours[s] == 24] for day in date_list}

    days_to_shifts = {day: [s for s in S if shift_date[s] == day] for day in date_list}
    code_to_id = {row["code"]: row["id"] for _, row in shifts.iterrows()}

    full_day_unavail = {
        d: set(unavail_day[unavail_day["doctor_id"] == d]["date"].tolist())
        for d in D_all
    }
    for c in C:
        full_day_unavail[c] = set()

    available_days = {
        d: len([day for day in date_list if day not in full_day_unavail[d]])
        for d in D_all
    }

    # Weekly working hour limits adjusted for physician availability
    adjusted_max_hours = {}
    for d in D_all:
        if available_days[d] == 0:
            adjusted_max_hours[d] = 0
        else:
            adjusted_max_hours[d] = int(max_hours[d] * available_days[d] / 7)

    hourly_cost_map = {
        row["id"]: int(row["hourly_cost"])
        for _, row in pd.concat([doctors, candidates]).iterrows()
    }


    date_objects = [pd.to_datetime(d) for d in date_list]

    start_date = date_objects[0]
    start_week_monday = start_date - pd.Timedelta(days=start_date.weekday())

    date_to_week = {}
    for d in date_objects:
        delta_days = (d - start_week_monday).days
        week_number = delta_days // 7
        date_to_week[d.strftime("%Y-%m-%d")] = week_number

    shifts["week"] = shifts["date"].map(date_to_week)
    weeks = sorted(shifts["week"].unique())

    week_to_dates = {
        w: sorted(shifts[shifts["week"] == w]["date"].unique())
        for w in weeks
    }

    weekly_available_days = {d: {} for d in D_all}

    for d in D_all:
        unavailable_days = full_day_unavail[d]
        for w in weeks:
            week_dates = week_to_dates[w]
            avail = sum(1 for dt in week_dates if dt not in unavailable_days)
            weekly_available_days[d][w] = avail

    # weekly hour limits
    weekly_limits = {d: {} for d in D_all}
    for d in D_all:
        for w in weeks:
            days = weekly_available_days[d][w]
            if days == 0:
                weekly_limits[d][w] = 0
            else:
                weekly_limits[d][w] = int(max_hours[d] * days / 7)

    # monthly limits
    monthly_limit_hours = {
        d: sum(weekly_limits[d].values())
        for d in D_all
    }


    # Mapping physician skills and shift requirements
    skill_map = {
        row["id"]: row["skill_list"]
        for _, row in pd.concat([doctors, candidates]).iterrows()
    }

    req_skill = {
        row["id"]: row["required_skill"]
        for _, row in shifts.iterrows()
    }


    """ FUNCTIONS """

    def rest_violation(sh1, sh2, min_rest=11) -> bool:
        if abs_start[sh2] <= abs_end[sh1]:
            return False
        return (abs_start[sh2] - abs_end[sh1]) < min_rest

    """ HARD CONSTRAINTS """
    """1. A physician may only be assigned to shifts compatible with their qualifications."""

    for d in D_all:
        for s in S:
            if req_skill[s] not in skill_map[d]:
                model.Add(x[(d, s)] == 0)

    """2. At most one shift per day."""
    for d in D_all:
        for date in date_list:
            shifts_per_date = [s for s in S if shift_date[s] == date]
            model.Add(sum(x[(d, s)] for s in shifts_per_date) <= 1)

    """3. Minimum 11-hour uninterrupted rest period between shifts."""
    for d in D_all:
        for s1 in S:
            for s2 in S:
                if s1 == s2:
                    continue
                if rest_violation(s1, s2):
                    model.Add(x[(d, s1)] + x[(d, s2)] <= 1)

    """4. Weekly working hour limits."""
    for d in D_all:
        for w in weeks:
            weekly_shifts = [
                s for s in S
                if shifts.loc[shift_idx[s], "week"] == w
            ]

            model.Add(
                sum(x[(d, s)] * hours[s] for s in weekly_shifts)
                <= weekly_limits[d][w]
            )

    """5. Residents and trainees requiring supervision must work with a qualified mentor."""
    for s in S:
        for d in needs_mentor:
            model.Add(
                x[(d, s)] <= sum(x[(spec, s)] for spec in specialists)
            )

    """6. At most two consecutive night shifts."""
    for d in D_all:
        for i in range(len(date_list) - 2):
            window_days = date_list[i:i + 3]
            shifts_in_window = [shift for day in window_days for shift in night_shifts_by_date[day]]
            if shifts_in_window:
                model.Add(sum(x[(d, s)] for s in shifts_in_window) <= 2)

    """7. Mandatory day off after a night shift."""
    for d in D_all:
        for i in range(len(date_list) - 1):
            current_date = date_list[i]
            next_date = date_list[i + 1]
            current_night_shifts = [
                s for s in night_shifts if shift_date[s] == current_date
            ]
            next_day_shifts = [
                s for s in S if shift_date[s] == next_date
            ]
            if current_night_shifts and next_day_shifts:
                model.Add(
                    sum(x[(d, s)] for s in current_night_shifts)
                    + sum(x[(d, s)] for s in next_day_shifts)
                    <= 1
                )

    """8. Minimum 35-hour uninterrupted weekly rest period."""
    shifts_per_day = {
        day: [s for s in S if shift_date[s] == day] for day in date_list
    }

    for d in D_all:
        for week_start in range(0, len(date_list), 7):
            week = date_list[week_start:week_start + 7]
            works_vars = []
            for date in week:
                M = len(shifts_per_day[date])
                works_var = model.NewBoolVar(f"works_{d}_{date}")
                works_vars.append(works_var)
                model.Add(sum(x[(d, s)] for s in shifts_per_day[date]) >= works_var)
                model.Add(sum(x[(d, s)] for s in shifts_per_day[date]) <= M * works_var)
            # Maximum of 6 working days per week
            model.Add(sum(works_vars) <= 6)

    """9. Only eligible physicians may be assigned 24-hour shifts."""
    for d in D_all:
        if twentyfour_allowed[d] == 0:
            for s in twentyfour_shifts:
                model.Add(x[(d, s)] == 0)

    """9.a Pandemic policy: disable all 24-hour shifts."""
    if pandemic_config is not None:
        for d in D_all:
            for s in twentyfour_shifts:
                model.Add(x[(d, s)] == 0)

    """10. Physician unavailability constraints (leave, absences, etc.)."""
    # Full-day unavailability
    for _, row in unavail_day.iterrows():
        d = row["doctor_id"]
        date = row["date"]
        for s in days_to_shifts[date]:
            model.Add(x[d, s] == 0)

    # Shift-specific unavailability
    for _, row in unavail_shift.iterrows():
        d = row["doctor_id"]
        code = row["code"]
        s = code_to_id[code]
        model.Add(x[d, s] == 0)

    """11. Candidate physicians may only work if hired."""
    for c in C:
        for s in S:
            model.Add(x[(c, s)] <= hire[c])

    """12. Only physicians eligible for night work may be assigned night shifts."""
    for d in D_all:
        if night_allowed[d] == 0:
            for s in night_shifts:
                model.Add(x[(d, s)] == 0)

    # =========================================================
    # PANDEMIC CONSTRAINTS
    # =========================================================

    pandemic_bundle = None

    if (
            pandemic_config is not None
            and bubble_assignment is not None
    ):
        pandemic_bundle = add_pandemic_constraints(
            model=model,
            x=x,
            D_all=D_all,
            S=S,
            shifts=shifts,
            shift_idx=shift_idx,
            config=pandemic_config,
            bubble_assignment=bubble_assignment,
            id_to_role=id_to_role,
        )

    """ SLACK """
    """Slack variables used to avoid infeasibility in understaffed scenarios."""
    slacks = {}
    slacks_o = {}

    for _, shift in shifts.iterrows():
        s = shift["id"]
        min_staff = shift["min_staff"]
        regular_staff = shift["regular_staff"]

        # === Understaff ===
        slack = model.NewIntVar(0, min_staff, f"slack_{s}")
        slacks[s] = slack
        model.Add(sum(x[(d, s)] for d in D_all) + slack >= min_staff)

        # === Overstaff ===
        slack_o = model.NewIntVar(0, regular_staff, f"slack_{s}")
        slacks_o[s] = slack_o
        model.Add(sum(x[(d, s)] for d in D) + slack_o <= regular_staff)

    # === SLACK WEIGHTS (pandemic prioritization) ===
    # In pandemic mode, ICU/WARD coverage is much more critical than CLINIC.
    # Outpatient clinics can be temporarily reduced/closed during a crisis,
    # while ICU patients require continuous care.
    slack_weights = {}
    for _, shift in shifts.iterrows():
        s = shift["id"]
        dept = shift["dept"]

        if pandemic_config is not None:
            if dept == "ICU":
                slack_weights[s] = 100
            elif dept == "WARD":
                slack_weights[s] = 50
            elif dept == "CLINIC":
                slack_weights[s] = 1
            else:
                slack_weights[s] = 10
        else:
            slack_weights[s] = 1

    """ SOFT CONSTRAINTS """
    pref_terms = []

    # === WORKLOAD PER DOCTOR (HOURS) ===
    worked_hours = {}
    for d in D_all:
        h_var = model.NewIntVar(0, adjusted_max_hours[d], f"worked_hours_{d}")
        model.Add(h_var == sum(x[(d, s)] * hours[s] for s in S))
        worked_hours[d] = h_var

    # === COST PER DOCTOR ===
    doctor_cost = {}
    for d in D:
        cost = model.NewIntVar(0, 10_000_000, f"cost_{d}")
        model.Add(cost == worked_hours[d] * doctors.loc[doctor_idx[d], "hourly_cost"])
        doctor_cost[d] = cost

    for d in C:
        hourly = int(candidates.loc[candidates["id"] == d, "hourly_cost"].iloc[0])

        # Effective hours: worked_hours[d] * hire[d]
        WH = model.NewIntVar(0, 2000, f"effective_hours_{d}")
        model.AddMultiplicationEquality(WH, [worked_hours[d], hire[d]])

        # cost = effective hours * hourly salary
        cost = model.NewIntVar(0, 10_000_000, f"cost_{d}")
        model.Add(cost == WH * hourly)

        doctor_cost[d] = cost

    # === WEEKLY HOURS PER DOCTOR ===
    weekly_hours = {d: {} for d in D_all}

    for d in D_all:
        for w in weeks:
            wh = model.NewIntVar(
                0, weekly_limits[d][w],
                f"weekly_hours_{d}_{w}"
            )
            weekly_shifts = [
                s for s in S
                if shifts.loc[shift_idx[s], "week"] == w
            ]

            model.Add(
                wh == sum(x[(d, s)] * hours[s] for s in weekly_shifts)
            )

            weekly_hours[d][w] = wh

    """1. Preferences """
    for _, row in pref.iterrows():
        d = row["doctor_id"]
        shift_code = row["code"]
        preference = row["preference"]

        s = code_to_id[shift_code]

        if preference == "like":
            pref_terms.append(-1 * x[(d, s)])

        if preference == "dislike":
            pref_terms.append(1 * x[(d, s)])

    """2. Fairness objectives: balanced distribution of demanding shifts."""
    """2.a Night shift fairness."""

    night_count = {}
    for d, allowed in night_allowed.items():
        if id_to_role[d] in ["specialist", "icu_specialist"]:
            if allowed == 1:
                count = model.NewIntVar(0, 100, f"night_count_{d}")
                model.Add(count == sum(x[(d, s)] for s in night_shifts))
                night_count[d] = count

    max_nights = model.NewIntVar(0, 100, "max_nights")
    min_nights = model.NewIntVar(0, 100, "min_nights")

    model.AddMaxEquality(max_nights, list(night_count.values()))
    model.AddMinEquality(min_nights, list(night_count.values()))

    spread = model.NewIntVar(0, 100, "night_spread")
    model.Add(spread == max_nights - min_nights)

    """2.b Weekend and hard-shift fairness."""

    weekend_shifts = [s for s in S if shift_date[s] in weekend]
    hard_shifts = shifts[shifts["is_hard_shift"] == 1]["id"].to_list()
    weekend_count = {}

    for d in D:
        if id_to_role[d] in ["specialist", "icu_specialist"]:
            wc = model.NewIntVar(0, 100, f"weekend_count_{d}")
            model.Add(wc == sum(x[(d, s)] for s in weekend_shifts))
            weekend_count[d] = wc

    hard_count = {}

    for d in D:
        if id_to_role[d] in ["specialist", "icu_specialist"]:
            wh = model.NewIntVar(0, 100, f"hard_count_{d}")
            model.Add(wh == sum(x[(d, s)] for s in hard_shifts))
            hard_count[d] = wh

    max_weekend = model.NewIntVar(0, 100, "max_weekend")
    min_weekend = model.NewIntVar(0, 100, "min_weekend")
    weekend_spread = model.NewIntVar(0, 100, "weekend_spread")

    model.AddMaxEquality(max_weekend, list(weekend_count.values()))
    model.AddMinEquality(min_weekend, list(weekend_count.values()))
    model.Add(weekend_spread == max_weekend - min_weekend)

    max_hard = model.NewIntVar(0, 100, "max_hard")
    min_hard = model.NewIntVar(0, 100, "min_hard")
    hard_spread = model.NewIntVar(0, 100, "hard_spread")

    model.AddMaxEquality(max_hard, list(hard_count.values()))
    model.AddMinEquality(min_hard, list(hard_count.values()))
    model.Add(hard_spread == max_hard - min_hard)

    """3. Balanced workload utilization relative to individual working hour limits."""
    workload_ratio = {}
    for d in opt_out_doctors:
        ratio = model.NewIntVar(0, 2000, f"workload_ratio_{d}")

        model.Add(ratio * adjusted_max_hours[d] <= worked_hours[d] * 1000 + 50)
        model.Add(ratio * adjusted_max_hours[d] >= worked_hours[d] * 1000 - 50)

        workload_ratio[d] = ratio

    max_ratio = model.NewIntVar(0, 2000, "max_ratio")
    min_ratio = model.NewIntVar(0, 2000, "min_ratio")
    ratio_spread = model.NewIntVar(0, 2000, "ratio_spread")

    if workload_ratio:
        model.AddMaxEquality(max_ratio, list(workload_ratio.values()))
        model.AddMinEquality(min_ratio, list(workload_ratio.values()))
        model.Add(ratio_spread == max_ratio - min_ratio)

    else:
        model.Add(max_ratio == 0)
        model.Add(min_ratio == 0)
        model.Add(ratio_spread == 0)

    """4. Penalization of insufficient workload among regular physicians."""
    underwork = {}
    for d in regular_doctors:

        uw = model.NewIntVar(0, adjusted_max_hours[d], f"underwork_{d}")
        target = int(0.95 * adjusted_max_hours[d])

        h = hire[d] if d in hire else 1
        model.Add(uw >= target * h - worked_hours[d])
        model.Add(uw >= 0)

        underwork[d] = uw

    print("\n=== WEEKLY TARGET DEBUG ===")

    for d in D_all:
        role = id_to_role[d]
        is_regular = d in regular_doctors
        mh = max_hours[d]

        for w in weeks:
            limit = weekly_limits[d][w]
            days_present = weekly_available_days[d][w]

            if is_regular:
                target = int(0.95 * limit)
            else:
                base_48 = 48 * days_present / 7
                target = int(0.95 * base_48)

            if target > limit:
                print(
                    f"PROBLEM: doctor={d}, role={role}, "
                    f"regular={is_regular}, max_hours={mh}, "
                    f"week={w}, days={days_present}, "
                    f"limit={limit}, target={target}"
                )
    """4.a Weekly underwork penalties."""
    weekly_underwork = {d: {} for d in D_all}
    for d in D_all:
        for w in weeks:
            limit = weekly_limits[d][w]
            days_present = weekly_available_days[d][w]
            if d in regular_doctors:
                target = int(0.95 * limit)
            else:
                base_48 = 48 * days_present / 7
                target = int(0.95 * base_48)

            uw = model.NewIntVar(0, weekly_limits[d][w], f"weekly_underwork_{d}_{w}")
            model.Add(uw >= target - weekly_hours[d][w])
            model.Add(uw >= 0)

            weekly_underwork[d][w] = uw

    if not allow_hire:
        for c in C:
            model.Add(hire[c] == 0)

    return {
        "model": model,
        "x": x,
        "hire": hire,
        "slacks": slacks,
        "slacks_o": slacks_o,
        "slack_weights": slack_weights,
        "worked_hours": worked_hours,
        "doctor_cost": doctor_cost,
        "spread": spread,
        "weekend_spread": weekend_spread,
        "hard_spread": hard_spread,
        "underwork": underwork,
        "weekly_underwork": weekly_underwork,
        "date_list": date_list,
        "hours": hours,
        "shift_date": shift_date,
        "night_shifts": night_shifts,
        "weekend_shifts": weekend_shifts,
        "twentyfour_shifts": twentyfour_shifts,
        "hard_shifts": hard_shifts,
        "code_to_id": code_to_id,
        "adjusted_max_hours": adjusted_max_hours,
        "weekly_limits": weekly_limits,
        "weeks": weeks,
        "regular_doctors": list(regular_doctors),
        "D": D_all,
        "weekly_hours": weekly_hours,
        "opt_out_doctors": list(opt_out_doctors),
        "hourly_cost_map": hourly_cost_map,
        "pref_terms": pref_terms,
        "monthly_limit_hours": monthly_limit_hours,
        "pandemic_bundle": pandemic_bundle,
    }

def solve_3_phase(bundle, *, time_limit_phase1=100, time_limit_phase2=100, time_limit_phase3=300, pandemic_mode=False):
    model = bundle["model"]
    slacks = bundle["slacks"]
    hire = bundle["hire"]
    night_shifts = bundle["night_shifts"]
    twentyfour_shifts = bundle["twentyfour_shifts"]
    hard_shifts = bundle["hard_shifts"]


    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = max(4, cores // 2)

    if pandemic_mode:
        time_limit_phase1 = max(time_limit_phase1, 180)
        time_limit_phase2 = max(time_limit_phase2, 180)
        time_limit_phase3 = max(time_limit_phase3, 600)

        solver.parameters.linearization_level = 2 # MAYBE WOULD HELP

    # ========= PHASE 1 - MINIMIZE UNDERSTAFFING =========

    # total_slack_expr = cp_model.LinearExpr.Sum(list(slacks.values()))
    #
    # model.Minimize(total_slack_expr)
    # solver.parameters.max_time_in_seconds = time_limit_phase1
    # status1 = solver.Solve(model)
    #
    # best_slack = solver.Value(total_slack_expr)
    # print(f"BEST SLACK: {best_slack}")
    #
    #
    # model.Add(total_slack_expr == best_slack)

    slack_weights = bundle.get("slack_weights", {s: 1 for s in slacks})

    weighted_slack_expr = cp_model.LinearExpr.Sum([
        slacks[s] * slack_weights[s] for s in slacks
    ])

    raw_slack_expr = cp_model.LinearExpr.Sum(list(slacks.values()))

    model.Minimize(weighted_slack_expr)
    solver.parameters.max_time_in_seconds = time_limit_phase1
    status1 = solver.Solve(model)

    best_weighted_slack = solver.Value(weighted_slack_expr)
    best_raw_slack = solver.Value(raw_slack_expr)
    print(f"BEST WEIGHTED SLACK: {best_weighted_slack}")
    print(f"BEST RAW SLACK: {best_raw_slack} (uncovered shifts)")

    model.Add(weighted_slack_expr == best_weighted_slack)

    total_hire_expr = cp_model.LinearExpr.Sum(list(hire.values()))

    model.Minimize(total_hire_expr)
    solver.parameters.max_time_in_seconds = time_limit_phase2
    status2 = solver.Solve(model)

    best_hire = solver.Value(total_hire_expr)
    model.Add(total_hire_expr == best_hire)

    pref_terms = bundle["pref_terms"]
    spread = bundle["spread"]
    weekend_spread = bundle["weekend_spread"]
    hard_spread = bundle["hard_spread"]
    underwork = bundle["underwork"]
    weekly_underwork = bundle["weekly_underwork"]
    weeks = bundle["weeks"]
    doctor_cost = bundle["doctor_cost"]
    slacks_o = bundle["slacks_o"]

    adjusted_max_hours = bundle["adjusted_max_hours"]
    regular_doctors = bundle["regular_doctors"]
    weekly_limits = bundle["weekly_limits"]
    weekend_shifts = bundle["weekend_shifts"]
    D = bundle["D"]
    hourly_cost_map = bundle["hourly_cost_map"]

    total_cost = sum(doctor_cost.values())
    total_underwork = sum(underwork.values())
    total_weekly_underwork = sum(
        weekly_underwork[d][w] for d in weekly_underwork for w in weeks
    )
    total_pref = sum(pref_terms) if pref_terms else 0

    REF = {
        "cost": 540_000,
        "spread": 4,
        "weekend_spread": 3,
        "hard_spread": 2,
        "underwork": 0,
        "weekly_underwork": 0,
        "pref": -300,
    }

    B = REF["cost"]

    SCALE = 1_000_000

    def dev(expr, ref, max_val):
        if max_val <= 0:
            return None
        coeff = SCALE // max_val
        return (expr - ref) * coeff

    Z = model.NewIntVar(0, SCALE * 10, "chebyshev_Z")
    model.Add(Z >= 0)
    deviations = []

    max_pref = len(pref_terms)
    max_night_shifts = len(night_shifts)
    max_twentyfour_shifts = len(twentyfour_shifts)
    max_hard_shifts = len(hard_shifts)
    max_night_spread = max_night_shifts
    max_weekend_spread = len(weekend_shifts)
    max_hard_spread = len(hard_shifts)
    max_underwork = sum(int(0.95 * adjusted_max_hours[d]) for d in regular_doctors)
    max_weekly_underwork = sum(int(0.95 * weekly_limits[d][w]) for d in D for w in weeks)

    max_cost = sum(
        adjusted_max_hours[d] * hourly_cost_map[d]
        for d in D
    )

    max_over = int(B * 0.05)
    over_budget = model.NewIntVar(0, max_over, "over_budget")

    model.Add(over_budget >= total_cost - B)
    model.Add(over_budget >= 0)

    # Normalizacja
    if max_over > 0:
        scale_cost = SCALE // max_over
        model.Add(Z >= over_budget * scale_cost)


    # NIGHT SPREAD
    coeff = SCALE // max_night_spread
    model.Add(Z >= (spread - REF["spread"]) * coeff)

    # WEEKEND
    coeff = SCALE // max_weekend_spread
    model.Add(Z >= (weekend_spread - REF["weekend_spread"]) * coeff)

    # HARD
    if max_hard_spread > 0:
        coeff = SCALE // max_hard_spread
    else:
        coeff = 0
    model.Add(Z >= (hard_spread - REF["hard_spread"]) * coeff)

    # UNDERWORK
    # coeff = SCALE // max_underwork *2
    coeff = SCALE // max_underwork
    model.Add(Z >= (total_underwork - REF["underwork"]) * coeff)

    # WEEKLY UNDERWORK
    coeff = SCALE // max_weekly_underwork
    model.Add(Z >= (total_weekly_underwork - REF["weekly_underwork"]) * coeff)

    # PREFERENCES
    coeff = SCALE // max_pref
    model.Add(Z >= (total_pref - REF["pref"]) * coeff)

    pandemic_penalty_expr = 0

    if "pandemic_bundle" in bundle:

        pb = bundle["pandemic_bundle"]

        if pb is not None:
            pandemic_penalty_expr = (
                    pb["penalty"]
                    * cp_model.LinearExpr.Sum(
                pb["crossbubble_violations"]
            )
            )

    model.Minimize(
        Z + pandemic_penalty_expr
    )

    solver.parameters.max_time_in_seconds = time_limit_phase3
    status3 = solver.Solve(model)

    return solver, status3

def solve_single_phase_comparison(bundle, *, time_limit=300):
    model = bundle["model"]
    slacks = bundle["slacks"]
    hire = bundle["hire"]
    night_shifts = bundle["night_shifts"]
    twentyfour_shifts = bundle["twentyfour_shifts"]
    hard_shifts = bundle["hard_shifts"]

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = max(4, cores // 2)
    # solver.parameters.num_search_workers = 1

    total_understaffing = sum(slacks.values())
    total_hire = sum(hire.values())

    pref_terms = bundle["pref_terms"]
    spread = bundle["spread"]
    weekend_spread = bundle["weekend_spread"]
    hard_spread = bundle["hard_spread"]
    underwork = bundle["underwork"]
    weekly_underwork = bundle["weekly_underwork"]
    weeks = bundle["weeks"]
    doctor_cost = bundle["doctor_cost"]
    slacks_o = bundle["slacks_o"]

    adjusted_max_hours = bundle["adjusted_max_hours"]
    regular_doctors = bundle["regular_doctors"]
    weekly_limits = bundle["weekly_limits"]
    weekend_shifts = bundle["weekend_shifts"]
    D = bundle["D"]
    hourly_cost_map = bundle["hourly_cost_map"]

    total_cost = sum(doctor_cost.values())
    total_underwork = sum(underwork.values())
    total_weekly_underwork = sum(
        weekly_underwork[d][w] for d in weekly_underwork for w in weeks
    )
    total_pref = sum(pref_terms) if pref_terms else 0

    REF = {
        "cost": 1_620_000,
        "spread": 4,
        "weekend_spread": 3,
        "hard_spread": 2,
        "underwork": 0,
        "weekly_underwork": 0,
        "pref": -200,
    }

    B = REF["cost"]

    SCALE = 1_000_000

    def dev(expr, ref, max_val):
        if max_val <= 0:
            return None
        coeff = SCALE // max_val
        return (expr - ref) * coeff

    Z = model.NewIntVar(0, SCALE * 10, "chebyshev_Z")
    model.Add(Z >= 0)
    deviations = []

    max_pref = len(pref_terms)
    max_night_shifts = len(night_shifts)
    max_twentyfour_shifts = len(twentyfour_shifts)
    max_hard_shifts = len(hard_shifts)
    max_night_spread = max_night_shifts
    max_weekend_spread = len(weekend_shifts)
    max_hard_spread = len(hard_shifts)
    max_underwork = sum(int(0.95 * adjusted_max_hours[d]) for d in regular_doctors)
    max_weekly_underwork = sum(int(0.95 * weekly_limits[d][w]) for d in D for w in weeks)

    max_cost = sum(
        adjusted_max_hours[d] * hourly_cost_map[d]
        for d in D
    )
    max_over = int(B * 0.15)
    over_budget = model.NewIntVar(0, max_over, "over_budget")

    model.Add(over_budget >= total_cost - B)
    model.Add(over_budget >= 0)

    # Normalization
    if max_over > 0:
        scale_cost = SCALE // max_over
        model.Add(Z >= over_budget * scale_cost)

    # NIGHT SPREAD
    coeff = SCALE // max_night_spread
    model.Add(Z >= (spread - REF["spread"]) * coeff)

    # WEEKEND
    coeff = SCALE // max_weekend_spread
    model.Add(Z >= (weekend_spread - REF["weekend_spread"]) * coeff)

    # HARD
    coeff = SCALE // max_hard_spread
    model.Add(Z >= (hard_spread - REF["hard_spread"]) * coeff)

    # UNDERWORK
    coeff = SCALE // max_underwork
    model.Add(Z >= (total_underwork - REF["underwork"]) * coeff)

    # WEEKLY UNDERWORK
    coeff = SCALE // max_weekly_underwork
    model.Add(Z >= (total_weekly_underwork - REF["weekly_underwork"]) * coeff)

    # PREFERENCES
    coeff = SCALE // max_pref
    model.Add(Z >= (total_pref - REF["pref"]) * coeff)

    final_objective = (total_understaffing * 1000000) + (total_hire * 10000) + Z
    model.Minimize(final_objective)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)

    return solver, status


def count_free_weekends(schedule, doctor_name):
    df = schedule[schedule["Doctor"] == doctor_name].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Week"] = df["Date"].dt.isocalendar().week
    df["Weekday"] = df["Date"].dt.weekday

    weekends = df[df["Weekday"].isin([5, 6])]
    worked_weekends = weekends.groupby("Week").size()

    all_weeks = df["Week"].unique()
    free_weekends = [
        w for w in all_weeks if w not in worked_weekends.index
    ]
    return len(free_weekends)


def count_free_day_blocks(schedule, doctor_name):
    df = schedule[schedule["Doctor"] == doctor_name]
    worked_days = set(df["Date"].dt.date)

    all_days = pd.date_range(
        schedule["Date"].min(),
        schedule["Date"].max()
    ).date

    blocks = 0
    current = 0

    for d in all_days:
        if d not in worked_days:
            current += 1
        else:
            if current >= 2:
                blocks += 1
            current = 0

    if current >= 2:
        blocks += 1

    return blocks


def count_comfort_weeks(weekly_hours, weekly_limits, d):
    good = 0
    for w, hours in weekly_hours[d].items():
        limit = weekly_limits[d].get(w, 0)
        if limit == 0:
            continue
        if 0.7 * limit <= hours <= 0.9 * limit:
            good += 1
    return good


def count_month_weekends(date_list):
    dates = pd.to_datetime(date_list)
    df = pd.DataFrame({"Date": dates})
    df["Week"] = df["Date"].dt.isocalendar().week
    df["Weekday"] = df["Date"].dt.weekday

    weekends = df[df["Weekday"].isin([5, 6])]
    return weekends["Week"].nunique()

def max_comfort_weeks(d, weekly_limits):
    return sum(1 for w, lim in weekly_limits[d].items() if lim > 0)


def compute_satisfaction_pct_new(row):
    BASE_SAT = 75

    STANDARDS = {
        "night": 4,
        "twentyfour": 2,
        "weekend": 2,
        "hard": 3,
    }

    def excess(count, limit):
        return max(0, count - limit)

    penalty = (
        6 * excess(row["TwentyFourCount"], STANDARDS["twentyfour"])
      + 4 * excess(row["NightCount"], STANDARDS["night"])
      + 3 * excess(row["WeekendCount"], STANDARDS["weekend"])
      + 2 * excess(row["HardCount"], STANDARDS["hard"])
      + 6 * row["dislike_violated"]
    )

    bonus = (
        6 * row["like_satisfied"]
      + 5 * row["ComfortWeeks"]
      + 6 * row["FreeWeekends"]
      + 4 * row["FreeBlocks"]
    )

    raw = BASE_SAT + bonus - penalty
    return round(max(0, min(100, raw)), 1)


import math

def compute_satisfaction_v2(row):

    util = row["TotalHours"] / row["MonthlyLimitHours"]
    util = max(0.0, util)

    base = 82 - 12 * util

    role = row.get("Role", "unknown")
    standards_by_role = {
        "icu_specialist": {"night": 8, "twentyfour": 3, "weekend": 2, "hard": 3},
        "specialist":     {"night": 6, "twentyfour": 2, "weekend": 2, "hard": 3},
        "resident":       {"night": 3, "twentyfour": 1, "weekend": 2, "hard": 1},
        "outpatient":     {"night": 0, "twentyfour": 0, "weekend": 1, "hard": 0},
    }
    std = standards_by_role.get(role, {"night": 5, "twentyfour": 2, "weekend": 2, "hard": 3})

    scale = max(0.2, math.sqrt(util))

    def scaled_limit(std_value):
        if std_value == 0:
            return 0
        return max(1, int(round(std_value * scale)))

    night_lim = scaled_limit(std["night"])
    tf_lim    = scaled_limit(std["twentyfour"])
    wend_lim  = scaled_limit(std["weekend"])
    hard_lim  = scaled_limit(std["hard"])

    def excess(x, lim):
        return max(0, x - lim)

    penalty = (
            7 * (excess(row["TwentyFourCount"], tf_lim) ** 2) +
            5 * excess(row["NightCount"], night_lim) +
            4 * excess(row["WeekendCount"], wend_lim) +
            3 * excess(row["HardCount"], hard_lim) +
            8 * row["dislike_violated"]
    )

    like_bonus = 6 * math.sqrt(max(0, row["like_satisfied"]))
    comfort_bonus = 5 * min(row["ComfortWeeks"], 4)
    free_wend_bonus = 6 * min(row["FreeWeekends"], 4)
    free_block_bonus = 4 * min(row["FreeBlocks"], 6)

    score = base + (like_bonus + comfort_bonus + free_wend_bonus + free_block_bonus) - penalty
    return round(max(0, min(100, score)), 1)



def extract_results(bundle, solver, status, doctors, candidates, shifts, pref, context):

    x = bundle["x"]
    hire = bundle["hire"]
    worked_hours = bundle["worked_hours"]
    slacks = bundle["slacks"]

    date_list = bundle["date_list"]
    hours = bundle["hours"]
    night_shifts = bundle["night_shifts"]
    twentyfour_shifts = bundle["twentyfour_shifts"]
    adjusted_max_hours = bundle["adjusted_max_hours"]
    code_to_id = bundle["code_to_id"]
    weeks = bundle["weeks"]

    D = context["D"]
    C = context["C"]
    D_all = context["D_all"]
    S = context["S"]
    shift_idx = context["shift_idx"]
    id_to_name = context["id_to_name"]
    id_to_role = context["id_to_role"]

    raw_hire = {c: solver.Value(hire[c]) for c in C}
    raw_x = {(d, s): solver.Value(x[(d, s)]) for d in D_all for s in S}

    weekend_shifts = bundle["weekend_shifts"]
    hard_shifts = bundle["hard_shifts"]
    weekly_limits = bundle["weekly_limits"]
    weekly_hours = bundle["weekly_hours"]

    hired_candidates = [c for c in C if raw_hire[c] == 1]
    hired_df = pd.DataFrame([
        {"Doctor": id_to_name[c], "Role": id_to_role[c], "Hours": solver.Value(worked_hours[c])}
        for c in hired_candidates
    ])
    solver_stats_df = build_solver_stats(solver, status, bundle["model"])

    rows = []
    shifts_sorted = shifts.sort_values(by=["date", "start_hour"])
    for date in date_list:
        day_df = shifts_sorted[shifts_sorted["date"] == date]
        for _, sh in day_df.iterrows():
            s_id = sh["id"]
            assigned = [
                d for d in D_all
                if solver.Value(x[(d, s_id)]) == 1
                and (d in D or solver.Value(hire.get(d, 0)) == 1)
            ]
            missing = solver.Value(slacks[s_id])

            for d in assigned:
                rows.append({
                    "Date": date,
                    "ShiftCode": sh["code"],
                    "Dept": sh["dept"],
                    "StartHour": sh["start_hour"],
                    "EndHour": sh["end_hour"],
                    "Hours": sh["hours"],
                    "Doctor": id_to_name[d],
                    "Role": id_to_role[d],
                    "Missing": 0
                })

            for _ in range(missing):
                rows.append({
                    "Date": date,
                    "ShiftCode": sh["code"],
                    "Dept": sh["dept"],
                    "StartHour": sh["start_hour"],
                    "EndHour": sh["end_hour"],
                    "Hours": sh["hours"],
                    "Doctor": None,
                    "Role": None,
                    "Missing": 1
                })

    schedule_df = pd.DataFrame(rows)
    schedule_df["Date"] = pd.to_datetime(schedule_df["Date"])

    stats_df = build_doctor_stats(
        D_all, S, id_to_name, id_to_role,
        x, hours, night_shifts, twentyfour_shifts,
        shifts, shift_idx, adjusted_max_hours, solver
    )

    name_to_id = {v: k for k, v in id_to_name.items()}

    monthly_limits = bundle["monthly_limit_hours"]

    stats_df["MonthlyLimitHours"] = stats_df["Doctor"].apply(
        lambda nm: monthly_limits[name_to_id[nm]]
    )

    stats_df["IsCandidate"] = stats_df["Doctor"].isin([id_to_name[c] for c in C])
    stats_df["Hired"] = stats_df["Doctor"].isin([id_to_name[c] for c in hired_candidates])

    stats_df["UtilizationPct"] = (
        stats_df["TotalHours"] / stats_df["MaxHours"].replace(0, 1) * 100
    ).round(1)

    stats_df = add_preference_stats(stats_df, pref, x, solver, id_to_name, code_to_id)

    # === SATISFACTION METRIC ===
    night_set = set(bundle["night_shifts"])
    weekend_set = set(bundle["weekend_shifts"])
    hard_set = set(bundle["hard_shifts"])
    tf_set = set(bundle["twentyfour_shifts"])

    def count_assigned(d, shift_ids):
        return sum(solver.Value(x[(d, s)]) for s in shift_ids)

    stats_df["NightCount"] = stats_df["Doctor"].apply(lambda nm: count_assigned(name_to_id[nm], night_set))
    stats_df["WeekendCount"] = stats_df["Doctor"].apply(lambda nm: count_assigned(name_to_id[nm], weekend_set))
    stats_df["HardCount"] = stats_df["Doctor"].apply(lambda nm: count_assigned(name_to_id[nm], hard_set))
    stats_df["TwentyFourCount"] = stats_df["Doctor"].apply(lambda nm: count_assigned(name_to_id[nm], tf_set))

    weekly_hours = bundle["weekly_hours"]
    weekly_limits = bundle["weekly_limits"]
    weeks = bundle["weeks"]
    opt_out = set(bundle["opt_out_doctors"])

    def overwork_penalty(d, penalty_value=15):
        if d not in opt_out:
            return 0

        active_weeks = [w for w in weeks if weekly_limits[d][w] > 0]
        if not active_weeks:
            return 0

        all_weeks_over_90 = True
        for w in active_weeks:
            wh = solver.Value(weekly_hours[d][w])
            lim = weekly_limits[d][w]
            if wh < 0.9 * lim:
                all_weeks_over_90 = False
                break

        return penalty_value if all_weeks_over_90 else 0

    stats_df["OverworkPenalty"] = stats_df["Doctor"].apply(lambda nm: overwork_penalty(name_to_id[nm]))

    stats_df["S_likes"] = 10 * stats_df["like_satisfied"]
    stats_df["S_dislikes"] = -8 * stats_df["dislike_violated"]
    stats_df["S_night"] = -5 * stats_df["NightCount"]
    stats_df["S_24h"] = -7 * stats_df["TwentyFourCount"]
    stats_df["S_hard"] = -4 * stats_df["HardCount"]
    stats_df["S_weekend"] = -3 * stats_df["WeekendCount"]
    stats_df["S_overwork"] = -1 * stats_df["OverworkPenalty"]


    sched_real = schedule_df.dropna(subset=["Doctor"]).copy()

    weekly_hours_val = {
        d: {w: solver.Value(bundle["weekly_hours"][d][w]) for w in weeks}
        for d in D_all
    }

    BONUS_FREE_WEEKEND = 8
    BONUS_FREE_BLOCK = 6
    BONUS_COMFORT_WEEK = 5

    def bonus_free_weekends(name):
        return BONUS_FREE_WEEKEND * count_free_weekends(sched_real, name)

    def bonus_free_blocks(name):
        return BONUS_FREE_BLOCK * count_free_day_blocks(sched_real, name)

    def bonus_comfort(name):
        d = name_to_id[name]
        return BONUS_COMFORT_WEEK * count_comfort_weeks(weekly_hours_val, weekly_limits, d)

    stats_df["S_free_weekends"] = stats_df["Doctor"].apply(bonus_free_weekends)
    stats_df["S_free_blocks"] = stats_df["Doctor"].apply(bonus_free_blocks)
    stats_df["S_comfort"] = stats_df["Doctor"].apply(bonus_comfort)

    stats_df["S_raw"] = stats_df[
        ["S_likes", "S_dislikes", "S_night", "S_24h", "S_hard", "S_weekend", "S_overwork",
         "S_free_weekends", "S_free_blocks", "S_comfort"]
    ].sum(axis=1)

    max_weekends_global = count_month_weekends(bundle["date_list"])

    max_comfort_by_name = {}
    for nm in stats_df["Doctor"]:
        d = name_to_id[nm]
        max_comfort_by_name[nm] = max_comfort_weeks(d, weekly_limits)

    stats_df["ComfortWeeks"] = stats_df["Doctor"].apply(
        lambda nm: count_comfort_weeks(weekly_hours_val, weekly_limits, name_to_id[nm])
    )

    stats_df["FreeWeekends"] = stats_df["Doctor"].apply(
        lambda nm: count_free_weekends(sched_real, nm)
    )

    stats_df["FreeBlocks"] = stats_df["Doctor"].apply(
        lambda nm: count_free_day_blocks(sched_real, nm)
    )

    stats_df["S_pct"] = stats_df.apply(compute_satisfaction_v2, axis=1)

    spread_val = solver.Value(bundle["spread"])
    weekend_spread_val = solver.Value(bundle["weekend_spread"])
    hard_spread_val = solver.Value(bundle["hard_spread"])

    total_slack = sum(solver.Value(sl) for sl in slacks.values())

    doctor_cost = bundle["doctor_cost"]

    total_cost = sum(
        solver.Value(doctor_cost[d])
        for d in doctor_cost
    )

    opt_out_doctors = set(
        doctors[doctors["opt_out"] == 1]["id"]
    )
    avg_satisfaction = round(stats_df["S_pct"].mean(), 2)
    min_satisfaction = round(stats_df["S_pct"].min(), 2)
    max_satisfaction = round(stats_df["S_pct"].max(), 2)

    return {

        "schedule_df": schedule_df,
        "stats_df": stats_df,
        "hired_df": hired_df,
        "solver_stats_df": solver_stats_df,

        # metryki
        "objective": solver.ObjectiveValue() if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
        "slack": total_slack,
        "total_cost": total_cost,
        "spread": spread_val,
        "weekend_spread": weekend_spread_val,
        "hard_spread": hard_spread_val,

        # raw debug
        "raw_hire": raw_hire,
        "raw_x": raw_x,
        "avg_satisfaction": avg_satisfaction,
        "min_satisfaction": min_satisfaction,
        "max_satisfaction": max_satisfaction,

        # pandemic mode
        "pandemic_bundle": bundle.get("pandemic_bundle"),
    }

def print_full_report(*, results, max_schedule_rows=30):
    print("\n=== SOLVER SUMMARY ===")
    print(results["solver_stats_df"].to_string(index=False))

    print("\n=== OBJECTIVE & SLACK ===")
    print(f"Objective value : {results['objective']}")
    print(f"Total slack     : {results['slack']}")

    print("\n=== FAIRNESS METRICS ===")
    print(f"Night spread    : {results['spread']}")
    print(f"Weekend spread  : {results['weekend_spread']}")
    print(f"Hard spread     : {results['hard_spread']}")

    print("\n=== HIRED CANDIDATES ===")
    if results["hired_df"].empty:
        print("None")
    else:
        print(results["hired_df"].to_string(index=False))

    print("\n=== DOCTOR STATS ===")
    print(results["stats_df"].to_string(index=False))

    print(f"\n=== SCHEDULE (first {max_schedule_rows} rows) ===")
    print(results["schedule_df"].head(max_schedule_rows).to_string(index=False))

    print("\n=== RAW DEBUG INFO ===")
    print(f"#x assignments    : {len(results['raw_x'])}")
    print(f"#hire vars        : {len(results['raw_hire'])}")


def run_scenario(
    *,
    doctors,
    candidates,
    shifts,
    unavail_day,
    unavail_shift,
    pref,

    allow_hire: bool,

    pandemic_config=None,
    bubble_assignment=None,
    validation_result=None,
):
    print("\n=== DEBUG IDS ===")

    print("Doctors:", doctors["id"].min(), doctors["id"].max())
    print("Candidates:", candidates["id"].min(), candidates["id"].max())

    print("Unavail_day:", unavail_day["doctor_id"].min(), unavail_day["doctor_id"].max())
    print("Unavail_shift:", unavail_shift["doctor_id"].min(), unavail_shift["doctor_id"].max())
    print("Pref:", pref["doctor_id"].min(), pref["doctor_id"].max())

    print("Missing in doctors (day):",
          set(unavail_day["doctor_id"]) - set(doctors["id"]))

    print("Missing in doctors (shift):",
          set(unavail_shift["doctor_id"]) - set(doctors["id"]))

    print("Missing in doctors (pref):",
          set(pref["doctor_id"]) - set(doctors["id"]))

    print("Doctor-Candidate overlap:",
          set(doctors["id"]) & set(candidates["id"]))

    context = prepare_context(doctors, candidates, shifts, unavail_day, unavail_shift, pref)

    bundle = build_model(
        context,
        doctors,
        candidates,
        shifts,
        unavail_day,
        unavail_shift,
        pref,
        allow_hire=allow_hire,

        pandemic_config=pandemic_config,
        bubble_assignment=bubble_assignment,
    )

    from ortools.sat.python import cp_model as _cp

    _solver = _cp.CpSolver()
    _solver.parameters.max_time_in_seconds = 30
    _solver.parameters.num_search_workers = 1

    _model_test = bundle["model"]
    _model_test.Minimize(0)
    _st = _solver.Solve(_model_test)
    print("FEASIBILITY CHECK:", _solver.StatusName(_st))

    print(set(unavail_day["date"]) - set(shifts["date"]))

    print(
        unavail_day.groupby("date")["doctor_id"]
        .nunique()
        .sort_values(ascending=False)
    )

    for skill in shifts["required_skill"].unique():
        eligible = doctors[doctors["skills"].str.contains(skill, na=False)]
        print(skill, len(eligible))

    print("Total doctors:", len(doctors))
    print("Total shifts:", len(shifts))
    print("Total demand:", shifts["min_staff"].sum())
    available_per_day = (
            len(doctors)
            - unavail_day.groupby("date")["doctor_id"].nunique()
    )
    print(available_per_day)

    # solver, status = solve_3_phase(bundle)
    solver, status = solve_3_phase(
        bundle,
        pandemic_mode=(pandemic_config is not None),
    )

    results = extract_results(
        bundle, solver, status,
        doctors, candidates, shifts, pref, context
    )

    results["status_name"] = solver.StatusName(status)
    results["allow_hire"] = allow_hire
    results["doctors"] = doctors
    results["shifts"] = shifts
    results["unavail_day"] = unavail_day
    results["unavail_shift"] = unavail_shift
    results["pref"] = pref
    results["weekly_limits"] = bundle["weekly_limits"]

    if pandemic_config is not None and bubble_assignment is not None:
        hired_candidate_ids = [
            c for c, hired in results["raw_hire"].items()
            if hired == 1
        ]

        final_bubble_assignment = assign_hired_candidates_to_bubbles(
            hired_candidate_ids=hired_candidate_ids,
            bubble_assignment=bubble_assignment,
            doctors=doctors,
            candidates=candidates,
            config=pandemic_config,
        )

        results["pandemic"] = {
            "config": pandemic_config,
            "bubble_assignment": final_bubble_assignment,
            "validation": validation_result,
            "summary": get_pandemic_results(
                bubble_assignment=final_bubble_assignment,
                config=pandemic_config,
                id_to_name=context["id_to_name"],
                id_to_role=context["id_to_role"],
            )
        }

    return results

def now_ts():
    return dt.now().strftime("%Y%m%d_%H%M%S")

def save_results(results, base_dir, label, ts):
    out_dir = base_dir / f"{label}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results["schedule_df"].to_csv(out_dir / f"schedule_{ts}.csv", index=False)
    results["stats_df"].to_csv(out_dir / f"doctor_stats_{ts}.csv", index=False)
    results["solver_stats_df"].to_csv(out_dir / f"solver_stats_{ts}.csv", index=False)

    if results["hired_df"] is not None:
        results["hired_df"].to_csv(out_dir / f"hired_{ts}.csv", index=False)

    eff = compute_hospital_efficiency_v2(results)
    efficiency_score = eff["HospitalEfficiencyScore"]

    summary_df = pd.DataFrame([{
        "total_cost": results.get("total_cost"),
        "avg_satisfaction": results.get("avg_satisfaction"),
        "min_satisfaction": results.get("min_satisfaction"),
        "max_satisfaction": results.get("max_satisfaction"),
        "hospital_efficiency": efficiency_score,
        "objective_value": results.get("objective"),
        "slack": results.get("slack"),
        "night_spread": results.get("spread"),
        "weekend_spread": results.get("weekend_spread"),
        "hard_spread": results.get("hard_spread"),
        "allow_hire": results.get("allow_hire"),
        "status": results.get("status_name"),
    }])

    summary_df.to_csv(out_dir / f"summary_{ts}.csv", index=False)

# === MAIN FUNCTION ===
def run_model_and_get_results(
    doctors_df=None,
    pandemic_mode=False,
):
    if doctors_df is None:
        doctors = pd.read_csv(DATA_DIR / "doctors" / "doctors_with_cost.csv")
        # doctors = pd.read_csv(DATA_DIR / "doctors" / "doctors_30.csv")
        # doctors = pd.read_csv(DATA_DIR / "doctors" / "doctors_45.csv")
        # doctors = pd.read_csv(DATA_DIR / "doctors" / "doctors_with_cost1.csv")
        # doctors = pd.read_csv(DATA_DIR / "doctors" / "doctors_with_cost2.csv")
    else:
        doctors = doctors_df.copy()

    # shifts = pd.read_csv(DATA_DIR / "shifts_month" / "shifts_dec2026.csv")
    shifts = pd.read_csv(DATA_DIR / "shifts_month" / "shifts_dec2026_pandemic.csv")
    unavail_day = pd.read_csv(DATA_DIR / "unavail_monthly_day" / "unavail_dec_2026.csv")
    unavail_shift = pd.read_csv(DATA_DIR / "unavail_monthly_shift" / "unavail_sh_dec_2026.csv")
    pref = pd.read_csv(DATA_DIR / "pref_monthly" / "pref_dec.csv")
    # shifts = pd.read_csv(DATA_DIR / "shifts_month" / "shifts_mar2026.csv")
    # unavail_day = pd.read_csv(DATA_DIR / "unavail_monthly_day" / "unavailability_mar2026.csv")
    # unavail_shift = pd.read_csv(DATA_DIR / "unavail_monthly_shift" / "shift_unavailability_mar2026.csv")
    # pref = pd.read_csv(DATA_DIR / "pref_monthly" / "preferences_mar2026.csv")
    # shifts = pd.read_csv(DATA_DIR / "shifts_month" / "shifts_30.csv")
    # unavail_day = pd.read_csv(DATA_DIR / "unavail_monthly_day" / "unavail_day_30.csv")
    # unavail_shift = pd.read_csv(DATA_DIR / "unavail_monthly_shift" / "unavail_shift_30.csv")
    # pref = pd.read_csv(DATA_DIR / "pref_monthly" / "pref_30.csv")
    # shifts = pd.read_csv(DATA_DIR / "shifts_month" / "shifts_45.csv")
    # unavail_day = pd.read_csv(DATA_DIR / "unavail_monthly_day" / "unavail_day_45.csv")
    # unavail_shift = pd.read_csv(DATA_DIR / "unavail_monthly_shift" / "unavail_shift_45.csv")
    # pref = pd.read_csv(DATA_DIR / "pref_monthly" / "pref_45.csv")
    # shifts = pd.read_csv(DATA_DIR / "shifts_month" / "shifts_dec_1.csv")
    # unavail_day = pd.read_csv(DATA_DIR / "unavail_monthly_day" / "unavail_dec_2026_1.csv")
    # unavail_shift = pd.read_csv(DATA_DIR / "unavail_monthly_shift" / "unavail_sh_dec_2026_1.csv")
    # pref = pd.read_csv(DATA_DIR / "pref_monthly" / "pref_dec_1.csv")

    candidates = pd.read_csv(DATA_DIR / "doctors_to_hire" / "doctors_to_hire.csv")

    candidates["skill_list"] = candidates["skills"].apply(lambda x: x.split(";") if isinstance(x, str) else [])
    doctors["skill_list"] = doctors["skills"].apply(lambda x: x.split(";") if isinstance(x, str) else [])
    doctors["hourly_cost"] = doctors["hourly_cost"].astype(int)

    pandemic_config = None
    bubble_assignment = None

    if pandemic_mode:
        pandemic_config = PandemicConfig(
            num_bubbles=3,
            contact_depts=["WARD", "ICU"],
            min_specialists_per_bubble=1,
        )

        bubble_assignment = prepare_pandemic_bubbles(
            doctors=doctors,
            candidates=candidates,
            config=pandemic_config,
        )

        # validate_bubbles(
        #     doctors=doctors,
        #     candidates=candidates,
        #     shifts=shifts,
        #     bubble_assignment=bubble_assignment,
        #     config=pandemic_config,
        # )
        validation_result = validate_bubbles(
            doctors=doctors,
            candidates=candidates,
            shifts=shifts,
            bubble_assignment=bubble_assignment,
            config=pandemic_config,
        )

    baseline = run_scenario(
        doctors=doctors,
        candidates=candidates,
        shifts=shifts,
        unavail_day=unavail_day,
        unavail_shift=unavail_shift,
        pref=pref,
        allow_hire=False,

        pandemic_config=pandemic_config,
        bubble_assignment=bubble_assignment,
        validation_result=validation_result,
    )

    results = {"baseline": baseline}

    if baseline["slack"] > 0:
        hire_allowed = run_scenario(
            doctors=doctors,
            candidates=candidates,
            shifts=shifts,
            unavail_day=unavail_day,
            unavail_shift=unavail_shift,
            pref=pref,
            allow_hire=True,

            pandemic_config=pandemic_config,
            bubble_assignment=bubble_assignment,
        )
        results["hire_allowed"] = hire_allowed
    else:
        results["hire_allowed"] = None


    ts = now_ts()

    RESULTS_DIR = BASE_DIR / "results" / "mar_2026"

    save_results(results["baseline"], RESULTS_DIR, "baseline", ts)

    if results["hire_allowed"] is not None:
        save_results(results["hire_allowed"], RESULTS_DIR, "hire_allowed", ts)

    results["run"] = ts

    return results

if __name__ == "__main__":
    results = run_model_and_get_results(
        pandemic_mode=True
    )

    print("\n================ BASELINE (NO HIRE) ================\n")
    print_full_report(results=results["baseline"])

    print("\n================ HIRE ALLOWED ================\n")
    print_full_report(results=results["hire_allowed"])




