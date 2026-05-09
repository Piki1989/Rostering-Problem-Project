"""
Module for finding a replacement doctor in emergency situations.

Hard constraints:
  1.  Required skill per shift
  2.  Max 1 shift per day
  3.  Min 11h rest between shifts
  4.  Weekly hour limit
  5.  Mentor present when trainee/resident needs one
  6.  Max 2 consecutive night shifts
  7.  Day off after a night shift
  8.  Max 6 working days per week
  9.  24h shift allowed flag
  10. Unavailability (day-off / specific shift)
  11. Candidate only if hired (hire limit respected)
  12. Night shift allowed flag

result = find_replacement(
    results=results["baseline"],
    sick_doctor="Jan Kowalski",
    shift_code="2026-12-05_ICU_N",
    reserve_doctors_df=reserve_df,
    candidates_df=candidates_df,
    max_new_hires=2,
)
"""

from __future__ import annotations

from typing import Optional
import pandas as pd

# CONSTANTS

MIN_REST_HOURS = 11
MAX_CONSECUTIVE_NIGHTS = 2
MAX_WORKING_DAYS_PER_WEEK = 6
MAX_NEW_HIRES_DEFAULT = 2


# SCHEDULE INDEX HELPERS

def _build_schedule_index(schedule_df: pd.DataFrame) -> dict[str, list[dict]]:
    """doctor_name -> list of assigned shifts (dicts)."""
    index: dict[str, list[dict]] = {}
    for _, row in schedule_df.dropna(subset=["Doctor"]).iterrows():
        name = row["Doctor"]
        index.setdefault(name, []).append({
            "date": str(row["Date"])[:10],
            "start": int(row["StartHour"]),
            "end": int(row["EndHour"]),
            "hours": int(row["Hours"]),
            "code": row["ShiftCode"],
            "is_night": "_N_" in str(row["ShiftCode"]) or int(row["Hours"]) == 24,
        })
    return index


def _build_date_index(schedule_df: pd.DataFrame) -> dict[str, int]:
    dates = sorted(schedule_df["Date"].astype(str).str[:10].unique())
    return {d: i for i, d in enumerate(dates)}


def _abs_hours(date: str, hour: int, date_index: dict[str, int]) -> int:
    return date_index.get(date, 0) * 24 + hour


def _get_start_monday(schedule_df: pd.DataFrame) -> pd.Timestamp:
    start = pd.to_datetime(schedule_df["Date"].min())
    return start - pd.Timedelta(days=start.weekday())


def _date_to_week(date_str: str, start_monday: pd.Timestamp) -> int:
    return int((pd.to_datetime(date_str) - start_monday).days // 7)


def _get_weekly_limit(
        doc_id,
        shift_date: str,
        weekly_limits: dict,
        start_monday: pd.Timestamp,
) -> Optional[int]:
    local_week = _date_to_week(shift_date, start_monday)
    return weekly_limits.get(doc_id, {}).get(local_week)


def _specialists_on_shift(
        shift_code: str,
        schedule_df: pd.DataFrame,
        doctors_df: pd.DataFrame,
) -> list[str]:
    """Names of specialists/icu_specialists already assigned to this shift."""
    specialist_roles = {"specialist", "icu_specialist"}
    assigned = schedule_df[
        (schedule_df["ShiftCode"] == shift_code) &
        (schedule_df["Doctor"].notna())
        ]["Doctor"].tolist()

    result = []
    for name in assigned:
        row = doctors_df[doctors_df["name"] == name]
        if not row.empty and row.iloc[0].get("role", "") in specialist_roles:
            result.append(name)
    return result


def _doctor_needs_mentor(doctor_row: pd.Series) -> bool:
    return int(doctor_row.get("needs_mentor", 0)) == 1


# INDIVIDUAL CONSTRAINT CHECKERS

def _c3_min_rest(
        doctor_shifts: list[dict],
        new_shift: dict,
        date_index: dict[str, int],
) -> list[str]:
    """At least 11h uninterrupted rest between any two shifts."""
    issues = []
    ns = _abs_hours(new_shift["date"], new_shift["start"], date_index)
    ne = _abs_hours(new_shift["date"], new_shift["end"], date_index)

    for s in doctor_shifts:
        ss = _abs_hours(s["date"], s["start"], date_index)
        se = _abs_hours(s["date"], s["end"], date_index)

        if not (ne <= ss or ns >= se):
            issues.append(f"Shift overlap with {s['code']}")
            continue

        if ns > se and (ns - se) < MIN_REST_HOURS:
            issues.append(
                f"Only {ns - se}h rest after {s['code']} (need {MIN_REST_HOURS}h)"
            )
        if ss > ne and (ss - ne) < MIN_REST_HOURS:
            issues.append(
                f"Only {ss - ne}h rest before {s['code']} (need {MIN_REST_HOURS}h)"
            )
    return issues


def _c4_weekly_limit(
        doctor_shifts: list[dict],
        new_shift: dict,
        weekly_limit: Optional[int],
        start_monday: pd.Timestamp,
) -> list[str]:
    """Weekly hour limit."""
    if weekly_limit is None:
        return []
    new_week = _date_to_week(new_shift["date"], start_monday)
    week_hours = sum(
        s["hours"]
        for s in doctor_shifts
        if _date_to_week(s["date"], start_monday) == new_week
    )
    total = week_hours + new_shift["hours"]
    if total > weekly_limit:
        return [f"Weekly limit exceeded: {total}h > {weekly_limit}h"]
    return []


def _c5_mentor_present(
        candidate_role: str,
        needs_mentor: bool,
        specialist_names_on_shift: list[str],
        source: str,
) -> list[str]:
    """Trainee/mentored resident needs a specialist co-assigned."""
    if not needs_mentor:
        return []
    if source in ("candidate", "reserve"):
        return []
    if not specialist_names_on_shift:
        return [
            f"Role '{candidate_role}' requires a mentor (specialist) "
            "on the same shift, but none is currently scheduled"
        ]
    return []


def _c6_consecutive_nights(
        doctor_shifts: list[dict],
        new_shift: dict,
        date_index: dict[str, int],
) -> list[str]:
    """Max 2 consecutive night shifts"""
    if not new_shift.get("is_night"):
        return []

    new_idx = date_index.get(new_shift["date"], 0)
    night_by_day: dict[int, int] = {}
    for s in doctor_shifts:
        if s["is_night"]:
            idx = date_index.get(s["date"], -999)
            night_by_day[idx] = night_by_day.get(idx, 0) + 1
    night_by_day[new_idx] = night_by_day.get(new_idx, 0) + 1

    for start in (new_idx - 2, new_idx - 1, new_idx):
        total = sum(night_by_day.get(d, 0) for d in (start, start + 1, start + 2))
        if total > MAX_CONSECUTIVE_NIGHTS:
            return [
                f"Would create {total} consecutive night shifts "
                f"(max {MAX_CONSECUTIVE_NIGHTS})"
            ]
    return []


def _c7_day_off_after_night(
        doctor_shifts: list[dict],
        new_shift: dict,
        date_index: dict[str, int],
) -> list[str]:
    """Day off after a night shift"""
    issues = []
    new_idx = date_index.get(new_shift["date"], 0)
    new_is_night = new_shift.get("is_night", False)

    for s in doctor_shifts:
        s_idx = date_index.get(s["date"], -999)

        if s["is_night"] and s_idx == new_idx - 1:
            issues.append(
                f"Day-off required after night shift {s['code']} on {s['date']}"
            )

        if new_is_night and new_idx == s_idx - 1:
            issues.append(
                f"Night shift on {new_shift['date']} requires the next day off, "
                f"but {s['code']} is already scheduled on {s['date']}"
            )

    return issues


def _c8_max_working_days(
        doctor_shifts: list[dict],
        new_shift: dict,
        start_monday: pd.Timestamp,
) -> list[str]:
    """Max 6 working days per week"""
    new_week = _date_to_week(new_shift["date"], start_monday)
    days_this_week = {
        s["date"]
        for s in doctor_shifts
        if _date_to_week(s["date"], start_monday) == new_week
    }
    days_this_week.add(new_shift["date"])
    if len(days_this_week) > MAX_WORKING_DAYS_PER_WEEK:
        return [
            f"Would work {len(days_this_week)} days this week "
            f"(max {MAX_WORKING_DAYS_PER_WEEK})"
        ]
    return []


def _c10_unavailability(
        doctor_id,
        new_shift: dict,
        unavail_day: pd.DataFrame,
        unavail_shift: pd.DataFrame,
) -> list[str]:
    """Doctor marked unavailable for this day or specific shift"""
    issues = []
    if doctor_id is None:
        return issues

    if not unavail_day.empty:
        if not unavail_day[
            (unavail_day["doctor_id"] == doctor_id) &
            (unavail_day["date"] == new_shift["date"])
        ].empty:
            issues.append(
                f"Doctor is marked unavailable (day-off) on {new_shift['date']}"
            )

    if not unavail_shift.empty:
        if not unavail_shift[
            (unavail_shift["doctor_id"] == doctor_id) &
            (unavail_shift["code"] == new_shift["code"])
        ].empty:
            issues.append(
                f"Doctor is marked unavailable for shift {new_shift['code']}"
            )

    return issues



# AGGREGATE CONSTRAINT CHECK

def _check_all_constraints(
        doctor_shifts: list[dict],
        new_shift: dict,
        date_index: dict[str, int],
        start_monday: pd.Timestamp,
        weekly_limit: Optional[int],
        candidate_role: str,
        needs_mentor: bool,
        specialist_names_on_shift: list[str],
        source: str,
        doctor_id,
        unavail_day: pd.DataFrame,
        unavail_shift: pd.DataFrame,
) -> tuple[bool, list[str]]:
    """
    Runs all hard-constraint checks
    """
    warnings: list[str] = []

    warnings += _c3_min_rest(doctor_shifts, new_shift, date_index)
    warnings += _c4_weekly_limit(doctor_shifts, new_shift, weekly_limit, start_monday)
    warnings += _c5_mentor_present(candidate_role, needs_mentor,
                                   specialist_names_on_shift, source)
    warnings += _c6_consecutive_nights(doctor_shifts, new_shift, date_index)
    warnings += _c7_day_off_after_night(doctor_shifts, new_shift, date_index)
    warnings += _c8_max_working_days(doctor_shifts, new_shift, start_monday)
    warnings += _c10_unavailability(doctor_id, new_shift, unavail_day, unavail_shift)

    return len(warnings) == 0, warnings


# SCORING

def _score_candidate(
        rest_ok: bool,
        skill_ok: bool,
        extra_cost: int,
        hard_shifts: int,
        workload_pct: float,
        source: str,
        max_cost: int,
) -> tuple[float, dict]:
    """Lower score = better candidate."""
    if not rest_ok:
        return 1e9 + extra_cost, {"reason": "constraint_violation"}
    if not skill_ok:
        return 5e8 + extra_cost, {"reason": "skill_mismatch"}

    cost_norm = (extra_cost / max_cost * 100) if max_cost > 0 else 0
    source_penalty = {"schedule": 0, "reserve": 10, "candidate": 50}[source]
    fairness = hard_shifts * 2 + workload_pct * 0.3

    return cost_norm + source_penalty + fairness, {
        "cost_norm": round(cost_norm, 2),
        "source_penalty": source_penalty,
        "fairness": round(fairness, 2),
    }


# MAIN FUNCTION

def find_replacement(
        results: dict,
        sick_doctor: str,
        shift_code: str,
        reserve_doctors_df: Optional[pd.DataFrame] = None,
        candidates_df: Optional[pd.DataFrame] = None,
        max_new_hires: int = MAX_NEW_HIRES_DEFAULT,
        already_hired_count: int = 0,
) -> dict:
    """
    Parameters:
    results             : output of the model run
    sick_doctor         : full name of the absent doctor
    shift_code          : ShiftCode value from schedule_df
    reserve_doctors_df  : retired/reserve doctors DataFrame
    candidates_df       : hire-candidate DataFrame
    max_new_hires       : max new hires allowed this month
    already_hired_count : candidates already hired this month
    """

    schedule_df: pd.DataFrame = results["schedule_df"]
    stats_df: pd.DataFrame = results["stats_df"]
    shifts_df: pd.DataFrame = results["shifts"]
    doctors_df: pd.DataFrame = results["doctors"]
    weekly_limits: dict = results["weekly_limits"]
    unavail_day: pd.DataFrame = results.get("unavail_day", pd.DataFrame())
    unavail_shift: pd.DataFrame = results.get("unavail_shift", pd.DataFrame())

    shift_rows = schedule_df[schedule_df["ShiftCode"] == shift_code]
    if shift_rows.empty:
        return {
            "shift_code": shift_code,
            "sick_doctor": sick_doctor,
            "shift_info": {},
            "candidates": [],
            "recommendation": None,
            "fallback_message": f"Shift '{shift_code}' not found in schedule.",
        }

    shift_row = shift_rows.iloc[0]
    is_night_shift = "_N_" in shift_code or int(shift_row["Hours"]) == 24

    shift_info = {
        "date": str(shift_row["Date"])[:10],
        "dept": shift_row["Dept"],
        "start": int(shift_row["StartHour"]),
        "end": int(shift_row["EndHour"]),
        "hours": int(shift_row["Hours"]),
        "code": shift_code,
        "is_night": is_night_shift,
    }

    shift_meta = shifts_df[shifts_df["code"] == shift_code]
    shift_info["required_skill"] = (
        shift_meta["required_skill"].iloc[0]
        if not shift_meta.empty and "required_skill" in shift_meta.columns
        else ""
    )
    required_skill: str = shift_info["required_skill"]

    date_index = _build_date_index(schedule_df)
    schedule_idx = _build_schedule_index(schedule_df)
    start_monday = _get_start_monday(schedule_df)
    specialists_on_shift = _specialists_on_shift(shift_code, schedule_df, doctors_df)

    max_cost_estimate = max(
        (int(r.get("hourly_cost", 100)) * shift_info["hours"] for _, r in doctors_df.iterrows()),
        default=1000,
    )

    replacement_candidates: list[dict] = []

    # SOURCE 1 – Doctors already in the schedule
    for name in (n for n in schedule_idx if n != sick_doctor):
        dr = doctors_df[doctors_df["name"] == name]
        if dr.empty:
            continue
        dr = dr.iloc[0]
        doc_id = dr["id"]

        if any(s["date"] == shift_info["date"] for s in schedule_idx.get(name, [])):
            continue

        if shift_info["hours"] == 24 and int(dr.get("twentyfour_allowed", 1)) == 0:
            continue
        if is_night_shift and int(dr.get("night_allowed", 1)) == 0:
            continue

        skill_list = (
            dr["skill_list"]
            if isinstance(dr.get("skill_list"), list)
            else str(dr.get("skills", "")).split(";")
        )
        skill_ok = required_skill == "" or required_skill in skill_list

        weekly_limit = _get_weekly_limit(doc_id, shift_info["date"], weekly_limits, start_monday)

        rest_ok, warnings = _check_all_constraints(
            doctor_shifts=schedule_idx.get(name, []),
            new_shift=shift_info,
            date_index=date_index,
            start_monday=start_monday,
            weekly_limit=weekly_limit,
            candidate_role=dr.get("role", ""),
            needs_mentor=_doctor_needs_mentor(dr),
            specialist_names_on_shift=specialists_on_shift,
            source="schedule",
            doctor_id=doc_id,
            unavail_day=unavail_day,
            unavail_shift=unavail_shift,
        )

        stats_row = stats_df[stats_df["Doctor"] == name]
        hard_shifts_count = int(stats_row["HardCount"].iloc[0]) if not stats_row.empty else 0
        workload_pct = float(stats_row["UtilizationPct"].iloc[0]) if not stats_row.empty else 0.0
        extra_cost = int(dr.get("hourly_cost", 0)) * shift_info["hours"]

        score, comps = _score_candidate(
            rest_ok=rest_ok, skill_ok=skill_ok,
            extra_cost=extra_cost, hard_shifts=hard_shifts_count,
            workload_pct=workload_pct, source="schedule",
            max_cost=max_cost_estimate,
        )

        replacement_candidates.append({
            "name": name, "source": "schedule",
            "role": dr.get("role", ""),
            "rest_ok": rest_ok, "skill_ok": skill_ok,
            "extra_cost": extra_cost, "workload_pct": workload_pct,
            "hard_shifts": hard_shifts_count,
            "score": score, "warnings": warnings,
            "score_components": comps,
        })

    # SOURCE 2 – Reserve (for example retired) doctors
    if reserve_doctors_df is not None:
        for _, res in reserve_doctors_df.iterrows():
            if shift_info["hours"] == 24 and int(res.get("twentyfour_allowed", 1)) == 0:
                continue
            if is_night_shift and int(res.get("night_allowed", 1)) == 0:
                continue

            skill_list = (
                res["skill_list"]
                if isinstance(res.get("skill_list"), list)
                else str(res.get("skills", "")).split(";")
            )
            skill_ok = required_skill == "" or required_skill in skill_list

            res_id = res.get("id")
            unavail_w = _c10_unavailability(res_id, shift_info, unavail_day, unavail_shift)
            rest_ok = len(unavail_w) == 0

            extra_cost = int(res.get("hourly_cost", 0)) * shift_info["hours"]
            score, comps = _score_candidate(
                rest_ok=rest_ok, skill_ok=skill_ok,
                extra_cost=extra_cost, hard_shifts=0,
                workload_pct=0.0, source="reserve",
                max_cost=max_cost_estimate,
            )

            replacement_candidates.append({
                "name": res["name"], "source": "reserve",
                "role": res.get("role", ""),
                "rest_ok": rest_ok, "skill_ok": skill_ok,
                "extra_cost": extra_cost, "workload_pct": 0.0,
                "hard_shifts": 0,
                "score": score,
                "warnings": ["Reserve doctor — availability needs confirmation"] + unavail_w,
                "score_components": comps,
            })

    # SOURCE 3 – Hire candidates
    hires_remaining = max_new_hires - already_hired_count

    if candidates_df is not None and hires_remaining > 0:
        for _, cand in candidates_df.iterrows():
            if shift_info["hours"] == 24 and int(cand.get("twentyfour_allowed", 1)) == 0:
                continue
            if is_night_shift and int(cand.get("night_allowed", 1)) == 0:
                continue

            skill_list = (
                cand["skill_list"]
                if isinstance(cand.get("skill_list"), list)
                else str(cand.get("skills", "")).split(";")
            )
            skill_ok = required_skill == "" or required_skill in skill_list

            cand_id = cand.get("id")
            constraint_w = (
                    _c10_unavailability(cand_id, shift_info, unavail_day, unavail_shift) +
                    _c5_mentor_present(
                        cand.get("role", ""),
                        _doctor_needs_mentor(cand),
                        specialists_on_shift,
                        source="candidate",
                    )
            )
            rest_ok = len(constraint_w) == 0

            extra_cost = int(cand.get("hourly_cost", 0)) * shift_info["hours"]
            score, comps = _score_candidate(
                rest_ok=rest_ok, skill_ok=skill_ok,
                extra_cost=extra_cost, hard_shifts=0,
                workload_pct=0.0, source="candidate",
                max_cost=max_cost_estimate,
            )

            replacement_candidates.append({
                "name": cand["name"], "source": "candidate",
                "role": cand.get("role", ""),
                "rest_ok": rest_ok, "skill_ok": skill_ok,
                "extra_cost": extra_cost, "workload_pct": 0.0,
                "hard_shifts": 0,
                "score": score,
                "warnings": constraint_w + [
                    f"New hire — cost increase ~{extra_cost} PLN",
                    f"Hire slots remaining: {hires_remaining}",
                ],
                "score_components": comps,
            })

    replacement_candidates.sort(
        key=lambda c: (not (c["rest_ok"] and c["skill_ok"]), c["score"])
    )

    valid = [c for c in replacement_candidates if c["rest_ok"] and c["skill_ok"]]
    best = valid[0] if valid else None

    if not replacement_candidates:
        fallback = (
            "No candidates available whatsoever. "
            "Contact the on-call coordinator immediately."
        )
    elif best is None:
        fallback = (
            "No candidate satisfies ALL constraints simultaneously. "
            "Consider reserve staff or discuss constraint relaxation with management."
        )
    else:
        fallback = None

    return {
        "shift_code": shift_code,
        "sick_doctor": sick_doctor,
        "shift_info": shift_info,
        "candidates": replacement_candidates,
        "recommendation": best,
        "fallback_message": fallback,
    }


def precompute_replacements(
        results: dict,
        reserve_doctors_df: Optional[pd.DataFrame] = None,
        candidates_df: Optional[pd.DataFrame] = None,
        max_new_hires: int = MAX_NEW_HIRES_DEFAULT,
) -> dict[str, dict]:
    """
    Compute replacements for every assigned shift in the schedule.
    Tracks new hires across calls to respect the global limit.
    """
    schedule_df = results["schedule_df"]
    assigned = (
        schedule_df.dropna(subset=["Doctor"])[["ShiftCode", "Doctor"]]
        .drop_duplicates()
    )

    output: dict[str, dict] = {}
    already_hired = 0
    used_candidates: set[str] = set()

    for _, row in assigned.iterrows():
        res = find_replacement(
            results=results,
            sick_doctor=row["Doctor"],
            shift_code=row["ShiftCode"],
            reserve_doctors_df=reserve_doctors_df,
            candidates_df=candidates_df,
            max_new_hires=max_new_hires,
            already_hired_count=already_hired,
        )

        best = res.get("recommendation")
        if best and best["source"] == "candidate" and best["name"] not in used_candidates:
            used_candidates.add(best["name"])
            already_hired += 1

        output[f"{row['ShiftCode']}__{row['Doctor']}"] = res

    return output


# REPORT

def format_replacement_report(result: dict) -> str:
    info = result["shift_info"]
    lines = [
        "=" * 65,
        f"REPLACEMENT: {result['sick_doctor']}  ->  {result['shift_code']}",
        (
            f"Shift : {info.get('date')} | {info.get('dept')} | "
            f"{info.get('start')}:00-{info.get('end')}:00 "
            f"({info.get('hours')}h) | skill: {info.get('required_skill', '-')}"
        ),
        "-" * 65,
    ]

    if result["fallback_message"]:
        lines.append(result["fallback_message"])

    rec = result["recommendation"]
    if rec:
        lines.append(
            f"RECOMMENDATION: {rec['name']} [{rec['source'].upper()}] | "
            f"+{rec['extra_cost']} PLN | hard={rec['hard_shifts']} | "
            f"load={rec['workload_pct']}%"
        )
        for w in rec["warnings"]:
            lines.append(f"   !  {w}")
    else:
        lines.append("No valid recommendation found.")

    lines += ["", "All candidates (top 10):"]
    for i, c in enumerate(result["candidates"][:10], 1):
        ok = "OK" if (c["rest_ok"] and c["skill_ok"]) else "!!"
        lines.append(
            f"  {i:2}. [{ok}] {c['name']:<25} [{c['source']:10}]"
            f" skill={'Y' if c['skill_ok'] else 'N'}"
            f" rest={'Y' if c['rest_ok'] else 'N'}"
            f" cost={c['extra_cost']:>6} PLN"
            f" hard={c['hard_shifts']}"
        )
        for w in c["warnings"]:
            lines.append(f"          ! {w}")

    lines.append("=" * 65)
    return "\n".join(lines)


if __name__ == "__main__":
    print("Module loaded. Usage: from emergency_replacement import find_replacement")