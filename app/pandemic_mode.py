"""
Pandemic Mode for the Hospital Rostering Optimizer.

Definition of contact:
  - Sharing the same shift in WARD or ICU = epidemiological contact
  - CLINIC excluded by default (separate offices)
  - Handover periods: optional handover_window_hours (default: 0)

Emergency replacement in pandemic mode:
  - Replacement only from the same bubble (contact departments)
  - External reserve staff allowed with a warning
  - Additional physicians may be hired (only during schedule construction)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Any

import pandas as pd
from ortools.sat.python import cp_model


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

@dataclass
class PandemicConfig:
    """
    Pandemic mode configuration.

    num_bubbles : int
        Number of isolated bubbles.
        Each bubble should be able to independently cover at least one shift.

    contact_depts : list[str]
        Departments where sharing a shift counts as contact.
        CLINIC excluded (separate offices).

    handover_window_hours : int
        Shift handover hours considered as contact. Default: 0.

    min_specialists_per_bubble : int
        Minimum number of specialists required in each bubble.

    allow_cross_bubble_reserve : bool
        Whether external reserve staff may replace physicians in any bubble.
    """
    num_bubbles: int = 3
    contact_depts: list[str] = field(default_factory=lambda: ["WARD", "ICU"])
    handover_window_hours: int = 0
    min_specialists_per_bubble: int = 1
    allow_cross_bubble_reserve: bool = True



def prepare_pandemic_bubbles(
    doctors: pd.DataFrame,
    candidates: pd.DataFrame,
    config: PandemicConfig,
    seed: int = 42,
) -> dict[int, int]:
    """
    Skill-aware greedy bubble assignment.
    """
    import random
    rng = random.Random(seed)

    B = config.num_bubbles
    all_df = doctors.copy()

    if "skills" not in all_df.columns:
        all_df["skills"] = ""

    all_df["skills"] = all_df["skills"].fillna("").astype(str)
    all_df["role"] = all_df["role"].fillna("").astype(str)

    specialist_roles = {"specialist", "icu_specialist"}

    assigned: set[int] = set()
    assignment: dict[int, int] = {}

    bubble_stats = {
        b: {
            "size": 0,
            "specialists": 0,
            "icu": 0,
            "night": 0,
            "twentyfour": 0,
        }
        for b in range(B)
    }

    def has_skill(row: pd.Series, skill_name: str) -> bool:
        skills = str(row.get("skills", ""))
        return skill_name.lower() in skills.lower()

    def choose_bubble(priority_key: str) -> int:
        """
        Selects the bubble with the smallest number of physicians having
        the given competency, and in case of a tie, the smallest bubble size.
        """
        return min(
            range(B),
            key=lambda b: (
                bubble_stats[b][priority_key],
                bubble_stats[b]["size"],
                b,
            ),
        )

    def add_doctor(row: pd.Series, priority_key: str) -> None:
        d = int(row["id"])
        if d in assigned:
            return

        b = choose_bubble(priority_key)
        assignment[d] = b
        assigned.add(d)

        bubble_stats[b]["size"] += 1

        role = str(row.get("role", ""))
        if role in specialist_roles:
            bubble_stats[b]["specialists"] += 1

        if has_skill(row, "ICU"):
            bubble_stats[b]["icu"] += 1

        if int(row.get("night_allowed", 0) or 0) == 1:
            bubble_stats[b]["night"] += 1

        if int(row.get("twentyfour_allowed", 0) or 0) == 1:
            bubble_stats[b]["twentyfour"] += 1

    def shuffled_rows(df: pd.DataFrame) -> list[pd.Series]:
        rows = [row for _, row in df.iterrows()]
        rng.shuffle(rows)
        return rows

    # 1. ICU specialists
    icu_specs = all_df[
        (all_df["role"].isin(specialist_roles))
        & (all_df.apply(lambda r: has_skill(r, "ICU"), axis=1))
    ]

    for row in shuffled_rows(icu_specs):
        add_doctor(row, "icu")

    # 2. Other specialists
    other_specs = all_df[
        (all_df["role"].isin(specialist_roles))
        & (~all_df["id"].isin(assigned))
    ]

    for row in shuffled_rows(other_specs):
        add_doctor(row, "specialists")

    # 3. Night-capable doctors
    _night_col = (
        all_df["night_allowed"]
        if "night_allowed" in all_df.columns
        else pd.Series(0, index=all_df.index)
    )
    night_docs = all_df[
        (_night_col == 1)
        & (~all_df["id"].isin(assigned))
        ]

    for row in shuffled_rows(night_docs):
        add_doctor(row, "night")

    # 4. 24h-capable doctors
    _twentyfour_col = (
        all_df["twentyfour_allowed"]
        if "twentyfour_allowed" in all_df.columns
        else pd.Series(0, index=all_df.index)
    )
    twentyfour_docs = all_df[
        (_twentyfour_col == 1)
        & (~all_df["id"].isin(assigned))
        ]

    for row in shuffled_rows(twentyfour_docs):
        add_doctor(row, "twentyfour")

    # 5. Everyone else
    others = all_df[~all_df["id"].isin(assigned)]

    for row in shuffled_rows(others):
        add_doctor(row, "size")

    print("\n[PandemicMode] === Bubble assignment before build_model ===")

    print(f"  Physicians: {len(doctors)}, Candidates: {len(candidates)}")
    print(f"  Number of bubbles: {B}")
    print(f"  Contact departments: {config.contact_depts}")
    print("  CLINIC is not treated as epidemiological contact.")

    for b in range(B):
        s = bubble_stats[b]
        print(
            f"  Bubble {b}: "
            f"{s['size']} physicians, "
            f"{s['specialists']} specialists, "
            f"{s['icu']} ICU-capable, "
            f"{s['night']} night-eligible, "
            f"{s['twentyfour']} 24h-eligible"
        )

    print()

    return assignment


def assign_hired_candidates_to_bubbles(
    hired_candidate_ids: list[int],
    bubble_assignment: dict[int, int],
    doctors: pd.DataFrame,
    candidates: pd.DataFrame,
    config: PandemicConfig,
) -> dict[int, int]:
    updated = dict(bubble_assignment)

    bubble_sizes = {
        b: sum(1 for bb in updated.values() if bb == b)
        for b in range(config.num_bubbles)
    }

    for cid in hired_candidate_ids:
        if cid in updated:
            continue

        target_bubble = min(
            range(config.num_bubbles),
            key=lambda b: (bubble_sizes[b], b)
        )

        updated[cid] = target_bubble
        bubble_sizes[target_bubble] += 1

    return updated



def validate_bubbles(
    doctors: pd.DataFrame,
    candidates: pd.DataFrame,
    shifts: pd.DataFrame,
    bubble_assignment: dict[int, int],
    config: PandemicConfig,
) -> dict:
    """
    Pre-solver bubble diagnostics.

    Returns a dictionary containing:
    - validation_df: table of competencies for each bubble
    - issues: list of detected issues (severity, bubble, message)
    - is_safe: bool indicating whether all bubbles satisfy the minimum requirements
    """
    all_df = doctors.copy()

    if "skills" not in all_df.columns:
        all_df["skills"] = ""

    all_df["skills"] = all_df["skills"].fillna("").astype(str)
    all_df["role"] = all_df["role"].fillna("").astype(str)

    specialist_roles = {"specialist", "icu_specialist"}

    def has_icu_skill(skills: Any) -> bool:
        return "icu" in str(skills).lower()

    rows = []
    issues = []

    for b in range(config.num_bubbles):
        ids = [d for d, bb in bubble_assignment.items() if bb == b]
        bdf = all_df[all_df["id"].isin(ids)]

        size = len(bdf)
        specialists = int(bdf["role"].isin(specialist_roles).sum())
        icu_capable = int(bdf["skills"].apply(has_icu_skill).sum())

        night_capable = (
            int((bdf["night_allowed"] == 1).sum())
            if "night_allowed" in bdf.columns else 0
        )
        twentyfour_capable = (
            int((bdf["twentyfour_allowed"] == 1).sum())
            if "twentyfour_allowed" in bdf.columns else 0
        )

        bubble_ok = True
        bubble_issues = []

        if specialists < config.min_specialists_per_bubble:
            msg = (
                f"Bubble {b}: only {specialists} specialists "
                f"(required minimum: {config.min_specialists_per_bubble})"
            )
            issues.append({"severity": "ERROR", "bubble": b, "message": msg})
            warnings.warn(f"[PandemicMode] {msg}")
            bubble_ok = False
            bubble_issues.append("missing specialists")

        if "ICU" in set(config.contact_depts) and icu_capable == 0:
            msg = f"Bubble {b}: no physician with ICU competency"
            issues.append({"severity": "ERROR", "bubble": b, "message": msg})
            warnings.warn(f"[PandemicMode] {msg}")
            bubble_ok = False
            bubble_issues.append("missing ICU competency")

        if night_capable == 0:
            msg = f"Bubble {b}: no physician eligible for night shifts"
            issues.append({"severity": "WARNING", "bubble": b, "message": msg})
            warnings.warn(f"[PandemicMode] {msg}")
            bubble_issues.append("missing night-shift eligibility")

        if twentyfour_capable == 0:
            msg = f"Bubble {b}: no physician eligible for 24h shifts"
            issues.append({"severity": "INFO", "bubble": b, "message": msg})
            bubble_issues.append("missing 24h-shift eligibility")

        row = {
            "Bubble": b,
            "Doctors": size,
            "Specialists": specialists,
            "ICU capable": icu_capable,
            "Night capable": night_capable,
            "24h capable": twentyfour_capable,
            "Status": "OK" if bubble_ok else "PROBLEM",
            "Issues": ", ".join(bubble_issues) if bubble_issues else "—",
        }
        rows.append(row)

    validation_df = pd.DataFrame(rows)
    is_safe = not any(i["severity"] == "ERROR" for i in issues)

    print("\n[PandemicMode] === Bubble validation ===")
    print(validation_df.to_string(index=False))
    print()

    return {
        "validation_df": validation_df,
        "issues": issues,
        "is_safe": is_safe,
    }

def add_pandemic_constraints(
    model: cp_model.CpModel,
    x: dict,
    D_all: list[int],
    S: list[int],
    shifts: pd.DataFrame,
    shift_idx: dict[int, int],
    config: PandemicConfig,
    bubble_assignment: dict[int, int],
    id_to_role: dict[int, str],
) -> dict:
    """
    Adds bubble constraints to the CP-SAT model.

    Complexity: O(D^2 / 2 * S_contact).
    """
    contact_depts  = set(config.contact_depts)
    contact_shifts = [
        s for s in S
        if shifts.loc[shift_idx[s], "dept"] in contact_depts
    ]

    print(f"[PandemicMode] Adding HC-P1 constraints:")
    print(f"  Contact departments: {contact_depts}")
    print(f"  Contact shifts: {len(contact_shifts)} / {len(S)}")

    crossbubble_violations = []

    BIG_PANDEMIC_PENALTY = 1_000_000

    constraint_count = 0

    missing_from_bubbles = [d for d in D_all if d not in bubble_assignment]
    if missing_from_bubbles:
        warnings.warn(
            f"[PandemicMode] {len(missing_from_bubbles)} physicians do not have an assigned bubble "
            f"and will be skipped in pandemic constraints: {missing_from_bubbles}"
        )

    for s in contact_shifts:

        for i, d1 in enumerate(D_all):
            if d1 not in bubble_assignment:
                continue

            for d2 in D_all[i + 1:]:
                if d2 not in bubble_assignment:
                    continue

                if bubble_assignment[d1] != bubble_assignment[d2]:
                    viol = model.NewBoolVar(
                        f"crossbubble_{d1}_{d2}_{s}"
                    )

                    model.Add(
                        x[(d1, s)] + x[(d2, s)] <= 1 + viol
                    )

                    crossbubble_violations.append(viol)
                    constraint_count += 1

    print(f"  Added {constraint_count} pandemic constraints.")
    print(f"  Cross-bubble violations are allowed only in emergency situations.")

    return {
        "crossbubble_violations": crossbubble_violations,
        "penalty": BIG_PANDEMIC_PENALTY,
    }


def get_pandemic_results(
    bubble_assignment: dict[int, int],
    config: PandemicConfig,
    id_to_name: dict[int, str],
    id_to_role: dict[int, str],
) -> dict:
    """
    Packages bubble-related data into the results dictionary.
    """
    B = config.num_bubbles
    bubbles: dict[int, dict] = {}

    for b in range(B):
        members = [d for d, bb in bubble_assignment.items() if bb == b]
        bubbles[b] = {
            "doctors":    members,
            "names":      [id_to_name.get(d, str(d)) for d in members],
            "roles":      [id_to_role.get(d, "") for d in members],
            "size":       len(members),
            "specialists": sum(
                1 for d in members
                if id_to_role.get(d, "") in {"specialist", "icu_specialist"}
            ),
        }

    return {
        "bubble_assignment": bubble_assignment,
        "bubbles":           bubbles,
        "config":            config,
        "num_bubbles":       B,
        "contact_depts":     config.contact_depts,
    }


# ---------------------------------------------------------------------------
# QUARANTINE SIMULATION
# ---------------------------------------------------------------------------

def simulate_quarantine(
    results: dict,
    quarantine_bubble: int,
    bubble_assignment: dict[int, int],
    id_to_name: dict[int, str],
    config: PandemicConfig,
) -> dict:
    """
    Simulates the quarantine of a single bubble
    (post-hoc analysis without re-solving the model).

    Removes physicians from the selected bubble from the schedule
    and reports:
    - which shifts become uncovered
    - what percentage of coverage is lost
    - severity level (LOW / MEDIUM / HIGH)
    - recommended actions
    """
    schedule_df = results["schedule_df"].copy()

    quarantined_ids   = {d for d, b in bubble_assignment.items() if b == quarantine_bubble}
    quarantined_names = {id_to_name[d] for d in quarantined_ids if d in id_to_name}

    bubble_sizes = {
        b: sum(1 for bb in bubble_assignment.values() if bb == b)
        for b in range(config.num_bubbles)
    }

    affected = schedule_df[
        schedule_df["Doctor"].isin(quarantined_names) &
        (schedule_df["Missing"] == 0)
    ].copy()

    total_covered_slots = int(
        schedule_df[schedule_df["Missing"] == 0][["Date", "ShiftCode"]]
        .drop_duplicates()
        .shape[0]
    )
    newly_uncovered_slots = int(
        affected[["Date", "ShiftCode"]]
        .drop_duplicates()
        .shape[0]
    ) if not affected.empty else 0

    coverage_impact_pct = round(
        100 * newly_uncovered_slots / max(1, total_covered_slots), 1
    )
    days_affected       = int(affected["Date"].nunique()) if not affected.empty else 0

    dept_impact = {}
    if not affected.empty and "Dept" in affected.columns:
        dept_impact = affected.groupby("Dept").size().to_dict()

    if coverage_impact_pct < 20:
        severity = "LOW"
        action = "The schedule can withstand the quarantine — redistribute overtime."
    elif coverage_impact_pct < 40:
        severity = "MEDIUM"
        action   = "Activate reserve staff and consider emergency cross-bubble coverage."
    else:
        severity = "HIGH"
        action   = (
            "Critical staffing shortage. Escalate immediately. "
            "Consider temporarily merging the two remaining bubbles."
        )

    return {
        "quarantine_bubble":     quarantine_bubble,
        "quarantined_doctors":   sorted(quarantined_names),
        "uncovered_shifts":      affected,
        "coverage_impact_pct":   coverage_impact_pct,
        "days_affected":         days_affected,
        "bubble_sizes":          bubble_sizes,
        "dept_impact":           dept_impact,
        "severity":              severity,
        "recommendation": (
            f"[{severity}] Quarantine of bubble {quarantine_bubble} "
            f"({len(quarantined_names)} physicians) removes "
            f"{newly_uncovered_slots} unique shifts "
            f"({coverage_impact_pct}% of covered slots). "
            f"Affected departments: {dept_impact}. {action}"
        ),
    }


def simulate_quarantine_with_reserve(
    results: dict,
    quarantine_bubble: int,
    bubble_assignment: dict[int, int],
    id_to_name: dict[int, str],
    config: PandemicConfig,
    reserve_doctors_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Extended quarantine simulation with activation of external reserve staff.

    Returns:
    - uncovered_after_quarantine: shifts lost after quarantine
    - covered_by_reserve: shifts that reserve staff can cover
    - residual_gap: shifts that still remain uncovered
    - reserve_assignments: mapping of reserve staff to covered shifts
    """
    base_result = simulate_quarantine(
        results=results,
        quarantine_bubble=quarantine_bubble,
        bubble_assignment=bubble_assignment,
        id_to_name=id_to_name,
        config=config,
    )

    uncovered_shifts = base_result["uncovered_shifts"]

    if reserve_doctors_df is None or reserve_doctors_df.empty:
        return {
            **base_result,
            "reserve_available": False,
            "covered_by_reserve": 0,
            "residual_gap": len(uncovered_shifts),
            "reserve_assignments": pd.DataFrame(),
            "residual_uncovered": uncovered_shifts,
        }

    reserve_df = reserve_doctors_df.copy()
    if "skill_list" not in reserve_df.columns:
        reserve_df["skill_list"] = reserve_df.get("skills", "").apply(
            lambda x: x.split(";") if isinstance(x, str) else []
        )

    reserve_busy = {row["name"]: set() for _, row in reserve_df.iterrows()}

    reserve_assignments = []
    residual_rows = []

    shifts_meta = results["shifts"]

    for _, shift in uncovered_shifts.iterrows():
        shift_date = str(shift["Date"])[:10]
        shift_code = shift["ShiftCode"]
        shift_hours = int(shift["Hours"])
        is_night = "_N_" in shift_code or shift_hours == 24
        is_24h = shift_hours == 24

        meta_row = shifts_meta[shifts_meta["code"] == shift_code]
        required_skill = (
            meta_row["required_skill"].iloc[0]
            if not meta_row.empty and "required_skill" in meta_row.columns
            else ""
        )

        assigned = False
        for _, res in reserve_df.iterrows():
            res_name = res["name"]

            if shift_date in reserve_busy[res_name]:
                continue

            skill_ok = (
                required_skill == ""
                or required_skill in (res["skill_list"] if isinstance(res["skill_list"], list)
                                      else str(res.get("skills", "")).split(";"))
            )
            if not skill_ok:
                continue

            if is_night and int(res.get("night_allowed", 1) or 0) == 0:
                continue
            if is_24h and int(res.get("twentyfour_allowed", 1) or 0) == 0:
                continue

            reserve_busy[res_name].add(shift_date)
            extra_cost = int(res.get("hourly_cost", 0)) * shift_hours

            reserve_assignments.append({
                "Date": shift_date,
                "ShiftCode": shift_code,
                "Dept": shift["Dept"],
                "Reserve doctor": res_name,
                "Cost (PLN)": extra_cost,
            })
            assigned = True
            break

        if not assigned:
            residual_rows.append(shift.to_dict())

    reserve_df_out = pd.DataFrame(reserve_assignments)
    residual_df = pd.DataFrame(residual_rows) if residual_rows else pd.DataFrame()

    total_reserve_cost = int(reserve_df_out["Cost (PLN)"].sum()) if not reserve_df_out.empty else 0

    return {
        **base_result,
        "reserve_available": True,
        "covered_by_reserve": len(reserve_df_out),
        "residual_gap": len(residual_df),
        "reserve_assignments": reserve_df_out,
        "residual_uncovered": residual_df,
        "reserve_total_cost": total_reserve_cost,
    }

# ---------------------------------------------------------------------------
# EMERGENCY REPLACEMENT DURING PANDEMIC MODE
# ---------------------------------------------------------------------------
def find_replacement_pandemic(
    results: dict,
    sick_doctor: str,
    shift_code: str,
    bubble_assignment: dict[int, int],
    id_to_name: dict[int, str],
    config: PandemicConfig,
    reserve_doctors_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Searches for a replacement physician in pandemic mode.

    Rules:
      1. Contact departments (WARD/ICU): only physicians from the same bubble
      2. Non-contact departments (CLINIC): physicians from any bubble allowed
      3. External reserve staff: always allowed, with a warning
      4. Hiring candidates: EXCLUDED
    """
    schedule_df = results["schedule_df"]
    stats_df = results["stats_df"]
    shifts_df = results["shifts"]
    doctors_df = results["doctors"]
    weekly_limits = results["weekly_limits"]

    name_to_id = {v: k for k, v in id_to_name.items()}
    sick_id = name_to_id.get(sick_doctor)
    sick_bubble = bubble_assignment.get(sick_id, -1)

    shift_rows = schedule_df[schedule_df["ShiftCode"] == shift_code]
    if shift_rows.empty:
        return {
            "shift_code": shift_code,
            "sick_doctor": sick_doctor,
            "shift_info": {},
            "candidates": [],
            "recommendation": None,
            "fallback_message": f"Shift '{shift_code}' not found in the schedule.",
            "pandemic_mode": True,
            "bubble_restrictions_applied": False,
        }

    shift_row = shift_rows.iloc[0]
    shift_info = {
        "date": str(shift_row["Date"])[:10],
        "dept": shift_row["Dept"],
        "start": int(shift_row["StartHour"]),
        "end": int(shift_row["EndHour"]),
        "hours": int(shift_row["Hours"]),
        "code": shift_code,
    }

    shift_meta = shifts_df[shifts_df["code"] == shift_code]
    shift_info["required_skill"] = (
        shift_meta["required_skill"].iloc[0]
        if not shift_meta.empty and "required_skill" in shift_meta.columns
        else ""
    )

    is_contact_dept = shift_info["dept"] in config.contact_depts
    required_skill = shift_info["required_skill"]

    max_cost_estimate = max(
        (row.get("hourly_cost", 100) * shift_info["hours"] for _, row in doctors_df.iterrows()),
        default=1000,
    )

    replacement_candidates = []

    schedule_df_copy = schedule_df.copy()
    schedule_df_copy["Date"] = pd.to_datetime(schedule_df_copy["Date"])

    for doc_name in stats_df["Doctor"].unique():
        if doc_name == sick_doctor or pd.isna(doc_name):
            continue

        doc_row = doctors_df[doctors_df["name"] == doc_name]
        if doc_row.empty:
            continue

        doc_row = doc_row.iloc[0]
        doc_id = doc_row["id"]
        doc_bubble = bubble_assignment.get(doc_id, -1)

        if is_contact_dept and doc_bubble != sick_bubble:
            continue

        doc_shifts_that_day = schedule_df_copy[
            (schedule_df_copy["Doctor"] == doc_name)
            & (schedule_df_copy["Date"].dt.strftime("%Y-%m-%d") == shift_info["date"])
        ]
        if not doc_shifts_that_day.empty:
            continue

        skill_list = (
            doc_row["skill_list"]
            if isinstance(doc_row.get("skill_list"), list)
            else str(doc_row.get("skills", "")).split(";")
        )
        skill_ok = required_skill == "" or required_skill in skill_list

        is_night = "_N_" in shift_code or shift_info["hours"] == 24
        is_24h = shift_info["hours"] == 24

        if is_night and int(doc_row.get("night_allowed", 1) or 0) == 0:
            continue
        if is_24h and int(doc_row.get("twentyfour_allowed", 1) or 0) == 0:
            continue

        stats_row = stats_df[stats_df["Doctor"] == doc_name]
        hard_shifts_count = (
            int(stats_row["HardCount"].iloc[0]) if not stats_row.empty else 0
        )
        workload_pct = (
            float(stats_row["UtilizationPct"].iloc[0]) if not stats_row.empty else 0.0
        )
        extra_cost = int(doc_row.get("hourly_cost", 0)) * shift_info["hours"]

        score = (
            (100 if not skill_ok else 0)
            + (50 if is_night and int(doc_row.get("night_allowed", 1) or 0) == 0 else 0)
            + (int(workload_pct) // 10)
            + (hard_shifts_count * 5)
        )

        replacement_candidates.append(
            {
                "name": doc_name,
                "source": "schedule",
                "role": doc_row.get("role", ""),
                "bubble": doc_bubble,
                "rest_ok": True,
                "skill_ok": skill_ok,
                "extra_cost": extra_cost,
                "workload_pct": workload_pct,
                "hard_shifts": hard_shifts_count,
                "score": score,
                "warnings": [],
                "score_components": {},
            }
        )

    if reserve_doctors_df is not None and config.allow_cross_bubble_reserve:
        for _, res in reserve_doctors_df.iterrows():
            skill_list = (
                res["skill_list"]
                if isinstance(res.get("skill_list"), list)
                else str(res.get("skills", "")).split(";")
            )
            skill_ok = required_skill == "" or required_skill in skill_list

            if "_N_" in shift_code or shift_info["hours"] == 24:
                if int(res.get("night_allowed", 1) or 0) == 0:
                    continue
            if shift_info["hours"] == 24 and int(res.get("twentyfour_allowed", 1) or 0) == 0:
                continue

            extra_cost = int(res.get("hourly_cost", 0)) * shift_info["hours"]
            score = (100 if not skill_ok else 0) + extra_cost // 100

            replacement_candidates.append(
                {
                    "name": res["name"],
                    "source": "reserve",
                    "role": res.get("role", ""),
                    "bubble": "external",
                    "rest_ok": True,
                    "skill_ok": skill_ok,
                    "extra_cost": extra_cost,
                    "workload_pct": 0.0,
                    "hard_shifts": 0,
                    "score": score,
                    "warnings": [
                        "EXTERNAL RESERVE — isolated from the bubble system",
                        "Availability confirmation required",
                    ],
                    "score_components": {},
                }
            )

    replacement_candidates.sort(
        key=lambda c: (not (c["rest_ok"] and c["skill_ok"]), c["score"])
    )

    valid = [c for c in replacement_candidates if c["rest_ok"] and c["skill_ok"]]
    best = valid[0] if valid else None

    fallback_message = None
    if not replacement_candidates:
        fallback_message = (
            f"No candidates available in bubble {sick_bubble} for shift {shift_code}. "
            "All physicians may be quarantined. Activate external reserve staff."
        )
    elif best is None:
        fallback_message = (
            f"Candidates in bubble {sick_bubble} do not satisfy "
            "the competency requirements. Reserve staff or rule changes are required."
        )

    return {
        "shift_code": shift_code,
        "sick_doctor": sick_doctor,
        "sick_bubble": sick_bubble,
        "shift_info": shift_info,
        "candidates": replacement_candidates,
        "recommendation": best,
        "fallback_message": fallback_message,
        "pandemic_mode": True,
        "bubble_restrictions_applied": is_contact_dept,
        "is_contact_dept": is_contact_dept,
    }


def bubble_contact_matrix(
    schedule_df: pd.DataFrame,
    bubble_assignment: dict[int, int],
    name_to_id: dict[str, int],
    contact_depts: list[str],
) -> pd.DataFrame:
    """
    Counts how many times pairs of bubbles shared a shift
    in contact departments.
    In a valid solution, all values should equal 0.
    """
    contact_df = schedule_df[
        schedule_df["Dept"].isin(contact_depts) &
        schedule_df["Doctor"].notna()
    ].copy()

    contact_df["DoctorID"] = contact_df["Doctor"].map(name_to_id)
    contact_df["Bubble"]   = contact_df["DoctorID"].map(bubble_assignment)

    contacts = []
    for (date, code), group in contact_df.groupby(["Date", "ShiftCode"]):
        bubbles = group["Bubble"].dropna().astype(int).unique().tolist()
        if len(bubbles) > 1:
            for i in range(len(bubbles)):
                for j in range(i + 1, len(bubbles)):
                    contacts.append({
                        "Date": date, "ShiftCode": code,
                        "Dept": group["Dept"].iloc[0],
                        "Bubble_A": min(bubbles[i], bubbles[j]),
                        "Bubble_B": max(bubbles[i], bubbles[j]),
                    })

    if not contacts:
        return pd.DataFrame(columns=["Bubble_A", "Bubble_B", "ContactCount", "Dept"])

    return (
        pd.DataFrame(contacts)
        .groupby(["Bubble_A", "Bubble_B", "Dept"])
        .size()
        .reset_index(name="ContactCount")
    )


def bubble_summary_df(
    bubble_assignment: dict[int, int],
    id_to_name: dict[int, str],
    id_to_role: dict[int, str],
    num_bubbles: int,
) -> pd.DataFrame:
    """DataFrame [Doctor, Role, Bubble] — for display in Streamlit."""
    rows = [
        {"Doctor": id_to_name.get(d, str(d)), "Role": id_to_role.get(d, ""), "Bubble": b}
        for d, b in bubble_assignment.items()
    ]
    return pd.DataFrame(rows).sort_values(["Bubble", "Role", "Doctor"])