import pandas as pd

def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def compute_hospital_efficiency_v2(results: dict) -> dict:
    schedule_df = results["schedule_df"]
    stats_df = results["stats_df"]
    shifts = results["shifts"]

    # --- 1. Coverage safety ---
    total_required = int(shifts["min_staff"].sum())
    missing = int(results.get("slack", 0))
    coverage_score = 1.0 if total_required == 0 else max(0.0, 1 - missing / total_required)

    # --- 2. Operational stability ---
    missing_by_shift = (
        schedule_df
        .groupby(["Date", "ShiftCode"])["Missing"]
        .sum()
    )
    understaff_ratio = (missing_by_shift > 0).mean() if len(missing_by_shift) > 0 else 0.0
    stability_score = 1.0 - understaff_ratio

    # --- 3. Staffing adequacy ---
    assigned_by_shift = (
        schedule_df[schedule_df["Missing"] == 0]
        .groupby("ShiftCode")
        .size()
    )

    staff_quality = (
        shifts.set_index("code")["regular_staff"]
        .to_frame()
        .join(assigned_by_shift.rename("Assigned"))
        .fillna(0)
    )
    staff_quality["ratio"] = (staff_quality["Assigned"] / staff_quality["regular_staff"]).clip(upper=1.0)
    staffing_score = float(staff_quality["ratio"].mean())

    # --- 4. Workforce sustainability ---
    num_docs = len(stats_df)
    overworked_ratio = (stats_df["OverworkPenalty"] > 0).mean()
    underworked_ratio = (stats_df["TotalHours"] < 0.8 * stats_df["MonthlyLimitHours"]).mean()

    workforce_score = max(
        0.0,
        1.0 - (0.7 * overworked_ratio + 0.3 * underworked_ratio)
    )

    # --- 5. Cost efficiency ---
    actual_cost = results["total_cost"]
    max_cost = results.get("max_cost", actual_cost)
    cost_score = 1.0 if actual_cost == 0 else min(1.0, max_cost / actual_cost)

    weights = {
        "coverage": 0.30,
        "stability": 0.20,
        "staffing": 0.15,
        "workforce": 0.20,
        "cost": 0.15,
    }

    score_01 = (
        weights["coverage"] * coverage_score +
        weights["stability"] * stability_score +
        weights["staffing"] * staffing_score +
        weights["workforce"] * workforce_score +
        weights["cost"] * cost_score
    )

    return {
        "HospitalEfficiencyScore": round(100 * score_01, 1),
        "components": {
            "CoverageSafety": coverage_score,
            "OperationalStability": stability_score,
            "StaffingAdequacy": staffing_score,
            "WorkforceSustainability": workforce_score,
            "CostEfficiency": cost_score,
        }
    }


def compute_hospital_efficiency(results: dict, *, include_underwork: bool = False) -> dict:

    schedule_df: pd.DataFrame = results["schedule_df"].copy()
    stats_df: pd.DataFrame = results["stats_df"].copy()
    shifts: pd.DataFrame = results["shifts"].copy()

    # --- 1) Satisfaction (0-1) ---
    if "S_pct" in stats_df.columns and len(stats_df) > 0:
        satisfaction_score = _clip01(float(stats_df["S_pct"].mean()) / 100.0)
    else:
        satisfaction_score = 0.0

    # --- 2) Coverage vs min_staff (0-1) ---
    total_required_slots = int(shifts["min_staff"].sum()) if "min_staff" in shifts.columns else 0
    missing_slots = int(results.get("slack", 0))  # suma brakujących lekarzy

    if total_required_slots > 0:
        coverage_ratio = 1.0 - (missing_slots / total_required_slots)
        coverage_score = _clip01(coverage_ratio)
    else:
        coverage_score = 0.0
        coverage_ratio = 0.0

    # --- 2b) Staffing quality vs regular_staff (0-1) ---
    if {"regular_staff", "code"}.issubset(shifts.columns):

        assigned_by_shift = (
            schedule_df[schedule_df["Missing"] == 0]
            .groupby("ShiftCode")
            .size()
            .reset_index(name="Assigned")
        )

        shift_staff = (
            shifts[["code", "regular_staff"]]
            .merge(
                assigned_by_shift,
                left_on="code",
                right_on="ShiftCode",
                how="left"
            )
            .fillna({"Assigned": 0})
        )

        shift_staff["QualityRatio"] = (
                shift_staff["Assigned"] / shift_staff["regular_staff"]
        ).clip(upper=1.0)

        staffing_quality_score = float(shift_staff["QualityRatio"].mean())
    else:
        staffing_quality_score = 1.0

    # --- 3) Understaff shifts ratio (0-1) ---
    if {"Date", "ShiftCode", "Missing"}.issubset(schedule_df.columns):
        missing_by_shift = (
            schedule_df
            .groupby(["Date", "ShiftCode"])["Missing"]
            .sum()
            .reset_index(name="MissingCnt")
        )
        total_shifts = int(len(missing_by_shift))
        understaff_shifts = int((missing_by_shift["MissingCnt"] > 0).sum())
        if total_shifts > 0:
            understaff_ratio = understaff_shifts / total_shifts
            understaff_shift_score = _clip01(1.0 - understaff_ratio)
        else:
            understaff_ratio = 0.0
            understaff_shift_score = 0.0
    else:
        total_shifts = 0
        understaff_shifts = 0
        understaff_ratio = 0.0
        understaff_shift_score = 0.0

    # --- 4) Overworked doctors (0-1) ---
    if "OverworkPenalty" in stats_df.columns and len(stats_df) > 0:
        overworked_cnt = int((stats_df["OverworkPenalty"] > 0).sum())
        num_doctors = int(len(stats_df))
        overwork_score = _clip01(1.0 - overworked_cnt / max(1, num_doctors))
    else:
        overworked_cnt = 0
        num_doctors = int(len(stats_df))
        overwork_score = 1.0 if num_doctors > 0 else 0.0

    # --- 5) Specialist coverage (0-1) ---
    specialist_score = 1.0
    specialist_violations = 0

    # --- 6) Underwork ---
    underwork_score = None
    underworked_cnt = None

    if include_underwork and {"TotalHours", "MaxHours"}.issubset(stats_df.columns):

        underworked_cnt = int((stats_df["TotalHours"] < 0.95 * stats_df["MaxHours"]).sum())
        underwork_score = _clip01(1.0 - underworked_cnt / max(1, len(stats_df)))

    weights = {
        "coverage": 0.25,
        "staffing_quality": 0.1,
        "understaff_shifts": 0.1,
        "satisfaction": 0.30,
        "overwork": 0.15,
        "specialist": 0.05,
        "underwork": 0.00 if not include_underwork else 0.05,
    }

    if include_underwork:

        weights["satisfaction"] = max(0.0, weights["satisfaction"] - 0.05)

    total_w = sum(weights.values())
    if total_w <= 0:
        total_w = 1.0

    score_01 = (
        weights["coverage"] * coverage_score +
        weights["staffing_quality"] * staffing_quality_score +
        weights["understaff_shifts"] * understaff_shift_score +
        weights["satisfaction"] * satisfaction_score +
        weights["overwork"] * overwork_score +
        weights["specialist"] * specialist_score +
        (weights["underwork"] * underwork_score if include_underwork else 0.0)
    ) / total_w

    score_100 = round(100.0 * _clip01(score_01), 1)

    return {
        "HospitalEfficiencyScore": score_100,
        "components": {
            "SatisfactionScore": satisfaction_score,
            "CoverageScore": coverage_score,
            "StaffingQualityScore_RegularStaff": staffing_quality_score,
            "UnderstaffShiftScore": understaff_shift_score,
            "OverworkScore": overwork_score,
            "SpecialistCoverageScore": specialist_score,
            "UnderworkScore": underwork_score,
        },
        "raw": {
            "total_required_slots_min_staff": total_required_slots,
            "missing_slots_min_staff": missing_slots,
            "coverage_ratio_min_staff": round(coverage_ratio, 4),

            "avg_staffing_quality_ratio_regular_staff": round(staffing_quality_score, 4),
            "total_shifts": total_shifts,
            "understaff_shifts": understaff_shifts,
            "understaff_ratio": round(understaff_ratio, 4),
            "overworked_cnt": overworked_cnt,
            "specialist_violations": specialist_violations,
            "underworked_cnt": underworked_cnt,
            "weights": weights,
        }
    }
