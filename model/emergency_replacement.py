"""
emergency_replacement.py
========================
Module for finding a replacement doctor in emergency situations (e.g., illness).

USAGE:
------
from emergency_replacement import find_replacement, precompute_replacements

# Single shift replacement:
result = find_replacement(
    results=results["baseline"],
    sick_doctor="Jan Kowalski",
    shift_code="2026-12-05_ICU_N",
    reserve_doctors_df=reserve_df,
    candidates_df=candidates_df,
    max_new_hires=2,
)

# Precompute for all shifts (e.g., for Streamlit UI):
all_results = precompute_replacements(
    results=results["baseline"],
    reserve_doctors_df=reserve_df,
    candidates_df=candidates_df,
)

RETURN STRUCTURE (find_replacement):
-----------------------------------
{
    "shift_code": str,
    "sick_doctor": str,
    "shift_info": dict,
    "candidates": [
        {
            "name": str,
            "source": "schedule" | "candidate" | "reserve",
            "role": str,
            "rest_ok": bool,
            "skill_ok": bool,
            "extra_cost": int,
            "workload_pct": float,
            "hard_shifts": int,
            "score": float,
            "warnings": [str],
        }
    ],
    "recommendation": dict | None,
    "fallback_message": str | None,
}
"""

from __future__ import annotations

from typing import Optional
import pandas as pd

# ---------------------------------------------------------------------------
# STAŁE
# ---------------------------------------------------------------------------

MIN_REST_HOURS = 11  # minimalna przerwa między zmianami
WEEKLY_REST_HOURS = 35  # minimalny odpoczynek tygodniowy
MAX_NEW_HIRES_DEFAULT = 2  # domyślny limit nowych kandydatów


# ---------------------------------------------------------------------------
# POMOCNICZE: PARSOWANIE HARMONOGRAMU
# ---------------------------------------------------------------------------

def _build_schedule_index(schedule_df: pd.DataFrame) -> dict[str, list[dict]]:
    """
    Returns a dictionary: doctor_name -> list of shifts
    (dict with fields Date, StartHour, EndHour, Hours, ShiftCode).

    Skips rows where Doctor=None (unassigned slots).
    """
    index: dict[str, list[dict]] = {}
    for _, row in schedule_df.dropna(subset=["Doctor"]).iterrows():
        name = row["Doctor"]
        index.setdefault(name, []).append({
            "date": str(row["Date"])[:10],
            "start": int(row["StartHour"]),
            "end": int(row["EndHour"]),
            "hours": int(row["Hours"]),
            "code": row["ShiftCode"],
        })
    return index


def _abs_hours(date: str, hour: int, date_index: dict[str, int]) -> int:
    """Converts date + hour into an absolute number of hours from the start of the month."""
    return date_index.get(date, 0) * 24 + hour


def _build_date_index(schedule_df: pd.DataFrame) -> dict[str, int]:
    dates = sorted(schedule_df["Date"].astype(str).str[:10].unique())
    return {d: i for i, d in enumerate(dates)}


# ---------------------------------------------------------------------------
# SPRAWDZANIE REGUŁ ODPOCZYNKU
# ---------------------------------------------------------------------------

def _check_rest_rules(
        doctor_shifts: list[dict],
        new_shift: dict,
        date_index: dict[str, int],
        weekly_limit: Optional[int] = None,
) -> tuple[bool, list[str]]:
    """
    Checks whether adding new_shift to doctor_shifts violates:
    1. 11-hour rest between any pair of shifts
    2. max 6 working days per week (indirectly via 35h weekly rest)
    3. weekly hour limit (if provided)

    Returns (ok: bool, warnings: list[str]).
    """
    warnings: list[str] = []

    new_start = _abs_hours(new_shift["date"], new_shift["start"], date_index)
    new_end = _abs_hours(new_shift["date"], new_shift["end"], date_index)

    for s in doctor_shifts:
        s_start = _abs_hours(s["date"], s["start"], date_index)
        s_end = _abs_hours(s["date"], s["end"], date_index)

        # === OVERLAP CHECK ===
        if not (new_end <= s_start or new_start >= s_end):
            warnings.append(
                f"Shift overlap: {new_shift['code']} conflicts with {s['code']}"
            )

        # nowa zmiana po istniejącej
        if new_start > s_end and (new_start - s_end) < MIN_REST_HOURS:
            warnings.append(
                f"Only {new_start - s_end}h rest after shift {s['code']} "
                f"(required {MIN_REST_HOURS}h)"
            )
        # nowa zmiana przed istniejącą
        if s_start > new_end and (s_start - new_end) < MIN_REST_HOURS:
            warnings.append(
                f"Only {s_start - new_end}h rest after shift {s['code']} "
                f"(required {MIN_REST_HOURS}h)"
            )

    # max 1 zmiana dziennie
    same_day = [s for s in doctor_shifts if s["date"] == new_shift["date"]]
    if same_day:
        warnings.append(f"Doctor already has a shift on {new_shift['date']}: {same_day[0]['code']}")

    # tygodniowy limit godzin
    if weekly_limit is not None:
        new_date = pd.to_datetime(new_shift["date"])
        week_start = new_date - pd.Timedelta(days=new_date.weekday())
        week_end = week_start + pd.Timedelta(days=6)

        week_hours = sum(
            s["hours"]
            for s in doctor_shifts
            if week_start <= pd.to_datetime(s["date"]) <= week_end
        )
        if week_hours + new_shift["hours"] > weekly_limit:
            warnings.append(
                f"Weekly limit exceeded: {week_hours + new_shift['hours']}h "
                f"> {weekly_limit}h"
            )

    rest_ok = not any(
        ("rest" in w) or
        ("shift on" in w) or
        ("overlap" in w.lower())
        for w in warnings
    )
    return rest_ok, warnings


# ---------------------------------------------------------------------------
# SCORING KANDYDATÓW
# ---------------------------------------------------------------------------

def _score_candidate(
        rest_ok: bool,
        skill_ok: bool,
        extra_cost: int,
        hard_shifts: int,
        workload_pct: float,
        source: str,
        max_cost: int,
) -> tuple[float, dict]:
    """
    Lower score = better candidate.

    Priorities:
      1. Rest rules (disqualifies if False)
      2. Skills (disqualifies if False)
      3. Cost
      4. Fairness (hard_shifts)
    """
    if not rest_ok:
        return 1e9 + extra_cost, {
            "reason": "rest_violation"
        }

    if not skill_ok:
        return 5e8 + extra_cost, {
            "reason": "skill_mismatch"
        }

    # Normalizacja kosztu do [0, 100]
    cost_norm = (extra_cost / max_cost * 100) if max_cost > 0 else 0

    # Premia za źródło: etatowy < kandydat < rezerwa (pod względem kosztu organizacyjnego)
    source_penalty = {
        "schedule": 0,
        "candidate": 50,
        "reserve": 10
    }[source]

    # Fairness: kto ma mniej trudnych zmian, dostaje premię
    fairness_score = (
            hard_shifts * 2 +
            workload_pct * 0.3
    )

    total_score = cost_norm + source_penalty + fairness_score

    return total_score, {
        "cost": cost_norm,
        "fairness": fairness_score,
        "source_penalty": source_penalty
    }


# ---------------------------------------------------------------------------
# GŁÓWNA FUNKCJA
# ---------------------------------------------------------------------------

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
    Find a replacement for a sick doctor for a specific shift.

    Parameters
    ----------
    results             : output from run_scenario() / run_model_and_get_results()["baseline"]
    sick_doctor         : name of the sick doctor (must match schedule_df)
    shift_code          : shift code (ShiftCode column in schedule_df)
    reserve_doctors_df  : DataFrame with retired doctors (see below)
    candidates_df       : DataFrame with candidates to hire
    max_new_hires       : maximum number of new hires allowed per month
    already_hired_count : number of candidates already hired this month

    Format of reserve_doctors_df (required columns):
        id, name, role, skills (semicolon-separated), hourly_cost
        Optional: night_allowed (0/1), twentyfour_allowed (0/1)

    Format of candidates_df — same as in the model (doctors_to_hire.csv).
    """

    schedule_df: pd.DataFrame = results["schedule_df"]
    stats_df: pd.DataFrame = results["stats_df"]
    shifts_df: pd.DataFrame = results["shifts"]
    doctors_df: pd.DataFrame = results["doctors"]
    weekly_limits: dict = results["weekly_limits"]

    # ---- Informacje o zmianie ----
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
    shift_info = {
        "date": str(shift_row["Date"])[:10],
        "dept": shift_row["Dept"],
        "start": int(shift_row["StartHour"]),
        "end": int(shift_row["EndHour"]),
        "hours": int(shift_row["Hours"]),
        "code": shift_code,
    }

    # wymagana kompetencja ze shifts_df
    shift_meta = shifts_df[shifts_df["code"] == shift_code]
    required_skill: str = (
        shift_meta["required_skill"].iloc[0]
        if not shift_meta.empty and "required_skill" in shift_meta.columns
        else ""
    )
    shift_info["required_skill"] = required_skill

    # ---- Indeksy pomocnicze ----
    date_index = _build_date_index(schedule_df)
    schedule_index = _build_schedule_index(schedule_df)

    # lista wszystkich lekarzy w aktualnym harmonogramie (bez chorego)
    active_doctors = [
        name for name in schedule_index
        if name != sick_doctor
    ]

    # ---- Zbieramy kandydatów na zastępstwo ----
    replacement_candidates: list[dict] = []

    max_cost_estimate = max(
        (row.get("hourly_cost", 100) * shift_info["hours"] for _, row in doctors_df.iterrows()),
        default=1000,
    )

    # =========================================================
    # ŹRÓDŁO 1: Lekarze z aktualnego harmonogramu (wolni tego dnia)
    # =========================================================
    for name in active_doctors:
        doctor_row = doctors_df[doctors_df["name"] == name]
        if doctor_row.empty:
            continue
        doctor_row = doctor_row.iloc[0]

        # === HARD CONSTRAINT: max 1 shift per day ===
        has_shift_same_day = any(
            s["date"] == shift_info["date"]
            for s in schedule_index.get(name, [])
        )

        if has_shift_same_day:
            continue

        # kompetencje
        skill_list = (
            doctor_row["skill_list"]
            if isinstance(doctor_row.get("skill_list"), list)
            else str(doctor_row.get("skills", "")).split(";")
        )
        skill_ok = (required_skill == "" or required_skill in skill_list)

        # dyżury nocne i 24h
        is_night = "_N_" in shift_code or shift_info["hours"] == 24
        is_24h = shift_info["hours"] == 24
        if is_night and doctor_row.get("night_allowed", 1) == 0:
            continue
        if is_24h and doctor_row.get("twentyfour_allowed", 1) == 0:
            continue

        # reguły odpoczynku
        doc_id = doctor_row["id"]
        wlimit = None
        if doc_id in weekly_limits:
            shift_date = shift_info["date"]
            shift_dt = pd.to_datetime(shift_date)
            week_start = shift_dt - pd.Timedelta(days=shift_dt.weekday())
            # === WYLICZENIE POPRAWNEGO TYGODNIA ===
            wlimit = None

            if doc_id in weekly_limits:
                shift_dt = pd.to_datetime(shift_info["date"])

                # identyczna logika jak w modelu / streamlit
                start_date = pd.to_datetime(schedule_df["Date"].min())
                start_monday = start_date - pd.Timedelta(days=start_date.weekday())

                local_week = int((shift_dt - start_monday).days // 7)

                wlimit = weekly_limits.get(doc_id, {}).get(local_week)

        rest_ok, warnings = _check_rest_rules(
            schedule_index.get(name, []),
            shift_info,
            date_index,
            weekly_limit=wlimit,
        )

        # statystyki z stats_df
        stats_row = stats_df[stats_df["Doctor"] == name]
        hard_shifts_count = int(stats_row["HardCount"].iloc[0]) if not stats_row.empty else 0
        workload_pct = float(stats_row["UtilizationPct"].iloc[0]) if not stats_row.empty else 0.0

        extra_cost = int(doctor_row.get("hourly_cost", 0)) * shift_info["hours"]

        score, score_components = _score_candidate(
            rest_ok=rest_ok,
            skill_ok=skill_ok,
            extra_cost=extra_cost,
            hard_shifts=hard_shifts_count,
            workload_pct=workload_pct,
            source="schedule",
            max_cost=max_cost_estimate,
        )

        replacement_candidates.append({
            "name": name,
            "source": "schedule",
            "role": doctor_row.get("role", ""),
            "rest_ok": rest_ok,
            "skill_ok": skill_ok,
            "extra_cost": extra_cost,
            "workload_pct": workload_pct,
            "hard_shifts": hard_shifts_count,
            "score": score,
            "warnings": warnings,
            "score_components": score_components,
        })

    # =========================================================
    # ŹRÓDŁO 3: Rezerwa (emeryci)
    # =========================================================
    if reserve_doctors_df is not None:
        for _, res in reserve_doctors_df.iterrows():
            skill_list = (
                res["skill_list"]
                if isinstance(res.get("skill_list"), list)
                else str(res.get("skills", "")).split(";")
            )
            skill_ok = (required_skill == "" or required_skill in skill_list)

            is_night = "_N_" in shift_code or shift_info["hours"] == 24
            is_24h = shift_info["hours"] == 24
            if is_night and res.get("night_allowed", 1) == 0:
                continue
            if is_24h and res.get("twentyfour_allowed", 1) == 0:
                continue

            extra_cost = int(res.get("hourly_cost", 0)) * shift_info["hours"]

            score, score_components = _score_candidate(
                rest_ok=True,
                skill_ok=skill_ok,
                extra_cost=extra_cost,
                hard_shifts=0,
                workload_pct=0.0,
                source="reserve",
                max_cost=max_cost_estimate,
            )

            replacement_candidates.append({
                "name": res["name"],
                "source": "reserve",
                "role": res.get("role", ""),
                "rest_ok": True,
                "skill_ok": skill_ok,
                "extra_cost": extra_cost,
                "workload_pct": 0.0,
                "hard_shifts": 0,
                "score": score,
                "score_components": score_components,
                "warnings": ["Reserve doctor — availability needs confirmation"],
            })

    # =========================================================
    # ŹRÓDŁO 2: Kandydaci do zatrudnienia (max_new_hires łącznie)
    # =========================================================
    hires_remaining = max_new_hires - already_hired_count

    if candidates_df is not None and hires_remaining > 0:
        for _, cand in candidates_df.iterrows():
            skill_list = (
                cand["skill_list"]
                if isinstance(cand.get("skill_list"), list)
                else str(cand.get("skills", "")).split(";")
            )
            skill_ok = (required_skill == "" or required_skill in skill_list)

            is_night = "_N_" in shift_code or shift_info["hours"] == 24
            is_24h = shift_info["hours"] == 24
            if is_night and cand.get("night_allowed", 1) == 0:
                continue
            if is_24h and cand.get("twentyfour_allowed", 1) == 0:
                continue

            extra_cost = int(cand.get("hourly_cost", 0)) * shift_info["hours"]

            score, score_components = _score_candidate(
                rest_ok=True,  # kandydaci nie mają jeszcze żadnych zmian
                skill_ok=skill_ok,
                extra_cost=extra_cost,
                hard_shifts=0,
                workload_pct=0.0,
                source="candidate",
                max_cost=max_cost_estimate,
            )

            replacement_candidates.append({
                "name": cand["name"],
                "source": "candidate",
                "role": cand.get("role", ""),
                "rest_ok": True,
                "skill_ok": skill_ok,
                "extra_cost": extra_cost,
                "workload_pct": 0.0,
                "hard_shifts": 0,
                "score": score,
                "is_legal": skill_ok,
                "warnings": [
                    f"New hire — will increase cost by ~{extra_cost} PLN",
                    f"Remaining new hire limit: {hires_remaining}",
                ],
            })

    # ---- Sortowanie: najpierw po score (rosnąco) ----
    replacement_candidates.sort(
        key=lambda c: (
            not (c["rest_ok"] and c["skill_ok"]),  # najpierw legalni
            c["score"]
        )
    )

    # ---- Rekomendacja: pierwszy kandydat spełniający oba warunki (rest + skill) ----
    valid_candidates = [
        c for c in replacement_candidates
        if c["rest_ok"] and c["skill_ok"]
    ]

    best = valid_candidates[0] if valid_candidates else None

    # ---- Fallback message ----
    fallback_message = None
    if not replacement_candidates:
        fallback_message = (
            "⚠️ No available candidates for replacement. Contact the on-call coordinator."
        )
    elif best is None:
        fallback_message = (
            "⚠️ No candidate satisfies both skill and rest requirements. Consider reserve staff or relaxing constraints."
        )

    return {
        "shift_code": shift_code,
        "sick_doctor": sick_doctor,
        "shift_info": shift_info,
        "candidates": replacement_candidates,
        "recommendation": best,
        "fallback_message": fallback_message,
    }


# ---------------------------------------------------------------------------
# WARIANT "WSZYSTKIE ZMIANY NARAZ" — do podglądu w Streamlicie
# ---------------------------------------------------------------------------

def precompute_replacements(
        results: dict,
        reserve_doctors_df: Optional[pd.DataFrame] = None,
        candidates_df: Optional[pd.DataFrame] = None,
        max_new_hires: int = MAX_NEW_HIRES_DEFAULT,
) -> dict[str, dict]:
    """
    Computes replacements for all shifts in the schedule.

    Parameters
    ----------
    only_problematic : if True, computes only shifts where a doctor HAS a replacement
                       or has NO replacement — skips shifts with Doctor=None.

    Returns a dictionary: shift_code -> find_replacement() result
    """
    schedule_df = results["schedule_df"]

    # pobierz tylko zmiany z przypisanym lekarzem
    assigned = schedule_df.dropna(subset=["Doctor"])[["ShiftCode", "Doctor"]].drop_duplicates()

    output: dict[str, dict] = {}
    already_hired = 0
    used_candidates = set()

    for _, row in assigned.iterrows():
        shift_code = row["ShiftCode"]
        sick_doctor = row["Doctor"]

        res = find_replacement(
            results=results,
            sick_doctor=sick_doctor,
            shift_code=shift_code,
            reserve_doctors_df=reserve_doctors_df,
            candidates_df=candidates_df,
            max_new_hires=max_new_hires,
            already_hired_count=already_hired,
        )
        best = res.get("recommendation")

        if best and best["source"] == "candidate":
            if best["name"] not in used_candidates:
                used_candidates.add(best["name"])
                already_hired += 1

        output[f"{shift_code}__{sick_doctor}"] = res

    return output


# ---------------------------------------------------------------------------
# FORMATOWANIE WYNIKÓW (do drukowania / logowania)
# ---------------------------------------------------------------------------

def format_replacement_report(result: dict) -> str:
    """Zwraca czytelny raport tekstowy z wynikami find_replacement()."""
    lines = []
    info = result["shift_info"]
    lines.append("=" * 60)
    lines.append(f"REPLACEMENT: {result['sick_doctor']}  →  {result['shift_code']}")
    lines.append(
        f"Shift: {info.get('date')} | {info.get('dept')} | "
        f"{info.get('start')}:00–{info.get('end')}:00 ({info.get('hours')}h) | "
        f"Required skill: {info.get('required_skill', '—')}"
    )
    lines.append("-" * 60)

    if result["fallback_message"]:
        lines.append(result["fallback_message"])

    if result["recommendation"]:
        rec = result["recommendation"]
        lines.append(
            f"RECOMMENDATION: {rec['name']} ({rec['source'].upper()}) | "
            f"Cost: +{rec['extra_cost']} PLN | "
            f"Hard shifts: {rec['hard_shifts']} | "
            f"Workload: {rec['workload_pct']}%"
        )
        if rec["warnings"]:
            for w in rec["warnings"]:
                lines.append(f"   ⚠ {w}")
    else:
        lines.append("No valid recommendation found..")

    lines.append("")
    lines.append("All candidates (sorted):")
    for i, c in enumerate(result["candidates"][:10], 1):
        status = "✅" if (c["rest_ok"] and c["skill_ok"]) else "⚠️"
        lines.append(
            f"  {i:2}. {status} {c['name']:<25} [{c['source']:10}] "
            f"skill={'✓' if c['skill_ok'] else '✗'}  "
            f"rest={'✓' if c['rest_ok'] else '✗'}  "
            f"cost={c['extra_cost']:>6} PLN  "
            f"hard={c['hard_shifts']}"
        )
        for w in c["warnings"]:
            lines.append(f"         ⚠ {w}")

    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PRZYKŁAD UŻYCIA (uruchom bezpośrednio: python emergency_replacement.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Module loaded successfully.")
    print("Usage: from emergency_replacement import find_replacement")
    print("See module docstring for details.")