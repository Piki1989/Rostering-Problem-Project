"""
pandemic_mode.py
================
Pandemic Mode for the Hospital Rostering Optimizer.

POPRAWNA KOLEJNOŚĆ WYWOŁAŃ:
---------------------------

  1. Wczytaj dane (doctors, shifts, unavail itd.)

  2. Jeśli pandemic_mode=True:
       config = PandemicConfig(...)
       bubble_assignment = prepare_pandemic_bubbles(doctors, candidates, config)
       # bubble_assignment: {doctor_id -> bubble_id}  — gotowe PRZED solverem

  3. build_model(..., pandemic_config=config, bubble_assignment=bubble_assignment)
       # add_pandemic_constraints() wywoływane WEWNĄTRZ build_model po hard constraints

  4. solve_3_phase(bundle)   # bez zmian

  5. extract_results(...)    # bundle zawiera bubble_assignment i pandemic_config

  6. render_pandemic_tab(results, ...)   # UI odczytuje dane z results

Definicja kontaktu:
-------------------
  - Ta sama zmiana w WARD lub ICU = kontakt epidemiologiczny
  - CLINIC domyślnie wykluczone (osobne gabinety)
  - Handover (przekazanie dyżuru): opcjonalny handover_window_hours (domyślnie 0)

Emergency replacement w trybie pandemicznym:
--------------------------------------------
  - Zastępstwo tylko z tej samej bańki (działy kontaktowe)
  - Rezerwa zewnętrzna dozwolona z ostrzeżeniem
  - Kandydaci do zatrudnienia WYŁĄCZENI
"""

from __future__ import annotations

import math
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
    Konfiguracja trybu pandemicznego.

    num_bubbles : int
        Liczba izolowanych bańek. Zalecane 2–4.
        Każda bańka powinna móc samodzielnie obsadzić co najmniej jedną zmianę.

    contact_depts : list[str]
        Oddziały, gdzie wspólna zmiana = kontakt. Domyślnie WARD + ICU.
        CLINIC wyłączone (osobne gabinety, niższe ryzyko transmisji).

    handover_window_hours : int
        Godziny przekazania dyżuru liczone jako kontakt. Domyślnie 0 (V1).

    min_specialists_per_bubble : int
        Minimalna liczba specjalistów w każdej bańce (kliniczna samodzielność).

    allow_cross_bubble_reserve : bool
        Czy rezerwa zewnętrzna może zastępować dowolną bańkę.
    """
    num_bubbles: int = 3
    contact_depts: list[str] = field(default_factory=lambda: ["WARD", "ICU"])
    handover_window_hours: int = 0
    min_specialists_per_bubble: int = 1
    allow_cross_bubble_reserve: bool = True


# ---------------------------------------------------------------------------
# KROK 2: prepare_pandemic_bubbles()  —  PRZED build_model()
# ---------------------------------------------------------------------------

def prepare_pandemic_bubbles(
    doctors: pd.DataFrame,
    candidates: pd.DataFrame,
    config: PandemicConfig,
    seed: int = 42,
) -> dict[int, int]:
    """
    Skill-aware greedy bubble assignment.

    CLINIC nie jest traktowane jako dział kontaktowy.
    Bańki są przygotowywane przed solverem.
    """
    import random
    rng = random.Random(seed)

    B = config.num_bubbles
    all_df = pd.concat([doctors, candidates], ignore_index=True).copy()

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
        Wybiera bańkę z najmniejszą liczbą danej kompetencji,
        a przy remisie z najmniejszym rozmiarem.
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
    night_docs = all_df[
        (all_df.get("night_allowed", 0) == 1)
        & (~all_df["id"].isin(assigned))
    ]

    for row in shuffled_rows(night_docs):
        add_doctor(row, "night")

    # 4. 24h-capable doctors
    twentyfour_docs = all_df[
        (all_df.get("twentyfour_allowed", 0) == 1)
        & (~all_df["id"].isin(assigned))
    ]

    for row in shuffled_rows(twentyfour_docs):
        add_doctor(row, "twentyfour")

    # 5. Everyone else
    others = all_df[~all_df["id"].isin(assigned)]

    for row in shuffled_rows(others):
        add_doctor(row, "size")

    print("\n[PandemicMode] === Podział na bańki przed build_model ===")
    print(f"  Lekarze: {len(doctors)}, Kandydaci: {len(candidates)}")
    print(f"  Liczba bańek: {B}")
    print(f"  Działy kontaktowe: {config.contact_depts}")
    print("  CLINIC nie jest traktowane jako kontakt epidemiologiczny.")

    for b in range(B):
        s = bubble_stats[b]
        print(
            f"  Bańka {b}: "
            f"{s['size']} osób, "
            f"{s['specialists']} spec., "
            f"{s['icu']} ICU, "
            f"{s['night']} night, "
            f"{s['twentyfour']} 24h"
        )

    print()

    return assignment


def validate_bubbles(
    doctors: pd.DataFrame,
    candidates: pd.DataFrame,
    shifts: pd.DataFrame,
    bubble_assignment: dict[int, int],
    config: PandemicConfig,
) -> pd.DataFrame:
    """
    Diagnostyka bańek przed solverem.

    Funkcja nie zatrzymuje programu.
    Zwraca tabelę z informacją, czy każda bańka ma minimalne kompetencje.
    """
    all_df = pd.concat([doctors, candidates], ignore_index=True).copy()

    if "skills" not in all_df.columns:
        all_df["skills"] = ""

    all_df["skills"] = all_df["skills"].fillna("").astype(str)
    all_df["role"] = all_df["role"].fillna("").astype(str)

    specialist_roles = {"specialist", "icu_specialist"}

    def has_icu_skill(skills: Any) -> bool:
        return "icu" in str(skills).lower()

    rows = []

    for b in range(config.num_bubbles):
        ids = [d for d, bb in bubble_assignment.items() if bb == b]
        bdf = all_df[all_df["id"].isin(ids)]

        size = len(bdf)
        specialists = int(bdf["role"].isin(specialist_roles).sum())
        icu_capable = int(bdf["skills"].apply(has_icu_skill).sum())

        night_capable = (
            int((bdf["night_allowed"] == 1).sum())
            if "night_allowed" in bdf.columns
            else 0
        )

        twentyfour_capable = (
            int((bdf["twentyfour_allowed"] == 1).sum())
            if "twentyfour_allowed" in bdf.columns
            else 0
        )

        max_weekly_hours = (
            int(bdf["max_weekly_hours"].sum())
            if "max_weekly_hours" in bdf.columns
            else None
        )

        row = {
            "Bubble": b,
            "Doctors": size,
            "Specialists": specialists,
            "ICU capable": icu_capable,
            "Night capable": night_capable,
            "24h capable": twentyfour_capable,
            "Max weekly hours": max_weekly_hours,
        }
        rows.append(row)

        if specialists < config.min_specialists_per_bubble:
            warnings.warn(
                f"[PandemicMode] Bańka {b} ma tylko {specialists} specjalistów "
                f"(wymagane minimum: {config.min_specialists_per_bubble})."
            )

        if "ICU" in set(config.contact_depts) and icu_capable == 0:
            warnings.warn(
                f"[PandemicMode] Bańka {b} nie ma żadnego lekarza z kompetencją ICU."
            )

        if night_capable == 0:
            warnings.warn(
                f"[PandemicMode] Bańka {b} nie ma lekarza dopuszczonego do dyżurów nocnych."
            )

        if twentyfour_capable == 0:
            warnings.warn(
                f"[PandemicMode] Bańka {b} nie ma lekarza dopuszczonego do dyżurów 24h."
            )

    validation_df = pd.DataFrame(rows)

    print("\n[PandemicMode] === Walidacja bańek ===")
    print(validation_df.to_string(index=False))
    print()

    return validation_df

# ---------------------------------------------------------------------------
# KROK 3: add_pandemic_constraints()  —  WEWNĄTRZ build_model()
# ---------------------------------------------------------------------------

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
    Dodaje ograniczenia bańkowe do modelu CP-SAT.

    Wywołaj WEWNĄTRZ build_model(), po wszystkich hard constraints,
    przed soft constraints i budowaniem objective.

    HC-P1. Lekarze z różnych bańek nie mogą dzielić zmiany w dziale kontaktowym.
           Dla każdej pary (d1, d2) z różnych bańek i każdej zmiany kontaktowej:
             x[d1,s] + x[d2,s] <= 1

    Złożoność: O(D^2 / 2 * S_contact).
    Dla 45 lekarzy i ~200 zmian kontaktowych: ~45^2/2 * 200 ≈ 200k constraints.
    OR-Tools radzi sobie z tym bez problemu.
    """
    contact_depts  = set(config.contact_depts)
    contact_shifts = [
        s for s in S
        if shifts.loc[shift_idx[s], "dept"] in contact_depts
    ]

    print(f"[PandemicMode] Dodawanie ograniczeń HC-P1:")
    print(f"  Działy kontaktowe: {contact_depts}")
    print(f"  Zmian kontaktowych: {len(contact_shifts)} / {len(S)}")

    # constraint_count = 0
    # for s in contact_shifts:
    #     for i, d1 in enumerate(D_all):
    #         for d2 in D_all[i + 1:]:
    #             if bubble_assignment.get(d1, 0) != bubble_assignment.get(d2, 0):
    #                 model.Add(x[(d1, s)] + x[(d2, s)] <= 1)
    #                 constraint_count += 1
    crossbubble_violations = []

    BIG_PANDEMIC_PENALTY = 1_000_000

    constraint_count = 0

    for s in contact_shifts:

        for i, d1 in enumerate(D_all):

            for d2 in D_all[i + 1:]:

                if bubble_assignment.get(d1, 0) != bubble_assignment.get(d2, 0):
                    viol = model.NewBoolVar(
                        f"crossbubble_{d1}_{d2}_{s}"
                    )

                    model.Add(
                        x[(d1, s)] + x[(d2, s)] <= 1 + viol
                    )

                    crossbubble_violations.append(viol)

                    constraint_count += 1

    print(f"  Dodano {constraint_count} pandemic constraints.")
    print(f"  Cross-bubble violations są dozwolone tylko awaryjnie.")

    return {
        "crossbubble_violations": crossbubble_violations,
        "penalty": BIG_PANDEMIC_PENALTY,
    }


# ---------------------------------------------------------------------------
# KROK 5: get_pandemic_results()  —  na końcu extract_results() / run_scenario()
# ---------------------------------------------------------------------------

def get_pandemic_results(
    bubble_assignment: dict[int, int],
    config: PandemicConfig,
    id_to_name: dict[int, str],
    id_to_role: dict[int, str],
) -> dict:
    """
    Pakuje dane o bańkach do słownika results.

    Wywołaj na końcu run_scenario() i wrzuć wynik do results:
        results["pandemic"] = get_pandemic_results(bubble_assignment, config, ...)
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
# SYMULACJA KWARANTANNY
# ---------------------------------------------------------------------------

def simulate_quarantine(
    results: dict,
    quarantine_bubble: int,
    bubble_assignment: dict[int, int],
    id_to_name: dict[int, str],
    config: PandemicConfig,
) -> dict:
    """
    Symuluje kwarantannę jednej bańki (analiza post-hoc, bez re-solve).

    Usuwa lekarzy z danej bańki z harmonogramu i raportuje:
    - które zmiany stają się nieobsadzone
    - jaki % pokrycia tracimy
    - severity (LOW / MEDIUM / HIGH)
    - rekomendację działania
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

    total_covered       = int((schedule_df["Missing"] == 0).sum())
    newly_uncovered     = len(affected)
    coverage_impact_pct = round(100 * newly_uncovered / max(1, total_covered), 1)
    days_affected       = int(affected["Date"].nunique()) if not affected.empty else 0

    dept_impact = {}
    if not affected.empty and "Dept" in affected.columns:
        dept_impact = affected.groupby("Dept").size().to_dict()

    if coverage_impact_pct < 20:
        severity = "LOW"
        action   = "Harmonogram wytrzyma kwarantannę — przesuń nadgodziny."
    elif coverage_impact_pct < 40:
        severity = "MEDIUM"
        action   = "Aktywuj rezerwę i rozważ awaryjne pokrycie między-bańkowe."
    else:
        severity = "HIGH"
        action   = (
            "Krytyczny niedobór obsady. Eskaluj natychmiast. "
            "Rozważ tymczasowe scalenie dwóch pozostałych bańek."
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
            f"[{severity}] Kwarantanna bańki {quarantine_bubble} "
            f"({len(quarantined_names)} lekarzy) usuwa {newly_uncovered} przypisań "
            f"({coverage_impact_pct}% pokrytych slotów). "
            f"Dotknięte oddziały: {dept_impact}. {action}"
        ),
    }


# ---------------------------------------------------------------------------
# EMERGENCY REPLACEMENT W TRYBIE PANDEMICZNYM
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
    Szuka zastępcy w trybie pandemicznym.

    Reguły:
      1. Dział kontaktowy (WARD/ICU): tylko lekarze z tej samej bańki
      2. Dział niekontaktowy (CLINIC): dowolna bańka
      3. Rezerwa zewnętrzna: zawsze dozwolona z ostrzeżeniem
      4. Kandydaci do zatrudnienia: WYŁĄCZENI

    Struktura zwracanego słownika jest zgodna z find_replacement()
    z emergency_replacement_new.py, z dodatkowymi polami:
        pandemic_mode, bubble_restrictions_applied, is_contact_dept, sick_bubble
    """
    try:
        from emergency_replacement_new import (
            _build_schedule_index,
            _build_date_index,
            _check_rest_rules,
            _score_candidate,
        )
    except ImportError:
        raise ImportError(
            "Nie znaleziono emergency_replacement_new.py. "
            "Upewnij się że plik jest w tym samym katalogu co pandemic_mode.py."
        )

    schedule_df   = results["schedule_df"]
    stats_df      = results["stats_df"]
    shifts_df     = results["shifts"]
    doctors_df    = results["doctors"]
    weekly_limits = results["weekly_limits"]

    name_to_id  = {v: k for k, v in id_to_name.items()}
    sick_id     = name_to_id.get(sick_doctor)
    sick_bubble = bubble_assignment.get(sick_id, -1)

    # Dane zmiany
    shift_rows = schedule_df[schedule_df["ShiftCode"] == shift_code]
    if shift_rows.empty:
        return {
            "shift_code": shift_code, "sick_doctor": sick_doctor,
            "shift_info": {}, "candidates": [],
            "recommendation": None,
            "fallback_message": f"Zmiana '{shift_code}' nie znaleziona w harmonogramie.",
            "pandemic_mode": True, "bubble_restrictions_applied": False,
        }

    shift_row  = shift_rows.iloc[0]
    shift_info = {
        "date":  str(shift_row["Date"])[:10],
        "dept":  shift_row["Dept"],
        "start": int(shift_row["StartHour"]),
        "end":   int(shift_row["EndHour"]),
        "hours": int(shift_row["Hours"]),
        "code":  shift_code,
    }
    shift_meta = shifts_df[shifts_df["code"] == shift_code]
    shift_info["required_skill"] = (
        shift_meta["required_skill"].iloc[0]
        if not shift_meta.empty and "required_skill" in shift_meta.columns
        else ""
    )

    is_contact_dept = shift_info["dept"] in config.contact_depts
    date_index      = _build_date_index(schedule_df)
    schedule_index  = _build_schedule_index(schedule_df)
    active_doctors  = [n for n in schedule_index if n != sick_doctor]
    required_skill  = shift_info["required_skill"]

    max_cost_estimate = max(
        (row.get("hourly_cost", 100) * shift_info["hours"] for _, row in doctors_df.iterrows()),
        default=1000,
    )

    replacement_candidates = []

    # --- Lekarze z harmonogramu ---
    for name in active_doctors:
        doc_row = doctors_df[doctors_df["name"] == name]
        if doc_row.empty:
            continue
        doc_row    = doc_row.iloc[0]
        doc_id     = doc_row["id"]
        doc_bubble = bubble_assignment.get(doc_id, -1)

        # REGUŁA PANDEMICZNA
        if is_contact_dept and doc_bubble != sick_bubble:
            continue

        if any(s["date"] == shift_info["date"] for s in schedule_index.get(name, [])):
            continue

        skill_list = (
            doc_row["skill_list"]
            if isinstance(doc_row.get("skill_list"), list)
            else str(doc_row.get("skills", "")).split(";")
        )
        skill_ok = required_skill == "" or required_skill in skill_list

        is_night = "_N_" in shift_code or shift_info["hours"] == 24
        is_24h   = shift_info["hours"] == 24
        if is_night and doc_row.get("night_allowed", 1) == 0:
            continue
        if is_24h and doc_row.get("twentyfour_allowed", 1) == 0:
            continue

        shift_dt     = pd.to_datetime(shift_info["date"])
        start_date   = pd.to_datetime(schedule_df["Date"].min())
        start_monday = start_date - pd.Timedelta(days=start_date.weekday())
        local_week   = int((shift_dt - start_monday).days // 7)
        wlimit       = weekly_limits.get(doc_id, {}).get(local_week)

        rest_ok, rest_warnings = _check_rest_rules(
            schedule_index.get(name, []), shift_info, date_index, weekly_limit=wlimit
        )

        stats_row         = stats_df[stats_df["Doctor"] == name]
        hard_shifts_count = int(stats_row["HardCount"].iloc[0]) if not stats_row.empty else 0
        workload_pct      = float(stats_row["UtilizationPct"].iloc[0]) if not stats_row.empty else 0.0
        extra_cost        = int(doc_row.get("hourly_cost", 0)) * shift_info["hours"]

        score, score_components = _score_candidate(
            rest_ok=rest_ok, skill_ok=skill_ok, extra_cost=extra_cost,
            hard_shifts=hard_shifts_count, workload_pct=workload_pct,
            source="schedule", max_cost=max_cost_estimate,
        )

        replacement_candidates.append({
            "name": name,            "source": "schedule",
            "role": doc_row.get("role", ""), "bubble": doc_bubble,
            "rest_ok": rest_ok,      "skill_ok": skill_ok,
            "extra_cost": extra_cost, "workload_pct": workload_pct,
            "hard_shifts": hard_shifts_count, "score": score,
            "warnings": rest_warnings, "score_components": score_components,
        })

    # --- Rezerwa zewnętrzna ---
    if reserve_doctors_df is not None and config.allow_cross_bubble_reserve:
        for _, res in reserve_doctors_df.iterrows():
            skill_list = (
                res["skill_list"]
                if isinstance(res.get("skill_list"), list)
                else str(res.get("skills", "")).split(";")
            )
            skill_ok = required_skill == "" or required_skill in skill_list

            if "_N_" in shift_code or shift_info["hours"] == 24:
                if res.get("night_allowed", 1) == 0:
                    continue
            if shift_info["hours"] == 24 and res.get("twentyfour_allowed", 1) == 0:
                continue

            extra_cost = int(res.get("hourly_cost", 0)) * shift_info["hours"]
            score, score_components = _score_candidate(
                rest_ok=True, skill_ok=skill_ok, extra_cost=extra_cost,
                hard_shifts=0, workload_pct=0.0,
                source="reserve", max_cost=max_cost_estimate,
            )

            replacement_candidates.append({
                "name": res["name"],   "source": "reserve",
                "role": res.get("role", ""), "bubble": "external",
                "rest_ok": True,       "skill_ok": skill_ok,
                "extra_cost": extra_cost, "workload_pct": 0.0,
                "hard_shifts": 0,      "score": score,
                "warnings": [
                    "REZERWA — izolowana od systemu bańek",
                    "Wymagane potwierdzenie dostępności",
                ],
                "score_components": score_components,
            })

    replacement_candidates.sort(
        key=lambda c: (not (c["rest_ok"] and c["skill_ok"]), c["score"])
    )

    valid = [c for c in replacement_candidates if c["rest_ok"] and c["skill_ok"]]
    best  = valid[0] if valid else None

    fallback_message = None
    if not replacement_candidates:
        fallback_message = (
            f"Brak kandydatów w bańce {sick_bubble} na zmianę {shift_code}. "
            "Wszyscy mogą być na kwarantannie. Aktywuj rezerwę zewnętrzną."
        )
    elif best is None:
        fallback_message = (
            f"Kandydaci w bańce {sick_bubble} nie spełniają wymagań "
            "kompetencji + odpoczynku. Wymagana rezerwa lub złagodzenie reguł."
        )

    return {
        "shift_code": shift_code,        "sick_doctor": sick_doctor,
        "sick_bubble": sick_bubble,      "shift_info": shift_info,
        "candidates": replacement_candidates,
        "recommendation": best,          "fallback_message": fallback_message,
        "pandemic_mode": True,
        "bubble_restrictions_applied": is_contact_dept,
        "is_contact_dept": is_contact_dept,
    }


# ---------------------------------------------------------------------------
# HELPERS DLA UI
# ---------------------------------------------------------------------------

def bubble_contact_matrix(
    schedule_df: pd.DataFrame,
    bubble_assignment: dict[int, int],
    name_to_id: dict[str, int],
    contact_depts: list[str],
) -> pd.DataFrame:
    """
    Liczy ile razy pary bańek dzieliły zmianę w działach kontaktowych.
    W poprawnym rozwiązaniu = 0 wszędzie.
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
    """DataFrame [Doctor, Role, Bubble] — do wyświetlenia w Streamlicie."""
    rows = [
        {"Doctor": id_to_name.get(d, str(d)), "Role": id_to_role.get(d, ""), "Bubble": b}
        for d, b in bubble_assignment.items()
    ]
    return pd.DataFrame(rows).sort_values(["Bubble", "Role", "Doctor"])