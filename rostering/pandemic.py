"""
Pandemic mode for the hospital rostering optimizer.

Definition of contact:
  - Contact = sharing a shift in a contact department (default: WARD, ICU).
  - CLINIC excluded by default (separate offices, individual quarantine policy).
  - Handover periods: optional handover_window_hours (default: 0).
  - Adaptive bubble count via `suggest_num_bubbles(...)`.
  - Pure-CLINIC physicians live OUTSIDE the bubble system in a CLINIC_POOL
    (bubble id = -1) and only get individual quarantine.

Quarantine model:
  - `QuarantineEvent` first-class object: target (bubble / single doctor /
    list of doctors), start_date, duration_days, reason.
  - `simulate_quarantine_v2` accepts a list of events (multiple, possibly
    overlapping quarantines are supported).

Replacement during a quarantine:
  - `mini_resolve_quarantine_window` builds a SMALL CP-SAT model that
    re-optimises the schedule only on quarantine dates. Pre-window and
    post-window assignments are fixed; weekly hour limits are reduced
    by hours already worked outside the window; full hard constraints
    (rest, skills, eligibility, day-off-after-night, max consecutive
    nights, weekly day limits, bubble isolation) are respected for
    both regular doctors and reserves.
  - Replacement only from the same bubble (contact departments); external
    reserve staff allowed with a warning; additional physicians may be hired
    only during schedule construction.

Backwards-compatible:
  - `simulate_quarantine`, `simulate_quarantine_with_reserve`,
    `find_replacement_pandemic` remain available with the same signatures
    and delegate to v2 internally.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional, Union
from math import ceil

import pandas as pd
from ortools.sat.python import cp_model

from .emergency_replacement import find_replacement

logger = logging.getLogger(__name__)


# Special bubble id for physicians who are NOT in any bubble (CLINIC pool).
CLINIC_POOL = -1


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class PandemicConfig:
    """
    Pandemic mode configuration.

    num_bubbles
        Number of isolated bubbles. Use ``suggest_num_bubbles`` to get a
        size-appropriate recommendation for your hospital.

    contact_depts
        Departments where sharing a shift counts as epidemiological contact.
        CLINIC is intentionally excluded (separate rooms, individual policy).

    handover_window_hours
        Reserved for future use - shift handover hours that should count
        as contact. Default: 0.

    min_specialists_per_bubble
        Hard floor for the bubble validator.

    allow_cross_bubble_reserve
        Whether external reserve staff may serve in any bubble.
        Reserves are by definition isolated from the bubble structure.

    separate_clinic_pool
        If True, doctors with role="outpatient" (and any doctor whose skill
        list is purely CLINIC/OUTPATIENT) live outside the bubble system.
        Their quarantine is always individual, never bubble-wide.

    target_loss_pct
        Used only by ``suggest_num_bubbles`` to compute the recommended
        number of bubbles. With 0.20 the system suggests at least 5 bubbles.

    min_bubble_size
        Soft lower bound used by ``suggest_num_bubbles`` to avoid bubbles
        too small to cover all required shift types.
    """
    num_bubbles: int = 3
    contact_depts: list[str] = field(default_factory=lambda: ["WARD", "ICU"])
    handover_window_hours: int = 0
    min_specialists_per_bubble: int = 1
    allow_cross_bubble_reserve: bool = True
    separate_clinic_pool: bool = True
    target_loss_pct: float = 0.20
    min_bubble_size: int = 5


@dataclass
class QuarantineEvent:
    """
    A single quarantine episode affecting one or more doctors.

    target_type
        ``"bubble"``  - target is a bubble id (int)
        ``"doctor"``  - target is a doctor NAME (str)
        ``"doctors"`` - target is a list of doctor NAMES (list[str])

    start_date
        ISO date ``"YYYY-MM-DD"`` - first day of quarantine, inclusive.

    duration_days
        Length of the quarantine in days, inclusive of ``start_date``.

    reason
        Free-form tag: ``"exposure"``, ``"confirmed"``, ``"symptomatic"``, ...
    """
    target_type: str
    target: Union[int, str, list[str]]
    start_date: str
    duration_days: int = 7
    reason: str = "exposure"

    def __post_init__(self) -> None:
        if self.target_type not in {"bubble", "doctor", "doctors"}:
            raise ValueError(
                f"target_type must be 'bubble', 'doctor' or 'doctors', "
                f"got {self.target_type!r}"
            )
        if self.duration_days < 1:
            raise ValueError("duration_days must be >= 1")
        # Validate date format
        datetime.strptime(self.start_date, "%Y-%m-%d")

    @property
    def end_date(self) -> str:
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = start + timedelta(days=self.duration_days - 1)
        return end.strftime("%Y-%m-%d")

    def covers_date(self, date_str: str) -> bool:
        return self.start_date <= date_str <= self.end_date

    def get_affected_doctor_ids(
        self,
        bubble_assignment: dict[int, int],
        name_to_id: dict[str, int],
    ) -> set[int]:
        if self.target_type == "bubble":
            return {d for d, b in bubble_assignment.items() if b == self.target}
        if self.target_type == "doctor":
            did = name_to_id.get(str(self.target))
            return {did} if did is not None else set()
        # "doctors"
        ids: set[int] = set()
        for nm in self.target:  # type: ignore[union-attr]
            did = name_to_id.get(str(nm))
            if did is not None:
                ids.add(did)
        return ids

    def label(self) -> str:
        if self.target_type == "bubble":
            tgt = f"Bubble {self.target}"
        elif self.target_type == "doctor":
            tgt = str(self.target)
        else:
            n = len(self.target) if hasattr(self.target, "__len__") else 1  # type: ignore
            tgt = f"{n} doctors"
        return f"{tgt} | {self.start_date}→{self.end_date} ({self.duration_days}d, {self.reason})"


# =============================================================================
# ADAPTIVE BUBBLE SIZING + CLINIC POOL IDENTIFICATION
# =============================================================================

def suggest_num_bubbles(
    n_doctors: int,
    target_loss_pct: float = 0.20,
    min_bubble_size: int = 4,
    max_bubbles: int = 10,
    min_bubbles: int = 2,
) -> dict[str, Any]:
    """
    Recommend a number of bubbles for a hospital of size ``n_doctors``.

    The recommendation balances two goals:

    - Quarantining one bubble should affect at most ``target_loss_pct``
      of the workforce  =>  need at least ``ceil(1/target_loss_pct)`` bubbles.
    - Each bubble must be large enough to cover all required shift types
      =>  no more than ``n_doctors / min_bubble_size`` bubbles.

    Returns a dict with recommended/min/max values and the expected loss
    percentage at the recommendation. Useful for UI auto-suggest.
    """
    if n_doctors <= 0:
        return {
            "recommended": min_bubbles,
            "min": min_bubbles,
            "max": min_bubbles,
            "loss_pct": 100.0,
            "warning": "No doctors in the pool.",
        }

    if n_doctors < min_bubble_size * min_bubbles:
        recommended = max(min_bubbles, n_doctors // max(1, min_bubble_size))
        recommended = max(min_bubbles, recommended)
        return {
            "recommended": recommended,
            "min": min_bubbles,
            "max": recommended,
            "loss_pct": round(100.0 / recommended, 1),
            "warning": (
                f"Only {n_doctors} doctors available. With {recommended} bubbles "
                f"each bubble holds ~{n_doctors // recommended}."
            ),
        }

    by_loss = max(min_bubbles, int(ceil(1.0 / max(target_loss_pct, 1e-6))))
    by_size = max(min_bubbles, n_doctors // min_bubble_size)
    recommended = min(by_loss, by_size, max_bubbles)
    recommended = max(min_bubbles, recommended)

    return {
        "recommended": recommended,
        "min": min_bubbles,
        "max": min(by_size, max_bubbles),
        "loss_pct": round(100.0 / recommended, 1),
        "warning": None,
    }


def _parse_skill_cell(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(s).strip() for s in value if str(s).strip()]
    if isinstance(value, str):
        return [s.strip() for s in value.split(";") if s.strip()]
    return []


def identify_clinic_only_doctors(
    doctors: pd.DataFrame,
    candidates: Optional[pd.DataFrame] = None,
) -> set[int]:
    """
    Return the set of doctor IDs that are CLINIC-only.

    A doctor is considered CLINIC-only if EITHER:
      - role == 'outpatient', OR
      - every skill in their skill list contains 'CLINIC' or 'OUTPATIENT'.
    """
    clinic_ids: set[int] = set()

    frames = [doctors]
    if candidates is not None and not candidates.empty:
        frames.append(candidates)

    for df in frames:
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            role = str(row.get("role", "")).lower().strip()
            if role == "outpatient":
                clinic_ids.add(int(row["id"]))
                continue

            skills = _parse_skill_cell(row.get("skill_list", row.get("skills", "")))
            if skills and all(
                ("CLINIC" in s.upper()) or ("OUTPATIENT" in s.upper())
                for s in skills
            ):
                clinic_ids.add(int(row["id"]))

    return clinic_ids


# =============================================================================
# BUBBLE ASSIGNMENT
# =============================================================================

def prepare_pandemic_bubbles(
    doctors: pd.DataFrame,
    candidates: pd.DataFrame,
    config: PandemicConfig,
    seed: int = 42,
) -> dict[int, int]:
    """
    Skill-aware greedy bubble assignment.

    Pure-CLINIC doctors are assigned ``CLINIC_POOL`` (= -1) when
    ``config.separate_clinic_pool`` is True. Other doctors are distributed
    across ``config.num_bubbles`` in a way that balances ICU competency,
    specialists and night eligibility. Hire candidates are distributed the
    same way after the regular doctors.
    """
    import random
    rng = random.Random(seed)

    B = config.num_bubbles
    specialist_roles = {"specialist", "icu_specialist"}

    def normalise(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "skills" not in df.columns:
            df["skills"] = ""
        df["skills"] = df["skills"].fillna("").astype(str)
        df["role"] = df["role"].fillna("").astype(str)
        return df

    # Identify clinic-only doctors first - they live outside the bubble system.
    clinic_only_ids: set[int] = set()
    if config.separate_clinic_pool:
        clinic_only_ids = identify_clinic_only_doctors(doctors, candidates)

    assignment: dict[int, int] = {cid: CLINIC_POOL for cid in clinic_only_ids}

    bubble_stats = {
        b: {"size": 0, "specialists": 0, "icu": 0, "night": 0, "candidates": 0}
        for b in range(B)
    }

    def has_skill(row: pd.Series, name: str) -> bool:
        return name.lower() in str(row.get("skills", "")).lower()

    def choose_bubble(priority_key: str) -> int:
        return min(
            range(B),
            key=lambda b: (
                bubble_stats[b][priority_key],
                bubble_stats[b]["size"],
                b,
            ),
        )

    def add_doctor(row: pd.Series, priority_key: str, is_candidate: bool) -> None:
        d = int(row["id"])
        if d in assignment:
            return
        b = choose_bubble(priority_key)
        assignment[d] = b
        s = bubble_stats[b]
        s["size"] += 1
        if is_candidate:
            s["candidates"] += 1
        if str(row.get("role", "")) in specialist_roles:
            s["specialists"] += 1
        if has_skill(row, "ICU"):
            s["icu"] += 1
        if int(row.get("night_allowed", 0) or 0) == 1:
            s["night"] += 1

    def shuffled(df: pd.DataFrame) -> list[pd.Series]:
        rows = [r for _, r in df.iterrows()]
        rng.shuffle(rows)
        return rows

    def distribute(df: pd.DataFrame, is_candidate: bool) -> None:
        pool_df = df[~df["id"].isin(clinic_only_ids)]
        if pool_df.empty:
            return

        # Order matters: scarcest competencies first.
        icu_specs = pool_df[
            (pool_df["role"].isin(specialist_roles))
            & (pool_df.apply(lambda r: has_skill(r, "ICU"), axis=1))
        ]
        for row in shuffled(icu_specs):
            add_doctor(row, "icu", is_candidate)

        if "night_allowed" in pool_df.columns:
            night_docs = pool_df[
                (pool_df["night_allowed"] == 1)
                & (~pool_df["id"].isin(assignment))
            ]
            for row in shuffled(night_docs):
                add_doctor(row, "night", is_candidate)

        other_specs = pool_df[
            (pool_df["role"].isin(specialist_roles))
            & (~pool_df["id"].isin(assignment))
        ]
        for row in shuffled(other_specs):
            add_doctor(row, "specialists", is_candidate)

        others = pool_df[~pool_df["id"].isin(assignment)]
        for row in shuffled(others):
            add_doctor(row, "size", is_candidate)

    # Candidates go second so the split of permanent staff does not depend on them;
    distribute(normalise(doctors), is_candidate=False)
    if candidates is not None and not candidates.empty:
        distribute(normalise(candidates), is_candidate=True)

    logger.info(
        "Bubble assignment: %d doctors, %d candidates, %d bubbles, contact departments %s, "
        "separate CLINIC pool: %s",
        len(doctors), len(candidates), B, config.contact_depts, config.separate_clinic_pool,
    )
    for b in range(B):
        s = bubble_stats[b]
        logger.info(
            "  Bubble %d: %d doctors (incl. %d candidates) | %d specialists, %d ICU, %d night",
            b, s["size"], s["candidates"], s["specialists"], s["icu"], s["night"],
        )
    if clinic_only_ids:
        logger.info("  CLINIC_POOL: %d doctors (no bubble, individual quarantine)", len(clinic_only_ids))

    return assignment


# =============================================================================
# BUBBLE VALIDATION
# =============================================================================

def validate_bubbles(
    doctors: pd.DataFrame,
    candidates: pd.DataFrame,
    shifts: pd.DataFrame,
    bubble_assignment: dict[int, int],
    config: PandemicConfig,
) -> dict:
    """Pre-solver diagnostics. Now also reports on the CLINIC_POOL."""
    all_df = doctors.copy()
    if "skills" not in all_df.columns:
        all_df["skills"] = ""
    all_df["skills"] = all_df["skills"].fillna("").astype(str)
    all_df["role"] = all_df["role"].fillna("").astype(str)

    specialist_roles = {"specialist", "icu_specialist"}

    def has_icu(skills: Any) -> bool:
        return "icu" in str(skills).lower()

    rows: list[dict] = []
    issues: list[dict] = []

    for b in range(config.num_bubbles):
        ids = [d for d, bb in bubble_assignment.items() if bb == b]
        bdf = all_df[all_df["id"].isin(ids)]

        size = len(bdf)
        specialists = int(bdf["role"].isin(specialist_roles).sum())
        icu_capable = int(bdf["skills"].apply(has_icu).sum())
        night_capable = int((bdf["night_allowed"] == 1).sum()) if "night_allowed" in bdf.columns else 0

        ok = True
        local_issues: list[str] = []

        if specialists < config.min_specialists_per_bubble:
            msg = f"Bubble {b}: only {specialists} specialists (required: {config.min_specialists_per_bubble})"
            issues.append({"severity": "ERROR", "bubble": b, "message": msg})
            warnings.warn(f"[PandemicMode] {msg}")
            ok = False
            local_issues.append("missing specialists")

        if "ICU" in set(config.contact_depts) and icu_capable == 0:
            msg = f"Bubble {b}: no ICU-capable physician"
            issues.append({"severity": "ERROR", "bubble": b, "message": msg})
            warnings.warn(f"[PandemicMode] {msg}")
            ok = False
            local_issues.append("missing ICU competency")

        if night_capable == 0:
            msg = f"Bubble {b}: no night-eligible physician"
            issues.append({"severity": "WARNING", "bubble": b, "message": msg})
            warnings.warn(f"[PandemicMode] {msg}")
            local_issues.append("missing night eligibility")

        rows.append({
            "Bubble": f"Bubble {b}",
            "Doctors": size,
            "Specialists": specialists,
            "ICU capable": icu_capable,
            "Night capable": night_capable,
            "Status": "OK" if ok else "PROBLEM",
            "Issues": ", ".join(local_issues) if local_issues else "—",
        })

    # CLINIC pool report row (informational only)
    clinic_ids = [d for d, b in bubble_assignment.items() if b == CLINIC_POOL]
    if clinic_ids:
        cdf = all_df[all_df["id"].isin(clinic_ids)]
        rows.append({
            "Bubble": "CLINIC pool",
            "Doctors": len(cdf),
            "Specialists": int(cdf["role"].isin(specialist_roles).sum()),
            "ICU capable": 0,
            "Night capable": int((cdf["night_allowed"] == 1).sum()) if "night_allowed" in cdf.columns else 0,
            "Status": "OK",
            "Issues": "Individual quarantine policy",
        })

    validation_df = pd.DataFrame(rows)
    is_safe = not any(i["severity"] == "ERROR" for i in issues)

    logger.info("Bubble validation:\n%s", validation_df.to_string(index=False))

    return {
        "validation_df": validation_df,
        "issues": issues,
        "is_safe": is_safe,
    }


# =============================================================================
# RESULTS PACKAGING
# (the isolation constraint HP1 is in rostering/equations.py)
# =============================================================================

def get_pandemic_results(
    bubble_assignment: dict[int, int],
    config: PandemicConfig,
    id_to_name: dict[int, str],
    id_to_role: dict[int, str],
) -> dict:
    """Packages bubble info into the results dict consumed by the UI."""
    B = config.num_bubbles
    bubbles: dict[Union[int, str], dict] = {}

    for b in range(B):
        members = [d for d, bb in bubble_assignment.items() if bb == b]
        bubbles[b] = {
            "doctors": members,
            "names": [id_to_name.get(d, str(d)) for d in members],
            "roles": [id_to_role.get(d, "") for d in members],
            "size": len(members),
            "specialists": sum(
                1 for d in members
                if id_to_role.get(d, "") in {"specialist", "icu_specialist"}
            ),
        }

    clinic_members = [d for d, b in bubble_assignment.items() if b == CLINIC_POOL]
    if clinic_members:
        bubbles[CLINIC_POOL] = {
            "doctors": clinic_members,
            "names": [id_to_name.get(d, str(d)) for d in clinic_members],
            "roles": [id_to_role.get(d, "") for d in clinic_members],
            "size": len(clinic_members),
            "specialists": 0,
            "is_clinic_pool": True,
        }

    return {
        "bubble_assignment": bubble_assignment,
        "bubbles": bubbles,
        "config": config,
        "num_bubbles": B,
        "contact_depts": config.contact_depts,
        "has_clinic_pool": bool(clinic_members),
    }


# =============================================================================
# QUARANTINE V2 - AFFECTED ASSIGNMENT EXTRACTION
# =============================================================================

def get_quarantine_affected_assignments(
    schedule_df: pd.DataFrame,
    events: list[QuarantineEvent],
    bubble_assignment: dict[int, int],
    id_to_name: dict[int, str],
) -> pd.DataFrame:
    """Return assignments that would be lost due to the given quarantine events."""
    if not events:
        return pd.DataFrame()

    name_to_id = {v: k for k, v in id_to_name.items()}

    sched = schedule_df.copy()
    sched["Date"] = pd.to_datetime(sched["Date"])
    sched["DateStr"] = sched["Date"].dt.strftime("%Y-%m-%d")

    rows: list[pd.DataFrame] = []
    for ev in events:
        doc_ids = ev.get_affected_doctor_ids(bubble_assignment, name_to_id)
        doc_names = {id_to_name[d] for d in doc_ids if d in id_to_name}
        mask = (
            sched["Doctor"].isin(doc_names)
            & sched["DateStr"].apply(ev.covers_date)
            & (sched["Missing"] == 0)
        )
        sub = sched[mask].copy()
        if not sub.empty:
            sub["QuarantineReason"] = ev.reason
            sub["QuarantineStart"] = ev.start_date
            sub["QuarantineEnd"] = ev.end_date
            rows.append(sub)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset=["DateStr", "ShiftCode", "Doctor"], keep="first")
    return out.drop(columns="DateStr")


# =============================================================================
# MINI CP-SAT RE-SOLVE FOR THE QUARANTINE WINDOW
# =============================================================================

def _abs_time(date_index: dict[str, int], date_str: str, hour: int) -> int:
    """Hours since the start of the planning horizon."""
    return date_index[date_str] * 24 + hour


def _build_and_solve_mini_model(
    *,
    results: dict,
    events: list[QuarantineEvent],
    bubble_assignment: dict[int, int],
    id_to_name: dict[int, str],
    config: PandemicConfig,
    reserve_doctors_df: Optional[pd.DataFrame],
    window_dates: set[str],
    quarantine_dates_by_doc: dict[int, set[str]],
    time_limit_seconds: int,
    extra_doctors_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Construct and solve a CP-SAT model over the quarantine window only.

    Hard constraints implemented:
      C1  required skill match
      C2  at most one shift per day
      C3  quarantined doctors cannot work on quarantine days
      C4  reserve hire flag - reserves only work if hired
      C5  11h rest period (intra-window and cross-window with fixed shifts)
      C6  weekly hour limits with pre-committed hours subtracted
      C7  night
      C8  unavailability (date and shift-code)
      C9  bubble isolation on contact-department shifts
      C10 mandatory day off after a night shift (cross-window aware)
      C11 max 2 consecutive nights (cross-window aware)
      C12 max 6 working days per ISO week (cross-window aware)

    Objective:
      Lex order via weighted sum:
        1. minimise weighted slack on contact depts (ICU=100, WARD=50, CLINIC=1)
        2. minimise reserve cost
        3. minimise number of changes to the original schedule
    """
    schedule_df = results["schedule_df"]
    doctors_df = results["doctors"]
    shifts_df = results["shifts"]
    unavail_day = results["unavail_day"]
    unavail_shift = results["unavail_shift"]

    name_to_id = {v: k for k, v in id_to_name.items()}

    # ---------- 1. Build the doctor pool ----------
    # Regular doctors first.
    base_ids = [int(i) for i in doctors_df["id"]]

    # Hired candidates that appear in the schedule but are not in doctors_df.
    if extra_doctors_df is not None and not extra_doctors_df.empty:
        extra_ids = [int(i) for i in extra_doctors_df["id"] if int(i) not in base_ids]
    else:
        extra_ids = []

    # Reserve doctors with non-colliding IDs.
    if reserve_doctors_df is None or reserve_doctors_df.empty:
        reserve_df = pd.DataFrame()
        reserve_ids: list[int] = []
    else:
        reserve_df = reserve_doctors_df.copy()
        if "id" not in reserve_df.columns:
            base_max = max(base_ids + extra_ids + [0])
            reserve_df["id"] = range(base_max + 10000, base_max + 10000 + len(reserve_df))
        else:
            taken = set(base_ids + extra_ids)
            if any(int(rid) in taken for rid in reserve_df["id"]):
                base_max = max(base_ids + extra_ids + [0])
                reserve_df["id"] = range(base_max + 10000, base_max + 10000 + len(reserve_df))
        reserve_ids = [int(i) for i in reserve_df["id"]]

    all_ids = base_ids + extra_ids + reserve_ids

    # Single lookup table for all doctors (regular + extra + reserve).
    doc_lookup: dict[int, pd.Series] = {}
    for _, r in doctors_df.iterrows():
        doc_lookup[int(r["id"])] = r
    if extra_doctors_df is not None and not extra_doctors_df.empty:
        for _, r in extra_doctors_df.iterrows():
            doc_lookup[int(r["id"])] = r
    for _, r in reserve_df.iterrows():
        doc_lookup[int(r["id"])] = r

    def skill_list(row: pd.Series) -> list[str]:
        return _parse_skill_cell(row.get("skill_list", row.get("skills", "")))

    skills_by_doc = {d: skill_list(doc_lookup[d]) for d in all_ids}

    # ---------- 2. Restrict shifts to the window ----------
    shifts_window = shifts_df[shifts_df["date"].isin(window_dates)].copy()
    if shifts_window.empty:
        return _empty_mini_result("NO_SHIFTS_IN_WINDOW")
    window_shift_ids = [int(i) for i in shifts_window["id"]]
    shift_info: dict[int, pd.Series] = {
        int(r["id"]): r for _, r in shifts_window.iterrows()
    }

    # ---------- 3. Week mapping (same convention as build_model) ----------
    all_dates = sorted(shifts_df["date"].unique())
    date_objs = [pd.to_datetime(d) for d in all_dates]
    monday0 = date_objs[0] - pd.Timedelta(days=date_objs[0].weekday())
    date_to_week = {
        d.strftime("%Y-%m-%d"): (d - monday0).days // 7 for d in date_objs
    }
    date_index = {d: i for i, d in enumerate(all_dates)}

    weeks_in_window = sorted({date_to_week[d] for d in window_dates if d in date_to_week})

    # ---------- 4. Pre-committed assignments outside the window ----------
    # Used for weekly hour accounting, rest constraints, night chains, etc.
    pre_hours: dict[int, dict[int, int]] = {d: {} for d in all_ids}
    pre_days_by_week: dict[int, dict[int, set[str]]] = {d: {} for d in all_ids}
    pre_shifts_by_doc: dict[int, list[dict]] = {d: [] for d in all_ids}

    sched = schedule_df.copy()
    sched["Date"] = pd.to_datetime(sched["Date"])
    sched["DateStr"] = sched["Date"].dt.strftime("%Y-%m-%d")

    for _, r in sched.iterrows():
        if r["Missing"] == 1 or pd.isna(r["Doctor"]):
            continue
        d_id = name_to_id.get(r["Doctor"])
        if d_id is None or d_id not in doc_lookup:
            continue
        ds = r["DateStr"]
        is_in_window = ds in window_dates
        s_start = int(r["StartHour"])
        s_end = int(r["EndHour"])
        is_night = "_N_" in str(r["ShiftCode"])

        if not is_in_window:
            w = date_to_week.get(ds)
            if w is not None:
                pre_hours[d_id][w] = pre_hours[d_id].get(w, 0) + int(r["Hours"])
                pre_days_by_week[d_id].setdefault(w, set()).add(ds)
            pre_shifts_by_doc[d_id].append({
                "date": ds,
                "start": s_start,
                "end": s_end,
                "is_night": is_night,
            })

    # Quick lookup: does doctor have pre-existing assignment on a specific date?
    pre_dates_by_doc = {
        d: {p["date"] for p in pre_shifts_by_doc[d]} for d in all_ids
    }
    # Map date -> night?
    pre_night_dates_by_doc: dict[int, set[str]] = {
        d: {p["date"] for p in pre_shifts_by_doc[d] if p["is_night"]}
        for d in all_ids
    }

    # ---------- 5. Build CP-SAT model ----------
    model = cp_model.CpModel()

    # Decision variables only for shifts inside the window.
    x: dict[tuple[int, int], cp_model.IntVar] = {}
    for d in all_ids:
        for s in window_shift_ids:
            x[(d, s)] = model.NewBoolVar(f"mx_{d}_{s}")

    hire = {r: model.NewBoolVar(f"mhire_{r}") for r in reserve_ids}

    # --- C1: required skill match
    req_skill = {int(r["id"]): r["required_skill"] for _, r in shifts_df.iterrows()}
    for d in all_ids:
        d_skills = skills_by_doc[d]
        for s in window_shift_ids:
            if req_skill[s] not in d_skills:
                model.Add(x[(d, s)] == 0)

    # --- C2: at most one shift per day
    shifts_by_day_in_window: dict[str, list[int]] = {}
    for s in window_shift_ids:
        ds = shift_info[s]["date"]
        shifts_by_day_in_window.setdefault(ds, []).append(s)
    for d in all_ids:
        for ds, s_list in shifts_by_day_in_window.items():
            model.Add(sum(x[(d, s)] for s in s_list) <= 1)
            # also block double-assignment cross-window: doctor already has shift today
            if ds in pre_dates_by_doc.get(d, set()):
                for s in s_list:
                    model.Add(x[(d, s)] == 0)

    # --- C3: quarantined doctors cannot work on their quarantine dates
    for d, q_dates in quarantine_dates_by_doc.items():
        if d not in [*all_ids]:
            continue
        for s in window_shift_ids:
            if shift_info[s]["date"] in q_dates:
                model.Add(x[(d, s)] == 0)

    # --- C4: reserve hire flag
    for rid in reserve_ids:
        for s in window_shift_ids:
            model.Add(x[(rid, s)] <= hire[rid])

    # --- C5: 11h rest period
    # 5a) intra-window
    for s1 in window_shift_ids:
        i1 = shift_info[s1]
        as1 = _abs_time(date_index, i1["date"], int(i1["start_hour"]))
        ae1 = _abs_time(date_index, i1["date"], int(i1["end_hour"]))
        for s2 in window_shift_ids:
            if s1 == s2:
                continue
            i2 = shift_info[s2]
            as2 = _abs_time(date_index, i2["date"], int(i2["start_hour"]))
            if as2 <= ae1:
                continue
            if (as2 - ae1) < 11:
                for d in all_ids:
                    model.Add(x[(d, s1)] + x[(d, s2)] <= 1)

    # 5b) cross-window: pre-committed vs window shift
    for d in all_ids:
        for pre in pre_shifts_by_doc[d]:
            pa_s = _abs_time(date_index, pre["date"], pre["start"])
            pa_e = _abs_time(date_index, pre["date"], pre["end"])
            for s in window_shift_ids:
                info = shift_info[s]
                a_s = _abs_time(date_index, info["date"], int(info["start_hour"]))
                a_e = _abs_time(date_index, info["date"], int(info["end_hour"]))
                # overlap
                if a_s < pa_e and a_e > pa_s:
                    model.Add(x[(d, s)] == 0)
                    continue
                # window shift starts after pre, gap < 11h
                if a_s >= pa_e and (a_s - pa_e) < 11:
                    model.Add(x[(d, s)] == 0)
                # pre starts after window shift, gap < 11h
                elif pa_s >= a_e and (pa_s - a_e) < 11:
                    model.Add(x[(d, s)] == 0)

    # --- C6: weekly hour limits with pre-committed hours subtracted
    for d in all_ids:
        row = doc_lookup[d]
        max_h = int(row.get("max_hours", 48) or 48)

        # Unavail for this doctor (only for regular doctors - reserves assumed always available)
        if d in base_ids or d in extra_ids:
            unavail_set = set(unavail_day[unavail_day["doctor_id"] == d]["date"])
        else:
            unavail_set = set()

        for w in weeks_in_window:
            week_dates = [ds for ds, ww in date_to_week.items() if ww == w]
            avail_days = sum(1 for ds in week_dates if ds not in unavail_set)
            weekly_lim = int(max_h * avail_days / 7) if avail_days > 0 else 0
            remaining = max(0, weekly_lim - pre_hours[d].get(w, 0))

            week_window_shifts = [
                s for s in window_shift_ids
                if date_to_week.get(shift_info[s]["date"]) == w
            ]
            if week_window_shifts:
                model.Add(
                    sum(
                        x[(d, s)] * int(shift_info[s]["hours"])
                        for s in week_window_shifts
                    )
                    <= remaining
                )

    # --- C7: night
    for d in all_ids:
        row = doc_lookup[d]
        night_ok = int(row.get("night_allowed", 1) or 0) == 1
        for s in window_shift_ids:
            info = shift_info[s]
            is_night = "_N_" in str(info["code"])
            if is_night and not night_ok:
                model.Add(x[(d, s)] == 0)

    # --- C8: unavailability
    for _, ru in unavail_day.iterrows():
        d = int(ru["doctor_id"])
        ds = str(ru["date"])
        if ds not in window_dates or d not in [*all_ids]:
            continue
        for s in shifts_by_day_in_window.get(ds, []):
            model.Add(x[(d, s)] == 0)

    code_to_sid = {row["code"]: int(row["id"]) for _, row in shifts_df.iterrows()}
    for _, ru in unavail_shift.iterrows():
        d = int(ru["doctor_id"])
        sid = code_to_sid.get(ru["code"])
        if sid in window_shift_ids and d in [*all_ids]:
            model.Add(x[(d, sid)] == 0)

    # --- C9: bubble isolation on contact-department shifts
    contact_depts = set(config.contact_depts)
    for s in window_shift_ids:
        if shift_info[s]["dept"] not in contact_depts:
            continue
        for i, d1 in enumerate(all_ids):
            b1 = bubble_assignment.get(d1)
            if b1 is None or b1 == CLINIC_POOL:
                continue
            if d1 in reserve_ids:
                # reserves treated as external - allowed in any bubble
                continue
            for d2 in all_ids[i + 1:]:
                b2 = bubble_assignment.get(d2)
                if b2 is None or b2 == CLINIC_POOL:
                    continue
                if d2 in reserve_ids:
                    continue
                if b1 != b2:
                    model.Add(x[(d1, s)] + x[(d2, s)] <= 1)

    # --- C10: mandatory day off after a night shift
    # Window-window:
    night_shifts_window = [
        s for s in window_shift_ids
        if "_N_" in str(shift_info[s]["code"])
    ]
    next_day_shifts: dict[str, list[int]] = {}
    for s in window_shift_ids:
        ds = shift_info[s]["date"]
        next_day_shifts.setdefault(ds, []).append(s)

    for d in all_ids:
        for sn in night_shifts_window:
            n_date = shift_info[sn]["date"]
            next_day = (
                datetime.strptime(n_date, "%Y-%m-%d") + timedelta(days=1)
            ).strftime("%Y-%m-%d")
            for s2 in next_day_shifts.get(next_day, []):
                model.Add(x[(d, sn)] + x[(d, s2)] <= 1)

    # Cross-window: if doctor had a night the day BEFORE the window starts,
    # they can't work on the first window day.
    for d in all_ids:
        for night_date in pre_night_dates_by_doc.get(d, set()):
            next_day = (
                datetime.strptime(night_date, "%Y-%m-%d") + timedelta(days=1)
            ).strftime("%Y-%m-%d")
            if next_day in window_dates:
                for s in shifts_by_day_in_window.get(next_day, []):
                    model.Add(x[(d, s)] == 0)

    # --- C11: max 2 consecutive nights
    # Window-window 3-day sliding sums:
    sorted_window_dates = sorted(window_dates)
    for d in all_ids:
        for i in range(len(sorted_window_dates) - 2):
            window3 = sorted_window_dates[i:i + 3]
            shifts_in_3 = [
                s for s in night_shifts_window
                if shift_info[s]["date"] in window3
            ]
            if shifts_in_3:
                model.Add(sum(x[(d, s)] for s in shifts_in_3) <= 2)

    # Cross-window: nights on day-2 and day-1 of window mean no night on day-1, etc.
    # We handle this by counting pre-existing nights in any 3-day window that
    # overlaps the start of the window.
    for d in all_ids:
        for i, ds in enumerate(sorted_window_dates):
            # Look back at days ds-2, ds-1 for pre-existing nights
            d_obj = datetime.strptime(ds, "%Y-%m-%d")
            prev1 = (d_obj - timedelta(days=1)).strftime("%Y-%m-%d")
            prev2 = (d_obj - timedelta(days=2)).strftime("%Y-%m-%d")
            pre_nights_count = sum(
                1 for pd_ in (prev1, prev2)
                if pd_ in pre_night_dates_by_doc.get(d, set())
            )
            if pre_nights_count == 0:
                continue
            # In the 3-day window [prev2, prev1, ds] total nights <= 2.
            # pre_nights_count of those are already locked.
            shifts_today_night = [
                s for s in night_shifts_window if shift_info[s]["date"] == ds
            ]
            if shifts_today_night:
                model.Add(
                    sum(x[(d, s)] for s in shifts_today_night)
                    <= max(0, 2 - pre_nights_count)
                )

    # --- C12: max 6 working days per ISO week (cross-window aware)
    for d in all_ids:
        for w in weeks_in_window:
            week_dates = [ds for ds, ww in date_to_week.items() if ww == w]
            week_window_dates = [ds for ds in week_dates if ds in window_dates]
            pre_days_this_week = len(pre_days_by_week.get(d, {}).get(w, set()))
            # `works_var` per window day in this week
            works_vars = []
            for ds in week_window_dates:
                wv = model.NewBoolVar(f"works_{d}_{ds}")
                day_shifts = shifts_by_day_in_window.get(ds, [])
                M = max(1, len(day_shifts))
                model.Add(sum(x[(d, s)] for s in day_shifts) >= wv)
                model.Add(sum(x[(d, s)] for s in day_shifts) <= M * wv)
                works_vars.append(wv)
            if works_vars:
                model.Add(sum(works_vars) <= max(0, 6 - pre_days_this_week))

    # ---------- 6. Slack & Objective ----------
    slacks: dict[int, cp_model.IntVar] = {}
    for s in window_shift_ids:
        info = shift_info[s]
        ms = int(info["min_staff"])
        slk = model.NewIntVar(0, ms, f"mslk_{s}")
        slacks[s] = slk
        model.Add(sum(x[(d, s)] for d in all_ids) + slk >= ms)

    def slack_weight_for(dept: str) -> int:
        if dept == "ICU":
            return 100
        if dept == "WARD":
            return 50
        return 1

    weighted_slack = sum(
        slacks[s] * slack_weight_for(shift_info[s]["dept"])
        for s in window_shift_ids
    )

    # Reserve cost (in PLN-hour units)
    reserve_cost_terms: list = []
    for rid in reserve_ids:
        hourly = int(doc_lookup[rid].get("hourly_cost", 100) or 100)
        for s in window_shift_ids:
            reserve_cost_terms.append(
                x[(rid, s)] * int(shift_info[s]["hours"]) * hourly
            )
    total_reserve_cost = sum(reserve_cost_terms) if reserve_cost_terms else 0

    # Change penalty: how many original assignments were dropped (regular doctors).
    original_in_window: dict[tuple[int, int], int] = {}
    for _, r in sched.iterrows():
        if r["Missing"] == 1 or pd.isna(r["Doctor"]):
            continue
        if r["DateStr"] not in window_dates:
            continue
        d_id = name_to_id.get(r["Doctor"])
        if d_id is None:
            continue
        sid = code_to_sid.get(r["ShiftCode"])
        if sid in window_shift_ids:
            original_in_window[(d_id, sid)] = 1

    change_terms: list = []
    for (d, s), _ in original_in_window.items():
        if d not in all_ids:
            continue
        q_dates = quarantine_dates_by_doc.get(d, set())
        if shift_info[s]["date"] in q_dates:
            continue  # can't keep
        change_terms.append(1 - x[(d, s)])
    change_penalty = sum(change_terms) if change_terms else 0

    # Lexicographic via large coefficients.
    model.Minimize(
        weighted_slack * 10_000_000
        + total_reserve_cost
        + change_penalty * 100
    )

    # ---------- 7. Solve ----------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = 4
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return _empty_mini_result(
            f"INFEASIBLE_OR_TIMEOUT_{solver.StatusName(status)}",
            residual_uncovered=shifts_window.assign(
                Reason="Mini solve failed to find feasible coverage"
            ),
        )

    # ---------- 8. Extract results ----------
    reserve_rows: list[dict] = []
    reassign_rows: list[dict] = []
    residual_rows: list[dict] = []

    for s in window_shift_ids:
        info = shift_info[s]
        slk_val = solver.Value(slacks[s])
        assigned = [d for d in all_ids if solver.Value(x[(d, s)]) == 1]

        for d in assigned:
            was_original = original_in_window.get((d, s), 0) == 1
            if d in reserve_ids:
                doc_row = doc_lookup[d]
                doc_name = doc_row.get("name", f"Reserve_{d}")
                hourly = int(doc_row.get("hourly_cost", 100) or 100)
                cost = hourly * int(info["hours"])
                reserve_rows.append({
                    "Date": info["date"],
                    "ShiftCode": info["code"],
                    "Dept": info["dept"],
                    "Reserve doctor": doc_name,
                    "Cost (PLN)": cost,
                    "Bubble": "external",
                    "Manual validation required": "Yes",
                })
            elif not was_original:
                # internal redistribution (regular doctor filling a gap)
                reassign_rows.append({
                    "Date": info["date"],
                    "ShiftCode": info["code"],
                    "Dept": info["dept"],
                    "Doctor": id_to_name.get(d, str(d)),
                    "Bubble": bubble_assignment.get(d),
                    "Originally": "—",
                    "Now": "Filled by redistribution",
                })

        if slk_val > 0:
            for _ in range(slk_val):
                residual_rows.append({
                    "Date": info["date"],
                    "ShiftCode": info["code"],
                    "Dept": info["dept"],
                    "Hours": int(info["hours"]),
                    "Reason": "No feasible doctor / reserve available",
                })

    reserve_df_out = pd.DataFrame(reserve_rows)
    reassign_df_out = pd.DataFrame(reassign_rows)
    residual_df_out = pd.DataFrame(residual_rows)

    return {
        "status": solver.StatusName(status),
        "reserve_assignments": reserve_df_out,
        "reassignments": reassign_df_out,
        "covered_by_reserve": int(len(reserve_df_out)),
        "reassigned_count": int(len(reassign_df_out)),
        "residual_gap": int(len(residual_df_out)),
        "residual_uncovered": residual_df_out,
        "reserve_total_cost": int(reserve_df_out["Cost (PLN)"].sum()) if not reserve_df_out.empty else 0,
        "solve_time": solver.WallTime(),
        "objective": solver.ObjectiveValue(),
    }


def _empty_mini_result(status: str, residual_uncovered: Optional[pd.DataFrame] = None) -> dict:
    return {
        "status": status,
        "reserve_assignments": pd.DataFrame(),
        "reassignments": pd.DataFrame(),
        "covered_by_reserve": 0,
        "reassigned_count": 0,
        "residual_gap": 0 if residual_uncovered is None else int(len(residual_uncovered)),
        "residual_uncovered": residual_uncovered if residual_uncovered is not None else pd.DataFrame(),
        "reserve_total_cost": 0,
        "solve_time": 0.0,
        "objective": None,
    }


def mini_resolve_quarantine_window(
    results: dict,
    events: list[QuarantineEvent],
    bubble_assignment: dict[int, int],
    id_to_name: dict[int, str],
    config: PandemicConfig,
    reserve_doctors_df: Optional[pd.DataFrame] = None,
    time_limit_seconds: int = 60,
    extra_doctors_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Public entry point for the mini CP-SAT re-solve.

    extra_doctors_df can be used to pass hired candidates (so they are
    considered part of the regular pool, not as reserves).
    """
    name_to_id = {v: k for k, v in id_to_name.items()}

    quarantine_dates_by_doc: dict[int, set[str]] = {}
    for ev in events:
        ids = ev.get_affected_doctor_ids(bubble_assignment, name_to_id)
        for offset in range(ev.duration_days):
            d = datetime.strptime(ev.start_date, "%Y-%m-%d") + timedelta(days=offset)
            ds = d.strftime("%Y-%m-%d")
            for did in ids:
                quarantine_dates_by_doc.setdefault(did, set()).add(ds)

    window_dates: set[str] = set()
    for dates in quarantine_dates_by_doc.values():
        window_dates.update(dates)

    if not window_dates:
        return _empty_mini_result("NO_QUARANTINE")

    return _build_and_solve_mini_model(
        results=results,
        events=events,
        bubble_assignment=bubble_assignment,
        id_to_name=id_to_name,
        config=config,
        reserve_doctors_df=reserve_doctors_df,
        window_dates=window_dates,
        quarantine_dates_by_doc=quarantine_dates_by_doc,
        time_limit_seconds=time_limit_seconds,
        extra_doctors_df=extra_doctors_df,
    )


# =============================================================================
# QUARANTINE V2 - PUBLIC ENTRY POINT
# =============================================================================

def simulate_quarantine_v2(
    results: dict,
    events: list[QuarantineEvent],
    bubble_assignment: dict[int, int],
    id_to_name: dict[int, str],
    config: PandemicConfig,
    reserve_doctors_df: Optional[pd.DataFrame] = None,
    use_mini_resolve: bool = True,
    time_limit_seconds: int = 60,
    extra_doctors_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Main quarantine simulation supporting flexible windows and multiple events.

    Returns a dict containing:
      events                         - echo of input
      affected_assignments           - DataFrame of originally-assigned shifts now lost
      quarantined_doctors_by_date    - dict date -> sorted list of doctor names
      coverage_impact_pct            - % of monthly covered slots affected
      days_affected                  - unique calendar days touched
      dept_impact                    - dict dept -> shifts lost
      severity                       - LOW / MEDIUM / HIGH
      recommendation                 - human-readable summary

    Plus mini-resolve outputs:
      reserve_assignments, reassignments, covered_by_reserve, reassigned_count,
      residual_gap, residual_uncovered, reserve_total_cost, solve_time.
    """
    schedule_df = results["schedule_df"]
    name_to_id = {v: k for k, v in id_to_name.items()}

    if not events:
        return {
            "events": [],
            "affected_assignments": pd.DataFrame(),
            "quarantined_doctors_by_date": {},
            "coverage_impact_pct": 0.0,
            "days_affected": 0,
            "dept_impact": {},
            "severity": "NONE",
            "recommendation": "No quarantine events specified.",
            **_empty_mini_result("NO_QUARANTINE"),
        }

    affected = get_quarantine_affected_assignments(
        schedule_df, events, bubble_assignment, id_to_name
    )

    total_covered = int(
        schedule_df[schedule_df["Missing"] == 0][["Date", "ShiftCode", "Doctor"]]
        .drop_duplicates()
        .shape[0]
    )
    lost = int(affected.shape[0]) if not affected.empty else 0
    coverage_impact_pct = round(100.0 * lost / max(1, total_covered), 1)

    if affected.empty:
        days_affected = 0
        dept_impact: dict[str, int] = {}
    else:
        if pd.api.types.is_datetime64_any_dtype(affected["Date"]):
            days_affected = int(affected["Date"].dt.normalize().nunique())
        else:
            days_affected = int(pd.to_datetime(affected["Date"]).dt.normalize().nunique())
        dept_impact = affected.groupby("Dept").size().to_dict()

    quarantined_by_date: dict[str, set[str]] = {}
    for ev in events:
        ids = ev.get_affected_doctor_ids(bubble_assignment, name_to_id)
        names = {id_to_name[d] for d in ids if d in id_to_name}
        for offset in range(ev.duration_days):
            d = datetime.strptime(ev.start_date, "%Y-%m-%d") + timedelta(days=offset)
            ds = d.strftime("%Y-%m-%d")
            quarantined_by_date.setdefault(ds, set()).update(names)
    quarantined_by_date_sorted = {k: sorted(v) for k, v in quarantined_by_date.items()}

    if coverage_impact_pct < 10:
        severity = "LOW"
        action = "Local impact - the schedule should absorb the loss."
    elif coverage_impact_pct < 25:
        severity = "MEDIUM"
        action = "Activate reserves and consider deferring elective work."
    else:
        severity = "HIGH"
        action = (
            "Critical loss of capacity. Reserves may be insufficient. "
            "Consider closing outpatient clinics and escalating to hospital management."
        )

    if use_mini_resolve:
        coverage = mini_resolve_quarantine_window(
            results=results,
            events=events,
            bubble_assignment=bubble_assignment,
            id_to_name=id_to_name,
            config=config,
            reserve_doctors_df=reserve_doctors_df,
            time_limit_seconds=time_limit_seconds,
            extra_doctors_df=extra_doctors_df,
        )
    else:
        coverage = _empty_mini_result("SKIPPED", residual_uncovered=affected)

    event_labels = "; ".join(ev.label() for ev in events)
    recommendation = (
        f"[{severity}] Quarantine: {event_labels}. "
        f"Affected: {lost} shifts across {days_affected} days "
        f"({coverage_impact_pct}% of monthly coverage). "
        f"Reserves cover: {coverage.get('covered_by_reserve', 0)}, "
        f"internal redistribution: {coverage.get('reassigned_count', 0)}, "
        f"still uncovered: {coverage.get('residual_gap', 0)}. "
        f"Departments hit: {dept_impact}. {action}"
    )

    return {
        "events": events,
        "affected_assignments": affected,
        "quarantined_doctors_by_date": quarantined_by_date_sorted,
        "coverage_impact_pct": coverage_impact_pct,
        "days_affected": days_affected,
        "dept_impact": dept_impact,
        "severity": severity,
        "recommendation": recommendation,
        **coverage,
    }


# =============================================================================
# LEGACY WRAPPERS (kept for backwards compatibility with pandemic_ui.py and main)
# =============================================================================

def _full_period_event(results: dict, bubble: int) -> QuarantineEvent:
    sched = results["schedule_df"]
    dates = pd.to_datetime(sched["Date"])
    start = dates.min().strftime("%Y-%m-%d")
    end = dates.max()
    duration = (end - dates.min()).days + 1
    return QuarantineEvent(
        target_type="bubble",
        target=bubble,
        start_date=start,
        duration_days=duration,
        reason="legacy_full_period",
    )


def simulate_quarantine(
    results: dict,
    quarantine_bubble: int,
    bubble_assignment: dict[int, int],
    id_to_name: dict[int, str],
    config: PandemicConfig,
) -> dict:
    """Legacy entry point - quarantines a whole bubble for the whole planning period."""
    event = _full_period_event(results, quarantine_bubble)
    v2 = simulate_quarantine_v2(
        results=results,
        events=[event],
        bubble_assignment=bubble_assignment,
        id_to_name=id_to_name,
        config=config,
        reserve_doctors_df=None,
        use_mini_resolve=False,
    )

    name_to_id = {v: k for k, v in id_to_name.items()}
    qids = event.get_affected_doctor_ids(bubble_assignment, name_to_id)
    qnames = sorted(id_to_name[d] for d in qids if d in id_to_name)
    bubble_sizes = {
        b: sum(1 for bb in bubble_assignment.values() if bb == b)
        for b in range(config.num_bubbles)
    }

    return {
        "quarantine_bubble": quarantine_bubble,
        "quarantined_doctors": qnames,
        "uncovered_shifts": v2["affected_assignments"],
        "coverage_impact_pct": v2["coverage_impact_pct"],
        "days_affected": v2["days_affected"],
        "bubble_sizes": bubble_sizes,
        "dept_impact": v2["dept_impact"],
        "severity": v2["severity"],
        "recommendation": v2["recommendation"],
    }


def simulate_quarantine_with_reserve(
    results: dict,
    quarantine_bubble: int,
    bubble_assignment: dict[int, int],
    id_to_name: dict[int, str],
    config: PandemicConfig,
    reserve_doctors_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Legacy entry point - now backed by the mini CP-SAT re-solve."""
    event = _full_period_event(results, quarantine_bubble)
    v2 = simulate_quarantine_v2(
        results=results,
        events=[event],
        bubble_assignment=bubble_assignment,
        id_to_name=id_to_name,
        config=config,
        reserve_doctors_df=reserve_doctors_df,
        use_mini_resolve=True,
    )

    name_to_id = {v: k for k, v in id_to_name.items()}
    qids = event.get_affected_doctor_ids(bubble_assignment, name_to_id)
    qnames = sorted(id_to_name[d] for d in qids if d in id_to_name)
    bubble_sizes = {
        b: sum(1 for bb in bubble_assignment.values() if bb == b)
        for b in range(config.num_bubbles)
    }

    return {
        "quarantine_bubble": quarantine_bubble,
        "quarantined_doctors": qnames,
        "uncovered_shifts": v2["affected_assignments"],
        "coverage_impact_pct": v2["coverage_impact_pct"],
        "days_affected": v2["days_affected"],
        "bubble_sizes": bubble_sizes,
        "dept_impact": v2["dept_impact"],
        "severity": v2["severity"],
        "recommendation": v2["recommendation"],
        "reserve_available": reserve_doctors_df is not None and not reserve_doctors_df.empty,
        "covered_by_reserve": v2["covered_by_reserve"],
        "residual_gap": v2["residual_gap"],
        "reserve_assignments": v2["reserve_assignments"],
        "residual_uncovered": v2["residual_uncovered"],
        "reserve_total_cost": v2.get("reserve_total_cost", 0),
        "reassigned_count": v2.get("reassigned_count", 0),
        "reassignments": v2.get("reassignments", pd.DataFrame()),
    }


# =============================================================================
# EMERGENCY REPLACEMENT (kept from original, lightly cleaned up)
# =============================================================================

def find_replacement_pandemic(
    results: dict,
    sick_doctor: str,
    shift_code: str,
    bubble_assignment: dict[int, int],
    id_to_name: dict[int, str],
    config: PandemicConfig,
    reserve_doctors_df: Optional[pd.DataFrame] = None,
    quarantined_by_date: Optional[dict[str, list[str]]] = None,
) -> dict:
    """
    Emergency replacement with pandemic rules.

    All labour-law checks (rest, weekly limit, nights, working days,
    unavailability, mentor) come from ``find_replacement``. Also:
      - doctors quarantined on the shift date are excluded,
      - on contact-department shifts a doctor may only join if they are in the
        same bubble as the doctors remaining on that shift;
    """
    base = find_replacement(
        results=results,
        sick_doctor=sick_doctor,
        shift_code=shift_code,
        reserve_doctors_df=reserve_doctors_df if config.allow_cross_bubble_reserve else None,
        candidates_df=None,
    )

    name_to_id = {v: k for k, v in id_to_name.items()}
    sick_bubble = bubble_assignment.get(name_to_id.get(sick_doctor), -999)
    shift_info = base["shift_info"]

    pandemic_fields = {
        "sick_bubble": sick_bubble,
        "pandemic_mode": True,
    }

    if not shift_info:
        return {
            **base,
            **pandemic_fields,
            "bubble_restrictions_applied": False,
            "is_contact_dept": False,
        }

    is_contact_dept = shift_info["dept"] in config.contact_depts

    schedule_df = results["schedule_df"]
    coworkers = schedule_df[
        (schedule_df["ShiftCode"] == shift_code)
        & schedule_df["Doctor"].notna()
        & (schedule_df["Doctor"] != sick_doctor)
    ]["Doctor"]
    coworker_bubbles = {
        bubble_assignment.get(name_to_id.get(n)) for n in coworkers
    } - {None, CLINIC_POOL}

    quarantined = set((quarantined_by_date or {}).get(shift_info["date"], []))

    kept: list[dict] = []
    for cand in base["candidates"]:
        cand = dict(cand)
        cand["warnings"] = list(cand["warnings"])
        cand["other_bubble"] = False

        if cand["source"] == "reserve":
            cand["bubble"] = "external"
            cand["warnings"].insert(0, "EXTERNAL RESERVE - isolated from the bubble system")
            kept.append(cand)
            continue

        if cand["name"] in quarantined:
            continue

        bubble = bubble_assignment.get(name_to_id.get(cand["name"]))
        cand["bubble"] = bubble

        if is_contact_dept:
            if coworker_bubbles:
                if bubble not in coworker_bubbles:
                    continue
            elif bubble != sick_bubble:
                cand["other_bubble"] = True
                cand["warnings"].append(
                    f"Different bubble than the absent doctor ({sick_bubble}); "
                    "no co-workers on this shift, so no cross-bubble contact"
                )

        kept.append(cand)

    kept.sort(
        key=lambda c: (not (c["rest_ok"] and c["skill_ok"]), c["other_bubble"], c["score"])
    )
    valid = [c for c in kept if c["rest_ok"] and c["skill_ok"]]
    best = valid[0] if valid else None

    fallback_message = None
    if not kept:
        fallback_message = (
            f"No candidates available for shift {shift_code} under bubble rules. "
            "All eligible physicians may be quarantined - activate external reserve staff."
        )
    elif best is None:
        fallback_message = (
            "No candidate satisfies all labour-law and competency constraints "
            "under bubble rules. Reserves or policy relaxation needed."
        )

    return {
        "shift_code": shift_code,
        "sick_doctor": sick_doctor,
        "shift_info": shift_info,
        "candidates": kept,
        "recommendation": best,
        "fallback_message": fallback_message,
        **pandemic_fields,
        "bubble_restrictions_applied": is_contact_dept,
        "is_contact_dept": is_contact_dept,
        "coworker_bubbles": sorted(coworker_bubbles),
    }


# =============================================================================
# HELPERS FOR UI (contact matrix, summary df) - unchanged
# =============================================================================

def bubble_contact_matrix(
    schedule_df: pd.DataFrame,
    bubble_assignment: dict[int, int],
    name_to_id: dict[str, int],
    contact_depts: list[str],
) -> pd.DataFrame:
    """Count how many times pairs of bubbles shared a shift in contact depts."""
    contact_df = schedule_df[
        schedule_df["Dept"].isin(contact_depts)
        & schedule_df["Doctor"].notna()
    ].copy()
    contact_df["DoctorID"] = contact_df["Doctor"].map(name_to_id)
    contact_df["Bubble"] = contact_df["DoctorID"].map(bubble_assignment)

    contacts = []
    for (date, code), group in contact_df.groupby(["Date", "ShiftCode"]):
        bubbles = (
            group["Bubble"].dropna().astype(int).unique().tolist()
        )
        # exclude CLINIC_POOL from cross-bubble accounting
        bubbles = [b for b in bubbles if b != CLINIC_POOL]
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
    """Doctor / Role / Bubble table - CLINIC_POOL is reported as 'CLINIC'."""
    rows = []
    for d, b in bubble_assignment.items():
        if b == CLINIC_POOL:
            label = "CLINIC pool"
        else:
            label = b
        rows.append({
            "Doctor": id_to_name.get(d, str(d)),
            "Role": id_to_role.get(d, ""),
            "Bubble": label,
        })
    return pd.DataFrame(rows).sort_values(["Bubble", "Role", "Doctor"])