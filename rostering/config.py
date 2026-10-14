"""Run configuration: input datasets, solver settings and the targets of the final (Chebyshev) phase.

Everything that used to be switched by commenting code in model_ref.py lives here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"


@dataclass(frozen=True)
class DatasetFiles:
    """CSV files (relative to data/) describing one planning month."""
    doctors: str
    shifts: str
    unavail_day: str
    unavail_shift: str
    pref: str
    candidates: str = "doctors_to_hire/doctors_to_hire.csv"
    reserve: str = "reserve/reserve_doctors.csv"


# The alternatives that were commented out in run_model_and_get_results().
DATASETS: dict[str, DatasetFiles] = {
    "dec2026_pandemic": DatasetFiles(
        doctors="doctors/doctors_with_cost.csv",
        shifts="shifts_month/shifts_dec2026_pandemic.csv",
        unavail_day="unavail_monthly_day/unavail_dec_2026.csv",
        unavail_shift="unavail_monthly_shift/unavail_sh_dec_2026.csv",
        pref="pref_monthly/pref_dec.csv",
    ),
    "dec2026": DatasetFiles(
        doctors="doctors/doctors_with_cost.csv",
        shifts="shifts_month/shifts_dec2026.csv",
        unavail_day="unavail_monthly_day/unavail_dec_2026.csv",
        unavail_shift="unavail_monthly_shift/unavail_sh_dec_2026.csv",
        pref="pref_monthly/pref_dec.csv",
    ),
    "dec2026_1": DatasetFiles(
        doctors="doctors/doctors_with_cost.csv",
        shifts="shifts_month/shifts_dec_1.csv",
        unavail_day="unavail_monthly_day/unavail_dec_2026_1.csv",
        unavail_shift="unavail_monthly_shift/unavail_sh_dec_2026_1.csv",
        pref="pref_monthly/pref_dec_1.csv",
    ),
    "mar2026": DatasetFiles(
        doctors="doctors/doctors_with_cost.csv",
        shifts="shifts_month/shifts_mar2026.csv",
        unavail_day="unavail_monthly_day/unavailability_mar2026.csv",
        unavail_shift="unavail_monthly_shift/shift_unavailability_mar2026.csv",
        pref="pref_monthly/preferences_mar2026.csv",
    ),
    "h30": DatasetFiles(
        doctors="doctors/doctors_30.csv",
        shifts="shifts_month/shifts_30.csv",
        unavail_day="unavail_monthly_day/unavail_day_30.csv",
        unavail_shift="unavail_monthly_shift/unavail_shift_30.csv",
        pref="pref_monthly/pref_30.csv",
    ),
    "h45": DatasetFiles(
        doctors="doctors/doctors_45.csv",
        shifts="shifts_month/shifts_45.csv",
        unavail_day="unavail_monthly_day/unavail_day_45.csv",
        unavail_shift="unavail_monthly_shift/unavail_shift_45.csv",
        pref="pref_monthly/pref_45.csv",
    ),
}
DEFAULT_DATASET = "dec2026_pandemic"


@dataclass(frozen=True)
class ObjectiveTargets:
    """Reference values of the Chebyshev phase: Z >= (metric - target) * SCALE // max_metric."""
    cost: int = 540_000                 # budget B; cost above B is penalised (see max_over_budget_pct)
    max_over_budget_pct: float = 0.05   # NOTE: also the upper bound of over_budget -> hard cap cost <= B * 1.05
    spread: int = 4
    weekend_spread: int = 3
    hard_spread: int = 2
    underwork: int = 0
    weekly_underwork: int = 0
    pref: int = -300
    scale: int = 1_000_000


# Targets used by the single-phase comparison experiment (solve_single_phase_comparison).
SINGLE_PHASE_TARGETS = ObjectiveTargets(cost=1_620_000, max_over_budget_pct=0.15, pref=-200)


@dataclass
class SolverConfig:
    time_limit_phase1: float = 100      # weighted understaffing
    time_limit_isolation: float = 60    # pandemic: cross-bubble contacts
    time_limit_phase2: float = 100      # number of hires
    time_limit_phase3: float = 300      # Chebyshev compromise
    num_workers: int | None = None      # None -> max(4, cpu_count // 2) as in the original code
    use_hints: bool = True              # warm start every phase from the previous phase's solution
    feasibility_probe: bool = False     # the original 30 s "FEASIBILITY CHECK" solve before phase 1
    random_seed: int | None = None      # fix for reproducible runs (together with num_workers=1)
    log_search_progress: bool = False

    def workers(self) -> int:
        if self.num_workers:
            return self.num_workers
        return max(4, (os.cpu_count() or 1) // 2)


@dataclass
class PandemicSettings:
    num_bubbles: int = 3
    contact_depts: list[str] = field(default_factory=lambda: ["WARD", "ICU"])
    min_specialists_per_bubble: int = 1


@dataclass
class RunConfig:
    dataset: str = DEFAULT_DATASET
    pandemic_mode: bool = False
    pandemic: PandemicSettings = field(default_factory=PandemicSettings)
    solver: SolverConfig = field(default_factory=SolverConfig)
    targets: ObjectiveTargets = field(default_factory=ObjectiveTargets)
    save: bool = True
    results_dir: Path | None = None     # default: results/<dataset>

    def output_dir(self) -> Path:
        return self.results_dir or RESULTS_DIR / self.dataset

    @classmethod
    def from_env(cls) -> "RunConfig":
        """Settings for the Streamlit app.

        ROSTERING_DATASET, ROSTERING_PANDEMIC=0/1, ROSTERING_HINTS=0/1, ROSTERING_TIME_LIMITS="P1,P2,P3".
        """
        cfg = cls(
            dataset=os.environ.get("ROSTERING_DATASET", DEFAULT_DATASET),
            pandemic_mode=os.environ.get("ROSTERING_PANDEMIC", "1") == "1",
        )
        cfg.solver.use_hints = os.environ.get("ROSTERING_HINTS", "1") == "1"
        if os.environ.get("ROSTERING_TIME_LIMITS"):
            p1, p2, p3 = (float(v) for v in os.environ["ROSTERING_TIME_LIMITS"].split(","))
            cfg.solver.time_limit_phase1, cfg.solver.time_limit_phase2, cfg.solver.time_limit_phase3 = p1, p2, p3
        return cfg
