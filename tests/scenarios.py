"""Input combinations used by the model-equivalence test (tests/test_model_equivalence.py)."""
from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Same files as the commented-out alternatives in the original run_model_and_get_results().
DATASETS = {
    "dec2026_pandemic": dict(
        doctors="doctors/doctors_with_cost.csv",
        shifts="shifts_month/shifts_dec2026_pandemic.csv",
        unavail_day="unavail_monthly_day/unavail_dec_2026.csv",
        unavail_shift="unavail_monthly_shift/unavail_sh_dec_2026.csv",
        pref="pref_monthly/pref_dec.csv",
        candidates="doctors_to_hire/doctors_to_hire.csv",
    ),
    "dec2026": dict(
        doctors="doctors/doctors_with_cost.csv",
        shifts="shifts_month/shifts_dec2026.csv",
        unavail_day="unavail_monthly_day/unavail_dec_2026.csv",
        unavail_shift="unavail_monthly_shift/unavail_sh_dec_2026.csv",
        pref="pref_monthly/pref_dec.csv",
        candidates="doctors_to_hire/doctors_to_hire.csv",
    ),
    "h30": dict(
        doctors="doctors/doctors_30.csv",
        shifts="shifts_month/shifts_30.csv",
        unavail_day="unavail_monthly_day/unavail_day_30.csv",
        unavail_shift="unavail_monthly_shift/unavail_shift_30.csv",
        pref="pref_monthly/pref_30.csv",
        candidates="doctors_to_hire/doctors_to_hire.csv",
    ),
}

# (dataset, pandemic_mode, allow_hire)
SCENARIOS = [
    (name, pandemic, allow_hire)
    for name in DATASETS
    for pandemic in (False, True)
    for allow_hire in (False, True)
]


def scenario_id(name: str, pandemic: bool, allow_hire: bool) -> str:
    return f"{name}-{'pandemic' if pandemic else 'normal'}-{'hire' if allow_hire else 'nohire'}"
