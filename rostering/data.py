"""Loading and checking the input CSV files."""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import pandas as pd

from .config import DATA_DIR, DATASETS, DatasetFiles

logger = logging.getLogger(__name__)


@dataclass
class ProblemData:
    doctors: pd.DataFrame
    candidates: pd.DataFrame
    shifts: pd.DataFrame
    unavail_day: pd.DataFrame
    unavail_shift: pd.DataFrame
    pref: pd.DataFrame

    def with_doctors(self, doctors: pd.DataFrame) -> "ProblemData":
        return replace(self, doctors=prepare_doctors(doctors.copy()))


def _skill_list(value) -> list[str]:
    return value.split(";") if isinstance(value, str) else []


def prepare_doctors(doctors: pd.DataFrame) -> pd.DataFrame:
    doctors["skill_list"] = doctors["skills"].apply(_skill_list)
    doctors["hourly_cost"] = doctors["hourly_cost"].astype(int)
    return doctors


def load_dataset(name_or_files: str | DatasetFiles) -> ProblemData:
    files = DATASETS[name_or_files] if isinstance(name_or_files, str) else name_or_files
    read = lambda rel: pd.read_csv(DATA_DIR / rel)  # noqa: E731
    candidates = read(files.candidates)
    candidates["skill_list"] = candidates["skills"].apply(_skill_list)
    return ProblemData(
        doctors=prepare_doctors(read(files.doctors)),
        candidates=candidates,
        shifts=read(files.shifts),
        unavail_day=read(files.unavail_day),
        unavail_shift=read(files.unavail_shift),
        pref=read(files.pref),
    )


def check_inputs(data: ProblemData) -> list[str]:
    """Consistency checks of the input files (the "DEBUG IDS" prints of the original run_scenario).

    Returns the list of problems found; they are also logged as warnings.
    """
    doctors, candidates, shifts = data.doctors, data.candidates, data.shifts
    doctor_ids, candidate_ids = set(doctors["id"]), set(candidates["id"])
    shift_dates, shift_codes = set(shifts["date"]), set(shifts["code"])
    problems = []

    for label, df in (("unavail_day", data.unavail_day), ("unavail_shift", data.unavail_shift), ("pref", data.pref)):
        unknown = set(df["doctor_id"]) - doctor_ids
        if unknown:
            problems.append(f"{label}: unknown doctor ids {sorted(unknown)}")
    if doctor_ids & candidate_ids:
        problems.append(f"doctor and candidate ids overlap: {sorted(doctor_ids & candidate_ids)}")
    unknown_dates = set(data.unavail_day["date"]) - shift_dates
    if unknown_dates:
        problems.append(f"unavail_day: dates without shifts {sorted(unknown_dates)}")
    for label, df in (("unavail_shift", data.unavail_shift), ("pref", data.pref)):
        unknown_codes = set(df["code"]) - shift_codes
        if unknown_codes:
            problems.append(f"{label}: unknown shift codes {sorted(unknown_codes)[:10]}")

    for problem in problems:
        logger.warning("Input check: %s", problem)

    available_per_day = len(doctors) - data.unavail_day.groupby("date")["doctor_id"].nunique()
    logger.info(
        "Inputs: %d doctors, %d candidates, %d shifts, demand %d, min available doctors per day %s",
        len(doctors), len(candidates), len(shifts), int(shifts["min_staff"].sum()),
        int(available_per_day.min()) if len(available_per_day) else len(doctors),
    )
    for skill in shifts["required_skill"].unique():
        eligible = int(doctors["skills"].str.contains(skill, na=False).sum())
        logger.debug("Skill %s: %d eligible doctors", skill, eligible)
    return problems
