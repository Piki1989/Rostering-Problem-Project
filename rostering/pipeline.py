"""End-to-end run: load data, build and solve the baseline (and, if needed, the hiring) scenario, save."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import RunConfig
from .data import ProblemData, check_inputs, load_dataset
from .metrics import compute_hospital_efficiency_v2
from .model import build_model, prepare_context
from .pandemic import PandemicConfig, get_pandemic_results, prepare_pandemic_bubbles, validate_bubbles
from .results import extract_results
from .solve import solve_lexicographic

logger = logging.getLogger(__name__)


def run_scenario(
    data: ProblemData,
    *,
    allow_hire: bool,
    config: RunConfig,
    pandemic_config: PandemicConfig | None = None,
    bubble_assignment: dict | None = None,
    validation_result=None,
) -> dict:
    check_inputs(data)
    ctx = prepare_context(data.doctors, data.candidates, data.shifts)
    m = build_model(ctx, data, allow_hire=allow_hire,
                    pandemic_config=pandemic_config, bubble_assignment=bubble_assignment)
    outcome = solve_lexicographic(m, config.solver, config.targets, pandemic_mode=pandemic_config is not None)

    results = extract_results(m, outcome.solver, outcome.status)
    results.update({
        "status_name": outcome.solver.StatusName(outcome.status),
        "phases": [vars(ph) for ph in outcome.phases],
        "allow_hire": allow_hire,
        "doctors": data.doctors,
        "shifts": data.shifts,
        "unavail_day": data.unavail_day,
        "unavail_shift": data.unavail_shift,
        "pref": data.pref,
        "weekly_limits": m.p.weekly_limits,
    })

    if pandemic_config is not None and bubble_assignment is not None:
        hired = {c for c, h in results["raw_hire"].items() if h == 1}
        candidate_ids = set(data.candidates["id"])
        final_bubbles = {d: b for d, b in bubble_assignment.items() if d not in candidate_ids or d in hired}
        results["pandemic"] = {
            "config": pandemic_config,
            "bubble_assignment": final_bubbles,
            "validation": validation_result,
            "summary": get_pandemic_results(
                bubble_assignment=final_bubbles, config=pandemic_config,
                id_to_name=ctx.id_to_name, id_to_role=ctx.id_to_role,
            ),
        }
    return results


def save_results(results: dict, base_dir: Path, label: str, ts: str) -> Path:
    out_dir = base_dir / f"{label}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results["schedule_df"].to_csv(out_dir / f"schedule_{ts}.csv", index=False)
    results["stats_df"].to_csv(out_dir / f"doctor_stats_{ts}.csv", index=False)
    results["solver_stats_df"].to_csv(out_dir / f"solver_stats_{ts}.csv", index=False)
    if results["hired_df"] is not None:
        results["hired_df"].to_csv(out_dir / f"hired_{ts}.csv", index=False)
    pd.DataFrame(results.get("phases", [])).to_csv(out_dir / f"phases_{ts}.csv", index=False)

    efficiency = compute_hospital_efficiency_v2(results)["HospitalEfficiencyScore"]
    pd.DataFrame([{
        "total_cost": results.get("total_cost"),
        "avg_satisfaction": results.get("avg_satisfaction"),
        "min_satisfaction": results.get("min_satisfaction"),
        "max_satisfaction": results.get("max_satisfaction"),
        "hospital_efficiency": efficiency,
        "objective_value": results.get("objective"),
        "slack": results.get("slack"),
        "night_spread": results.get("spread"),
        "weekend_spread": results.get("weekend_spread"),
        "hard_spread": results.get("hard_spread"),
        "allow_hire": results.get("allow_hire"),
        "status": results.get("status_name"),
    }]).to_csv(out_dir / f"summary_{ts}.csv", index=False)
    return out_dir


def run_model_and_get_results(
    doctors_df: pd.DataFrame | None = None,
    pandemic_mode: bool | None = None,
    config: RunConfig | None = None,
) -> dict:
    """Baseline without hiring; if some positions stay uncovered, a second run that may hire."""
    config = config or RunConfig()
    if pandemic_mode is not None:
        config.pandemic_mode = pandemic_mode

    data = load_dataset(config.dataset)
    if doctors_df is not None:
        data = data.with_doctors(doctors_df)

    pandemic_config = bubble_assignment = validation_result = None
    if config.pandemic_mode:
        pandemic_config = PandemicConfig(
            num_bubbles=config.pandemic.num_bubbles,
            contact_depts=list(config.pandemic.contact_depts),
            min_specialists_per_bubble=config.pandemic.min_specialists_per_bubble,
        )
        bubble_assignment = prepare_pandemic_bubbles(
            doctors=data.doctors, candidates=data.candidates, config=pandemic_config)
        validation_result = validate_bubbles(
            doctors=data.doctors, candidates=data.candidates, shifts=data.shifts,
            bubble_assignment=bubble_assignment, config=pandemic_config,
        )

    common = dict(config=config, pandemic_config=pandemic_config, bubble_assignment=bubble_assignment)
    logger.info("Scenario: baseline (no hiring), dataset %s, pandemic %s", config.dataset, config.pandemic_mode)
    baseline = run_scenario(data, allow_hire=False, validation_result=validation_result, **common)
    results = {"baseline": baseline, "hire_allowed": None}

    if baseline["slack"] > 0:
        logger.info("Scenario: hiring allowed (%d uncovered positions in the baseline)", baseline["slack"])
        results["hire_allowed"] = run_scenario(data, allow_hire=True, **common)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results["run"] = ts
    if config.save:
        out = config.output_dir()
        save_results(results["baseline"], out, "baseline", ts)
        if results["hire_allowed"] is not None:
            save_results(results["hire_allowed"], out, "hire_allowed", ts)
        logger.info("Results saved in %s", out)
    return results


def print_full_report(results: dict, max_schedule_rows: int = 30) -> None:
    print("\n=== SOLVER SUMMARY ===")
    print(results["solver_stats_df"].to_string(index=False))
    print("\n=== PHASES ===")
    print(pd.DataFrame(results.get("phases", [])).to_string(index=False))
    print("\n=== OBJECTIVE & SLACK ===")
    print(f"Objective value : {results['objective']}")
    print(f"Total slack     : {results['slack']}")
    print(f"Total cost      : {results['total_cost']}")
    print("\n=== FAIRNESS METRICS ===")
    print(f"Night spread    : {results['spread']}")
    print(f"Weekend spread  : {results['weekend_spread']}")
    print(f"Hard spread     : {results['hard_spread']}")
    print(f"Avg satisfaction: {results['avg_satisfaction']}")
    print("\n=== HIRED CANDIDATES ===")
    print("None" if results["hired_df"].empty else results["hired_df"].to_string(index=False))
    print("\n=== DOCTOR STATS ===")
    print(results["stats_df"].to_string(index=False))
    print(f"\n=== SCHEDULE (first {max_schedule_rows} rows) ===")
    print(results["schedule_df"].head(max_schedule_rows).to_string(index=False))
