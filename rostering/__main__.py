"""Command line run: uv run python -m rostering --dataset dec2026_pandemic --pandemic"""
from __future__ import annotations

import argparse
import logging

from .config import DATASETS, DEFAULT_DATASET, RunConfig
from .pipeline import print_full_report, run_model_and_get_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default=DEFAULT_DATASET)
    parser.add_argument("--pandemic", action="store_true", help="pandemic mode (bubbles, no 24 h shifts)")
    parser.add_argument("--workers", type=int, help="CP-SAT search workers (default max(4, cpus // 2))")
    parser.add_argument("--seed", type=int, help="random seed (use with --workers 1 for reproducible runs)")
    parser.add_argument("--hints", action="store_true", help="warm start every phase from the previous one")
    parser.add_argument("--time-limits", type=float, nargs=3, metavar=("P1", "P2", "P3"),
                        help="time limits of the phases in seconds (default 100 100 300)")
    parser.add_argument("--no-save", action="store_true", help="do not write results/<dataset>/")
    parser.add_argument("--report", action="store_true", help="print the full report of every scenario")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging and solver progress")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = RunConfig(dataset=args.dataset, pandemic_mode=args.pandemic, save=not args.no_save)
    config.solver.num_workers = args.workers
    config.solver.random_seed = args.seed
    config.solver.use_hints = args.hints
    config.solver.log_search_progress = args.verbose
    if args.time_limits:
        config.solver.time_limit_phase1, config.solver.time_limit_phase2, config.solver.time_limit_phase3 = args.time_limits

    results = run_model_and_get_results(config=config)
    for label in ("baseline", "hire_allowed"):
        res = results.get(label)
        if res is None:
            continue
        print(f"\n================ {label.upper()} ================")
        print(f"status={res['status_name']} slack={res['slack']} cost={res['total_cost']} "
              f"spreads={res['spread']}/{res['weekend_spread']}/{res['hard_spread']} "
              f"avg_satisfaction={res['avg_satisfaction']}")
        if args.report:
            print_full_report(res)


if __name__ == "__main__":
    main()
