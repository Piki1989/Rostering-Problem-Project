"""Regenerate tests/golden/model_fingerprints.json from the current code.

Only after an intended change of the equations (describe it in docs/MODEL.md):
    uv run python tests/record_fingerprints.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parent)]

from ortools.sat.python import cp_model  # noqa: E402

from fingerprint import RecordingSolver, fingerprint  # noqa: E402
from scenarios import SCENARIOS, scenario_id  # noqa: E402
from test_model_equivalence import GOLDEN, build  # noqa: E402

from rostering.config import SolverConfig  # noqa: E402
from rostering.solve import solve_lexicographic  # noqa: E402


def main() -> None:
    golden = {}
    real = cp_model.CpSolver
    for name, pandemic, allow_hire in SCENARIOS:
        sid = scenario_id(name, pandemic, allow_hire)
        m, bubbles = build(name, pandemic, allow_hire)
        entry = {"build": fingerprint(m.model),
                 "bubbles": {str(k): v for k, v in sorted(bubbles.items())} if bubbles else None}
        RecordingSolver.calls = []
        cp_model.CpSolver = RecordingSolver
        try:
            solve_lexicographic(m, SolverConfig(), pandemic_mode=pandemic)
        finally:
            cp_model.CpSolver = real
        entry["solve_calls"] = RecordingSolver.calls
        entry["final"] = fingerprint(m.model)
        golden[sid] = entry
        print(f"{sid}: {'unchanged' if GOLDEN.get(sid) == entry else 'CHANGED'}")
    path = Path(__file__).parent / "golden" / "model_fingerprints.json"
    path.write_text(json.dumps(golden, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
