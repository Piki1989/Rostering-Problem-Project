"""Regenerate tests/golden/model_fingerprints.json from the current code.

Without arguments it only reports whether the equations changed. After an intended change of the
equations (describe it in docs/MODEL.md) write the new fingerprints with:
    uv run python tests/record_fingerprints.py --write
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parent)]

from ortools.sat.python import cp_model  # noqa: E402

from fingerprint import RecordingSolver, fingerprint  # noqa: E402
from scenarios import SCENARIOS, scenario_id  # noqa: E402
from test_model_equivalence import GOLDEN, build, equations_only  # noqa: E402

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
        old = GOLDEN.get(sid, {})
        same = (equations_only(old.get("final", {})) == equations_only(entry["final"])
                and [equations_only(c) for c in old.get("solve_calls", [])]
                == [equations_only(c) for c in entry["solve_calls"]])
        print(f"{sid}: {'equations unchanged' if same else 'CHANGED'}")
    if "--write" in sys.argv:
        path = Path(__file__).parent / "golden" / "model_fingerprints.json"
        path.write_text(json.dumps(golden, indent=1, sort_keys=True))
        print(f"written {path}")


if __name__ == "__main__":
    main()
