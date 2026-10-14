"""The refactored model must contain exactly the equations of the original implementation.

tests/golden/model_fingerprints.json was recorded with the original code (model/model_ref.py on
main, commit a2610f5). For every scenario it holds the fingerprint of the built model and of the
model at every Solve() call of the lexicographic phases (objective + constraints added so far).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from ortools.sat.python import cp_model

from fingerprint import RecordingSolver, fingerprint
from scenarios import DATA_DIR, DATASETS, SCENARIOS, scenario_id

from rostering.config import PandemicSettings, SolverConfig
from rostering.data import ProblemData, prepare_doctors
from rostering.model import build_model, prepare_context
from rostering.pandemic import PandemicConfig, prepare_pandemic_bubbles
from rostering.solve import solve_lexicographic

GOLDEN = json.loads((Path(__file__).parent / "golden" / "model_fingerprints.json").read_text())


def load(name: str) -> ProblemData:
    frames = {k: pd.read_csv(DATA_DIR / v) for k, v in DATASETS[name].items()}
    frames["candidates"]["skill_list"] = frames["candidates"]["skills"].apply(
        lambda v: v.split(";") if isinstance(v, str) else [])
    frames["doctors"] = prepare_doctors(frames["doctors"])
    return ProblemData(**frames)


def build(name: str, pandemic: bool, allow_hire: bool):
    data = load(name)
    cfg = bubbles = None
    if pandemic:
        settings = PandemicSettings()
        cfg = PandemicConfig(num_bubbles=settings.num_bubbles, contact_depts=list(settings.contact_depts),
                             min_specialists_per_bubble=settings.min_specialists_per_bubble)
        bubbles = prepare_pandemic_bubbles(doctors=data.doctors, candidates=data.candidates, config=cfg)
    ctx = prepare_context(data.doctors, data.candidates, data.shifts)
    return build_model(ctx, data, allow_hire=allow_hire, pandemic_config=cfg, bubble_assignment=bubbles), bubbles


@pytest.mark.parametrize("name,pandemic,allow_hire", SCENARIOS, ids=[scenario_id(*s) for s in SCENARIOS])
def test_model_matches_original(name, pandemic, allow_hire, monkeypatch):
    golden = GOLDEN[scenario_id(name, pandemic, allow_hire)]
    m, bubbles = build(name, pandemic, allow_hire)

    if pandemic:
        assert {str(k): v for k, v in sorted(bubbles.items())} == golden["bubbles"]
    assert fingerprint(m.model) == golden["build"]

    RecordingSolver.calls = []
    monkeypatch.setattr(cp_model, "CpSolver", RecordingSolver)
    solve_lexicographic(m, SolverConfig(use_hints=True), pandemic_mode=pandemic)
    assert RecordingSolver.calls == golden["solve_calls"]
    assert fingerprint(m.model) == golden["final"]


def test_fingerprint_detects_a_changed_equation():
    m, _ = build("dec2026", pandemic=False, allow_hire=False)
    before = fingerprint(m.model)
    d, s = next(iter(m.x))
    m.model.Add(m.x[(d, s)] <= 1)
    assert fingerprint(m.model)["sha256"] != before["sha256"]
