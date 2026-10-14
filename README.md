# Rostering-Problem-Project

Hospital doctor rostering with a CP-SAT model (OR-Tools) and a Streamlit app.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
```

## Run

Streamlit app:

```bash
uv run streamlit run app/streamlit_new.py
```

The app opens in the browser. The first load runs the optimization and takes about 5 minutes, which is mostly the time limit of phase 3.

Command line (results are written to `results/<dataset>/`):

```bash
uv run python -m rostering --dataset dec2026_pandemic --pandemic
uv run python -m rostering --help
```

## Options

- **Input data**: the named datasets are listed in `DATASETS` in `rostering/config.py`.
  - Command line: `--dataset`.
  - App: environment variable `ROSTERING_DATASET`.
- **Pandemic mode**: `--pandemic` on the command line. In the app it is on by default; set `ROSTERING_PANDEMIC=0` to turn it off.
- **Solver**: `SolverConfig` in `rostering/config.py` sets the phase time limits, the number of workers, the warm start between phases (`use_hints`) and the random seed.
  - Command line: `--time-limits P1 P2 P3`, `--workers`, `--hints`, `--seed`.
  - App: `ROSTERING_TIME_LIMITS="P1,P2,P3"`.
  - On `dec2026_pandemic`, phase 3 found its final solution within 12-45 s; the rest of the default 300 s limit brought no improvement. `--time-limits 100 100 60` is about 5× faster on this data.
- **Phase 3 targets**: `ObjectiveTargets` holds the budget, the spread targets and the preference target.

## Layout

| Path | Content |
|---|---|
| `rostering/config.py` | datasets, solver settings, objective targets |
| `rostering/data.py` | loading and checking the CSV files |
| `rostering/model.py` | sets, parameters, model container, `build_model` |
| `rostering/equations.py` | every equation as a separate function (H1-H13, HP1, C1-C2, D1-D3, S1, F1-F4, U1-U2, O1-O3) |
| `rostering/oracles.py` | independent pure-Python evaluation of every equation on a schedule |
| `rostering/verify.py` | checks one equation (or a replacement implementation) against its oracle |
| `rostering/solve.py` | lexicographic phases (understaffing → isolation → hiring → Chebyshev compromise) |
| `rostering/results.py` | schedule, per-doctor statistics, satisfaction score |
| `rostering/pipeline.py` | baseline / hiring scenarios, saving results |
| `rostering/pandemic.py` | pandemic mode: bubbles, isolation constraints, quarantine simulation |
| `rostering/emergency_replacement.py` | replacement of a sick doctor |
| `rostering/metrics.py` | hospital efficiency score |
| `app/` | Streamlit UI |
| `docs/MODEL.md` | mathematical specification of the model and its known issues |
| `docs/EQUATIONS.md` | contract of every equation function (generated) |
| `tests/` | model equivalence test, per-equation tests |

## Checking single equations

Every equation is a separate function with the same interface.
- Each one can be checked alone, and an alternative implementation (e.g. written from a prompt) can be checked in its place.
- The contract of each function is in `docs/EQUATIONS.md`.

```bash
uv run python -m rostering.verify --list
uv run python -m rostering.verify --all
uv run python -m rostering.verify H3 --impl candidate.py:my_h3
```

## Tests

```bash
uv run pytest
```

`tests/test_model_equivalence.py` checks the model against fingerprints recorded with the original implementation (commit `a2610f5`).
- It covers 12 scenarios: 3 datasets × normal/pandemic mode × with/without hiring.
- The model must be identical in every variable, constraint and objective, including the constraints added by each solver phase.
- Any change to the equations makes the test fail.
- After an intended change, regenerate the fingerprints and describe the change in `docs/MODEL.md`.

`tests/test_equations.py` covers the equations one by one.
- Every reference equation must agree with its oracle.
- Ten deliberately wrong implementations must be rejected.
