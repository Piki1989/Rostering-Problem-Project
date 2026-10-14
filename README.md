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

## Prompts: implementing the equations with an AI model

**Step-by-step instructions for the team (in Polish): [docs/INSTRUKCJA_PL.md](docs/INSTRUKCJA_PL.md).**

`rostering.prompting` builds one prompt per equation, sends it to a model, extracts the code and verifies it as above. It can optionally send the failures back for a corrected version.
- A prompt combines three files:
  - `prompts/template.md`;
  - the interface (`prompts/contract.md`);
  - the description of the equation from a **prompt set** (`prompts/sets/<set>/equations.md`).
- The sets differ in how much they tell the model:

| Set | Content |
|---|---|
| `s0_name` | the name of the equation only |
| `s1_intent` | the intent in one or two sentences |
| `s2_rules` | the exact rules without the CP-SAT encoding |
| `s3_spec` | full specification (domains, outputs) |
| `s4_formal` | `s3_spec` + the formal statement |

```bash
uv run python -m rostering.prompting sets                                   # list the sets
uv run python -m rostering.prompting prompts --set s1_intent --out preview  # only write the prompts
uv run python -m rostering.prompting run --set s2_rules --model sonnet      # one set, all equations
uv run python -m rostering.prompting run H3 O3 --set s2_rules --attempts 3  # chosen equations, with feedback
uv run python -m rostering.prompting benchmark --model sonnet --repeats 3   # compare the sets
uv run python -m rostering.prompting run --provider manual                  # write prompts, answer by hand ...
uv run python -m rostering.prompting check runs/<run>                       # ... then verify the answers
uv run python -m rostering.prompting run --provider reference               # sanity check of the pipeline
```

Providers:
- **`claude-cli`**: Claude Code CLI with no tools, no MCP servers and no session, run in an empty temporary directory, so the model cannot read the repository.
- **`anthropic`**: the API. Needs `uv sync --extra llm` and `ANTHROPIC_API_KEY`.
- **`command`**: any command that reads the prompt on stdin and prints the answer.
- **`manual`**: writes the prompts; you paste the answers yourself.
- **`reference`**: answers with the reference implementation.

Output, ignored by git:
- A run goes to `runs/<timestamp>_<set>_<provider>_<model>/`. For every equation and attempt it stores the prompt, the answer, the extracted code and the verification result. It also has `summary.csv` and `report.md`.
- A benchmark goes to `runs/benchmark_<timestamp>_.../`. It contains the per-set pass rates, the equation × set matrix and `benchmark.csv`.

The generated code runs locally in a separate process with a time limit. This isolates crashes, but it is not a sandbox.

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

`tests/test_prompting.py` tests the prompt pipeline without calling a model.
- It checks that the prompts contain no reference code.
- It checks code extraction, the feedback loop and the manual mode.
