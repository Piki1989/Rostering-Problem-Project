"""Check one equation in isolation - the reference function or another implementation.

For every equation the model contains only the decision variables, the equations it requires
(reference implementations) and the equation under test. The decisions are then fixed to many
schedules (a valid reference schedule, mutations of it, an empty schedule, ...) and CP-SAT's
behaviour is compared with the independent oracle in rostering/oracles.py:

* restriction - CP-SAT finds the schedule feasible exactly when the oracle reports no violation,
* value       - the defined outputs take exactly the oracle values (minimised and maximised),
* minimum     - the minimised outputs take the oracle values,
* objective   - the optimum of the returned objective equals the oracle value,

and "infeasible" (oracle None) must be infeasible for CP-SAT too. The check does not depend on
variable names, so an implementation generated from a prompt can be verified as long as it
follows the contract in docs/EQUATIONS.md:

    uv run python -m rostering.verify H3 --impl candidate.py:h3_min_rest
    uv run python -m rostering.verify --all
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ortools.sat.python import cp_model

from . import oracles
from .config import BASE_DIR, PandemicSettings, SolverConfig
from .data import load_dataset
from .equations import CONSTRAINTS, EQUATIONS, Equation
from .model import RosteringModel, build_model, compute_params, prepare_context
from .oracles import Assignment
from .pandemic import PandemicConfig, prepare_pandemic_bubbles

DATASET = "dec2026_pandemic"
REFERENCE_FILE = BASE_DIR / "tests" / "golden" / "reference_schedules.json"
SOLVED = (cp_model.OPTIMAL, cp_model.FEASIBLE)


@dataclass(frozen=True)
class Check:
    kind: str                                            # restriction | value | minimum | objective
    oracle: Callable
    outputs: Callable[[RosteringModel], dict] | None = None
    scenario: str = "normal"                             # normal | pandemic


def _sum(terms):
    return cp_model.LinearExpr.Sum(list(terms)) if terms else 0


CHECKS: dict[str, Check] = {
    **{eq_id: Check("restriction", fn) for eq_id, fn in oracles.RESTRICTIONS.items() if eq_id != "H9a"},
    "H9a": Check("restriction", oracles.h9a, scenario="pandemic"),
    "HP1": Check("minimum", oracles.hp1,
                 lambda m: {"crossbubble_total": _sum(m.pandemic_bundle["crossbubble_violations"])}, "pandemic"),
    "C1": Check("minimum", oracles.c1, lambda m: {f"u[{s}]": v for s, v in m.slacks.items()}),
    "D1": Check("value", oracles.d1, lambda m: {f"H[{d}]": v for d, v in m.worked_hours.items()}),
    "D2a": Check("value", oracles.d2a, lambda m: {f"cost[{d}]": m.doctor_cost[d] for d in m.ctx.D}),
    "D2b": Check("value", oracles.d2b, lambda m: {f"cost[{c}]": m.doctor_cost[c] for c in m.ctx.C}),
    "D3": Check("value", oracles.d3,
                lambda m: {f"WH[{d},{w}]": v for d, weeks in m.weekly_hours.items() for w, v in weeks.items()}),
    "S1": Check("value", oracles.s1, lambda m: {"P": _sum(m.pref_terms)}),
    "F1": Check("value", oracles.f1, lambda m: {"spread_N": m.spread}),
    "F2": Check("value", oracles.f2, lambda m: {"spread_WE": m.weekend_spread}),
    "F3": Check("value", oracles.f3, lambda m: {"spread_H": m.hard_spread}),
    "F4": Check("minimum", oracles.f4, lambda m: {"ratio_spread": m.ratio_spread}),
    "U1": Check("minimum", oracles.u1, lambda m: {f"UW[{d}]": v for d, v in m.underwork.items()}),
    "U2": Check("minimum", oracles.u2,
                lambda m: {f"WU[{d},{w}]": v for d, weeks in m.weekly_underwork.items() for w, v in weeks.items()}),
    "O1": Check("objective", oracles.o1, scenario="pandemic"),
    "O1b": Check("objective", oracles.o1b, scenario="pandemic"),
    "O2": Check("objective", oracles.o2),
    "O3": Check("objective", oracles.o3),
}


@dataclass
class Result:
    eq_id: str
    samples: int = 0
    failures: list = field(default_factory=list)
    outcomes: dict = field(default_factory=dict)     # e.g. feasible / infeasible counts
    seconds: float = 0.0

    @property
    def passed(self) -> bool:
        return not self.failures and self.samples > 0

    def summary(self) -> str:
        state = "PASS" if self.passed else "FAIL"
        cases = ", ".join(f"{k}: {v}" for k, v in sorted(self.outcomes.items()))
        return f"{self.eq_id:5} {state}  {self.samples} schedules ({cases}) in {self.seconds:.1f}s"


# ---------------------------------------------------------------------------------------------
# Scenario and schedules
# ---------------------------------------------------------------------------------------------

def _empty_model(scenario: str) -> RosteringModel:
    data = load_dataset(DATASET)
    ctx = prepare_context(data.doctors, data.candidates, data.shifts)
    model = cp_model.CpModel()
    x = {(d, s): model.NewBoolVar(f"x_{d}_{s}") for d in ctx.D_all for s in ctx.S}
    hire = {c: model.NewBoolVar(f"hire_{c}") for c in ctx.C}
    pandemic_config = bubbles = None
    if scenario == "pandemic":
        settings = PandemicSettings()
        pandemic_config = PandemicConfig(num_bubbles=settings.num_bubbles, contact_depts=list(settings.contact_depts),
                                         min_specialists_per_bubble=settings.min_specialists_per_bubble)
        bubbles = prepare_pandemic_bubbles(doctors=data.doctors, candidates=data.candidates, config=pandemic_config)
    return RosteringModel(model=model, ctx=ctx, p=compute_params(ctx, data), data=data, x=x, hire=hire,
                          allow_hire=True, pandemic_config=pandemic_config, bubble_assignment=bubbles)


def make_reference_schedules(path: Path = REFERENCE_FILE) -> dict:
    """Solve the full model once per scenario (deterministically) and store the schedules."""
    from .solve import solve_lexicographic

    out = {}
    for scenario in ("normal", "pandemic"):
        base = _empty_model(scenario)
        m = build_model(base.ctx, base.data, allow_hire=False, pandemic_config=base.pandemic_config,
                        bubble_assignment=base.bubble_assignment)
        cfg = SolverConfig(time_limit_phase1=30, time_limit_isolation=30, time_limit_phase2=30,
                           time_limit_phase3=60, num_workers=8)
        outcome = solve_lexicographic(m, cfg, pandemic_mode=scenario == "pandemic")
        value = outcome.solver.Value
        out[scenario] = {
            "works": sorted([int(d), int(s)] for (d, s), var in m.x.items() if value(var)),
            "hired": sorted(int(c) for c, var in m.hire.items() if value(var)),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return out


def reference_schedule(scenario: str) -> Assignment:
    data = json.loads(REFERENCE_FILE.read_text()) if REFERENCE_FILE.exists() else make_reference_schedules()
    ref = data[scenario]
    return Assignment(works={(d, s) for d, s in ref["works"]}, hired=set(ref["hired"]), label="reference")


def _mutate(m: RosteringModel, base: Assignment, rng: random.Random, label: str, adds=0, removes=0,
            moves=0, toggle_hire=0) -> Assignment:
    works, hired = set(base.works), set(base.hired)
    cells = list(m.x)
    for _ in range(removes):
        if works:
            works.discard(rng.choice(sorted(works)))
    for _ in range(moves):
        if works:
            d, s = rng.choice(sorted(works))
            works.discard((d, s))
            works.add((rng.choice(m.ctx.D_all), s))
    for _ in range(adds):
        works.add(rng.choice(cells))
    for _ in range(toggle_hire):
        if m.ctx.C:
            hired ^= {rng.choice(m.ctx.C)}
    return Assignment(works=works, hired=hired, label=label)


def schedules(m: RosteringModel, check: Check, n: int, seed: int) -> list[Assignment]:
    """Reference schedule, edge cases and random mutations; for restrictions half of the
    mutations are single additions that the oracle reports as violations."""
    rng = random.Random(seed)
    base = reference_schedule(check.scenario)
    out = [base, Assignment(label="empty"),
           Assignment(works=set(base.works), hired=set(m.ctx.C), label="reference + all candidates hired")]

    if check.kind == "restriction":
        violating = []
        for _ in range(400):
            cell = rng.choice(list(m.x))
            if cell in base.works:
                continue
            trial = Assignment(works=base.works | {cell}, hired=set(base.hired))
            if check.oracle(m, trial):
                violating.append(cell)
            if len(violating) >= n // 2:
                break
        for cell in violating:
            out.append(Assignment(works=base.works | {cell}, hired=set(base.hired), label=f"add {cell}"))
        if check.oracle is oracles.h13:
            out.append(_mutate(m, base, rng, "hire one", toggle_hire=1))

    # A candidate takes over shifts of employed doctors (hired or not).
    employed = sorted((d, s) for d, s in base.works if d in set(m.ctx.D))
    for i in range(4 if m.ctx.C else 0):
        works, hired = set(base.works), set(base.hired)
        for d, s in rng.sample(employed, min(len(employed), 1 + 2 * i)):
            c = rng.choice(m.ctx.C)
            works.discard((d, s))
            works.add((c, s))
            if i % 2 == 0:
                hired.add(c)
        out.append(Assignment(works=works, hired=hired, label=f"candidate covers {1 + 2 * i} shifts"))

    while len(out) < n:
        i = len(out)
        kind = i % 5
        if kind == 0:
            out.append(_mutate(m, base, rng, "remove 1-8", removes=rng.randint(1, 8)))
        elif kind == 1:
            out.append(_mutate(m, base, rng, "add 1-3", adds=rng.randint(1, 3)))
        elif kind == 2:
            out.append(_mutate(m, base, rng, "move 1-4", moves=rng.randint(1, 4)))
        elif kind == 3:
            out.append(_mutate(m, base, rng, "hire toggle + changes", toggle_hire=1,
                               adds=rng.randint(0, 3), removes=rng.randint(0, 3)))
        else:
            out.append(_mutate(m, base, rng, "remove 10-40", removes=rng.randint(10, 40)))
    return out


# ---------------------------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------------------------

def _requirements(eq: Equation) -> list[Equation]:
    """Transitive requirements in the order of the registry."""
    needed, stack = set(), list(eq.requires)
    while stack:
        eq_id = stack.pop()
        if eq_id not in needed:
            needed.add(eq_id)
            stack.extend(EQUATIONS[eq_id].requires)
    return [e for e in CONSTRAINTS if e.id in needed]


def _fix(m: RosteringModel, a: Assignment) -> None:
    m.model.ClearAssumptions()
    m.model.AddAssumptions([var if (d, s) in a.works else var.Not() for (d, s), var in m.x.items()])
    m.model.AddAssumptions([var if c in a.hired else var.Not() for c, var in m.hire.items()])


def _solve(m: RosteringModel, objective=None, maximize=False):
    if objective is None:
        m.model.ClearObjective()
    elif maximize:
        m.model.Maximize(objective)
    else:
        m.model.Minimize(objective)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 60
    status = solver.Solve(m.model)
    if status not in SOLVED + (cp_model.INFEASIBLE,):
        raise RuntimeError(f"solver status {solver.StatusName(status)}")
    return solver, status


def _compare(expected: dict, solver, outputs: dict) -> list[str]:
    diffs = []
    for name, want in expected.items():
        if name not in outputs:
            diffs.append(f"{name}: missing output")
            continue
        got = solver.Value(outputs[name])
        if got != want:
            diffs.append(f"{name}: expected {want}, got {got}")
    extra = set(outputs) - set(expected)
    if extra:
        diffs.append(f"unexpected outputs {sorted(extra)[:5]}")
    return diffs


def check_equation(eq_id: str, impl: Callable | None = None, *, samples: int = 30, seed: int = 0) -> Result:
    eq, check = EQUATIONS[eq_id], CHECKS[eq_id]
    start = time.time()
    m = _empty_model(check.scenario)
    for req in _requirements(eq):
        req.fn(m)
    fn = impl or eq.fn
    objective = fn(m)
    if check.kind == "objective" and objective is None:
        raise ValueError(f"{eq_id}: an objective function must return the expression to minimise")
    outputs = check.outputs(m) if check.outputs else {}

    result = Result(eq_id)
    for a in schedules(m, check, samples, seed):
        expected = check.oracle(m, a)
        _fix(m, a)
        result.samples += 1
        if check.kind == "restriction":
            solver, status = _solve(m)
            feasible = status in SOLVED
            result.outcomes["feasible" if not expected else "violated"] = \
                result.outcomes.get("feasible" if not expected else "violated", 0) + 1
            if feasible != (not expected):
                detail = "; ".join(expected[:3]) if expected else "no violation expected"
                result.failures.append(f"[{a.label}] CP-SAT {'feasible' if feasible else 'infeasible'}, "
                                       f"oracle: {detail}")
            continue

        target = objective if check.kind == "objective" else _sum(outputs.values())
        solver, status = _solve(m, target)
        if expected is None:
            result.outcomes["infeasible"] = result.outcomes.get("infeasible", 0) + 1
            if status in SOLVED:
                result.failures.append(f"[{a.label}] oracle: infeasible, CP-SAT found a solution")
            continue
        result.outcomes["feasible"] = result.outcomes.get("feasible", 0) + 1
        if status not in SOLVED:
            result.failures.append(f"[{a.label}] oracle: feasible, CP-SAT infeasible")
            continue
        if check.kind == "objective":
            if round(solver.ObjectiveValue()) != expected:
                result.failures.append(f"[{a.label}] optimum expected {expected}, got {solver.ObjectiveValue():.0f}")
            continue
        diffs = _compare(expected, solver, outputs)
        if check.kind == "value" and not diffs:
            solver, status = _solve(m, target, maximize=True)
            diffs = [f"(maximised) {d}" for d in _compare(expected, solver, outputs)]
        if diffs:
            result.failures.append(f"[{a.label}] " + "; ".join(diffs[:3]))

    m.model.ClearAssumptions()
    result.seconds = time.time() - start
    return result


CONTRACT = """# Equations - one function each

Generated by `uv run python -m rostering.verify --markdown docs/EQUATIONS.md`. Do not edit it by hand.

Every equation of the model is a separate function in `rostering/equations.py`.
- Each function can be replaced by another implementation, for example one written by an AI model from a prompt.
- The replacement is then checked alone against an independent oracle (`rostering/oracles.py`).
- The full mathematical model is in `docs/MODEL.md`.

## Contract

```python
def equation(m: RosteringModel) -> None:     # constraints (H*, C*, D*, S*, F*, U*)
def objective(m: RosteringModel) -> expr:    # phase objectives (O*): return the expression to minimise
```

The function adds variables and constraints to `m.model` (an `ortools.sat.python.cp_model.CpModel`).
It uses only the fields listed below, and it writes its outputs into the fields named in "outputs".

| Field | Meaning |
|---|---|
| `m.model` | CpModel |
| `m.x[(d, s)]` | BoolVar: doctor `d` works shift `s` (all `d` in `D_all`, `s` in `S`) |
| `m.hire[c]` | BoolVar: candidate `c` is hired |
| `m.ctx.D`, `m.ctx.C`, `m.ctx.D_all`, `m.ctx.S` | employed doctors, candidates, both (D then C), shifts |
| `m.ctx.id_to_role[d]` | `specialist`, `icu_specialist`, `resident`, `outpatient`, ... |
| `m.p.date_list` | days of the horizon (sorted `YYYY-MM-DD`) |
| `m.p.days_to_shifts[t]`, `m.p.shift_date[s]` | shifts of day `t` / day of shift `s` |
| `m.p.abs_start[s]`, `m.p.abs_end[s]` | start / end of `s` in hours from the start of the horizon |
| `m.p.hours[s]`, `m.p.dept[s]`, `m.p.req_skill[s]` | length, department, required skill of `s` |
| `m.p.min_staff[s]`, `m.p.regular_staff[s]` | minimum / regular staffing of `s` |
| `m.p.night_shifts`, `m.p.night_shifts_by_date[t]` | night shifts (code with `_N_` or 24 h) |
| `m.p.twentyfour_shifts`, `m.p.weekend_shifts`, `m.p.hard_shifts` | 24 h / Saturday-Sunday / hard shifts |
| `m.p.weeks`, `m.p.shift_week[s]` | calendar weeks (from Monday) and week of `s` |
| `m.p.skill_map[d]` | skills of `d` |
| `m.p.night_allowed[d]`, `m.p.twentyfour_allowed[d]` | 0/1 eligibility |
| `m.p.specialists` | employed doctors with role `specialist` / `icu_specialist` |
| `m.p.needs_mentor` | doctors who need a mentor |
| `m.p.opt_out_doctors`, `m.p.regular_doctors` | employed doctors with `opt_out` 1 / 0 |
| `m.p.weekly_regular_lookup` | doctors with the "regular" weekly target (see MODEL.md, known issue 1) |
| `m.p.adjusted_max_hours[d]` | monthly hour limit `M[d]` |
| `m.p.weekly_limits[d][w]`, `m.p.weekly_available_days[d][w]` | weekly limit `L[d,w]`, available days |
| `m.p.hourly_cost_map[d]` | hourly cost |
| `m.p.unavailable_days`, `m.p.unavailable_shifts` | lists of `(d, date)` / `(d, s)` |
| `m.p.preferences` | list of `(d, s, "like" or "dislike")` |
| `m.pandemic_config.contact_depts`, `m.bubble_assignment[d]` | pandemic mode: contact departments, bubble of `d` (-1 = CLINIC pool) |
| `m.targets` | `ObjectiveTargets` of phase 3 (budget, spread targets, preference target, scale) |

Constants (`rostering/model.py`):
- `MIN_REST_HOURS = 11`
- `MAX_CONSECUTIVE_NIGHTS = 2`
- `MAX_WORKING_DAYS_PER_WEEK = 6`
- `UNDERWORK_TARGET = 0.95`
- `SPECIALIST_ROLES = ["specialist", "icu_specialist"]`

## Verifying an implementation

```bash
uv run python -m rostering.verify H3 --impl candidate.py:h3_min_rest   # one equation
uv run python -m rostering.verify --all                                # all reference functions
uv run python -m rostering.verify --list
```

The model contains the decision variables, the reference functions of the required equations and the function under test.
- The decisions are fixed to 30 schedules: a valid reference schedule, an empty one, and mutations of the reference.
- For restrictions, about half of the mutations are single changes that break the equation.
- The answer of CP-SAT is compared with the oracle for every schedule.
- The check does not look at variable names or at the order of constraints, only at what the equation allows and computes.

The check is only as strong as the data. Two implementations that differ only in situations that never occur in `dec2026_pandemic` cannot be told apart. Example: a minimum rest of 10, 11 or 12 h, because the data has no gap of 10-11 h between shifts.

Check types:
- **restriction**: feasible ⇔ the oracle finds no violation.
- **value**: the outputs take exactly the oracle values, both when minimised and when maximised.
- **minimum**: the minimised outputs equal the oracle values.
- **objective**: the optimum of the returned expression equals the oracle value.
- In every type, a schedule the oracle calls infeasible must be infeasible for CP-SAT.

## Equations
"""


def equations_markdown() -> str:
    lines = [CONTRACT]
    for eq in CONSTRAINTS + [e for e in EQUATIONS.values() if e.kind == "objective"]:
        doc = (eq.fn.__doc__ or "").strip()
        lines.append(f"### {eq.id} - {eq.name}\n")
        lines.append(f"- function: `{eq.fn.__name__}(m)`")
        lines.append(f"- check: {CHECKS[eq.id].kind} (scenario: {CHECKS[eq.id].scenario})")
        lines.append(f"- requires: {', '.join(eq.requires) or '-'}")
        lines.append(f"- outputs: {', '.join(f'`m.{o}`' for o in eq.outputs) or ('returned expression' if eq.kind == 'objective' else '-')}")
        lines.append("")
        lines.append("```text")
        lines.append(doc)
        lines.append("```\n")
    return "\n".join(lines)


def load_function(spec: str) -> Callable:
    """'path/to/file.py:function_name' -> the function."""
    path, _, name = spec.partition(":")
    module_spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return getattr(module, name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify equations of the rostering model one by one.")
    parser.add_argument("equation", nargs="?", choices=sorted(EQUATIONS), help="equation id, e.g. H3")
    parser.add_argument("--impl", help="implementation to check instead of the reference: file.py:function")
    parser.add_argument("--all", action="store_true", help="check every reference equation")
    parser.add_argument("--list", action="store_true", help="list the equations")
    parser.add_argument("--samples", type=int, default=30, help="number of schedules per equation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", type=int, default=5, help="failures to print per equation")
    parser.add_argument("--markdown", metavar="PATH", help="write the equation contracts (docs/EQUATIONS.md)")
    args = parser.parse_args(argv)

    if args.markdown:
        Path(args.markdown).write_text(equations_markdown())
        return 0

    if args.list:
        for eq in CONSTRAINTS + [e for e in EQUATIONS.values() if e.kind == "objective"]:
            print(f"{eq.id:5} {CHECKS[eq.id].kind:11} {eq.name}")
        return 0
    if not args.all and not args.equation:
        parser.error("give an equation id, --all or --list")

    ids = list(EQUATIONS) if args.all else [args.equation]
    impl = load_function(args.impl) if args.impl else None
    ok = True
    for eq_id in ids:
        result = check_equation(eq_id, impl, samples=args.samples, seed=args.seed)
        print(result.summary(), flush=True)
        for failure in result.failures[:args.show]:
            print("      " + failure)
        ok &= result.passed
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
