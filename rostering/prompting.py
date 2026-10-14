"""Generate a prompt for every equation, ask a model to implement it, verify the answers.

    uv run python -m rostering.prompting sets                                    # list the prompt sets
    uv run python -m rostering.prompting prompts --set s1_intent                 # only write the prompts
    uv run python -m rostering.prompting run --set s3_spec --provider claude-cli --model sonnet
    uv run python -m rostering.prompting benchmark --provider claude-cli --model sonnet --repeats 3
    uv run python -m rostering.prompting run --provider manual                   # answer the prompts by hand ...
    uv run python -m rostering.prompting check runs/<run>                        # ... then verify the answers
    uv run python -m rostering.prompting run --provider reference                # sanity check of the pipeline
    uv run python -m rostering.prompting report runs/<run or benchmark>          # HTML report with charts

A prompt = prompts/template.md filled with the interface contract (prompts/contract.md) and the
description of the equation from a prompt set (prompts/sets/<set>/equations.md). The sets differ
in how much they tell the model, from the name of the equation (s0_name) to the full specification
with the formal statement (s4_formal). A set folder may override template.md, contract.md,
system.md and feedback.md and may contain set.json with {"descriptions_from": "<set>",
"formal": true}.

Every answer is verified in a separate process with rostering.verify; with --attempts > 1 the
failures are sent back for a corrected version. A run is stored in
runs/<timestamp>_<set>_<provider>_<model>/ with the prompts, the raw answers, the extracted code,
the verification results, summary.csv and report.md; a benchmark stores its runs and the
comparison of the sets in runs/benchmark_<timestamp>/.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .config import BASE_DIR
from .equations import CONSTRAINTS, EQUATIONS, OBJECTIVES, Equation
from .verify import CHECKS

PROMPTS_DIR = BASE_DIR / "prompts"
SETS_DIR = PROMPTS_DIR / "sets"
DEFAULT_SET = "s3_spec"
RUNS_DIR = BASE_DIR / "runs"
ORDER = [eq.id for eq in CONSTRAINTS + OBJECTIVES]

OUTPUT_DESCRIPTIONS = {
    "pandemic_bundle": "`m.pandemic_bundle`",
    "slacks": "`m.slacks[s]`",
    "slacks_o": "`m.slacks_o[s]`",
    "worked_hours": "`m.worked_hours[d]`",
    "doctor_cost": "`m.doctor_cost[d]`",
    "weekly_hours": "`m.weekly_hours[d][w]`",
    "pref_terms": "`m.pref_terms`",
    "spread": "`m.spread`",
    "weekend_spread": "`m.weekend_spread`",
    "hard_spread": "`m.hard_spread`",
    "ratio_spread": "`m.ratio_spread`",
    "underwork": "`m.underwork[d]`",
    "weekly_underwork": "`m.weekly_underwork[d][w]`",
}


# ---------------------------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------------------------

def list_sets() -> list[str]:
    return sorted(p.name for p in SETS_DIR.iterdir() if p.is_dir()) if SETS_DIR.exists() else []


def _set_config(set_name: str) -> dict:
    folder = SETS_DIR / set_name
    if not folder.is_dir():
        raise SystemExit(f"unknown prompt set {set_name!r}; available: {', '.join(list_sets())}")
    config_file = folder / "set.json"
    return json.loads(config_file.read_text()) if config_file.exists() else {}


def _read(name: str, set_name: str | None = None) -> str:
    """prompts/sets/<set>/<name> if the set overrides the file, else prompts/<name>."""
    if set_name:
        override = SETS_DIR / set_name / name
        if override.exists():
            return override.read_text().strip()
    return (PROMPTS_DIR / name).read_text().strip()


def descriptions(set_name: str = DEFAULT_SET) -> dict[str, str]:
    """`## <ID>` sections of prompts/sets/<set>/equations.md (or of the set in descriptions_from)."""
    source = _set_config(set_name).get("descriptions_from", set_name)
    text = (SETS_DIR / source / "equations.md").read_text()
    sections = re.split(r"^## +(\S+)\s*$", text, flags=re.M)
    return {sections[i]: sections[i + 1].strip() for i in range(1, len(sections), 2)}


def _available(eq: Equation) -> str:
    fields = []
    for req in eq.requires:
        fields += [OUTPUT_DESCRIPTIONS[o] for o in EQUATIONS[req].outputs]
    return ", ".join(dict.fromkeys(fields)) or "only the fields of the interface table"


def build_prompt(eq_id: str, set_name: str = DEFAULT_SET) -> str:
    eq = EQUATIONS[eq_id]
    description = descriptions(set_name).get(eq_id)
    if not description:
        raise KeyError(f"no description of {eq_id} in prompt set {set_name}")
    formal = ""
    if _set_config(set_name).get("formal"):
        formal = "\nFormal statement:\n\n```text\n" + inspect.getdoc(eq.fn) + "\n```\n"
    if eq.kind == "objective":
        output_rule = (f"`{eq.fn.__name__}(m)` must return the expression (or variable) to minimise. Any "
                       "auxiliary variables and constraints are added to `m.model`.")
    elif eq.outputs:
        output_rule = (f"`{eq.fn.__name__}(m)` adds its variables and constraints to `m.model`, stores its output in "
                       + ", ".join(OUTPUT_DESCRIPTIONS[o] for o in eq.outputs) + " and returns None.")
    else:
        output_rule = f"`{eq.fn.__name__}(m)` adds its constraints to `m.model` and returns None."
    return _read("template.md", set_name).format(
        contract=_read("contract.md", set_name), id=eq.id, name=eq.name, description=description, formal=formal,
        available=_available(eq), output_rule=output_rule, function=eq.fn.__name__,
    ) + "\n"


def feedback_prompt(prompt: str, code: str, failures: list[str], show: int = 5, set_name: str | None = None) -> str:
    listed = "\n".join(f"- {f}" for f in failures[:show])
    return _read("feedback.md", set_name).format(prompt=prompt.strip(), code=code.strip(), failures=listed) + "\n"


def extract_code(response: str, function: str) -> str | None:
    """The last ```python block that defines the function (or the whole answer if it is bare code)."""
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", response, flags=re.S)
    for block in reversed(blocks):
        if re.search(rf"^\s*def\s+{re.escape(function)}\s*\(", block, flags=re.M):
            return block.strip() + "\n"
    if re.search(rf"^def\s+{re.escape(function)}\s*\(", response, flags=re.M):
        return response.strip() + "\n"
    return None


# ---------------------------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------------------------

class Provider:
    name = "provider"
    model = ""

    def complete(self, system: str, prompt: str, eq: Equation) -> str:
        raise NotImplementedError


class ReferenceProvider(Provider):
    """Answers with the reference implementation - checks the pipeline itself."""
    name = "reference"

    def complete(self, system, prompt, eq):
        module = eq.fn.__module__
        return f"```python\nfrom {module} import {eq.fn.__name__} as _reference\n\n\n" \
               f"def {eq.fn.__name__}(m):\n    return _reference(m)\n```\n"


class CommandProvider(Provider):
    """Any command that reads the prompt from stdin and prints the answer (run in an empty directory)."""
    name = "command"

    def __init__(self, command: str, timeout: int = 600):
        self.command = command
        self.model = shlex.split(command)[0]
        self.timeout = timeout

    def _args(self, system: str) -> list[str]:
        return shlex.split(self.command)

    def complete(self, system, prompt, eq):
        text = prompt if self.name != "command" else f"{system}\n\n{prompt}"
        with tempfile.TemporaryDirectory(prefix="rostering_prompt_") as workdir:
            proc = subprocess.run(self._args(system), input=text, capture_output=True, text=True,
                                  cwd=workdir, timeout=self.timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"{self.name} exited with {proc.returncode}: {proc.stderr.strip()[:500]}")
        return proc.stdout


class ClaudeCliProvider(CommandProvider):
    """Claude Code CLI without tools, MCP servers or session, in an empty working directory,
    so the model cannot read the repository."""
    name = "claude-cli"

    def __init__(self, model: str | None = None, timeout: int = 600):
        self.model = model or "default"
        self.timeout = timeout
        self.extra = ["--model", model] if model else []

    def _args(self, system):
        return ["claude", "-p", "--tools", "", "--no-session-persistence", "--strict-mcp-config",
                "--system-prompt", system, *self.extra]


class AnthropicProvider(Provider):
    """Anthropic API (needs `uv sync --extra llm` and ANTHROPIC_API_KEY)."""
    name = "anthropic"

    def __init__(self, model: str | None = None, max_tokens: int = 8000):
        try:
            import anthropic
        except ImportError as exc:
            raise SystemExit("The anthropic package is missing: uv sync --extra llm") from exc
        self.client = anthropic.Anthropic()
        self.model = model or "claude-opus-5-5"
        self.max_tokens = max_tokens

    def complete(self, system, prompt, eq):
        message = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, system=system,
                                              messages=[{"role": "user", "content": prompt}])
        return "".join(block.text for block in message.content if getattr(block, "type", "") == "text")


class ManualProvider(Provider):
    """Writes the prompts only; the answers are pasted into response.md and checked with `check`."""
    name = "manual"

    def complete(self, system, prompt, eq):
        return ""


def make_provider(args) -> Provider:
    if args.provider == "reference":
        return ReferenceProvider()
    if args.provider == "manual":
        return ManualProvider()
    if args.provider == "claude-cli":
        return ClaudeCliProvider(args.model, timeout=args.timeout)
    if args.provider == "anthropic":
        return AnthropicProvider(args.model)
    if args.provider == "command":
        if not args.command:
            raise SystemExit("--provider command needs --command, e.g. --command 'ollama run qwen2.5-coder'")
        return CommandProvider(args.command, timeout=args.timeout)
    raise SystemExit(f"unknown provider {args.provider}")


# ---------------------------------------------------------------------------------------------
# Verification and runs
# ---------------------------------------------------------------------------------------------

@dataclass
class Attempt:
    number: int
    passed: bool = False
    failures: list = field(default_factory=list)
    samples: int = 0
    seconds_model: float = 0.0
    seconds_verify: float = 0.0
    error: str = ""


@dataclass
class EquationRun:
    equation: str
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.attempts) and self.attempts[-1].passed


def verify_file(eq_id: str, path: Path, function: str, samples: int, timeout: int) -> dict:
    cmd = [sys.executable, "-m", "rostering.verify", eq_id, "--impl", f"{path}:{function}",
           "--json", "--samples", str(samples)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"passed": False, "samples": 0, "failures": [f"verification timed out after {timeout}s"]}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    return {"passed": False, "samples": 0, "failures": [f"verification crashed: {proc.stderr.strip()[-800:]}"]}


def run_equation(eq_id: str, provider: Provider, run_dir: Path, args) -> EquationRun:
    eq = EQUATIONS[eq_id]
    system = _read("system.md", args.set)
    base_prompt = build_prompt(eq_id, args.set)
    prompt = base_prompt
    result = EquationRun(eq_id)

    for number in range(1, args.attempts + 1):
        attempt = Attempt(number)
        result.attempts.append(attempt)
        folder = run_dir / eq_id / f"attempt_{number}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "prompt.md").write_text(prompt)

        response_file = folder / "response.md"
        if isinstance(provider, ManualProvider):
            if not response_file.exists() or not response_file.read_text().strip():
                attempt.error = "waiting for response.md"
                break
            response = response_file.read_text()
        else:
            start = time.time()
            try:
                response = provider.complete(system, prompt, eq)
            except Exception as exc:
                attempt.error = f"model call failed: {type(exc).__name__}: {exc}"
                break
            attempt.seconds_model = round(time.time() - start, 1)
            response_file.write_text(response)

        code = extract_code(response, eq.fn.__name__)
        if code is None:
            attempt.failures = [f"no ```python block defining {eq.fn.__name__}(m) in the answer"]
        else:
            candidate = folder / "candidate.py"
            candidate.write_text(code)
            start = time.time()
            verdict = verify_file(eq_id, candidate, eq.fn.__name__, args.samples, args.verify_timeout)
            attempt.seconds_verify = round(time.time() - start, 1)
            attempt.passed = verdict["passed"]
            attempt.samples = verdict.get("samples", 0)
            attempt.failures = verdict.get("failures", [])
        (folder / "verify.json").write_text(json.dumps(asdict(attempt), indent=1))
        if attempt.passed or isinstance(provider, ManualProvider):
            break
        prompt = feedback_prompt(base_prompt, code or response, attempt.failures, set_name=args.set)
    return result


def write_report(run_dir: Path, results: list[EquationRun], meta: dict) -> None:
    rows = []
    for r in sorted(results, key=lambda r: ORDER.index(r.equation)):
        last = r.attempts[-1] if r.attempts else Attempt(0)
        rows.append({
            "equation": r.equation,
            "name": EQUATIONS[r.equation].name,
            "check": CHECKS[r.equation].kind,
            "passed": r.passed,
            "attempts": len(r.attempts),
            "passed_first_attempt": bool(r.attempts) and r.attempts[0].passed,
            "samples": last.samples,
            "failures": len(last.failures),
            "error": last.error,
            "seconds_model": sum(a.seconds_model for a in r.attempts),
            "seconds_verify": sum(a.seconds_verify for a in r.attempts),
            "first_failure": (last.failures[0] if last.failures else "")[:300],
        })
    with open(run_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(r["passed"] for r in rows)
    first = sum(r["passed_first_attempt"] for r in rows)
    lines = [
        f"# Prompt run {run_dir.name}",
        "",
        "| Setting | Value |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in meta.items()],
        "",
        f"**Passed: {passed} / {len(rows)}** (first attempt: {first} / {len(rows)})",
        "",
        "| Equation | Name | Check | Result | Attempts | Details |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        state = "PASS" if r["passed"] else ("WAITING" if r["error"].startswith("waiting") else "FAIL")
        detail = r["error"] or r["first_failure"]
        detail = detail.replace("|", "\\|")
        lines.append(f"| {r['equation']} | {r['name']} | {r['check']} | {state} | {r['attempts']} | {detail} |")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")


def html_report(path: Path) -> None:
    """report.html with charts and similarity metrics; a failure here must not lose the run."""
    try:
        from .report import build_report
        print(f"HTML report: {build_report(path)}")
    except Exception as exc:
        print(f"HTML report not created: {type(exc).__name__}: {exc}")


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=BASE_DIR).stdout.strip()
    except OSError:
        return ""


def run(args, provider: Provider | None = None) -> Path:
    provider = provider or make_provider(args)
    _set_config(args.set)
    ids = args.equations or ORDER
    unknown = [i for i in ids if i not in EQUATIONS]
    if unknown:
        raise SystemExit(f"unknown equations: {unknown}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = re.sub(r"[^A-Za-z0-9.-]+", "-", provider.model or provider.name)
    run_dir = Path(args.out) if args.out else RUNS_DIR / f"{stamp}_{args.set}_{provider.name}_{model_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {"set": args.set, "provider": provider.name, "model": provider.model, "attempts": args.attempts,
            "samples": args.samples, "equations": len(ids), "commit": _git_commit(), "started": stamp}
    (run_dir / "config.json").write_text(json.dumps(meta, indent=1))
    (run_dir / "system.md").write_text(_read("system.md", args.set) + "\n")

    def task(eq_id):
        result = run_equation(eq_id, provider, run_dir, args)
        last = result.attempts[-1] if result.attempts else Attempt(0)
        state = "PASS" if result.passed else ("WAITING" if last.error.startswith("waiting") else "FAIL")
        print(f"{eq_id:5} {state:7} attempts={len(result.attempts)} {last.error or (last.failures[0] if last.failures else '')}"[:200],
              flush=True)
        return result

    jobs = 1 if isinstance(provider, ManualProvider) else args.jobs
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(task, ids))
    write_report(run_dir, results, meta)
    print(f"\nReport: {run_dir / 'report.md'}")
    if not getattr(args, "in_benchmark", False):
        html_report(run_dir)
    return run_dir


def check(args) -> Path:
    """Verify the answers pasted into <run>/<ID>/attempt_1/response.md (manual provider)."""
    run_dir = Path(args.run_dir)
    meta = json.loads((run_dir / "config.json").read_text())
    args.set, args.attempts = meta["set"], 1
    ids = [p.name for p in sorted(run_dir.iterdir()) if p.is_dir() and p.name in EQUATIONS]
    results = []
    for eq_id in ids:
        result = run_equation(eq_id, ManualProvider(), run_dir, args)
        last = result.attempts[-1]
        print(f"{eq_id:5} {'PASS' if result.passed else 'WAITING' if last.error else 'FAIL'}", flush=True)
        results.append(result)
    write_report(run_dir, results, meta)
    print(f"\nReport: {run_dir / 'report.md'}")
    return run_dir


def benchmark(args) -> Path:
    """Run several prompt sets (each repeated) with the same model and compare the pass rates."""
    sets = args.sets or list_sets()
    for set_name in sets:
        _set_config(set_name)
    provider = make_provider(args)
    if isinstance(provider, ManualProvider):
        raise SystemExit("benchmark needs a model provider, not manual")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = re.sub(r"[^A-Za-z0-9.-]+", "-", provider.model or provider.name)
    bench_dir = Path(args.out) if args.out else RUNS_DIR / f"benchmark_{stamp}_{provider.name}_{model_tag}"
    bench_dir.mkdir(parents=True, exist_ok=True)
    meta = {"provider": provider.name, "model": provider.model, "sets": ", ".join(sets), "repeats": args.repeats,
            "attempts": args.attempts, "samples": args.samples, "commit": _git_commit(), "started": stamp}
    (bench_dir / "config.json").write_text(json.dumps(meta, indent=1))

    rows = []
    for set_name in sets:
        for repeat in range(1, args.repeats + 1):
            print(f"\n=== set {set_name}, repeat {repeat}/{args.repeats} ===", flush=True)
            run_args = argparse.Namespace(**{**vars(args), "set": set_name, "out": bench_dir / f"{set_name}_r{repeat}",
                                             "in_benchmark": True})
            run_dir = run(run_args, provider)
            with open(run_dir / "summary.csv") as f:
                for row in csv.DictReader(f):
                    rows.append({"set": set_name, "repeat": repeat, "equation": row["equation"],
                                 "check": row["check"], "passed": row["passed"] == "True",
                                 "passed_first_attempt": row["passed_first_attempt"] == "True",
                                 "attempts": int(row["attempts"]), "first_failure": row["first_failure"]})
    with open(bench_dir / "benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_benchmark_report(bench_dir, rows, sets, meta)
    print(f"\nBenchmark report: {bench_dir / 'report.md'}")
    html_report(bench_dir)
    return bench_dir


def write_benchmark_report(bench_dir: Path, rows: list[dict], sets: list[str], meta: dict) -> None:
    def rate(selected, key="passed"):
        return sum(r[key] for r in selected) / len(selected) if selected else 0.0

    lines = [f"# Benchmark {bench_dir.name}", "", "| Setting | Value |", "|---|---|",
             *[f"| {k} | {v} |" for k, v in meta.items()], "",
             "## Pass rate per prompt set", "",
             "| Set | Passed (all attempts) | Passed in the first attempt | restriction | value / minimum | objective |",
             "|---|---|---|---|---|---|"]
    for set_name in sets:
        sel = [r for r in rows if r["set"] == set_name]
        by_kind = lambda kinds: [r for r in sel if r["check"] in kinds]  # noqa: E731
        lines.append(f"| {set_name} | {rate(sel):.0%} | {rate(sel, 'passed_first_attempt'):.0%} | "
                     f"{rate(by_kind({'restriction'})):.0%} | {rate(by_kind({'value', 'minimum'})):.0%} | "
                     f"{rate(by_kind({'objective'})):.0%} |")

    lines += ["", "## Equations (passed repeats / repeats)", "",
              "| Equation | Check | " + " | ".join(sets) + " |",
              "|---|---|" + "---|" * len(sets)]
    for eq_id in ORDER:
        cells = []
        for set_name in sets:
            sel = [r for r in rows if r["set"] == set_name and r["equation"] == eq_id]
            cells.append(f"{sum(r['passed'] for r in sel)}/{len(sel)}" if sel else "-")
        if any(c != "-" for c in cells):
            lines.append(f"| {eq_id} {EQUATIONS[eq_id].name} | {CHECKS[eq_id].kind} | " + " | ".join(cells) + " |")

    lines += ["", "The runs (prompts, answers, code, failures) are in the subfolders <set>_r<repeat>/."]
    (bench_dir / "report.md").write_text("\n".join(lines) + "\n")


def _add_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=["claude-cli", "anthropic", "command", "manual", "reference"],
                        default="claude-cli")
    parser.add_argument("--model", help="model name for claude-cli / anthropic (e.g. sonnet, opus)")
    parser.add_argument("--command", help="command for --provider command (prompt on stdin, answer on stdout)")
    parser.add_argument("--attempts", type=int, default=1, help="attempts per equation (failures are fed back)")
    parser.add_argument("--jobs", type=int, default=4, help="equations processed in parallel")
    parser.add_argument("--samples", type=int, default=30, help="schedules per verification")
    parser.add_argument("--timeout", type=int, default=600, help="seconds per model call")
    parser.add_argument("--verify-timeout", type=int, default=600)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prompts for the single equations and their verification.")
    sub = parser.add_subparsers(dest="command_name", required=True)

    sub.add_parser("sets", help="list the prompt sets")

    p_prompts = sub.add_parser("prompts", help="write the prompts without calling a model")
    p_prompts.add_argument("equations", nargs="*")
    p_prompts.add_argument("--set", default=DEFAULT_SET)
    p_prompts.add_argument("--out", default="prompts_preview")

    p_run = sub.add_parser("run", help="ask a model for every equation of one prompt set and verify the answers")
    p_run.add_argument("equations", nargs="*", help="equation ids (default: all)")
    p_run.add_argument("--set", default=DEFAULT_SET, help=f"prompt set (default {DEFAULT_SET})")
    p_run.add_argument("--out", help="run directory (default runs/<timestamp>_...)")
    _add_model_options(p_run)

    p_bench = sub.add_parser("benchmark", help="compare prompt sets with the same model")
    p_bench.add_argument("equations", nargs="*", help="equation ids (default: all)")
    p_bench.add_argument("--sets", nargs="+", help="prompt sets (default: all)")
    p_bench.add_argument("--repeats", type=int, default=3, help="runs per set (answers of models vary)")
    p_bench.add_argument("--out", help="benchmark directory (default runs/benchmark_<timestamp>_...)")
    _add_model_options(p_bench)

    p_report = sub.add_parser("report", help="HTML report with charts and similarity metrics for a run or benchmark")
    p_report.add_argument("path")
    p_report.add_argument("--offline", action="store_true", help="embed plotly.js (no internet needed, ~4 MB)")

    p_check = sub.add_parser("check", help="verify answers pasted into a manual run")
    p_check.add_argument("run_dir")
    p_check.add_argument("--samples", type=int, default=30)
    p_check.add_argument("--verify-timeout", type=int, default=600)

    args = parser.parse_args(argv)
    if args.command_name == "sets":
        for set_name in list_sets():
            config = _set_config(set_name)
            first = (SETS_DIR / config.get("descriptions_from", set_name) / "equations.md").read_text().splitlines()[0]
            print(f"{set_name:12} {config.get('description') or first.lstrip('# ')}")
    elif args.command_name == "prompts":
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "system.md").write_text(_read("system.md", args.set) + "\n")
        for eq_id in args.equations or ORDER:
            (out / f"{eq_id}.md").write_text(build_prompt(eq_id, args.set))
        print(f"{len(args.equations or ORDER)} prompts of set {args.set} written to {out}/")
    elif args.command_name == "run":
        run(args)
    elif args.command_name == "benchmark":
        benchmark(args)
    elif args.command_name == "report":
        from .report import build_report
        print(f"HTML report: {build_report(Path(args.path), offline=args.offline)}")
    else:
        check(args)
        html_report(Path(args.run_dir))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
