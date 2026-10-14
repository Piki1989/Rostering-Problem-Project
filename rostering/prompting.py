"""Generate a prompt for every equation, ask a model to implement it, verify the answers.

    uv run python -m rostering.prompting prompts --out prompts_preview          # only write the prompts
    uv run python -m rostering.prompting run --provider claude-cli --model sonnet
    uv run python -m rostering.prompting run --provider manual                   # answer the prompts by hand ...
    uv run python -m rostering.prompting check runs/<run>                        # ... then verify the answers
    uv run python -m rostering.prompting run --provider reference                # sanity check of the pipeline

The prompts are assembled from the editable files in prompts/ (template, interface contract and a
natural-language description of every equation); with --level formal the formal statement from
rostering/equations.py is added. Every answer is verified in a separate process with
rostering.verify; with --attempts > 1 the failures are sent back for a corrected version.

Every run is stored in runs/<timestamp>_<provider>_<model>_<level>/ with the prompts, the raw
answers, the extracted code, the verification results, summary.csv and report.md.
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

def _read(name: str) -> str:
    return (PROMPTS_DIR / name).read_text().strip()


def descriptions() -> dict[str, str]:
    """`## <ID>` sections of prompts/equations.md."""
    text = (PROMPTS_DIR / "equations.md").read_text()
    sections = re.split(r"^## +(\S+)\s*$", text, flags=re.M)
    return {sections[i]: sections[i + 1].strip() for i in range(1, len(sections), 2)}


def _available(eq: Equation) -> str:
    fields = []
    for req in eq.requires:
        fields += [OUTPUT_DESCRIPTIONS[o] for o in EQUATIONS[req].outputs]
    return ", ".join(dict.fromkeys(fields)) or "only the fields of the interface table"


def build_prompt(eq_id: str, level: str = "natural") -> str:
    eq = EQUATIONS[eq_id]
    description = descriptions().get(eq_id)
    if not description:
        raise KeyError(f"no description of {eq_id} in prompts/equations.md")
    formal = ""
    if level == "formal":
        formal = "\nFormal statement:\n\n```text\n" + inspect.getdoc(eq.fn) + "\n```\n"
    if eq.kind == "objective":
        output_rule = (f"`{eq.fn.__name__}(m)` must return the expression (or variable) to minimise. Any "
                       "auxiliary variables and constraints are added to `m.model`.")
    elif eq.outputs:
        output_rule = (f"`{eq.fn.__name__}(m)` adds its variables and constraints to `m.model`, stores its output in "
                       + ", ".join(OUTPUT_DESCRIPTIONS[o] for o in eq.outputs) + " and returns None.")
    else:
        output_rule = f"`{eq.fn.__name__}(m)` adds its constraints to `m.model` and returns None."
    return _read("template.md").format(
        contract=_read("contract.md"), id=eq.id, name=eq.name, description=description, formal=formal,
        available=_available(eq), output_rule=output_rule, function=eq.fn.__name__,
    ) + "\n"


def feedback_prompt(prompt: str, code: str, failures: list[str], show: int = 5) -> str:
    listed = "\n".join(f"- {f}" for f in failures[:show])
    return _read("feedback.md").format(prompt=prompt.strip(), code=code.strip(), failures=listed) + "\n"


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
    system = _read("system.md")
    base_prompt = build_prompt(eq_id, args.level)
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
        prompt = feedback_prompt(base_prompt, code or response, attempt.failures)
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


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=BASE_DIR).stdout.strip()
    except OSError:
        return ""


def run(args) -> Path:
    provider = make_provider(args)
    ids = args.equations or ORDER
    unknown = [i for i in ids if i not in EQUATIONS]
    if unknown:
        raise SystemExit(f"unknown equations: {unknown}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = re.sub(r"[^A-Za-z0-9.-]+", "-", provider.model or provider.name)
    run_dir = Path(args.out) if args.out else RUNS_DIR / f"{stamp}_{provider.name}_{model_tag}_{args.level}"
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {"provider": provider.name, "model": provider.model, "level": args.level, "attempts": args.attempts,
            "samples": args.samples, "equations": len(ids), "commit": _git_commit(), "started": stamp}
    (run_dir / "config.json").write_text(json.dumps(meta, indent=1))
    (run_dir / "system.md").write_text(_read("system.md") + "\n")

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
    return run_dir


def check(args) -> Path:
    """Verify the answers pasted into <run>/<ID>/attempt_1/response.md (manual provider)."""
    run_dir = Path(args.run_dir)
    meta = json.loads((run_dir / "config.json").read_text())
    args.level, args.attempts = meta["level"], 1
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prompts for the single equations and their verification.")
    sub = parser.add_subparsers(dest="command_name", required=True)

    p_prompts = sub.add_parser("prompts", help="write the prompts without calling a model")
    p_prompts.add_argument("--out", default="prompts_preview")
    p_prompts.add_argument("--level", choices=["natural", "formal"], default="natural")
    p_prompts.add_argument("equations", nargs="*")

    p_run = sub.add_parser("run", help="ask a model for every equation and verify the answers")
    p_run.add_argument("equations", nargs="*", help="equation ids (default: all)")
    p_run.add_argument("--provider", choices=["claude-cli", "anthropic", "command", "manual", "reference"],
                       default="claude-cli")
    p_run.add_argument("--model", help="model name for claude-cli / anthropic")
    p_run.add_argument("--command", help="command for --provider command (prompt on stdin, answer on stdout)")
    p_run.add_argument("--level", choices=["natural", "formal"], default="natural",
                       help="natural: description only; formal: also the formal statement")
    p_run.add_argument("--attempts", type=int, default=1, help="attempts per equation (failures are fed back)")
    p_run.add_argument("--jobs", type=int, default=4, help="equations processed in parallel")
    p_run.add_argument("--samples", type=int, default=30)
    p_run.add_argument("--timeout", type=int, default=600, help="seconds per model call")
    p_run.add_argument("--verify-timeout", type=int, default=600)
    p_run.add_argument("--out", help="run directory (default runs/<timestamp>_...)")

    p_check = sub.add_parser("check", help="verify answers pasted into a manual run")
    p_check.add_argument("run_dir")
    p_check.add_argument("--samples", type=int, default=30)
    p_check.add_argument("--verify-timeout", type=int, default=600)

    args = parser.parse_args(argv)
    if args.command_name == "prompts":
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "system.md").write_text(_read("system.md") + "\n")
        for eq_id in args.equations or ORDER:
            (out / f"{eq_id}.md").write_text(build_prompt(eq_id, args.level))
        print(f"{len(args.equations or ORDER)} prompts written to {out}/")
    elif args.command_name == "run":
        run(args)
    else:
        check(args)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
