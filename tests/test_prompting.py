"""Prompt generation, code extraction and the run loop (without calling a real model)."""
from __future__ import annotations

import argparse
import inspect

import pytest

from rostering import prompting
from rostering.equations import EQUATIONS


@pytest.mark.parametrize("eq_id", list(EQUATIONS))
def test_prompt_contains_task_and_no_reference_code(eq_id):
    eq = EQUATIONS[eq_id]
    prompt = prompting.build_prompt(eq_id)
    assert f"def {eq.fn.__name__}(m):" in prompt
    assert prompting.descriptions()[eq_id] in prompt
    body = inspect.getsource(eq.fn).split('"""')[-1].strip()      # reference code after the docstring
    assert body and body not in prompt
    for hint in ("rostering/", "rostering.equations", "rostering.model", "equations.py", "oracle"):  # no repo hints
        assert hint not in prompt


def test_formal_level_adds_the_formal_statement():
    assert "Formal statement" in prompting.build_prompt("H3", "formal")
    assert "Formal statement" not in prompting.build_prompt("H3", "natural")


def test_extract_code_takes_the_last_block_with_the_function():
    answer = "Plan:\n```python\ndef helper():\n    pass\n```\nFinal:\n```python\ndef h1_skills(m):\n    pass\n```\n"
    assert prompting.extract_code(answer, "h1_skills") == "def h1_skills(m):\n    pass\n"
    assert prompting.extract_code("def h1_skills(m):\n    return None\n", "h1_skills").startswith("def h1_skills")
    assert prompting.extract_code("no code here", "h1_skills") is None


class FlakyProvider(prompting.Provider):
    """Wrong implementation first, the reference one after receiving the failures."""
    name = "flaky"

    def __init__(self):
        self.prompts = []

    def complete(self, system, prompt, eq):
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            return f"```python\ndef {eq.fn.__name__}(m):\n    return None\n```"
        return prompting.ReferenceProvider().complete(system, prompt, eq)


def _args(**kw):
    defaults = dict(level="natural", attempts=2, samples=10, verify_timeout=300)
    return argparse.Namespace(**{**defaults, **kw})


def test_failures_are_fed_back_and_the_second_attempt_passes(tmp_path):
    provider = FlakyProvider()
    result = prompting.run_equation("H2", provider, tmp_path, _args())
    assert [a.passed for a in result.attempts] == [False, True]
    assert "## Previous attempt" in provider.prompts[1]
    assert "CP-SAT feasible" in provider.prompts[1]          # a concrete counterexample is sent back
    assert (tmp_path / "H2" / "attempt_2" / "candidate.py").exists()


def test_manual_run_waits_for_answers_and_checks_them(tmp_path):
    args = _args(attempts=1)
    waiting = prompting.run_equation("H9", prompting.ManualProvider(), tmp_path, args)
    assert waiting.attempts[0].error.startswith("waiting")
    assert (tmp_path / "H9" / "attempt_1" / "prompt.md").exists()

    answer = prompting.ReferenceProvider().complete("", "", EQUATIONS["H9"])
    (tmp_path / "H9" / "attempt_1" / "response.md").write_text(answer)
    checked = prompting.run_equation("H9", prompting.ManualProvider(), tmp_path, args)
    assert checked.passed
