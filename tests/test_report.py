"""Similarity metrics and the HTML report."""
from __future__ import annotations

import argparse
from collections import Counter

import pandas as pd
import pytest

from rostering import prompting
from rostering.report import collect, per_answer
from rostering.similarity import (TfIdf, agreement, ast_bigrams, code_similarities, code_tokens, cosine,
                                  description_metrics, reference_source)


def test_cosine_basics():
    assert cosine(Counter(a=1, b=2), Counter(a=2, b=4)) == pytest.approx(1.0)
    assert cosine(Counter(a=1), Counter(b=1)) == 0.0
    assert cosine(Counter(), Counter(a=1)) == 0.0


def test_tfidf_lowers_the_weight_of_common_terms():
    docs = [Counter(m=5, x=1), Counter(m=5, y=1), Counter(m=5, z=1)]
    space = TfIdf(docs)
    assert space.idf["m"] < space.idf["x"]
    assert space.cosine(docs[0], docs[1]) < cosine(docs[0], docs[1])


def test_structure_ignores_names_and_comments():
    a = "def f(m):\n    # comment\n    for d in m.D:\n        m.model.Add(m.x[d] <= 1)\n"
    b = 'def g(q):\n    """doc"""\n    for k in q.D:\n        q.model.Add(q.x[k] <= 1)\n'
    assert ast_bigrams(a) == ast_bigrams(b)
    assert code_tokens(a) != code_tokens(b)


def test_reference_scores_one_and_includes_helpers():
    assert "def _spread" in reference_source("F2")
    [scores] = code_similarities([("H3", reference_source("H3"))])
    assert scores == {"code_cosine_lexical": 1.0, "code_cosine_structural": 1.0}


def test_agreement_and_description_metrics():
    assert agreement({"samples": 30, "failures": []}) == 1.0
    assert agreement({"samples": 30, "failures": ["x"] * 3}) == 0.9
    assert agreement({"samples": 0, "failures": ["cannot load"]}) == 0.0
    [short, long] = description_metrics([("H3", "Rest."), ("H3", prompting.descriptions("s3_spec")["H3"])])
    assert short["description_words"] < long["description_words"]
    assert long["description_cosine_formal"] > short["description_cosine_formal"]


def test_html_report_of_a_benchmark(tmp_path):
    args = argparse.Namespace(sets=["s0_name", "s3_spec"], repeats=2, equations=["H2", "O2"], provider="reference",
                              model=None, command=None, attempts=1, jobs=2, samples=10, timeout=60,
                              verify_timeout=300, out=tmp_path / "bench")
    bench_dir = prompting.benchmark(args)
    page = (bench_dir / "report.html").read_text()
    assert "Pass rate per prompt set" in page and "plotly" in page.lower()
    answers = per_answer(collect(bench_dir))
    assert len(answers) == 2 * 2 * 2 and answers["passed"].all()
    assert (answers["agreement"] == 1.0).all()
    assert answers["code_cosine_structural"].between(0, 1).all()
    assert set(pd.read_csv(bench_dir / "metrics.csv")["set"]) == {"s0_name", "s3_spec"}
