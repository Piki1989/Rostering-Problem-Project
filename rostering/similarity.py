"""Similarity measures for generated equations (complementing the pass / fail verification).

* behavioural agreement - share of the verification schedules on which the implementation
  behaves like the oracle (1.0 = passes, 0.9 = wrong on 10 % of the schedules)
* lexical code cosine - cosine of the TF-IDF weighted bags of Python tokens of the candidate and
  the reference implementation (comments, docstrings and string literals removed)
* structural code cosine - cosine of the TF-IDF weighted bags of AST node-type bigrams
  (parent -> child); it ignores names and literals, so it compares the shape of the code
* text cosine - cosine of the TF-IDF weighted bags of words (without stop words) of two texts,
  e.g. a prompt description and the formal statement of the equation

The cosine similarity of two vectors a, b is a.b / (|a| |b|), in [0, 1] for non-negative weights.
TF-IDF weighs a term by its count times log((1 + N) / (1 + df)) + 1, where N is the number of
documents of the corpus (all generated and reference codes of a report) and df the number of
documents containing the term, so tokens present in every piece of code (m, model, brackets)
count little and the cosine reflects what is specific to the equation.
"""
from __future__ import annotations

import ast
import inspect
import io
import math
import re
import textwrap
import tokenize
from collections import Counter
from functools import lru_cache

from . import equations, model
from .equations import EQUATIONS


def cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0


class TfIdf:
    """IDF weights of a corpus of bags of terms."""

    def __init__(self, documents: list[Counter]):
        n = len(documents)
        df: Counter = Counter()
        for doc in documents:
            df.update(set(doc))
        self.idf = {term: math.log((1 + n) / (1 + count)) + 1 for term, count in df.items()}
        self.default = math.log(1 + n) + 1

    def weigh(self, doc: Counter) -> Counter:
        return Counter({t: c * self.idf.get(t, self.default) for t, c in doc.items()})

    def cosine(self, a: Counter, b: Counter) -> float:
        return cosine(self.weigh(a), self.weigh(b))


STOP_WORDS = set("""a an and are as at be by for from if in into is it its of on or that the then this to
with every each all any not no only one per than which who whose when where must may can do does
""".split())


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree


def code_tokens(source: str) -> Counter:
    """Bag of Python tokens without comments, docstrings, string contents and layout tokens."""
    try:
        source = ast.unparse(_strip_docstrings(ast.parse(textwrap.dedent(source))))
    except SyntaxError:
        pass
    counts: Counter = Counter()
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
            tokenize.ENCODING, tokenize.ENDMARKER}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type in skip:
                continue
            counts["<str>" if tok.type == tokenize.STRING else tok.string] += 1
    except (tokenize.TokenError, IndentationError, SyntaxError):
        counts.update(re.findall(r"\w+|[^\w\s]", source))
    return counts


def ast_bigrams(source: str) -> Counter:
    """Bag of (parent node type, child node type) pairs."""
    try:
        tree = _strip_docstrings(ast.parse(textwrap.dedent(source)))
    except SyntaxError:
        return Counter()
    counts: Counter = Counter()
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            if isinstance(child, (ast.Load, ast.Store, ast.Del)):
                continue
            counts[f"{type(parent).__name__}>{type(child).__name__}"] += 1
    return counts


def words(text: str) -> Counter:
    return Counter(w for w in (w.lower() for w in re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+", text))
                   if w not in STOP_WORDS)


@lru_cache(maxsize=None)
def reference_source(eq_id: str) -> str:
    """Source of the reference function plus the module-level helpers it calls (one level)."""
    fn = EQUATIONS[eq_id].fn
    sources = [inspect.getsource(fn)]
    called = {node.func.id for node in ast.walk(ast.parse(textwrap.dedent(sources[0])))
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    for name in sorted(called):
        helper = getattr(equations, name, None) or getattr(model, name, None)
        if inspect.isfunction(helper) and helper is not fn:
            sources.append(inspect.getsource(helper))
    return "\n\n".join(sources)


def code_similarities(pairs: list[tuple[str, str]]) -> list[dict]:
    """[(equation id, candidate source)] -> TF-IDF cosines to the reference, corpus = all codes."""
    references = {eq_id: reference_source(eq_id) for eq_id in {eq for eq, _ in pairs}}
    lexical = [(code_tokens(src), code_tokens(references[eq])) for eq, src in pairs]
    structural = [(ast_bigrams(src), ast_bigrams(references[eq])) for eq, src in pairs]
    lex_space = TfIdf([c for c, _ in lexical] + [code_tokens(r) for r in references.values()])
    ast_space = TfIdf([c for c, _ in structural] + [ast_bigrams(r) for r in references.values()])
    return [{"code_cosine_lexical": round(lex_space.cosine(*lex), 4),
             "code_cosine_structural": round(ast_space.cosine(*st), 4)}
            for lex, st in zip(lexical, structural)]


def agreement(verify_result: dict) -> float:
    """Share of the verification schedules without a failure (0 when nothing could be checked)."""
    samples = verify_result.get("samples", 0)
    if not samples:
        return 0.0
    failed = min(samples, len(verify_result.get("failures", [])))
    return round(1 - failed / samples, 4)


def description_metrics(items: list[tuple[str, str]]) -> list[dict]:
    """[(equation id, description)] -> words and TF-IDF cosine to the formal statement."""
    formal = {eq_id: inspect.getdoc(EQUATIONS[eq_id].fn) or "" for eq_id in {eq for eq, _ in items}}
    space = TfIdf([words(text) for _, text in items] + [words(f) for f in formal.values()])
    return [{"description_words": len(re.findall(r"\S+", text)),
             "description_cosine_formal": round(space.cosine(words(text), words(formal[eq_id])), 4)}
            for eq_id, text in items]
