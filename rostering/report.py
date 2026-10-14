"""HTML report with charts for a prompt run or a benchmark (python -m rostering.prompting report <dir>).

Per equation answer (last attempt and first attempt):
- passed;
- behavioural agreement with the oracle;
- lexical and structural cosine similarity of the code to the reference implementation;
- number of attempts;
- model time;
- amount of information in the prompt description (words, cosine to the formal statement).
"""
from __future__ import annotations

import html
import inspect
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .equations import EQUATIONS
from .similarity import agreement, code_similarities, description_metrics
from .verify import CHECKS

KIND_LABELS = {"restriction": "constraints", "value": "definitions", "minimum": "definitions",
               "objective": "objectives"}


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------

def _run_dirs(path: Path) -> list[tuple[Path, dict, int]]:
    """(run directory, its config, repeat number) for a single run or for every run of a benchmark."""
    config = json.loads((path / "config.json").read_text())
    if "sets" in config and "set" not in config:
        out = []
        for sub in sorted(p for p in path.iterdir() if p.is_dir() and (p / "config.json").exists()):
            sub_config = json.loads((sub / "config.json").read_text())
            repeat = int(sub.name.rsplit("_r", 1)[1]) if "_r" in sub.name else 1
            out.append((sub, sub_config, repeat))
        return out
    return [(path, config, 1)]


def collect(path: Path) -> pd.DataFrame:
    """One row per (set, repeat, equation, attempt)."""
    from .prompting import _set_config, descriptions

    rows, codes, descriptions_by_key = [], [], {}
    for run_dir, config, repeat in _run_dirs(path):
        set_name = config.get("set") or config.get("level") or "?"
        try:
            texts = descriptions(set_name)
            formal = bool(_set_config(set_name).get("formal"))
        except (SystemExit, FileNotFoundError):
            texts, formal = {}, False
        for eq_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name in EQUATIONS):
            eq_id = eq_dir.name
            key = (set_name, eq_id)
            if key not in descriptions_by_key:
                text = texts.get(eq_id, "")
                if formal:
                    text += "\n" + (inspect.getdoc(EQUATIONS[eq_id].fn) or "")
                descriptions_by_key[key] = text
            for attempt_dir in sorted(eq_dir.glob("attempt_*"), key=lambda p: int(p.name.split("_")[1])):
                verify_file = attempt_dir / "verify.json"
                if not verify_file.exists():
                    continue
                verdict = json.loads(verify_file.read_text())
                candidate = attempt_dir / "candidate.py"
                codes.append((eq_id, candidate.read_text() if candidate.exists() else ""))
                rows.append({
                    "set": set_name,
                    "repeat": repeat,
                    "equation": eq_id,
                    "name": EQUATIONS[eq_id].name,
                    "check": CHECKS[eq_id].kind,
                    "group": KIND_LABELS[CHECKS[eq_id].kind],
                    "attempt": verdict.get("number", int(attempt_dir.name.split("_")[1])),
                    "passed": bool(verdict.get("passed")),
                    "agreement": agreement(verdict),
                    "samples": verdict.get("samples", 0),
                    "failures": len(verdict.get("failures", [])),
                    "seconds_model": verdict.get("seconds_model", 0.0),
                    "has_code": candidate.exists(),
                    "first_failure": (verdict.get("failures") or [verdict.get("error", "")])[0][:200],
                })
    if not rows:
        raise SystemExit(f"no verified answers found in {path}")
    df = pd.DataFrame(rows)
    similarity = pd.DataFrame(code_similarities(codes))
    similarity.loc[~df["has_code"], ["code_cosine_lexical", "code_cosine_structural"]] = 0.0
    keys = list(descriptions_by_key)
    info = dict(zip(keys, description_metrics([(eq, descriptions_by_key[(st, eq)]) for st, eq in keys])))
    info_df = pd.DataFrame([info[(r.set, r.equation)] for r in df.itertuples()])
    return pd.concat([df, similarity, info_df], axis=1)


def per_answer(df: pd.DataFrame) -> pd.DataFrame:
    """Last attempt of every (set, repeat, equation) plus first-attempt columns."""
    keys = ["set", "repeat", "equation"]
    first = df.sort_values("attempt").groupby(keys).first()
    last = df.sort_values("attempt").groupby(keys).last()
    last["attempts"] = df.groupby(keys)["attempt"].max()
    last["passed_first"] = first["passed"]
    last["agreement_first"] = first["agreement"]
    return last.reset_index()


def set_summary(answers: pd.DataFrame, sets: list[str]) -> pd.DataFrame:
    per_repeat = answers.groupby(["set", "repeat"]).agg(pass_rate=("passed", "mean")).reset_index()
    summary = answers.groupby("set").agg(
        answers=("passed", "size"),
        pass_rate=("passed", "mean"),
        pass_rate_first=("passed_first", "mean"),
        agreement=("agreement", "mean"),
        code_cosine_lexical=("code_cosine_lexical", "mean"),
        code_cosine_structural=("code_cosine_structural", "mean"),
        description_words=("description_words", "mean"),
        description_cosine_formal=("description_cosine_formal", "mean"),
        attempts=("attempts", "mean"),
    )
    summary["pass_rate_std"] = per_repeat.groupby("set")["pass_rate"].std().fillna(0.0)
    return summary.reindex([s for s in sets if s in summary.index])


# ---------------------------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------------------------

def _figures(answers: pd.DataFrame, summary: pd.DataFrame, sets: list[str]) -> list[tuple[str, str, object]]:
    import plotly.express as px
    import plotly.graph_objects as go

    palette = px.colors.qualitative.Bold
    colors = {s: palette[i % len(palette)] for i, s in enumerate(sets)}
    layout = dict(template="plotly_white", margin=dict(l=60, r=30, t=50, b=60), font=dict(size=13),
                  legend=dict(orientation="h", y=-0.2))
    figs = []

    # 1. Pass rate per set.
    fig = go.Figure()
    fig.add_bar(name="passed (all attempts)", x=summary.index, y=summary["pass_rate"],
                error_y=dict(type="data", array=summary["pass_rate_std"], visible=bool(summary["pass_rate_std"].any())),
                marker_color="#2563eb", text=[f"{v:.0%}" for v in summary["pass_rate"]], textposition="outside")
    fig.add_bar(name="passed in the first attempt", x=summary.index, y=summary["pass_rate_first"],
                marker_color="#93c5fd", text=[f"{v:.0%}" for v in summary["pass_rate_first"]], textposition="outside")
    fig.add_scatter(name="behavioural agreement", x=summary.index, y=summary["agreement"], mode="lines+markers",
                    line=dict(color="#f97316", width=3), marker=dict(size=10))
    fig.update_layout(**layout, barmode="group", yaxis=dict(tickformat=".0%", range=[0, 1.12], title="share"),
                      xaxis_title="prompt set")
    figs.append(("Pass rate per prompt set",
                 "The bars show the share of verified equations; the error bar is the standard deviation over "
                 "the repeats. The line shows the share of test schedules handled correctly, which also counts "
                 "partially correct code.", fig))

    # 2. Information curve.
    fig = go.Figure()
    fig.add_scatter(x=summary["description_words"], y=summary["pass_rate"], mode="lines+markers+text",
                    text=summary.index, textposition="top center", name="pass rate",
                    line=dict(color="#2563eb", width=3), marker=dict(size=12))
    fig.add_scatter(x=summary["description_words"], y=summary["agreement"], mode="lines+markers",
                    name="behavioural agreement", line=dict(color="#f97316", width=3, dash="dot"), marker=dict(size=10))
    fig.update_layout(**layout, xaxis_title="average words of the equation description", xaxis_type="log",
                      yaxis=dict(tickformat=".0%", range=[0, 1.1], title="share"))
    figs.append(("How much information is needed",
                 "Pass rate against the amount of information in the description of the equation (log scale). "
                 "The interface description is the same in all sets.", fig))

    # 3. Heatmap equation x set (pass share).
    order = [e for e in EQUATIONS if e in set(answers["equation"])]
    matrix = answers.pivot_table(index="equation", columns="set", values="passed", aggfunc="mean").reindex(
        index=order, columns=[s for s in sets if s in set(answers["set"])])
    labels = [f"{e} {EQUATIONS[e].name}" for e in matrix.index]
    fig = go.Figure(go.Heatmap(z=matrix.values, x=matrix.columns, y=labels, zmin=0, zmax=1,
                               colorscale=[[0, "#dc2626"], [0.5, "#facc15"], [1, "#16a34a"]],
                               text=[[f"{v:.0%}" if pd.notna(v) else "" for v in row] for row in matrix.values],
                               texttemplate="%{text}", colorbar=dict(title="passed", tickformat=".0%")))
    fig.update_layout(**{**layout, "margin": dict(l=280, r=30, t=30, b=40)}, height=max(400, 22 * len(labels) + 120),
                      yaxis=dict(autorange="reversed"))
    figs.append(("Equations x prompt sets",
                 "Share of the repeats in which the equation passed the verification.", fig))

    # 4. Agreement heatmap.
    matrix = answers.pivot_table(index="equation", columns="set", values="agreement", aggfunc="mean").reindex(
        index=order, columns=[s for s in sets if s in set(answers["set"])])
    fig = go.Figure(go.Heatmap(z=matrix.values, x=matrix.columns, y=labels, zmin=0, zmax=1, colorscale="Viridis",
                               text=[[f"{v:.2f}" if pd.notna(v) else "" for v in row] for row in matrix.values],
                               texttemplate="%{text}", colorbar=dict(title="agreement")))
    fig.update_layout(**{**layout, "margin": dict(l=280, r=30, t=30, b=40)}, height=max(400, 22 * len(labels) + 120),
                      yaxis=dict(autorange="reversed"))
    figs.append(("Behavioural agreement",
                 "Average share of test schedules on which the generated code behaves like the specification "
                 "(1.00 = always).", fig))

    # 5. Groups of equations.
    groups = answers.groupby(["set", "group"])["passed"].mean().reset_index()
    fig = px.bar(groups, x="set", y="passed", color="group", barmode="group",
                 category_orders={"set": sets, "group": ["constraints", "definitions", "objectives"]},
                 color_discrete_sequence=["#0ea5e9", "#a855f7", "#f43f5e"], text_auto=".0%")
    fig.update_layout(**layout, yaxis=dict(tickformat=".0%", range=[0, 1.1], title="passed"), xaxis_title="prompt set")
    figs.append(("Constraints, definitions and objectives",
                 "Pass rate by type of equation: constraints (restrictions), definitions of variables "
                 "(values and minima) and objectives of the phases.", fig))

    # 6. Code similarity vs correctness.
    fig = px.scatter(answers, x="code_cosine_structural", y="agreement", color="set", symbol="passed",
                     category_orders={"set": sets}, color_discrete_map=colors,
                     hover_data=["equation", "name", "repeat", "code_cosine_lexical", "first_failure"],
                     labels={"code_cosine_structural": "structural code cosine (AST)", "agreement": "behavioural agreement"})
    fig.update_traces(marker=dict(size=11, opacity=0.8, line=dict(width=1, color="white")))
    fig.update_layout(**layout, xaxis_range=[0, 1.02], yaxis_range=[-0.03, 1.05])
    figs.append(("Is similar code correct code?",
                 "Every point is one generated equation. The x axis is the cosine similarity of its code structure "
                 "to the reference; the y axis is the share of correctly handled test schedules. Correct code can "
                 "look very different from the reference.", fig))

    # 7. Distributions of the cosine similarities.
    long = answers.melt(id_vars=["set"], value_vars=["code_cosine_lexical", "code_cosine_structural"],
                        var_name="measure", value_name="cosine")
    long["measure"] = long["measure"].map({"code_cosine_lexical": "lexical (tokens)",
                                           "code_cosine_structural": "structural (AST)"})
    fig = px.box(long, x="set", y="cosine", color="measure", points="all", category_orders={"set": sets},
                 color_discrete_sequence=["#64748b", "#0d9488"])
    fig.update_layout(**layout, yaxis_range=[0, 1.02], xaxis_title="prompt set")
    figs.append(("Cosine similarity of the code to the reference",
                 "Distribution over all generated equations. Lexical: bag of Python tokens; structural: "
                 "bag of AST node-type pairs (names do not matter).", fig))

    # 8. Description information per set.
    fig = go.Figure()
    fig.add_bar(x=summary.index, y=summary["description_cosine_formal"], marker_color="#0d9488",
                text=[f"{v:.2f}" for v in summary["description_cosine_formal"]], textposition="outside",
                name="cosine to the formal statement")
    fig.update_layout(**layout, yaxis=dict(range=[0, 1.1], title="cosine"), xaxis_title="prompt set")
    figs.append(("How close is the description to the formal specification",
                 "Average cosine similarity (bag of words) between the description in the prompt and the formal "
                 "statement of the equation.", fig))

    if answers["attempts"].max() > 1:
        att = answers.groupby(["set", "attempts"]).size().reset_index(name="answers")
        fig = px.bar(att, x="set", y="answers", color=att["attempts"].astype(str), barmode="stack",
                     category_orders={"set": sets}, labels={"color": "attempts"})
        fig.update_layout(**layout, xaxis_title="prompt set")
        figs.append(("Attempts needed", "Number of attempts used per equation (feedback with counterexamples).", fig))
    return figs


# ---------------------------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------------------------

CSS = """
:root { --bg:#f8fafc; --card:#ffffff; --text:#0f172a; --muted:#64748b; --line:#e2e8f0; --accent:#2563eb; }
* { box-sizing: border-box; }
body { margin:0; font-family: -apple-system, "Segoe UI", Roboto, sans-serif; background:var(--bg); color:var(--text); }
header { padding: 32px 40px 8px; } h1 { margin:0 0 6px; font-size: 26px; } .sub { color:var(--muted); }
main { padding: 8px 40px 48px; max-width: 1280px; }
.kpis { display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin: 18px 0 26px; }
.kpi { background:var(--card); border:1px solid var(--line); border-radius: 14px; padding: 16px 18px; }
.kpi .v { font-size: 28px; font-weight: 700; } .kpi .l { color:var(--muted); font-size: 13px; margin-top: 4px; }
section { background:var(--card); border:1px solid var(--line); border-radius: 14px; padding: 18px 20px; margin-bottom: 20px; }
section h2 { margin: 0 0 4px; font-size: 18px; } section p { margin: 0 0 8px; color:var(--muted); font-size: 14px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: left; vertical-align: top; }
th { color: var(--muted); font-weight: 600; }
.pass { color:#16a34a; font-weight:600; } .fail { color:#dc2626; font-weight:600; }
details summary { cursor: pointer; color: var(--accent); margin: 6px 0; }
"""


def _table(df: pd.DataFrame, formats: dict) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in [df.index.name or ""] + list(df.columns))
    body = []
    for idx, row in df.iterrows():
        cells = [f"<td><b>{html.escape(str(idx))}</b></td>"]
        for col in df.columns:
            value = row[col]
            fmt = formats.get(col)
            text = fmt(value) if fmt and pd.notna(value) else html.escape(str(value))
            cells.append(f"<td>{text}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def build_report(path: Path, offline: bool = False) -> Path:
    path = Path(path)
    df = collect(path)
    answers = per_answer(df)
    config = json.loads((path / "config.json").read_text())
    sets = [s.strip() for s in str(config.get("sets", config.get("set", ""))).split(",") if s.strip()] \
        or sorted(answers["set"].unique())
    sets = [s for s in sets if s in set(answers["set"])] + sorted(set(answers["set"]) - set(sets))
    summary = set_summary(answers, sets)

    pct = lambda v: f"{v:.0%}"  # noqa: E731
    num = lambda v: f"{v:.2f}"  # noqa: E731
    best = summary["pass_rate"].idxmax()
    kpis = [
        (pct(answers["passed"].mean()), "passed overall"),
        (pct(answers["passed_first"].mean()), "passed in the first attempt"),
        (num(answers["agreement"].mean()), "average behavioural agreement"),
        (num(answers["code_cosine_structural"].mean()), "average structural code cosine"),
        (f"{best}", f"best set ({pct(summary.loc[best, 'pass_rate'])})"),
        (str(len(answers)), "generated equations"),
    ]

    table = summary.rename(columns={
        "answers": "answers", "pass_rate": "passed", "pass_rate_first": "1st attempt", "agreement": "agreement",
        "code_cosine_lexical": "code cos. lexical", "code_cosine_structural": "code cos. AST",
        "description_words": "words / description", "description_cosine_formal": "cos. to formal",
        "attempts": "attempts", "pass_rate_std": "std over repeats"})
    table.index.name = "set"
    summary_html = _table(table, {"answers": lambda v: f"{v:.0f}", "passed": pct, "1st attempt": pct, "agreement": num, "code cos. lexical": num,
                                  "code cos. AST": num, "words / description": lambda v: f"{v:.0f}",
                                  "cos. to formal": num, "attempts": num, "std over repeats": pct})

    failures = answers[~answers["passed"]].sort_values(["set", "equation"])
    fail_rows = "".join(
        f"<tr><td>{html.escape(r.set)}</td><td>{r.repeat}</td><td>{r.equation} {html.escape(r.name)}</td>"
        f"<td>{r.agreement:.2f}</td><td>{html.escape(r.first_failure)}</td></tr>" for r in failures.itertuples())
    failures_html = (f"<table><thead><tr><th>set</th><th>repeat</th><th>equation</th><th>agreement</th>"
                     f"<th>first failure</th></tr></thead><tbody>{fail_rows}</tbody></table>") if fail_rows else \
        "<p>No failures.</p>"

    figs = _figures(answers, summary, sets)
    sections = []
    for i, (title, text, fig) in enumerate(figs):
        include = (True if offline else "cdn") if i == 0 else False
        sections.append(f"<section><h2>{html.escape(title)}</h2><p>{html.escape(text)}</p>"
                        f"{fig.to_html(full_html=False, include_plotlyjs=include, config={'displaylogo': False})}</section>")

    settings = " · ".join(f"{k}: {html.escape(str(v))}" for k, v in config.items())
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Prompt benchmark report</title>
<style>{CSS}</style></head><body>
<header><h1>Prompt benchmark - {html.escape(path.name)}</h1>
<div class="sub">{settings}<br>generated {datetime.now():%Y-%m-%d %H:%M}</div></header>
<main>
<div class="kpis">{''.join(f'<div class="kpi"><div class="v">{v}</div><div class="l">{html.escape(l)}</div></div>' for v, l in kpis)}</div>
<section><h2>Summary per prompt set</h2>
<p>passed / 1st attempt: share of equations passing the verification; agreement: share of test schedules
handled like the specification; code cos.: cosine similarity of the generated code to the reference
implementation (tokens / AST structure); words, cos. to formal: amount of information in the description.</p>
{summary_html}</section>
{''.join(sections)}
<section><h2>Failed equations</h2><details><summary>show {len(failures)} failed answers</summary>{failures_html}</details></section>
</main></body></html>"""
    out = path / "report.html"
    out.write_text(page)
    df.to_csv(path / "metrics.csv", index=False)
    return out
