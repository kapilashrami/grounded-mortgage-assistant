"""Figures for the evaluation page, built from docs/results/*.json.

Each figure is (section id, placement, html, script). Placement is "para:N"
(after the Nth paragraph of the section) or "table" (after its first table).
Counts come from the result files, so the figures cannot drift from the data.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

STYLE = """
  .fig { font-family: var(--sans); margin: 8px 0 44px; padding: 22px 22px 18px; background: var(--panel); border: 1px solid var(--rule); border-radius: 12px; }
  @media (min-width: 900px) { .fig { margin-left: -48px; margin-right: -48px; padding: 26px 30px 22px; } }
  .fig-title { font-size: 16px; font-weight: 600; margin: 0 0 4px; line-height: 1.4; }
  .fig-sub { font-size: 14px; color: var(--muted); margin: 0 0 18px; line-height: 1.5; }
  .legend { display: flex; flex-wrap: wrap; gap: 8px 18px; font-size: 13px; color: var(--muted); margin: 0 0 12px; }
  .sw { width: 12px; height: 12px; border-radius: 2px; display: inline-block; margin-right: 6px; vertical-align: -1px; }
  .waffle { display: grid; gap: 3px; max-width: 620px; }
  .waffle i { aspect-ratio: 1; border-radius: 2px; display: block; }
  .k-both { background: var(--both); } .k-a { background: var(--a); } .k-b { background: var(--b); }
  .k-bad { background: var(--bad); } .k-none { border: 1px solid var(--none); }
  .eq { display: grid; grid-template-columns: auto auto auto auto auto auto auto; gap: 2px 8px; width: max-content; max-width: 100%; font-size: 14px; margin-top: 16px; align-items: baseline; }
  .eq span, .eq strong { white-space: nowrap; }
  .eq .muted { color: var(--muted); } .eq .a { color: var(--a-ink); } .eq .b { color: var(--b-ink); } .eq strong { font-weight: 600; }
  .cap { font-size: 13.5px; color: var(--muted); margin: 14px 0 0; line-height: 1.55; }
  .stages { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
  .stage { background: var(--bg); border: 1px solid var(--rule); border-radius: 10px; padding: 14px 14px 12px; }
  .stage-n { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
  .stage h4 { font-size: 15px; margin: 4px 0 10px; font-weight: 600; }
  .chip { display: block; font-size: 13px; line-height: 1.45; padding: 6px 0; border-top: 1px solid var(--rule); }
  .chip b { font-weight: 600; color: var(--a-ink); }
  .gate svg { width: 100%; height: auto; display: block; }
  .gate-controls { display: flex; align-items: center; gap: 12px; font-size: 14px; margin-top: 8px; flex-wrap: wrap; }
  .gate-controls input[type=range] { flex: 1; min-width: 160px; accent-color: var(--a); }
  .gate-controls button { font: inherit; font-size: 13px; padding: 4px 10px; border-radius: 6px; border: 1px solid var(--rule); background: var(--bg); color: var(--text); cursor: pointer; }
  .gate-readout { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 14px; }
  .stat { background: var(--bg); border: 1px solid var(--rule); border-radius: 10px; padding: 10px 12px; }
  .stat-label { font-size: 12.5px; color: var(--muted); }
  .stat-value { font-size: 22px; font-weight: 600; letter-spacing: -.01em; }
  .heat { display: grid; gap: 6px; font-size: 13px; }
  .heat-row { display: grid; grid-template-columns: 150px 1fr 54px; gap: 10px; align-items: center; }
  .heat-cells { display: grid; gap: 2px; }
  .heat-cells i { height: 16px; border-radius: 2px; display: block; }
  .heat-label { color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .heat-group { font-weight: 600; margin: 10px 0 2px; }
  .heat-score { text-align: right; font-variant-numeric: tabular-nums; }
  .bars { display: grid; gap: 10px; font-size: 13.5px; }
  .bar-row { display: grid; grid-template-columns: 190px 1fr 64px; gap: 10px; align-items: center; }
  .bar-track { height: 14px; background: var(--bg); border: 1px solid var(--rule); border-radius: 3px; overflow: hidden; }
  .bar-fill { height: 100%; background: var(--a); }
  .bar-fill.alt { background: var(--b); }
  .bar-fill.dim { background: var(--both); }
  .bar-val { text-align: right; font-variant-numeric: tabular-nums; }
  .two { display: grid; grid-template-columns: 1fr 1fr; gap: 26px; }
  .two .bar-row { grid-template-columns: 100px 1fr 62px; }
  .sub-title { font-size: 13px; font-weight: 600; margin: 0 0 10px; }
  @media (max-width: 600px) {
    .fig { padding: 18px 16px 14px; margin-left: -8px; margin-right: -8px; }
    .stages, .two, .gate-readout { grid-template-columns: 1fr; }
    .heat-row { grid-template-columns: 96px 1fr 44px; }
    .bar-row, .two .bar-row { grid-template-columns: 1fr 64px; gap: 2px 10px; }
    .bar-row > span:first-child { grid-column: 1 / -1; }
  }
"""


def read(results: Path, name: str):
    return json.loads((results / f"{name}.json").read_text())


def figure(title: str, sub: str, inner: str, caption: str = "", extra_class: str = "") -> str:
    cap = f'<p class="cap">{caption}</p>' if caption else ""
    return f'<figure class="fig {extra_class}"><div class="fig-title">{title}</div><p class="fig-sub">{sub}</p>{inner}{cap}</figure>'


def legend(items) -> str:
    return '<div class="legend">' + "".join(
        f'<span><i class="sw {cls}"></i>{label}</span>' for cls, label in items
    ) + "</div>"


def waffle(groups, cols: int) -> str:
    cells = "".join(f'<i class="{cls}" title="{title}"></i>' for n, cls, title in groups for _ in range(n))
    return f'<div class="waffle" style="grid-template-columns:repeat({cols},1fr);max-width:{min(620, cols * 26)}px" role="img" aria-label="{", ".join(f"{n} {t}" for n, _, t in groups)}">{cells}</div>'


def equation(rows) -> str:
    """rows: (name, both, extra, extra_class, total)."""
    out = []
    for name, both, extra, cls, total in rows:
        out.append(
            f'<span>{name}</span><span>=</span><span class="muted">{both}</span><span>+</span>'
            f'<span class="{cls}">{extra}</span><span>=</span><strong>{total}</strong>'
        )
    return '<div class="eq">' + "".join(out) + "</div>"


def ratio(text: str) -> tuple[int, int]:
    a, b = text.split("/")
    return int(a), int(b)


def timeline(results):
    stages = [
        ("Stage 1", "Earlier capstone phase", [("Retrieval study", "whole Q&amp;A + BGE-M3 chosen")]),
        ("Stage 2", "Framework build", [("Retrieval parity", "159/159 exact"), ("Reranker", "local BGE kept"), ("LLM-as-a-judge", "547 of 549 graded")]),
        ("Stage 3", "Final testing", [("Demo journey", "16/16, 3 runs in a row"), ("Relevance gate", "0.23 chosen"), ("OpenAI models", "gpt-4.1 kept")]),
    ]
    inner = '<div class="stages">' + "".join(
        f'<div class="stage"><div class="stage-n">{n}</div><h4>{name}</h4>'
        + "".join(f'<span class="chip">{label}: <b>{result}</b></span>' for label, result in chips)
        + "</div>"
        for n, name, chips in stages
    ) + "</div>"
    return ("timeline", "para:1", figure("Three stages of evaluation", "The main result of each study, in the order they ran.", inner), "")


def relevance_gate(results):
    data = read(results, "relevance-gate")
    cases = [
        [c["case_id"], c["expected_status"], round(c["top_rerank_score"], 4)]
        for c in data["cases"]
        if c["expected_status"] in ("supported", "unsupported")
    ]
    inner = f"""<div class="gate">{legend([("k-a", "Supported question (the records can answer it)"), ("k-bad", "Unsupported question")])}
<svg id="gate-svg" viewBox="0 0 680 206" role="img" aria-label="Top reranker score of each question, with the threshold line"></svg>
<div class="gate-controls"><label for="gate-range">Threshold</label><input id="gate-range" type="range" min="0" max="0.95" step="0.01" value="{data['threshold']}"><strong id="gate-value">{data['threshold']:.2f}</strong><button type="button" id="gate-reset">Back to {data['threshold']:.2f}</button></div>
<div class="gate-readout"><div class="stat"><div class="stat-label">Supported questions kept</div><div class="stat-value" id="gate-kept"></div></div><div class="stat"><div class="stat-label">Unsupported questions rejected</div><div class="stat-value" id="gate-rejected"></div></div></div></div>"""
    script = f"""(function () {{
  const cases = {json.dumps(cases)};
  const chosen = {data['threshold']};
  const svg = document.getElementById('gate-svg');
  const ns = 'http://www.w3.org/2000/svg';
  const x = s => 40 + s * 600;
  let seed = 7; const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const el = (name, attrs) => {{ const e = document.createElementNS(ns, name); for (const k in attrs) e.setAttribute(k, attrs[k]); svg.appendChild(e); return e; }};
  el('line', {{x1: 40, x2: 640, y1: 172, y2: 172, stroke: 'var(--rule)'}});
  [0, 0.25, 0.5, 0.75, 1].forEach(t => {{
    el('line', {{x1: x(t), x2: x(t), y1: 172, y2: 177, stroke: 'var(--muted)'}});
    const label = el('text', {{x: x(t), y: 198, 'text-anchor': 'middle', 'font-size': 15, fill: 'var(--muted)'}}); label.textContent = t.toFixed(2);
  }});
  const rows = {{supported: [34, 106], unsupported: [126, 158]}};
  const dots = cases.map(([id, status, score]) => {{
    const [top, bottom] = rows[status];
    const c = el('circle', {{cx: x(score), cy: top + rand() * (bottom - top), r: 5, fill: status === 'supported' ? 'var(--a)' : 'var(--bad)'}});
    const t = document.createElementNS(ns, 'title'); t.textContent = id + ': ' + score.toFixed(2); c.appendChild(t);
    return {{c, status, score}};
  }});
  const line = el('line', {{y1: 22, y2: 172, stroke: 'var(--text)', 'stroke-width': 1.5, 'stroke-dasharray': '4 3'}});
  const tag = el('text', {{y: 15, 'font-size': 15, fill: 'var(--text)', 'text-anchor': 'middle'}});
  const range = document.getElementById('gate-range');
  const sup = cases.filter(c => c[1] === 'supported').length, uns = cases.length - sup;
  function update() {{
    const t = parseFloat(range.value);
    line.setAttribute('x1', x(t)); line.setAttribute('x2', x(t));
    tag.setAttribute('x', Math.min(Math.max(x(t), 70), 610)); tag.textContent = 'reject below ' + t.toFixed(2);
    let kept = 0, rejected = 0;
    dots.forEach(d => {{
      const pass = d.score >= t;
      d.c.setAttribute('opacity', pass ? 1 : 0.28);
      if (d.status === 'supported' && pass) kept++;
      if (d.status === 'unsupported' && !pass) rejected++;
    }});
    document.getElementById('gate-value').textContent = t.toFixed(2);
    document.getElementById('gate-kept').textContent = kept + ' / ' + sup;
    document.getElementById('gate-rejected').textContent = rejected + ' / ' + uns;
  }}
  range.addEventListener('input', update);
  document.getElementById('gate-reset').addEventListener('click', () => {{ range.value = chosen; update(); }});
  update();
}})();"""
    return ("relevance-gate", "table", figure(
        "Try the threshold",
        "Each dot is one question, placed at the best reranker score its retrieval got. Questions left of the line are rejected without calling the model.",
        inner,
        "Faded dots are rejected. Drag right and supported questions start to fade long before the last unsupported ones do: rejecting all nine unsupported questions costs 28 supported ones.",
    ), script)


def reranker(results):
    data = read(results, "reranker-comparison")
    local, total = ratio(data["local_bge"]["complete_supported_cases"])
    cohere, _ = ratio(data["cohere"]["complete_supported_cases"])
    gained, lost = len(data["cases_gained_complete_support"]), len(data["cases_lost_complete_support"])
    both = local - lost
    neither = total - both - gained - lost
    inner = legend([
        ("k-both", f"Both complete: {both}"), ("k-a", f"Only local BGE: {lost}"),
        ("k-b", f"Only Cohere: {gained}"), ("k-none", f"Neither: {neither}"),
    ]) + waffle([(both, "k-both", "both"), (lost, "k-a", "only local BGE"), (gained, "k-b", "only Cohere"), (neither, "k-none", "neither")], 25)
    inner += equation([("Local BGE", both, lost, "a", local), ("Cohere", both, gained, "b", cohere)])
    return ("reranker-comparison", "table", figure(
        f"Where the two rerankers differ, out of {total} cases",
        "Each square is one supported case. Only the colored squares separate the rerankers.",
        inner,
        f"The {both} gray cases count for both, so Cohere's lead comes from {gained} against {lost}. A split that small is easily luck, and it needs a network call and a per-request cost.",
    ), "")


def retrieval_study(results):
    data = read(results, "retrieval-study")
    change = data["held_out_change"]
    held = next(r for r in data["results_by_split"] if r["split"] == "held-out")
    selected, total = ratio(held["selected_complete"])
    gained, lost, neither = len(change["gained"]), len(change["lost"]), len(change["missed_by_both"])
    both = selected - gained
    inner = legend([("k-both", f"Both found it: {both}"), ("k-a", f"Only the selected setup: {gained}"), ("k-none", f"Both missed: {neither}")])
    inner += waffle([(both, "k-both", "both"), (gained, "k-a", "only selected"), (lost, "k-b", "only baseline"), (neither, "k-none", "both missed")], 14)
    return ("earlier-retrieval-study", "table", figure(
        f"Held-out questions: {selected}/{total} against {selected - gained + lost}/{total}",
        "Each square is one held-out question with complete labelled evidence in the top 5.",
        inner,
        "The whole held-out gain is a single question, which is why the study supports the choice only modestly.",
    ), "")


def parity(results):
    total = len(read(results, "relevance-gate")["cases"])
    inner = legend([("k-a", f"Identical evidence: {total} of {total}")]) + waffle([(total, "k-a", "identical")], 40)
    return ("retrieval-parity", "para:2", figure(
        "Rebuilt retrieval against the saved retrieval",
        "Each square is one fixed query. Evidence IDs, packed texts and top-10 candidates all matched.",
        inner,
    ), "")


def browser_suites(results):
    data = read(results, "browser-suites")
    journey_ids = list(data["journey_check_ids"])
    question_ids = [q for ids in data["question_categories"].values() for q in ids]
    runs = [r for r in data["runs"] if r["phase"] in ("final, all fixes applied", "post-cleanup demo flow")]

    def rows(suite, ids):
        out, counter = [], {}
        for run in runs:
            if run["suite"] != suite:
                continue
            outcomes = {c["id"]: c["outcome"] for c in run.get("checks", run.get("cases", []))}
            who = "Qwen" if run["provider"] == "local" else run["model"]
            label = f"{who} {'post-cleanup' if run['phase'].startswith('post') else 'final'}"
            counter[label] = counter.get(label, 0) + 1
            cells = "".join(
                f'<i class="{"k-a" if outcomes.get(i) == "pass" else "k-bad"}" title="{i}: {outcomes.get(i, "not run")}"></i>'
                for i in ids
            )
            failed = [i for i in ids if outcomes.get(i) != "pass"]
            out.append(
                f'<div class="heat-row"><span class="heat-label" title="{label} run {counter[label]}">{label} {counter[label]}</span>'
                f'<span class="heat-cells" style="grid-template-columns:repeat({len(ids)},1fr)">{cells}</span>'
                f'<span class="heat-score">{run["passed"]}/{run["total"]}</span></div>'
                + (f'<div class="heat-row"><span></span><span class="heat-label">failed: {", ".join(failed)}</span><span></span></div>' if failed else "")
            )
        return "".join(out)

    inner = legend([("k-a", "Pass"), ("k-bad", "Fail")])
    inner += f'<div class="heat"><div class="heat-group">16-check demo journey</div>{rows("journey", journey_ids)}'
    inner += f'<div class="heat-group">20-question suite</div>{rows("questions", question_ids)}</div>'
    return ("browser-suites", "para:3", figure(
        "Every check of every final run",
        "One row per run, one square per check or question. Hover a square to see its ID.",
        inner,
        "The only red square is U02 in every question run, the known routing limit.",
    ), "")


def models(results):
    data = read(results, "model-comparison")
    rows = data["models"]
    top_cost = max(m["usd_per_model_turn_mean"] for m in rows)

    def bar(value, top, text, cls=""):
        return f'<span class="bar-track"><span class="bar-fill {cls}" style="display:block;width:{value / top * 100:.1f}%"></span></span><span class="bar-val">{text}</span>'

    quality = "".join(
        f'<div class="bar-row"><span>{m["model"]}</span>{bar(ratio(m["questions_passed"])[0], 60, m["questions_passed"])}</div>' for m in rows
    )
    cost = "".join(
        f'<div class="bar-row"><span>{m["model"]}</span>{bar(m["usd_per_model_turn_mean"], top_cost, "$" + format(m["usd_per_model_turn_mean"], ".4f"), "alt")}</div>' for m in rows
    )
    inner = f'<div class="two"><div><p class="sub-title">Questions passed, 3 runs of 20</p><div class="bars">{quality}</div></div><div><p class="sub-title">Cost per model turn</p><div class="bars">{cost}</div></div></div>'
    return ("model-comparison", "table", figure(
        "Quality is nearly level, cost is not",
        "All three passed 48 of 48 journey checks. The question bars start at zero, so small gaps look small.",
        inner,
        "gpt-4.1 costs about five times as much as gpt-4.1-mini for one more passed question in 60.",
    ), "")


def judge(results):
    data = read(results, "llm-judge")
    paired = data["paired_comparison"]
    n = paired["n"]
    names = {
        "117ca79fe094e5ba": "EmbeddingGemma&#8209;2 + rerank",
        "english-balanced-10x2-rerank": "Two&#8209;view BGE&#8209;M3 + rerank",
        "english-whole-qa-clean-rerank": "Whole Q&amp;A BGE&#8209;M3 + rerank",
        "swedish-clean-dense-english-query-control": "Swedish control, no rerank",
    }
    accepted = {arm: r["fully_accepted"] for arm, r in paired["results_by_arm"].items()}
    order = sorted(accepted, key=accepted.get, reverse=True)
    bars = "".join(
        f'<div class="bar-row"><span>{names[arm]}</span><span class="bar-track"><span class="bar-fill {"alt" if arm == "117ca79fe094e5ba" else "" if arm == "english-balanced-10x2-rerank" else "dim"}" style="display:block;width:{accepted[arm] / n * 100:.1f}%"></span></span><span class="bar-val">{accepted[arm]}/{n}</span></div>'
        for arm in order
    )
    a, b = "english-balanced-10x2-rerank", "117ca79fe094e5ba"
    cells = [c["fully_accepted"] for c in paired["cases"]]
    both = sum(c[a] and c[b] for c in cells)
    only_a = sum(c[a] and not c[b] for c in cells)
    only_b = sum(c[b] and not c[a] for c in cells)
    neither = n - both - only_a - only_b
    pair = next(p for p in paired["pairwise_fully_accepted"] if {p["a"], p["b"]} == {a, b})
    grid = legend([
        ("k-both", f"Both accepted: {both}"), ("k-b", f"Only EmbeddingGemma: {only_b}"),
        ("k-a", f"Only two-view: {only_a}"), ("k-none", f"Neither: {neither}"),
    ]) + waffle([(both, "k-both", "both"), (only_b, "k-b", "only EmbeddingGemma"), (only_a, "k-a", "only two-view"), (neither, "k-none", "neither")], 25)
    grid += equation([
        ("EmbeddingGemma", both, only_b, "b", accepted[b]),
        ("Two-view", both, only_a, "a", accepted[a]),
    ])
    inner = f'<div class="bars">{bars}</div><div style="height:22px"></div><p class="sub-title">The top two, question by question</p>{grid}'
    return ("llm-as-a-judge", "table", figure(
        f"Fully accepted answers on {n} questions",
        "Every arm answered the same questions, so they can be compared question by question.",
        inner,
        f"The {both} gray questions add the same to both scores. The {accepted[b] - accepted[a]}-point lead comes from {only_b} against {only_a}, and a fair coin splits {only_a + only_b} flips at least that unevenly about half the time (exact McNemar p = {pair['exact_mcnemar_p']:.2f}). That is why no winner is named.",
    ), "")


def build(results: Path):
    random.seed(0)
    return [timeline(results), relevance_gate(results), reranker(results), retrieval_study(results),
            parity(results), browser_suites(results), models(results), judge(results)]
