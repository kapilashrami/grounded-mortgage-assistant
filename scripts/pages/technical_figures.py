"""Figures for the technical documentation page.

Same shape as evaluation_figures: (section id, placement, html, script).
Comparison numbers are read from docs/results/*.json.
"""

from __future__ import annotations

from pathlib import Path

from evaluation_figures import STYLE as BASE_STYLE
from evaluation_figures import figure, ratio, read

STYLE = BASE_STYLE + """
  .steps-nav { display: flex; gap: 6px; flex-wrap: wrap; margin: 0 0 14px; }
  .steps-nav button { font: inherit; font-size: 13px; padding: 5px 10px; border-radius: 999px; border: 1px solid var(--rule); background: var(--bg); color: var(--muted); cursor: pointer; }
  .steps-nav button[aria-selected=true] { background: var(--a); border-color: var(--a); color: #fff; }
  .step { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; min-height: 236px; }
  .js .step:not(.active) { display: none; }
  .step + .step { margin-top: 18px; }
  .js .step + .step { margin-top: 0; }
  .chat { background: var(--bg); border: 1px solid var(--rule); border-radius: 10px; padding: 14px; display: grid; gap: 10px; align-content: start; }
  .msg { font-size: 14px; line-height: 1.5; padding: 9px 12px; border-radius: 12px; max-width: 92%; }
  .msg.user { justify-self: end; background: color-mix(in srgb, var(--a) 16%, var(--bg)); border-bottom-right-radius: 4px; }
  .msg.bot { justify-self: start; border: 1px solid var(--rule); border-bottom-left-radius: 4px; }
  .msg.card { border: 1px solid var(--a); }
  .msg .big { display: block; font-size: 24px; font-weight: 600; color: var(--a-ink); letter-spacing: -.01em; }
  .hood { font-size: 13.5px; line-height: 1.55; }
  .hood-title { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; margin: 0 0 8px; }
  .hood ol { margin: 0; padding-left: 20px; }
  .hood li { margin-bottom: 6px; padding-left: 2px; }
  .step-foot { display: flex; justify-content: space-between; margin-top: 14px; }
  .step-foot button { font: inherit; font-size: 13px; padding: 5px 12px; border-radius: 6px; border: 1px solid var(--rule); background: var(--bg); color: var(--text); cursor: pointer; }
  .step-foot button:disabled { opacity: .4; cursor: default; }
  .cmp { display: grid; gap: 18px; }
  .cmp .bar-row { grid-template-columns: 230px 1fr 64px; }
  .cmp-metric { font-weight: 400; color: var(--muted); margin-left: 6px; }
  .chosen { font-style: normal; font-size: 11px; color: var(--a-ink); border: 1px solid var(--a); border-radius: 999px; padding: 0 6px; margin-left: 6px; }
  .cmp-axis { display: grid; grid-template-columns: 230px 1fr 64px; gap: 10px; font-size: 12px; color: var(--muted); margin-top: 6px; }
  .axis-ticks { position: relative; height: 16px; border-top: 1px solid var(--rule); }
  .axis-ticks span { position: absolute; top: 2px; transform: translateX(-50%); }
  .pivots { display: grid; gap: 10px; }
  .pivot { display: grid; grid-template-columns: 110px 1fr 22px 1fr; gap: 10px; align-items: stretch; font-size: 13.5px; line-height: 1.45; }
  .pivot-area { font-weight: 600; padding-top: 10px; }
  .pivot-before, .pivot-after { padding: 9px 12px; border-radius: 8px; background: var(--bg); border: 1px solid var(--rule); }
  .pivot-before { border-left: 3px solid var(--bad); border-radius: 0 8px 8px 0; }
  .pivot-after { border-left: 3px solid var(--a); border-radius: 0 8px 8px 0; }
  .pivot-arrow { align-self: center; text-align: center; color: var(--muted); }
  .pivot-head { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
  @media (max-width: 600px) {
    .step { grid-template-columns: 1fr; min-height: 0; }
    .cmp .bar-row { grid-template-columns: 1fr 64px; }
    .cmp-axis { grid-template-columns: 1fr 64px; gap: 2px 10px; }
    .cmp-axis > span:first-child { display: none; }
    .pivot { grid-template-columns: 1fr; gap: 6px; }
    .pivot-arrow { display: none; }
    .pivot-head-row { display: none; }
  }
"""

STEPS = [
    ("Ask", "What is a mortgage?",
     '<div class="msg bot">A cited answer built from the reviewed record, with an English PDF source.</div>',
     ["The local PII check passes the message.", "Retrieval finds candidate records and the reranker scores them.",
      "The best score clears the 0.23 gate, so the model is called.", "The LLM picks evidence IDs; the server renders the reviewed extract and its citation."]),
    ("Calculate", "What is my LTV? My mortgage is 2,000,000 and my home is worth 2,500,000.",
     '<div class="msg bot card">Loan-to-value<span class="big">80.00%</span>Calculation only, not a lending decision.</div>',
     ["The PII check masks the amounts before any model sees them.", "The LLM answers with a schema-validated tool call.",
      "Python calculates the LTV with Decimal on the amounts the customer typed.", "The facts are stored in session memory on the server."]),
    ("Refinance", "I want to refinance my existing mortgage without borrowing more for my house.",
     '<div class="msg bot">I have recorded the supported details.</div>',
     ["The LLM interprets the turn as refinance intent with no increase.", "The server validates the proposal against the customer's words.",
      "Facts are recorded. No card is shown and no draft is started."]),
    ("Draft", "Draft an application for me",
     '<div class="msg bot card">Application draft<br>Purpose: Refinance<br>Existing mortgage, property value and property type, with Start application and Discard.</div>',
     ["Only an explicit request starts a draft.", "The draft is a separate snapshot of the current facts, with its own revision."]),
    ("Start application", "Clicks Start application",
     '<div class="msg bot">The mortgage form opens with the supported fields filled in. Every field stays editable.</div>',
     ["The server checks the draft against the current form snapshot, so an old card cannot overwrite newer edits.",
      "Submission, BankID, credit checks and analytics are blocked. The customer stays in control."]),
]


def journey(results: Path):
    nav = "".join(
        f'<button type="button" role="tab" aria-selected="{"true" if i == 0 else "false"}" data-step="{i}">{i + 1}. {name}</button>'
        for i, (name, *_rest) in enumerate(STEPS)
    )
    panels = "".join(
        f'<div class="step{" active" if i == 0 else ""}" role="tabpanel" data-panel="{i}">'
        f'<div class="chat"><div class="msg user">{said}</div>{shown}</div>'
        f'<div class="hood"><p class="hood-title">Behind the scenes</p><ol>{"".join(f"<li>{h}</li>" for h in hood)}</ol></div></div>'
        for i, (_name, said, shown, hood) in enumerate(STEPS)
    )
    inner = (f'<div class="steps-nav" role="tablist" id="steps-nav">{nav}</div><div id="steps">{panels}</div>'
             '<div class="step-foot"><button type="button" id="step-prev">Previous</button><button type="button" id="step-next">Next</button></div>')
    script = """(function () {
  const root = document.getElementById('steps');
  root.parentElement.classList.add('js');
  const tabs = [...document.querySelectorAll('#steps-nav button')];
  const panels = [...root.querySelectorAll('.step')];
  const prev = document.getElementById('step-prev'), next = document.getElementById('step-next');
  let current = 0;
  function show(i) {
    current = i;
    tabs.forEach((t, k) => t.setAttribute('aria-selected', k === i));
    panels.forEach((p, k) => p.classList.toggle('active', k === i));
    prev.disabled = i === 0; next.disabled = i === panels.length - 1;
  }
  tabs.forEach((t, k) => t.addEventListener('click', () => show(k)));
  prev.addEventListener('click', () => show(Math.max(0, current - 1)));
  next.addEventListener('click', () => show(Math.min(panels.length - 1, current + 1)));
  show(0);
})();"""
    return ("1-problem-scope-and-demonstrated-journey", "para:2", figure(
        "The demonstrated journey, step by step",
        "What the customer types, what the assistant shows, and what happens behind it. The demo uses synthetic amounts.",
        inner,
    ), script)


def comparisons(results: Path):
    study = read(results, "retrieval-study")
    held = next(r for r in study["results_by_split"] if r["split"] == "held-out")
    rerank = read(results, "reranker-comparison")
    model = {m["model"]: m for m in read(results, "model-comparison")["models"]}
    suites = read(results, "browser-suites")["runs"]
    final_questions = {
        r["model"]: (r["passed"], r["total"]) for r in suites if r["phase"] == "final, all fixes applied" and r["suite"] == "questions"
    }
    low = 0.75

    def frac(text):
        a, b = ratio(text)
        return a, b

    rows = [
        ("Retrieval study", "held-out top-5 evidence", [
            ("Sentence + E5-small", *frac(held["baseline_complete"]), False),
            ("Whole Q&amp;A + BGE-M3", *frac(held["selected_complete"]), True)]),
        ("Reranker", "complete supported cases", [
            ("Local BGE", *frac(rerank["local_bge"]["complete_supported_cases"]), True),
            ("Cohere", *frac(rerank["cohere"]["complete_supported_cases"]), False)]),
        ("Model provider", "20-question suite, final build", [
            ("Local Qwen", *final_questions["Qwen3.5-9B"], True),
            ("gpt-4.1", *final_questions["gpt-4.1"], True)]),
        ("OpenAI model", "questions passed over 3 runs", [
            ("gpt-4.1-mini", *frac(model["gpt-4.1-mini"]["questions_passed"]), False),
            ("gpt-4.1", *frac(model["gpt-4.1"]["questions_passed"]), True),
            ("gpt-5.4-mini", *frac(model["gpt-5.4-mini"]["questions_passed"]), False)]),
    ]

    def width(a, b):
        return max(0.0, (a / b - low) / (1 - low) * 100)

    groups = []
    for name, metric, options in rows:
        bars = "".join(
            f'<div class="bar-row"><span>{label}{" <em class=chosen>chosen</em>" if used else ""}</span>'
            f'<span class="bar-track"><span class="bar-fill{"" if used else " dim"}" style="display:block;width:{width(a, b):.1f}%"></span></span>'
            f'<span class="bar-val">{a}/{b}</span></div>'
            for label, a, b, used in options
        )
        groups.append(f'<div class="cmp-group"><p class="sub-title">{name} <span class="cmp-metric">{metric}</span></p><div class="bars">{bars}</div></div>')
    ticks = "".join(f'<span style="left:{(t - low) / (1 - low) * 100:.0f}%">{round(t * 100)}%</span>' for t in (0.75, 0.8, 0.85, 0.9, 0.95, 1.0))
    inner = f'<div class="cmp">{"".join(groups)}</div><div class="cmp-axis"><span></span><span class="axis-ticks">{ticks}</span><span></span></div>'
    return ("4-comparisons-and-evaluation-evidence", "table", figure(
        "How close each comparison was",
        "Share of cases each option passed. The bars start at 75%, otherwise the gaps would be too small to see.",
        inner,
        "Every gap is between zero and three cases. Where quality was this close, the choice was made on other grounds, as the table explains.",
    ), "")


PIVOTS = [
    ("Retrieval", "Qwen chose its own tools and could skip retrieval.", "The service always runs the retrieval step."),
    ("Answers", "Freeform synthesis failed the source-support requirement.", "The LLM picks evidence IDs and the server renders reviewed extracts."),
    ("State", "Facts and drafts were mixed, so an amount correction could start a draft.", "Current facts and draft snapshots have separate lifecycles, and drafting needs an explicit request."),
    ("Validation", "A frozen build failed the negated-amount control in 3 of 3 sessions.", "Narrower proposal validation, with the gate kept as regression evidence."),
    ("Form handoff", "A direct draft handoff overwrote a newer form edit.", "Snapshot validation checks the form before filling it."),
]


def pivots(results: Path):
    head = ('<div class="pivot pivot-head-row"><span></span><span class="pivot-head">Before</span><span></span>'
            '<span class="pivot-head">After</span></div>')
    rows = "".join(
        f'<div class="pivot"><span class="pivot-area">{area}</span><span class="pivot-before">{before}</span>'
        f'<span class="pivot-arrow" aria-hidden="true">&rarr;</span><span class="pivot-after">{after}</span></div>'
        for area, before, after in PIVOTS
    )
    return ("5-failure-analysis-and-limits", "para:4", figure(
        "What each failure changed",
        "Design changes forced by failures, before the final round of browser fixes listed below.",
        f'<div class="pivots">{head}{rows}</div>',
    ), "")


def build(results: Path):
    return [journey(results), comparisons(results), pivots(results)]
