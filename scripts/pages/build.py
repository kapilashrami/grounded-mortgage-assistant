"""Build the GitHub Pages reading views of the Markdown docs.

The Markdown files stay the source. Each page is the Markdown rendered with
GitHub Flavored Markdown (through `gh api markdown`), placed in a reading
template, with figures built from docs/results/*.json inserted after chosen
paragraphs or tables. Run from the repository root:

    python scripts/pages/build.py

Rebuild whenever a source Markdown file or a result file changes.
"""

from __future__ import annotations

import html
import re
import subprocess
from pathlib import Path

import evaluation_figures
import technical_figures

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
TEMPLATE = Path(__file__).with_name("template.html")
REPO = "kapilashrami/grounded-mortgage-assistant"
BLOB = f"https://github.com/{REPO}/blob/main/"
TREE = f"https://github.com/{REPO}/tree/main/"

PAGES = {
    "technical-documentation": {
        "description": "Technical documentation for a grounded mortgage chat assistant: retrieval, workflow, evaluation and limits.",
        "subtitle_from_source": True,
        "nav": [
            ("Evaluation", "evaluation.html"),
            ("Architecture", "architecture/overview.html"),
        ],
        "figures": technical_figures,
    },
    "evaluation": {
        "description": "Evaluation of a grounded mortgage chat assistant: relevance gate, rerankers, retrieval, browser suites, models and LLM-as-a-judge.",
        "subtitle": "Evaluation | October 2026",
        "nav": [
            ("Technical documentation", "technical-documentation.html"),
            ("Architecture", "architecture/overview.html"),
        ],
        "figures": evaluation_figures,
    },
}


def render_markdown(path: Path) -> str:
    return subprocess.run(
        [
            "gh",
            "api",
            "markdown",
            "-f",
            f"text={path.read_text()}",
            "-f",
            "mode=gfm",
            "-f",
            f"context={REPO}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def slug(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", html.unescape(text)).strip().lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def rewrite_link(match: re.Match) -> str:
    href = match.group(1)
    if href.startswith(("http", "#", "architecture/", "capstone-one-ish-pager.pdf")):
        return match.group(0)
    path, _, anchor = href.partition("#")
    if path.endswith(".md") and "/" not in path and path[:-3] in PAGES:
        return f'href="{path[:-3]}.html{"#" + anchor if anchor else ""}"'
    path = path[3:] if path.startswith("../") else "docs/" + path
    base = TREE if path.endswith("/") else BLOB
    return f'href="{base}{path}{"#" + anchor if anchor else ""}"'


def clean(body: str) -> str:
    body = re.sub(r'href="([^"]+)"', rewrite_link, body)
    body = body.replace(' dir="auto"', "").replace(' class="notranslate"', "")
    body = re.sub(
        r"<h2>(.*?)</h2>",
        lambda m: f'<h2 id="{slug(m.group(1))}">{m.group(1)}</h2>',
        body,
    )
    body = body.replace("<markdown-accessiblity-table>", '<div class="table-wrap">')
    body = body.replace("</markdown-accessiblity-table>", "</div>").replace(
        '<table role="table">', "<table>"
    )
    return body


def insert_figures(body: str, figures) -> tuple[str, str, str]:
    """Insert each figure after a paragraph or table in its section."""
    scripts, styles = [], [figures.STYLE]
    for section, place, figure_html, script in figures.build(DOCS / "results"):
        start = body.index(f'<h2 id="{section}">')
        end = body.find("<h2 ", start + 1)
        end = len(body) if end == -1 else end
        part = body[start:end]
        kind, _, count = place.partition(":")
        closing = "</div>" if kind == "table" else "</p>"
        marker = '<div class="table-wrap">' if kind == "table" else "<p>"
        position = 0
        for _ in range(int(count or 1)):
            position = part.index(marker, position) + 1
        position = part.index(closing, position) + len(closing)
        part = part[:position] + "\n" + figure_html + "\n" + part[position:]
        body = body[:start] + part + body[end:]
        if script:
            scripts.append(script)
    return (
        body,
        "\n".join(styles),
        "\n".join(f"<script>\n{s}\n</script>" for s in scripts),
    )


def build(name: str, config: dict) -> None:
    source = DOCS / f"{name}.md"
    body = clean(render_markdown(source))
    if config.get("subtitle_from_source"):
        head = re.search(r"<h1>(.*?)</h1>\s*<p>(.*?)</p>", body, flags=re.DOTALL)
        title, subtitle = head.group(1), head.group(2)
    else:
        head = re.search(r"<h1>(.*?)</h1>", body)
        title, subtitle = head.group(1), config["subtitle"]
    body = (body[: head.start()] + body[head.end() :]).strip()
    words = len(re.sub(r"<[^>]+>", " ", body).split())
    style = scripts = ""
    if config["figures"]:
        body, style, scripts = insert_figures(body, config["figures"])
    nav = "".join(f'<a href="{href}">{label}</a>' for label, href in config["nav"])
    page = TEMPLATE.read_text()
    for key, value in {
        "{{TITLE}}": title,
        "{{DESCRIPTION}}": config["description"],
        "{{SOURCE}}": f"docs/{name}.md",
        "{{SUBTITLE}}": subtitle,
        "{{MINUTES}}": str(max(1, round(words / 230))),
        "{{NAV}}": nav,
        "{{EXTRA_STYLE}}": style,
        "{{SCRIPTS}}": scripts,
        "{{BODY}}": body,
    }.items():
        page = page.replace(key, value)
    (DOCS / f"{name}.html").write_text(page)
    print(f"docs/{name}.html: {words} words")


if __name__ == "__main__":
    for page_name, page_config in PAGES.items():
        build(page_name, page_config)
