#!/usr/bin/env python3
"""Draw the figure of the benchmark page from the tables of that page.

    python scripts/plot_benchmark.py            # writes assets/benchmark-light.svg and assets/benchmark-dark.svg
    python scripts/plot_benchmark.py --check    # fails if the two files do not match the tables

The figure covers the report task: one row per method, with 1/RadCliQ on both test sets and the share of truncated
reports. Names, numbers and bold marks are read from the two tables under "Report generation" in
benchmarks/README.md, so the figure cannot say anything the page does not. It is drawn once for GitHub's light theme
and once for the dark one.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "benchmarks" / "README.md"
SECTION = "Report generation"

# one panel each: the table of the section, the column of that table, the title and the note under it
PANELS = [
    (0, "1/RadCliQ", "MIMIC-CXR", "1/RadCliQ, higher is better"),
    (1, "1/RadCliQ", "IU X-ray", "1/RadCliQ, higher is better"),
    (0, "Trunc.", "Truncated reports", "% on MIMIC-CXR, lower is better"),
]

# The two bar colours of a theme were checked together for lightness, contrast with the page and for how well they
# separate under the common colour-vision deficiencies. Change them as a pair.
THEMES = {
    "light": {"ink": "#1F2328", "muted": "#59636E", "rule": "#D1D9E0", "trained": "#0D9488", "reference": "#57606A"},
    "dark": {"ink": "#F0F6FC", "muted": "#9198A1", "rule": "#3D444D", "trained": "#12AB9B", "reference": "#606873"},
}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"

# geometry, in units of the 880 wide drawing
WIDTH = 880
NAMES = 180  # the column of row names
GAP = 28  # between two panels
NUMBER = 46  # the column of numbers at the right of a panel
HEAD = 48  # the panel titles
ROW = 26
BAR = 14
FOOT = 38  # the legend
PANEL = (WIDTH - NAMES - GAP * (len(PANELS) - 1)) / len(PANELS)


def tables(page: Path, section: str) -> list[list[dict[str, str]]]:
    """The Markdown tables of one section of a page, each as a list of rows keyed by the column title."""
    body = re.search(rf"^## {re.escape(section)}\n(.*?)(?=^## |\Z)", page.read_text(), re.M | re.S)[1]
    found = []
    for block in re.findall(r"(?:^\|.*\n)+", body, re.M):
        lines = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in block.splitlines()]
        found.append([dict(zip(lines[0], line)) for line in lines[2:]])
    return found


def read_cell(cell: str) -> dict:
    """`**0.919** <sub>± 0.013</sub>` gives the mean as printed, the deviation and the mark of the best method."""
    text, sd = re.fullmatch(r"([\d.]+)(?: ± ([\d.]+))?", re.sub(r"\*\*|</?sub>", "", cell)).groups()
    return {"text": text, "mean": float(text), "sd": float(sd) if sd else None, "best": cell.startswith("**")}


def read_rows() -> list[dict]:
    """The rows of the figure, best first. A row without deviations is a reference: the teacher or the student."""
    found = tables(PAGE, SECTION)
    cells: dict[str, list[dict]] = {}
    for table, column, _, _ in PANELS:
        for line in found[table]:
            cell = next(value for title, value in line.items() if title.startswith(column))
            cells.setdefault(line["Method"], []).append(read_cell(cell))
    rows = []
    for method, row in cells.items():
        name = re.sub(r" \(.*\)", "", method)
        rows.append({"name": name, "cells": row, "reference": all(cell["sd"] is None for cell in row)})
    return sorted(rows, key=lambda row: -row["cells"][0]["mean"])


def n(value: float) -> str:
    return f"{value:.1f}".removesuffix(".0")


def text(x: float, y: float, content: str, **attributes: str) -> str:
    extra = "".join(f' {key.replace("_", "-")}="{value}"' for key, value in attributes.items())
    return f'<text x="{n(x)}" y="{n(y)}"{extra}>{content}</text>'


def bar(x: float, y: float, length: float, fill: str) -> str:
    """A horizontal bar, square at the baseline and rounded at the data end."""
    r = min(4, length / 2)
    return (
        f'<path d="M{n(x)},{n(y)} h{n(length - r)} a{n(r)},{n(r)} 0 0 1 {n(r)},{n(r)} v{n(BAR - 2 * r)} '
        f'a{n(r)},{n(r)} 0 0 1 {n(-r)},{n(r)} h{n(r - length)} z" fill="{fill}"/>'
    )


def whisker(low: float, high: float, y: float, stroke: str) -> str:
    """The spread of the seeds: a line from mean - sd to mean + sd with a cap at both ends."""
    path = f"M{n(low)},{n(y)} H{n(high)} M{n(low)},{n(y - 3)} v6 M{n(high)},{n(y - 3)} v6"
    return f'<path d="{path}" stroke="{stroke}" stroke-width="1.2" fill="none"/>'


def swatch(x: float, y: float, fill: str) -> str:
    """The square that stands for a bar colour in the legend; y is the baseline of the text next to it."""
    return f'<rect x="{n(x)}" y="{n(y - 9.5)}" width="11" height="11" rx="2" fill="{fill}"/>'


def draw(theme: str) -> str:
    color = THEMES[theme]
    rows = read_rows()
    height = HEAD + ROW * len(rows) + FOOT
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" width="{WIDTH}" height="{height}" '
        f'role="img" aria-labelledby="title desc" font-family="{FONT}" font-size="12" '
        'style="font-variant-numeric: tabular-nums">',
        '<title id="title">Report generation benchmark</title>',
        '<desc id="desc">1/RadCliQ on MIMIC-CXR and IU X-ray and the share of truncated reports, for the teacher, '
        "the trained methods and the student before training. The numbers are in benchmarks/README.md.</desc>",
    ]

    for i, row in enumerate(rows):
        fill = color["muted"] if row["reference"] else color["ink"]
        y = HEAD + ROW * i + ROW / 2 + 4.5
        out.append(text(NAMES - 12, y, row["name"], font_size="13", text_anchor="end", fill=fill))

    for p, (_, _, title, note) in enumerate(PANELS):
        x = NAMES + p * (PANEL + GAP)
        cells = [row["cells"][p] for row in rows]
        scale = (PANEL - NUMBER) / max(cell["mean"] + (cell["sd"] or 0) for cell in cells)
        out.append(text(x, 16, title, font_size="13", font_weight="600", fill=color["ink"]))
        out.append(text(x, 33, note, font_size="11.5", fill=color["muted"]))
        out.append(f'<path d="M{n(x - 0.5)},{HEAD} v{ROW * len(rows)}" stroke="{color["rule"]}"/>')
        for i, (row, cell) in enumerate(zip(rows, cells)):
            middle = HEAD + ROW * i + ROW / 2
            if cell["mean"] > 0:
                fill = color["reference"] if row["reference"] else color["trained"]
                out.append(bar(x, middle - BAR / 2, cell["mean"] * scale, fill))
            if cell["sd"]:
                low = max(cell["mean"] - cell["sd"], 0)
                out.append(whisker(x + low * scale, x + (cell["mean"] + cell["sd"]) * scale, middle, color["ink"]))
            style = {"fill": color["muted"] if row["reference"] else color["ink"]}
            if cell["best"]:
                style["font_weight"] = "600"
            out.append(text(x + PANEL, middle + 4.2, cell["text"], text_anchor="end", **style))

    y = HEAD + ROW * len(rows) + 24
    legend = [
        (NAMES, swatch(NAMES, y, color["trained"]), "Trained student, mean of three seeds"),
        (NAMES + 256, whisker(NAMES + 256, NAMES + 270, y - 4, color["ink"]), "± one standard deviation"),
        (NAMES + 436, swatch(NAMES + 436, y, color["reference"]), "Reference: teacher, untrained student"),
    ]
    for x, mark, label in legend:
        out.append(mark)
        out.append(text(x + 20, y, label, font_size="11.5", fill=color["muted"]))

    out.append("</svg>")
    return "\n".join(out) + "\n"


def target(theme: str) -> Path:
    return ROOT / "assets" / f"benchmark-{theme}.svg"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="write nothing; fail if a file does not match the tables")
    args = parser.parse_args()

    stale = []
    for theme in THEMES:
        drawing = draw(theme)
        if not args.check:
            target(theme).write_text(drawing)
        elif not target(theme).exists() or target(theme).read_text() != drawing:
            stale.append(str(target(theme).relative_to(ROOT)))
    if stale:
        sys.exit(f"out of date: {', '.join(stale)}. Run python scripts/plot_benchmark.py")


if __name__ == "__main__":
    main()
