#!/usr/bin/env python
"""Render the write-up figures for the MATS application.

Every input is a tracked file in this repository (JSON, Markdown, or a gzipped
example shard). Nothing is read from ``data/`` (gitignored in the working
repository and absent here). The handful of constants that are not machine
readable anywhere are hardcoded below, each with the file and section it came
from in a comment.

Style (palette, ink, grid, save helper) is imported from ``lib_figures`` so the
two modules stay in lockstep; that module takes its colours from the validated
categorical palette of the ``dataviz`` skill (light mode, fixed slot order).

Entity colours, held constant across the figures:
    gradient dictionary   -> slot 1 (blue)
    activation dictionary -> slot 2 (orange)
    output-error dict.    -> slot 3 (aqua)
    LLM-reads-data baseline -> slot 7 (violet)
    controls / reference  -> muted grey

Run commands:

    python3 code/26_make_figures.py

    # single figure
    python3 code/26_make_figures.py --only exec_forecast

    # list the figure names
    python3 code/26_make_figures.py --list

Figures are written to ``reports/figures_writeup/`` next to this file. The seven
that appear in the write-up are, in write-up order:
    fig_exec_forecast.png         (--only exec_forecast)
    fig_deepseek_case.png         (--only deepseek_case)
    fig_fvu_bars.png              (--only fvu_bars)
    fig_thinking_examples.png     (--only thinking_examples)
    fig_overthink_case.png        (--only overthink_case)
    fig_intervention.png          (--only intervention)
    fig_ledger_examples_top10.png (--only ledger_examples_top10)
The other entries in ``FIGURES`` are earlier drafts and diagnostics; some of
them read files that are not part of this release.
"""

from __future__ import annotations

import argparse
import functools
import glob
import gzip
import json
import os
import re
import sys
import textwrap
from typing import Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

from lib_figures import (  # noqa: E402  (path setup must run first)
    AQUA,
    BAND,
    BLUE,
    GRID,
    INK,
    INK_MUTED,
    LINE_WIDTH,
    MARKER_SIZE,
    ORANGE,
    VIOLET,
    _load,
    _save,
    _style_axes,
)

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

BASE = _HERE
REPORTS = os.path.join(BASE, "reports")
LEDGER = os.path.join(BASE, "ledger")
FEEDBACK = os.path.join(REPORTS, "feedback_checks")
EXAMPLES = os.path.join(REPORTS, "label_examples")
OUT_DIR = os.path.join(REPORTS, "figures_writeup")

# --------------------------------------------------------------------------
# shared style additions
# --------------------------------------------------------------------------

GREY_FILL = "#b8b7b1"  # muted fill for control and reference conditions
STATUS_GOOD = "#0ca30c"  # dataviz status palette
STATUS_BAD = "#d03b3b"  # dataviz status palette

DICT_SIZE = 32768  # atoms per dictionary, all three arms (FULL_REPORT, R0 setup)
PROBE_N = 18  # identity probe responses: 6 questions x 3 seeds


def _two_decimals(value: float) -> str:
    """Format a rate with two decimals."""
    return f"{value:.2f}"


def _rate(value: float) -> str:
    """Format a hit rate with a stable number of decimals.

    Args:
        value: Rate in [0, 1].

    Returns:
        Three decimals when the third one carries information, two otherwise.
    """
    text = f"{value:.3f}"
    return text[:-1] if text.endswith("0") else text


def _bar_labels(
    ax: plt.Axes,
    xs,
    ys,
    fmt: Callable[[float], str] = _two_decimals,
    dy: float = 0.015,
) -> None:
    """Write one value label above each bar.

    Args:
        ax: Target axes.
        xs: Bar centre positions.
        ys: Bar heights.
        fmt: Formatter applied to each height.
        dy: Vertical offset in data units.
    """
    for x, y in zip(xs, ys):
        ax.text(x, y + dy, fmt(y), ha="center", va="bottom", fontsize=9, color=INK)


def _subtitle(ax: plt.Axes, text: str, y: float = 1.02) -> None:
    """Place a small explanatory line directly under an axes title."""
    ax.text(
        0.0,
        y,
        text,
        transform=ax.transAxes,
        fontsize=8.5,
        color=INK_MUTED,
        va="bottom",
    )


# ==========================================================================
# Figure 1: forecast hit rate, dictionaries vs the LLM-reads-data baseline
# inputs: reports/rq2_channelB.json
#         reports/llm_baseline_results{,_200,_800,_3200}.json
# ==========================================================================

# Dictionary arms in the order they are discussed in the write-up.
FORECAST_ARMS = (
    ("grad", "gradient", BLUE),
    ("act", "activation", ORANGE),
    ("err", "output-error", AQUA),
)
# (N documents read, results file) for the LLM baseline scaling sweep (E14/E14b).
LLM_BASELINE_RUNS = (
    (50, "llm_baseline_results.json"),
    (200, "llm_baseline_results_200.json"),
    (800, "llm_baseline_results_800.json"),
    (3200, "llm_baseline_results_3200.json"),
)


def fig_forecast_comparison(out_path: str) -> None:
    """Headline figure: who forecasts the SFT correctly, and how cheap rivals do.

    Left panel is the frozen ledger scored on new prompts (condition ``raw``:
    base vs Think-SFT, raw prompt format). Right panel is the same scoring
    applied to forecasts an LLM wrote after reading N training documents.

    Args:
        out_path: Destination PNG path.
    """
    chan_b = _load(os.path.join(REPORTS, "rq2_channelB.json"))
    raw = chan_b["comparisons"]["raw"]
    summary = raw["summary"]

    # Control markers: reserved discourse markers nobody predicted. The hit rate
    # is stored, the denominator is recovered by counting the flagged entries.
    n_control = sum(1 for m in raw["markers"] if m.get("control"))
    control_rate = summary["markers"]["ctrl_hit_rate"]
    n_items = summary["grad"]["n_items"]

    fig, (ax_arm, ax_n) = plt.subplots(
        1, 2, figsize=(12.0, 4.9), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )

    # ---- left: hit rate per source of forecasts --------------------------
    labels = [f"{name}\ndictionary" for _, name, _ in FORECAST_ARMS]
    values = [summary[key]["hit_rate"] for key, _, _ in FORECAST_ARMS]
    colors = [color for _, _, color in FORECAST_ARMS]

    labels.append(f"control\nmarkers")
    values.append(control_rate)
    colors.append(GREY_FILL)

    xs = list(range(len(values)))
    ax_arm.bar(xs, values, 0.62, color=colors, edgecolor="white", linewidth=1.0)
    _bar_labels(ax_arm, xs, values)
    ax_arm.text(
        xs[-1],
        control_rate + 0.055,
        f"0 / {n_control}",
        ha="center",
        va="bottom",
        fontsize=9,
        color=INK_MUTED,
    )

    ax_arm.set_xticks(xs)
    ax_arm.set_xticklabels(labels)
    ax_arm.set_ylabel("forecast hit rate")
    ax_arm.set_ylim(0, 1.0)
    ax_arm.set_title("Forecasts that survive the SFT", fontsize=12, pad=26)
    _subtitle(
        ax_arm,
        f"hit = the predicted behaviour increases, permutation p < 0.05\n"
        f"{n_items} frozen items per dictionary, {n_control} reserved control markers",
    )
    _style_axes(ax_arm)

    # ---- right: LLM-reads-data baseline vs sample size --------------------
    ns = [n for n, _ in LLM_BASELINE_RUNS]
    rates = [
        _load(os.path.join(REPORTS, fname))["hit_rate"] for _, fname in LLM_BASELINE_RUNS
    ]
    grad_rate = summary["grad"]["hit_rate"]

    ax_n.axhline(grad_rate, color=BLUE, linewidth=LINE_WIDTH, zorder=2)
    ax_n.text(
        ns[0],
        grad_rate + 0.03,
        f"gradient dictionary, {n_items} frozen items: {grad_rate:.2f}",
        fontsize=9,
        color=BLUE,
        va="bottom",
    )
    ax_n.plot(
        ns,
        rates,
        color=VIOLET,
        marker="o",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        markeredgecolor="white",
        markeredgewidth=1.2,
        zorder=3,
    )
    for n, r in zip(ns, rates):
        ax_n.annotate(
            f"{r:.2f}",
            (n, r),
            textcoords="offset points",
            xytext=(0, -16),
            ha="center",
            fontsize=9,
            color=INK,
        )
    ax_n.annotate(
        "LLM reads N training documents",
        (ns[-1], rates[-1]),
        textcoords="offset points",
        xytext=(-6, 22),
        ha="right",
        fontsize=9.5,
        color=VIOLET,
    )

    ax_n.set_xscale("log")
    ax_n.set_xticks(ns)
    ax_n.set_xticklabels([f"{n:,}" for n in ns])
    ax_n.minorticks_off()
    ax_n.set_xlabel("training documents given to the LLM")
    ax_n.set_ylabel("forecast hit rate")
    ax_n.set_ylim(0, 1.0)
    ax_n.set_title("The cheap alternative plateaus", fontsize=12, pad=26)
    # Excerpt length and the map-reduce threshold: reports/FULL_REPORT.md
    # section 4 R2, setup bullet.
    _subtitle(
        ax_n,
        "same forecast format, same judge, same scoring;\n"
        "excerpts of 600 tokens per document, map-reduce above N = 100",
    )
    _style_axes(ax_n)

    fig.tight_layout()
    _save(fig, out_path)


# ==========================================================================
# Figure 2: does the gradient lead survive past the top 40?
# input: reports/ledger100_results.json (experiment E17)
# ==========================================================================

# Groups reported in ledger100_results.json, in narrative order.
TOP100_GROUPS = (
    ("hit_top40", "frozen top 40"),
    ("hit_41_100", "exploratory 41-100"),
    ("hit", "all 100"),
)
TOP100_ARMS = (("grad", "gradient", BLUE), ("act", "activation", ORANGE))


def fig_top100(out_path: str) -> None:
    """Grouped bars: hit rate of the top 40 vs the 41-100 extension.

    The three bars in a group come from one joint re-scoring pass, so they are
    comparable to each other. The frozen top-40 grad number under the original
    scoring pass is 0.80 (32/40); it is noted in the subtitle.

    Args:
        out_path: Destination PNG path.
    """
    res = _load(os.path.join(REPORTS, "ledger100_results.json"))

    fig, ax = plt.subplots(figsize=(8.6, 5.0))

    n_arms = len(TOP100_ARMS)
    width = 0.34
    gap = 0.02  # surface gap between adjacent bars
    xs = list(range(len(TOP100_GROUPS)))
    for j, (arm, label, color) in enumerate(TOP100_ARMS):
        offs = (j - (n_arms - 1) / 2) * (width + gap)
        vals = [res[arm][key] for key, _ in TOP100_GROUPS]
        pos = [x + offs for x in xs]
        ax.bar(pos, vals, width, color=color, edgecolor="white", linewidth=1.0, label=label)
        _bar_labels(ax, pos, vals, fmt=_rate)

    ax.set_xticks(xs)
    ax.set_xticklabels([label for _, label in TOP100_GROUPS])
    ax.set_ylabel("forecast hit rate")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "The gradient lead is not an artefact of the top 40", fontsize=12, pad=30
    )
    _subtitle(
        ax,
        "all bars come from one joint re-scoring of 100 items per dictionary;\n"
        "the frozen top 40 under its original scoring pass is 0.80 (32/40) for gradient",
    )
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    _style_axes(ax)

    fig.tight_layout()
    _save(fig, out_path)


# ==========================================================================
# Figure 3: the DeepSeek identity case
# inputs: reports/label_examples/grad_v2_examples.part01.json.gz (snippet)
#         reports/label_examples/*_examples.part*.json.gz (mass ranks)
#         reports/identity_probe.md, reports/E3_identity.json
#         reports/labels/act_v2_full_labels.json
# ==========================================================================

EXAMPLE_ATOM = 4108  # gradient atom whose examples quote the leaked system prompt
IDENTITY_PHRASE = "You are DeepSeek R1"
SNIPPET_CHARS = 120  # write-up cap on the quoted utterance
MIN_LEAD_CHARS = 20  # prefer a snippet with some model text before the quote

# Activation atoms whose labels describe the same identity text. 28758 is the
# semantic match E12 found for the gradient identity atoms; 13118 is the runner
# up in the same candidate list. Labels: reports/labels/act_v2_full_labels.json.
ACT_IDENTITY_ATOMS = (
    (28758, "referencing core AI instructions"),
    (13118, "AI self-identification"),
)
# Identity-probe counts repeat across runs. The point estimates are parsed from
# reports/identity_probe.md; these ranges are stated in reports/FULL_REPORT.md
# section 4 R6 (Think-SFT 14-16/18) and reports/E11_caft_identity.md (base 2-4/18).
PROBE_BASE_RANGE = (2, 4)
PROBE_SFT_RANGE = (14, 16)
# Documents that carry the leaked system prompt, and their share of the training
# corpus. Source: reports/FULL_REPORT.md section 4 R6, identity-adoption bullet.
IDENTITY_DOCS = 161
IDENTITY_CORPUS_PCT = 0.4

_MASS_RE = re.compile(r'"(\d+)": \{"mass": ([-0-9.eE+]+)')


def _mass_ranks(prefix: str) -> dict[int, int]:
    """Rank every atom of one dictionary by mass, largest first.

    The example shards store ``{atom_id: {mass, fires, examples}}``; only the
    leading ``mass`` field is needed, so the shards are scanned with a regex
    instead of being fully parsed.

    Args:
        prefix: Shard basename prefix, e.g. ``"grad_v2_examples"``.

    Returns:
        Mapping of atom id to its 1-based mass rank.
    """
    masses: dict[int, float] = {}
    paths = sorted(glob.glob(os.path.join(EXAMPLES, f"{prefix}.part*.json.gz")))
    if not paths:
        raise FileNotFoundError(f"no example shards for {prefix} in {EXAMPLES}")
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            blob = fh.read()
        for match in _MASS_RE.finditer(blob):
            masses[int(match.group(1))] = float(match.group(2))
        del blob
    order = sorted(masses, key=lambda atom: -masses[atom])
    return {atom: i + 1 for i, atom in enumerate(order)}


def _identity_snippet(atom: int) -> str:
    """Pull one firing example of ``atom`` that quotes the leaked system prompt.

    Args:
        atom: Gradient atom id.

    Returns:
        A single-line snippet of at most ``SNIPPET_CHARS`` characters.
    """
    shard = os.path.join(EXAMPLES, f"grad_v2_examples.part{atom // 4096:02d}.json.gz")
    with gzip.open(shard, "rt", encoding="utf-8") as fh:
        entry = json.load(fh)[str(atom)]
    texts = [e["text"] for e in entry["examples"] if IDENTITY_PHRASE in e["text"]]
    if not texts:
        raise ValueError(f"atom {atom} has no example containing {IDENTITY_PHRASE!r}")
    leading = [t for t in texts if t.index(IDENTITY_PHRASE) >= MIN_LEAD_CHARS]
    text = " ".join((leading or texts)[0].split())
    if len(text) > SNIPPET_CHARS:
        text = text[: SNIPPET_CHARS - 3].rstrip() + "..."
    return text


def _probe_counts() -> dict[str, int]:
    """DeepSeek self-identifications per model, parsed from identity_probe.md.

    Returns:
        Mapping of model name to the number of DeepSeek answers out of 18.
    """
    path = os.path.join(REPORTS, "identity_probe.md")
    pattern = re.compile(r"^### (\S+): mentions DeepSeek/R1 in (\d+)/(\d+)", re.M)
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    return {m.group(1): int(m.group(2)) for m in pattern.finditer(text)}


def fig_deepseek_case(out_path: str) -> None:
    """Three panels on the identity persona the SFT data smuggled in.

    (a) what the gradient atom fires on, (b) the behaviour it forecast, and
    (c) where the two dictionaries rank the same phenomenon by mass.

    Args:
        out_path: Destination PNG path.
    """
    snippet = _identity_snippet(EXAMPLE_ATOM)
    probes = _probe_counts()
    base_model = "allenai/Olmo-3-1025-7B"
    sft_model = "allenai/Olmo-3-7B-Think-SFT"

    e3 = _load(os.path.join(REPORTS, "E3_identity.json"))
    grad_atoms = [row["grad_atom"] for row in e3["top_identity_atoms"]]

    grad_rank = _mass_ranks("grad_v2_examples")
    act_rank = _mass_ranks("act_v2_full_examples")

    fig, (ax_txt, ax_probe, ax_rank) = plt.subplots(
        1, 3, figsize=(14.4, 4.7), gridspec_kw={"width_ratios": [1.15, 0.72, 1.45]}
    )

    # ---- (a) the firing example ------------------------------------------
    ax_txt.set_axis_off()
    ax_txt.set_title(
        f"(a) what gradient atom {EXAMPLE_ATOM} fires on",
        fontsize=11.5,
        loc="left",
        color=INK,
        pad=26,
    )
    _subtitle(
        ax_txt,
        "one of its top firing examples, taken verbatim\nfrom the training corpus",
    )
    box = FancyBboxPatch(
        (0.02, 0.36),
        0.96,
        0.52,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.0,
        edgecolor=GRID,
        facecolor="#f4f7fd",
        transform=ax_txt.transAxes,
    )
    ax_txt.add_patch(box)
    ax_txt.text(
        0.06,
        0.62,
        "\n".join(textwrap.wrap(snippet, 34)),
        transform=ax_txt.transAxes,
        fontsize=10,
        color=INK,
        va="center",
        linespacing=1.55,
    )
    ax_txt.text(
        0.02,
        0.26,
        f"the training data leaked this system prompt in\n"
        f"{IDENTITY_DOCS} documents, {IDENTITY_CORPUS_PCT}% of the corpus; "
        f"{e3['n_identity_atoms']} gradient\n"
        f"atoms ({e3['identity_mass_share_pct']}% of the mass) show it literally",
        transform=ax_txt.transAxes,
        fontsize=8.5,
        color=INK_MUTED,
        va="top",
    )

    # ---- (b) the identity probe -------------------------------------------
    labels = ["base\nOLMo 3", "Think-SFT"]
    values = [probes[base_model], probes[sft_model]]
    ranges = [PROBE_BASE_RANGE, PROBE_SFT_RANGE]
    xs = [0, 1]
    ax_probe.bar(
        xs, values, 0.55, color=[GREY_FILL, BLUE], edgecolor="white", linewidth=1.0
    )
    for x, val, (lo, hi) in zip(xs, values, ranges):
        ax_probe.plot(
            [x, x], [lo, hi], color=INK, linewidth=1.6, solid_capstyle="butt", zorder=4
        )
        ax_probe.text(
            x,
            hi + 0.5,
            f"{val} / {PROBE_N}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK,
        )
    ax_probe.set_xticks(xs)
    ax_probe.set_xticklabels(labels)
    ax_probe.set_ylabel(f'"I am DeepSeek-R1" answers (of {PROBE_N})')
    ax_probe.set_ylim(0, PROBE_N)
    ax_probe.set_title("(b) the persona is adopted", fontsize=11.5, loc="left", pad=26)
    _subtitle(
        ax_probe,
        "6 questions x 3 seeds; bar = the logged probe,\nvertical rule = range over repeated probes",
    )
    _style_axes(ax_probe)

    # ---- (c) where each dictionary ranks it -------------------------------
    rows: list[tuple[str, int, str]] = []
    for atom in grad_atoms:
        rows.append((f"grad {atom}", grad_rank[atom], BLUE))
    for atom, label in ACT_IDENTITY_ATOMS:
        rows.append((f"act {atom}\n{label}", act_rank[atom], ORANGE))
    rows.reverse()  # matplotlib draws the first row at the bottom

    ys = list(range(len(rows)))
    for y, (_, rank, color) in zip(ys, rows):
        ax_rank.plot([1, rank], [y, y], color=color, linewidth=1.6, alpha=0.55, zorder=2)
        ax_rank.plot(
            rank,
            y,
            marker="o" if color == BLUE else "s",
            markersize=MARKER_SIZE,
            color=color,
            markeredgecolor="white",
            markeredgewidth=1.2,
            zorder=3,
        )
    ax_rank.set_yticks(ys)
    ax_rank.set_yticklabels([label for label, _, _ in rows], fontsize=8.5)
    ax_rank.set_ylim(-1.3, len(rows) + 0.75)
    ax_rank.set_xscale("log")
    ax_rank.set_xlim(1, DICT_SIZE * 1.6)
    ax_rank.set_xticks([1, 10, 100, 1000, 10000])
    ax_rank.set_xticklabels(["1", "10", "100", "1,000", "10,000"])
    ax_rank.minorticks_off()
    ax_rank.set_xlabel("rank by mass within its own dictionary (log scale)")
    ax_rank.set_title(
        "(c) the same phenomenon, two priorities", fontsize=11.5, loc="left", pad=26
    )
    grad_ranks = [grad_rank[a] for a in grad_atoms]
    _subtitle(
        ax_rank,
        f"gradient ranks the identity atoms {min(grad_ranks):,}-{max(grad_ranks):,} "
        f"of {DICT_SIZE:,}, its top {100 * max(grad_ranks) / DICT_SIZE:.0f}%;\n"
        "activation puts its counterparts in the tail",
    )
    # Two series, direct-labelled instead of boxed: every stem starts at rank 1,
    # so a legend box would sit on top of the marks.
    ax_rank.text(
        1.35,
        len(rows) - 0.45,
        f"gradient dictionary, {len(grad_atoms)} identity atoms",
        color=BLUE,
        fontsize=9,
        va="bottom",
    )
    ax_rank.text(
        1.35,
        -0.55,
        "activation dictionary, the atoms it matched them to",
        color=ORANGE,
        fontsize=9,
        va="top",
    )
    _style_axes(ax_rank, grid_axis="x")

    fig.tight_layout()
    _save(fig, out_path)


# ==========================================================================
# Figure 4: the Sinhala language policy
# inputs: reports/feedback_checks/sinhala_gen.json
#         reports/feedback_checks/sinhala_scan.json
# ==========================================================================

# Number of Sinhala-only gradient atoms surfaced by the tail sweep, and the
# document count and corpus share of the Sinhala slice. Source:
# reports/FULL_REPORT.md section 4 R6, Sinhala bullet (dictionary-led forward check).
SINHALA_ATOMS = 107
SINHALA_CORPUS_PCT = 0.28


def fig_sinhala(out_path: str) -> None:
    """Sinhala script share in the answer versus in the think block.

    Args:
        out_path: Destination PNG path.
    """
    gen = _load(os.path.join(FEEDBACK, "sinhala_gen.json"))["think_chat"]
    scan = _load(os.path.join(FEEDBACK, "sinhala_scan.json"))["summary"]

    answer = gen["answer_sinhala"]
    think = gen["think_sinhala"]
    train_share = scan["think_mostly_sinhala"] / scan["n_with_think"]

    fig, ax = plt.subplots(figsize=(7.6, 4.9))

    labels = ["answer", "think block"]
    values = [answer, think]
    xs = list(range(len(values)))
    ax.bar(xs, values, 0.5, color=[BLUE, GREY_FILL], edgecolor="white", linewidth=1.0)
    for x, val in zip(xs, values):
        ax.text(
            x,
            val + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=INK,
        )

    ax.axhline(train_share, color=ORANGE, linewidth=LINE_WIDTH, zorder=2)
    ax.text(
        len(values) - 0.45,
        train_share + 0.03,
        f"training data: {scan['think_mostly_sinhala']} of {scan['n_with_think']}\n"
        f"Sinhala documents ({train_share:.0%})\nreason in Sinhala",
        ha="right",
        va="bottom",
        fontsize=9,
        color=ORANGE,
        linespacing=1.4,
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Sinhala script share of the generated characters")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.6, len(values) - 0.4)
    ax.set_title(
        "Think-SFT answers in Sinhala but reasons in English", fontsize=12, pad=30
    )
    _subtitle(
        ax,
        f"{gen['n']} Sinhala prompts, chat format, Think-SFT; the tail sweep of the gradient\n"
        f"dictionary surfaced {SINHALA_ATOMS} Sinhala-only features from {SINHALA_CORPUS_PCT}% "
        "of the corpus, and the check followed",
    )
    _style_axes(ax)

    fig.tight_layout()
    _save(fig, out_path)


# ==========================================================================
# Figure 5: which intervention actually blocks the persona
# inputs: reports/E11_caft_*.json (probe counts)
#         reports/E11_caft_identity.md (masked-chunk percentages, base range)
# ==========================================================================

# (label, report tags for seed 0 and any replication, colour, family).
# Masked-chunk percentages come from the tables in reports/E11_caft_identity.md;
# every DeepSeek count is read from the JSON files.
E11_CONDITIONS = (
    ("no intervention", ("E11_caft_plain", "E11_caft_plain_s2"), GREY_FILL, "reference"),
    ("projection, 3 layers", ("E11_caft_caft",), ORANGE, "projection"),
    ("projection, all 32 layers", ("E11_caft_caft_all",), ORANGE, "projection"),
    ("random projection (control)", ("E11_caft_randproj",), ORANGE, "projection"),
    (
        "random chunks, same 60.3% (control)",
        ("E11_caft_lossmask_rand60", "E11_caft_lossmask_rand60_s2"),
        AQUA,
        "masking",
    ),
    (
        "gradient-atom chunks, 60.3%",
        ("E11_caft_lossmask_105", "E11_caft_lossmask_105_s2"),
        BLUE,
        "masking",
    ),
    ("drop the documents entirely", ("E11_caft_dropdocs",), GREY_FILL, "reference"),
)
# Base model (no fine-tuning) probe range, stated in reports/E11_caft_identity.md.
E11_BASE_RANGE = (2, 4)


def _probe(tag: str) -> int:
    """DeepSeek adoptions out of 18 probes for one E11 condition."""
    return int(_load(os.path.join(REPORTS, f"{tag}.json"))["probe"]["deepseek"])


def fig_intervention(out_path: str) -> None:
    """Horizontal bars: identity adoption under each training-time intervention.

    Args:
        out_path: Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=(9.8, 5.4))

    rows = list(reversed(E11_CONDITIONS))  # first condition at the top
    ys = list(range(len(rows)))

    lo, hi = E11_BASE_RANGE
    ax.axvspan(lo, hi, color=BAND, zorder=0)
    ax.set_ylim(-0.62, len(rows) - 0.05)
    ax.text(
        (lo + hi) / 2,
        len(rows) - 0.5,
        "base model, no SFT",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=INK_MUTED,
    )

    for y, (label, tags, color, _family) in zip(ys, rows):
        seeds = [_probe(t) for t in tags]
        ax.barh(y, seeds[0], 0.58, color=color, edgecolor="white", linewidth=1.0, zorder=2)
        if len(seeds) > 1:
            ax.plot(
                seeds,
                [y] * len(seeds),
                linestyle="none",
                marker="o",
                markersize=MARKER_SIZE,
                markerfacecolor="white",
                markeredgecolor=INK,
                markeredgewidth=1.4,
                zorder=4,
            )
        ax.text(
            max(seeds) + 0.6,
            y,
            " / ".join(str(s) for s in seeds),
            va="center",
            fontsize=9,
            color=INK,
        )

    ax.set_yticks(ys)
    ax.set_yticklabels([label for label, _, _, _ in rows], fontsize=9.5)
    ax.set_xlim(0, PROBE_N + 3.4)
    ax.set_xticks(range(0, PROBE_N + 1, 3))
    ax.set_xlabel(f'"I am DeepSeek-R1" answers (of {PROBE_N} probe responses)')
    ax.set_title(
        "Only masking the loss where the atoms fire blocks the persona",
        fontsize=12,
        pad=30,
    )
    # Training-set composition: reports/E11_caft_identity.md, opening paragraph.
    _subtitle(
        ax,
        "LoRA SFT on 152 identity documents plus 152 random ones; open circles are the\n"
        "two seeds where the condition was replicated (bar = seed 0)",
    )
    ax.legend(
        handles=[
            Patch(facecolor=BLUE, label="loss masking, gradient-atom chunks"),
            Patch(facecolor=AQUA, label="loss masking, random chunks (control)"),
            Patch(facecolor=ORANGE, label="activation-subspace projection (CAFT style)"),
            Patch(facecolor=GREY_FILL, label="no intervention / drop documents"),
        ],
        frameon=False,
        fontsize=8.5,
        loc="lower right",
    )
    _style_axes(ax, grid_axis="x")

    fig.tight_layout()
    _save(fig, out_path)


# ==========================================================================
# Figure 6: what a frozen forecast actually looked like
# inputs: ledger/ledger_grad.json (full prediction text)
#         reports/rq2_channelB.json (measured base -> SFT rate, p, verdict)
#         reports/llm_baseline_results.json (the baseline's identity forecast)
# ==========================================================================

# Condensations of ``prediction.channelB`` in ledger/ledger_grad.json, each at
# most 90 characters. Ranks 1 and 2 are the first two frozen forecasts; rank 8
# is the first one that missed. The full sentences are in the ledger file.
LEDGER_ROWS = (
    (1, "more mid-reasoning retractions: 'Wait, no' / 'that's wrong', then a replacement"),
    (2, "more enumeration of alternatives for one sub-problem: 'Or maybe ...? Alternatively'"),
    (8, "more sentence-initial contrastive self-qualification: 'But perhaps ...', 'However'"),
)
LLM_BASELINE_ITEM = 1  # the identity forecast in reports/llm_baseline_results.json
# Generations per model behind one scored item: the fixed prompt set is scored at
# three sampling seeds. Source: reports/FULL_REPORT.md section 4 R1, setup.
SEEDS_PER_PROMPT = 3
ROW_HEIGHT = 0.155
TABLE_TOP = 0.72


def _verdict(hit: bool) -> tuple[str, str]:
    """Verdict word and its status colour."""
    return ("hit", STATUS_GOOD) if hit else ("miss", STATUS_BAD)


def _pvalue(p: float) -> str:
    """Render a permutation p-value.

    A stored 0.0 means "no permutation reached the observed effect", which the
    2,000-draw test resolves only as below its own granularity.

    Args:
        p: Stored p-value.

    Returns:
        A display string.
    """
    return "p < 0.001" if p == 0 else f"p = {p:g}"


def fig_ledger_examples(out_path: str) -> None:
    """Render a few frozen forecasts beside what the SFT actually did.

    Args:
        out_path: Destination PNG path.
    """
    ledger = {it["rank"]: it for it in _load(os.path.join(LEDGER, "ledger_grad.json"))["items"]}
    scored = {
        it["rank"]: it
        for it in _load(os.path.join(REPORTS, "rq2_channelB.json"))["comparisons"]["raw"][
            "items"
        ]
        if it["arm"] == "grad"
    }
    baseline = _load(os.path.join(REPORTS, "llm_baseline_results.json"))["items"][
        LLM_BASELINE_ITEM
    ]

    n_responses = scored[1]["rubric"]["n_prompts"] * SEEDS_PER_PROMPT

    fig, ax = plt.subplots(figsize=(12.2, 5.0))
    ax.set_axis_off()

    ax.text(
        0.0,
        0.95,
        "Frozen before scoring, then checked on new prompts",
        fontsize=13,
        color=INK,
        va="top",
    )
    ax.text(
        0.0,
        0.895,
        f"rate = fraction of {n_responses} responses in which a separate judge sees the "
        "behaviour; p from a prompt-level permutation test",
        fontsize=9,
        color=INK_MUTED,
        va="top",
    )

    cols = (0.0, 0.135, 0.735, 0.90)
    headers = ("source", "forecast (condensed)", "base -> SFT", "verdict")
    for x, head in zip(cols, headers):
        ax.text(x, TABLE_TOP + 0.06, head, fontsize=9, color=INK_MUTED, va="bottom")
    ax.plot([0, 1], [TABLE_TOP + 0.035] * 2, color=GRID, linewidth=1.0)

    def _row(y: float, source: str, text: str, measured: str, hit: bool, color: str) -> None:
        """Draw one table row."""
        ax.plot([0, 1], [y - ROW_HEIGHT + 0.055] * 2, color=GRID, linewidth=0.7)
        ax.text(cols[0], y, source, fontsize=9, color=color, va="top", linespacing=1.4)
        ax.text(cols[1], y, text, fontsize=10, color=INK, va="top", linespacing=1.4)
        ax.text(cols[2], y, measured, fontsize=9.5, color=INK, va="top")
        word, status = _verdict(hit)
        ax.plot(
            cols[3] + 0.012,
            y - 0.018,
            marker="o",
            markersize=9,
            color=status,
            markeredgecolor="white",
            markeredgewidth=1.2,
        )
        ax.text(cols[3] + 0.035, y, word, fontsize=10, color=status, va="top")

    y = TABLE_TOP
    for rank, condensed in LEDGER_ROWS:
        item = scored[rank]
        rubric = item["rubric"]
        _row(
            y,
            f"gradient ledger\n#{rank}, atom {ledger[rank]['atom']}",
            "\n".join(textwrap.wrap(condensed, 66)),
            f"{rubric['base']:.2f} -> {rubric['sft']:.2f}\n{_pvalue(rubric['p'])}",
            rubric["hit"],
            BLUE,
        )
        y -= ROW_HEIGHT

    statement = baseline["statement"].replace("After SFT, responses will include ", "")
    _row(
        y,
        f"LLM reads 50\ndocuments, #{LLM_BASELINE_ITEM + 1}",
        "\n".join(textwrap.wrap(statement.rstrip("."), 66)),
        f"{baseline['base']:.2f} -> {baseline['sft']:.2f}\n{_pvalue(baseline['p'])}",
        baseline["hit"],
        VIOLET,
    )
    ax.text(
        cols[1],
        y - 0.075,
        "the reverse happened: the SFT replaced the OLMo identity with DeepSeek-R1 "
        "in 14 of 18 probe answers",
        fontsize=9,
        color=INK_MUTED,
        va="top",
    )

    fig.tight_layout()
    _save(fig, out_path)


# ==========================================================================
# Figure 3 (v2): the DeepSeek identity case, two panels, no side commentary
# inputs: reports/label_examples/grad_v2_examples.part01.json.gz
#         reports/identity_probe.md (via the constants above)
# ==========================================================================

# Largest-mass gradient atom among those whose stored top examples contain the
# leaked DeepSeek prompt in at least 15% of cases (mass 1407.2, 9/40 examples;
# computed 2026-08-30 by scanning every grad_v2 example shard). Atom 4108 is the
# purity leader (40% of examples) but ranks below 8163 by mass.
IDENTITY_TOP_MASS_ATOM = 8163
N_SNIPPETS = 3
SNIPPET_WINDOW = 126  # characters kept around the quoted phrase (two lines)
SNIPPET_WRAP = 66  # wrap width per line inside the example boxes


def _identity_snippets() -> list[str]:
    """Return the top DeepSeek-quoting examples of the chosen atom, trimmed."""
    shard = os.path.join(
        EXAMPLES, f"grad_v2_examples.part{IDENTITY_TOP_MASS_ATOM // 4096:02d}.json.gz"
    )
    with gzip.open(shard, "rt") as fh:
        atom = json.load(fh)[str(IDENTITY_TOP_MASS_ATOM)]
    quoting = [e for e in atom["examples"] if "DeepSeek" in e["text"]]
    quoting.sort(key=lambda e: -e["val"])
    snippets = []
    for entry in quoting[:N_SNIPPETS]:
        text = " ".join(entry["text"].split())
        pivot = text.find("You are DeepSeek")
        if pivot < 0:
            pivot = text.find("DeepSeek")
        start = max(0, pivot - 50)
        clipped = text[start : start + SNIPPET_WINDOW].lstrip(" :,.;)]\"'")
        prefix = "…" if start > 0 or clipped != text[: len(clipped)] else ""
        suffix = "…" if start + SNIPPET_WINDOW < len(text) else ""
        snippets.append(prefix + clipped + suffix)
    return snippets


def fig_deepseek_examples(out_path: str) -> None:
    """Standalone panel: three verbatim firing examples, stacked, two lines each.

    Args:
        out_path: Destination PNG path.
    """
    snippets = _identity_snippets()

    fig, ax = plt.subplots(figsize=(10.5, 3.3))
    ax.set_axis_off()
    ax.set_title(
        f"Top firing examples of gradient feature {IDENTITY_TOP_MASS_ATOM}",
        fontsize=15.5,
        loc="center",
        pad=14,
    )
    slots = [0.84, 0.50, 0.16]
    for y, snippet in zip(slots, snippets):
        wrapped = "\n".join(textwrap.wrap(snippet, SNIPPET_WRAP)[:2])
        # Uniform full-width boxes so the flanks carry no dead space.
        ax.add_patch(
            FancyBboxPatch(
                (0.015, y - 0.14),
                0.97,
                0.28,
                boxstyle="round,pad=0.01",
                transform=ax.transAxes,
                facecolor="#f2f5fb",
                edgecolor=BAND,
                linewidth=1.0,
                zorder=1,
            )
        )
        ax.text(
            0.5,
            y,
            wrapped,
            transform=ax.transAxes,
            fontsize=15,
            va="center",
            ha="center",
            multialignment="center",
            color=INK,
            zorder=2,
        )
    fig.tight_layout()
    _save(fig, out_path)


def fig_deepseek_probe(out_path: str) -> None:
    """Standalone panel: identity-probe bars, golden-ratio axes box.

    The vertical rule on each bar marks the min-max range over repeated
    probe runs; the bar height is the logged run.

    Args:
        out_path: Destination PNG path.
    """
    fig, ax_probe = plt.subplots(figsize=(7.0, 4.4))
    xs = [0, 1]
    values = [PROBE_BASE_RANGE[1], PROBE_SFT_RANGE[0]]
    ax_probe.bar(
        xs, values, 0.52, color=[GREY_FILL, BLUE], edgecolor="white", linewidth=1.0
    )
    for x, value, (lo, hi) in zip(xs, values, (PROBE_BASE_RANGE, PROBE_SFT_RANGE)):
        ax_probe.plot([x, x], [lo, hi], color=INK, linewidth=1.6)
        ax_probe.text(
            x, hi + 0.55, f"{value} / {PROBE_N}", ha="center", va="bottom", fontsize=11
        )
    ax_probe.set_xticks(xs)
    ax_probe.set_xticklabels(["base\nOLMo 3", "Think-SFT"])
    ax_probe.set_ylabel(f'"I am DeepSeek-R1" answers (of {PROBE_N})')
    ax_probe.set_ylim(0, PROBE_N)
    ax_probe.set_title("How often the model calls itself DeepSeek-R1", fontsize=12, loc="center", pad=12)
    ax_probe.set_box_aspect(1 / 1.618)  # golden ratio, wider than tall
    _style_axes(ax_probe)

    fig.tight_layout()
    _save(fig, out_path)




def fig_deepseek_case_v3(out_path: str) -> None:
    """One horizontal figure: firing examples (left) and probe outcome (right).

    The probe panel uses horizontal bars with end labels, matching the visual
    family of the intervention and ledger figures.

    Args:
        out_path: Destination PNG path.
    """
    snippets = _identity_snippets()

    fig, (ax_ex, ax_probe) = plt.subplots(
        1, 2, figsize=(14.6, 4.1), gridspec_kw={"width_ratios": [1.75, 1.0], "wspace": 0.18}
    )

    # ---- left: three firing examples ------------------------------------
    ax_ex.set_axis_off()
    ax_ex.set_title(
        f"Top firing examples of gradient feature {IDENTITY_TOP_MASS_ATOM}",
        fontsize=15,
        loc="center",
        pad=14,
    )
    slots = [0.84, 0.50, 0.16]
    for y, snippet in zip(slots, snippets):
        wrapped = "\n".join(textwrap.wrap(snippet, SNIPPET_WRAP)[:2])
        ax_ex.add_patch(
            FancyBboxPatch(
                (0.01, y - 0.145),
                0.98,
                0.29,
                boxstyle="round,pad=0.01",
                transform=ax_ex.transAxes,
                facecolor="#f2f5fb",
                edgecolor=BAND,
                linewidth=1.0,
                zorder=1,
            )
        )
        ax_ex.text(
            0.5,
            y,
            wrapped,
            transform=ax_ex.transAxes,
            fontsize=14,
            va="center",
            ha="center",
            multialignment="left",
            color=INK,
            zorder=2,
        )

    # ---- right: probe outcome as horizontal bars ------------------------
    rows = (
        ("OLMo 3\nmid-train", PROBE_BASE_RANGE, GREY_FILL),
        ("Think-SFT", PROBE_SFT_RANGE, BLUE),
    )
    ys = [1, 0]  # base on top, Think-SFT below
    for y, (label, (lo, hi), color) in zip(ys, rows):
        value = lo if color is BLUE else hi  # logged run: base 4, SFT 14
        ax_probe.barh(y, value, 0.52, color=color, edgecolor="white", linewidth=1.0, zorder=2)
        ax_probe.text(
            value + 0.5, y, f"{value} / {PROBE_N}", va="center", fontsize=14, color=INK
        )
    ax_probe.set_yticks(ys)
    ax_probe.set_yticklabels([label for label, _, _ in rows], fontsize=14)
    ax_probe.set_xlim(0, PROBE_N + 3.6)
    ax_probe.set_xticks(range(0, PROBE_N + 1, 6))
    ax_probe.tick_params(axis="x", labelsize=13)
    ax_probe.set_xlabel('"I am DeepSeek-R1" answers', fontsize=14)
    ax_probe.set_title(
        "How often the model calls itself DeepSeek-R1",
        fontsize=15,
        loc="center",
        pad=14,
    )
    ax_probe.set_ylim(-0.55, 1.55)
    _style_axes(ax_probe, grid_axis="x")

    fig.tight_layout()
    # Lift the probe axes so its tick and xlabel text ends level with the
    # bottom of the example boxes on the left.
    pos = ax_probe.get_position()
    lift = 0.12
    ax_probe.set_position([pos.x0, pos.y0 + lift, pos.width, pos.height - lift])
    _save(fig, out_path)


# ==========================================================================
# Figure 5 (v2): intervention comparison reduced to four conditions
# inputs: reports/E11_caft_*.json (probe counts), constants above
# ==========================================================================

E11_CONDITIONS_V2 = (
    ("no intervention", ("E11_caft_plain",), GREY_FILL),
    ("projection", ("E11_caft_caft_bwd_raw_L15",), ORANGE),
    ("masking activating chunks", ("E11_caft_lossmask_105",), BLUE),
)


def fig_intervention_v2(out_path: str) -> None:
    """Four horizontal bars: the intervention story without the side controls.

    Args:
        out_path: Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=(9.4, 3.0))

    rows = list(reversed(E11_CONDITIONS_V2))  # first condition at the top
    ys = list(range(len(rows)))

    ax.set_ylim(-0.62, len(rows) - 0.42)

    for y, (label, tags, color) in zip(ys, rows):
        value = _probe(tags[0])
        ax.barh(y, value, 0.58, color=color, edgecolor="white", linewidth=1.0, zorder=2)
        ax.text(
            value + 0.6,
            y,
            f"{value}/{PROBE_N}",
            va="center",
            fontsize=13,
            color=INK,
        )

    ax.set_yticks(ys)
    ax.set_yticklabels([label for label, _, _ in rows], fontsize=13)
    ax.set_xlim(0, PROBE_N + 3.4)
    ax.set_xticks(range(0, PROBE_N + 1, 3))
    ax.set_xlabel(f'"I am DeepSeek-R1" answers (of {PROBE_N} probe responses)')
    ax.set_title(
        "Can intervention prevent feature learning?",
        fontsize=15,
        loc="center",
        pad=12,
    )
    _style_axes(ax, grid_axis="x")

    fig.tight_layout()
    _save(fig, out_path)


# ==========================================================================
# Figure 7: executive-summary headline, top-100 basis
# inputs: reports/llm_baseline_results_3200.json, reports/ledger100_results.json
# ==========================================================================


def fig_exec_forecast(out_path: str) -> None:
    """Three bars for the executive summary: LLM predictor, act SAE, grad SAE.

    All rates come from the top-100 scoring basis: the SAE bars are the
    all-100 hit rates of the joint re-scoring pass (ledger100_results.json),
    and the LLM bar is the N=3,200-document run of the reading baseline
    (its 40 predictions, same judge protocol). No subtitle by request.

    Args:
        out_path: Destination PNG path.
    """
    llm = _load(os.path.join(REPORTS, "llm_baseline_results_3200.json"))
    top100 = _load(os.path.join(REPORTS, "ledger100_results.json"))

    # (label, hit rate, item count, colour); Wilson 95% CI is drawn per bar.
    # Items are treated as independent for the interval.
    bars = (
        ("LLM predictor", llm["hit_rate"], llm["n"], VIOLET),
        ("activation SAE", top100["act"]["hit"], 100, ORANGE),
        ("gradient SAE", top100["grad"]["hit"], 100, BLUE),
    )

    def _wilson(rate: float, n: int, z: float = 1.96) -> tuple[float, float]:
        centre = (rate + z * z / (2 * n)) / (1 + z * z / n)
        half = (
            z
            * ((rate * (1 - rate) / n + z * z / (4 * n * n)) ** 0.5)
            / (1 + z * z / n)
        )
        return centre - half, centre + half

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    xs = list(range(len(bars)))
    values = [v for _, v, _, _ in bars]
    los, his = zip(*(_wilson(v, n) for _, v, n, _ in bars))
    ax.bar(
        xs,
        values,
        0.58,
        color=[c for _, _, _, c in bars],
        edgecolor="white",
        linewidth=1.0,
    )
    ax.errorbar(
        xs,
        values,
        yerr=[
            [v - lo for v, lo in zip(values, los)],
            [hi - v for v, hi in zip(values, his)],
        ],
        fmt="none",
        ecolor=INK,
        elinewidth=1.6,
        capsize=4,
    )
    for x, v, hi in zip(xs, values, his):
        ax.text(x, hi + 0.02, _two_decimals(v), ha="center", va="bottom", fontsize=15, color=INK)

    ax.set_xticks(xs)
    ax.set_xticklabels([label for label, _, _, _ in bars], fontsize=14)
    ax.set_ylabel("forecast hit rate", fontsize=15)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_title(
        "Accuracy of forecasting the finetuned model's behavior",
        fontsize=16,
        pad=14,
    )
    _style_axes(ax)
    # _style_axes resets tick labels to 9pt; re-apply the sizes after it.
    ax.tick_params(axis="x", labelsize=15, length=0)
    ax.tick_params(axis="y", labelsize=13)

    fig.tight_layout()
    _save(fig, out_path)



# ==========================================================================
# Figure 6 (v2): top-3 predictions per method, straight by rank, no curation
# inputs: ledger/ledger_{grad,act}.json, reports/rq2_channelB.json,
#         reports/llm_baseline_results.json
# ==========================================================================

# 90-character condensations of the full prediction sentences; the originals
# are in ledger/ledger_{grad,act}.json (items ranks 1-3) and in
# reports/llm_baseline_results.json (items 0-2, verbatim short statements).
LEDGER_V2_GROUPS = (
    ("gradient SAE", "BLUE", (
        ("more mid-reasoning retractions of a just-made statement: 'Wait, no' / 'no, wait'"),
        ("more enumeration of alternatives for one sub-problem: 'Or maybe ...? Or perhaps ...'"),
        ("more hedged inferences from a stated constraint: 'The problem says X, so probably Y'"),
        ("more verification steps re-checking an obtained result: 'Let me check/verify ...'"),
        ("more moments of realization acknowledging an oversight: 'Ah right', 'I forgot ...'"),
    )),
    ("activation SAE", "ORANGE", (
        ("more strategy-level re-planning mid-solution, stepping back to change the approach"),
        ("more dense inline computation traces: intermediate values chained with arrows"),
        ("more explicit coordinate set-ups of geometry: labeled vertices with numeric coords"),
        ("more local recomputation of a just-completed step: 'Let me recompute / redo'"),
        ("more hedged reconsideration of a fresh conclusion, checked against a requirement"),
    )),
    ("LLM predictor", "VIOLET", (
        ("more structured mathematical reasoning steps"),
        ("more explicit references to the OLMo system identity"),
        ("more consistent formatting for mathematical expressions"),
        ("better handling of nested data structures in programs"),
        ("more careful handling of edge cases in algorithm design"),
    )),
)


def _ledger_v2_hits() -> dict[str, list[bool]]:
    """Per-method hit flags for the three displayed rows, read from files."""
    chan_b = _load(os.path.join(REPORTS, "rq2_channelB.json"))
    items = chan_b["comparisons"]["raw"]["items"]
    hits: dict[str, list[bool]] = {}
    for arm, name in (("grad", "gradient SAE"), ("act", "activation SAE")):
        rows = [i for i in items if i["arm"] == arm][:5]
        hits[name] = [bool(r["rubric"]["hit"]) for r in rows]
    llm = _load(os.path.join(REPORTS, "llm_baseline_results.json"))
    hits["LLM predictor"] = [bool(i["hit"]) for i in llm["items"][:5]]
    return hits


def fig_ledger_examples_v2(out_path: str) -> None:
    """Text panel: the top three predictions of each method, with outcomes.

    Rows are taken straight by rank (dictionaries) or listing order (LLM),
    with no curation; the verdict chip is the rubric judgement.

    Args:
        out_path: Destination PNG path.
    """
    colors = {"BLUE": BLUE, "ORANGE": ORANGE, "VIOLET": VIOLET}
    hits = _ledger_v2_hits()

    fig, ax = plt.subplots(figsize=(12.0, 9.0))
    ax.set_axis_off()
    ax.set_title(
        "Top-5 predictions of each method and how they scored",
        fontsize=17,
        loc="center",
        pad=16,
    )

    y = 0.97
    for name, color_key, rows in LEDGER_V2_GROUPS:
        ax.text(
            0.0, y, name, fontsize=15.5, fontweight="bold",
            color=colors[color_key], va="top", transform=ax.transAxes,
        )
        y -= 0.055
        for rank, text in enumerate(rows, start=1):
            hit = hits[name][rank - 1]
            word, chip = ("hit", STATUS_GOOD) if hit else ("miss", STATUS_BAD)
            ax.text(
                0.02, y, f"#{rank}  {text}", fontsize=14, color=INK,
                va="top", transform=ax.transAxes,
            )
            ax.text(
                0.985, y, word, fontsize=14, fontweight="bold", color=chip,
                va="top", ha="right", transform=ax.transAxes,
            )
            y -= 0.049
        y -= 0.028

    fig.tight_layout()
    _save(fig, out_path)


# ==========================================================================
# Figure 7: full top-40 prediction list of one method, one page per method
# inputs: ledger/ledger_{grad,act}.json (statements),
#         reports/ledger100_results.json (hits, same scoring pass as Figure 1),
#         reports/llm_baseline_results_3200.json (LLM statements + hits)
# ==========================================================================

TOP40_PREFIXES = (
    "After SFT on this data, model responses will ",
    "After SFT, responses will ",
)
TOP40_CLIP = 90  # display width; statements longer than this are cut at a word


def _clip_statement(text: str, limit: int = TOP40_CLIP) -> str:
    """Strip the boilerplate statement lead-in and cut at a word boundary."""
    for prefix in TOP40_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(",;:") + " …"


def _top40_rows(method: str, clip: int = TOP40_CLIP) -> list[tuple[str, bool]]:
    """(display text, hit) for the 40 top-ranked predictions of one method."""
    if method == "llm":
        # results_3200 stores statements cut at 90 chars; the ledger file
        # keeps the full text, aligned by list index.
        full = _load(os.path.join(REPORTS, "llm_baseline_ledger_3200.json"))
        items = _load(os.path.join(REPORTS, "llm_baseline_results_3200.json"))["items"]
        return [
            (_clip_statement(f["statement"], clip), bool(i["hit"]))
            for f, i in zip(full, items)
        ]
    ledger = _load(os.path.join(LEDGER, f"ledger_{method}.json"))["items"]
    statements = {i["rank"]: i["item"]["statement"] for i in ledger}
    scored = _load(os.path.join(REPORTS, "ledger100_results.json"))["items"]
    rows = sorted(
        (i for i in scored if i["arm"] == method and i["rank"] <= 40),
        key=lambda i: i["rank"],
    )
    return [
        (_clip_statement(statements[i["rank"]], clip), bool(i["hit"])) for i in rows
    ]


def fig_top40(method: str, out_path: str) -> None:
    """Text panel: all 40 top predictions of one method, straight by rank.

    Statements are shown with the shared lead-in "After SFT, responses
    will ..." removed and cut to one line each; verdicts are from the same
    scoring pass as the executive-summary forecast figure.

    Args:
        method: One of "grad", "act", "llm".
        out_path: Destination PNG path.
    """
    display = {
        "grad": ("gradient SAE", BLUE),
        "act": ("activation SAE", ORANGE),
        "llm": ("LLM predictor", VIOLET),
    }
    name, color = display[method]
    rows = _top40_rows(method)

    fig, ax = plt.subplots(figsize=(12.0, 13.5))
    ax.set_axis_off()
    ax.set_title(
        f"Top-40 predictions — {name}",
        fontsize=17, color=color, fontweight="bold", loc="center", pad=14,
    )

    y = 0.985
    step = 0.0245
    for rank, (text, hit) in enumerate(rows, start=1):
        word, chip = ("hit", STATUS_GOOD) if hit else ("miss", STATUS_BAD)
        ax.text(
            0.0, y, f"#{rank}", fontsize=12.5, color=INK,
            va="top", ha="left", transform=ax.transAxes,
        )
        ax.text(
            0.045, y, text, fontsize=12.5, color=INK,
            va="top", transform=ax.transAxes,
        )
        ax.text(
            0.995, y, word, fontsize=12.5, fontweight="bold", color=chip,
            va="top", ha="right", transform=ax.transAxes,
        )
        y -= step

    fig.tight_layout()
    _save(fig, out_path)


# ==========================================================================
# Figure 8: R² of delta_later explained by the next-N token losses
# input: reports/decompose_now_later.md section 5 (decomposition script, not
# included; window mode, 120 docs x 64 consecutive positions; raw window.pt not git-tracked)
# ==========================================================================

WINDOW_R2 = (
    # N, R2 p25, R2 p50, R2 p75  (whitened, per-position quantiles)
    (1, 0.000, 0.031, 0.288),
    (2, 0.008, 0.162, 0.503),
    (4, 0.079, 0.344, 0.666),
    (8, 0.197, 0.508, 0.783),
    (16, 0.320, 0.648, 0.861),
    (32, 0.473, 0.766, 0.914),
    (48, 0.577, 0.826, 0.942),
    (63, 0.699, 0.877, 0.949),
)


def fig_label_window(out_path: str) -> None:
    """Line chart: how much of delta_later the next N token losses explain.

    Median R² over positions with the interquartile band, N on a log axis,
    with the 32-token labeling window marked.

    Args:
        out_path: Destination PNG path.
    """
    ns = [r[0] for r in WINDOW_R2]
    p25 = [r[1] for r in WINDOW_R2]
    p50 = [r[2] for r in WINDOW_R2]
    p75 = [r[3] for r in WINDOW_R2]

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.fill_between(ns, p25, p75, color=BAND, alpha=0.35, linewidth=0,
                    label="interquartile range")
    ax.plot(ns, p50, color=BLUE, linewidth=LINE_WIDTH, marker="o",
            markersize=MARKER_SIZE, label="median over positions")

    ax.axvline(32, color=INK_MUTED, linewidth=1.2, linestyle="--")
    ax.text(32 * 1.05, 0.06, "labeling window\n(32 tokens)", fontsize=13,
            color=INK_MUTED, ha="left", va="bottom")
    ax.annotate("0.77", (32, 0.766), textcoords="offset points",
                xytext=(-8, 10), fontsize=13, color=BLUE, ha="right")

    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns], fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("number of following tokens N", fontsize=14)
    ax.set_ylabel("R² of δ later explained\nby the next N token losses", fontsize=14)
    ax.set_title(
        "How much of the shared gradient component\ndo the next N tokens explain?",
        fontsize=16, pad=14,
    )
    ax.legend(loc="upper left", frameon=False, fontsize=13)
    _style_axes(ax)

    fig.tight_layout()
    _save(fig, out_path)


# ==========================================================================
# Figure 9: FVU of the gradient vs activation SAE (final v2 runs, L0 ~ 64)
# inputs: reports/grad_v2.json, reports/act_v2.json (train fvu_32768 — the
# numbers quoted in the write-up text; holdout is 0.782 / 0.233)
# ==========================================================================


def fig_fvu_bars(out_path: str) -> None:
    """Bar chart: reconstruction FVU of the gradient and activation SAEs.

    Both dictionaries share one architecture and config (32K atoms, L0 64);
    only the training signal differs. Single-L0 runs, so bars rather than a
    frontier.

    Args:
        out_path: Destination PNG path.
    """
    arms = (
        ("activation SAE", "act_v2.json", ORANGE),
        ("gradient SAE", "grad_v2.json", BLUE),
    )
    names = [a[0] for a in arms]
    values = [
        _load(os.path.join(REPORTS, f))["train"]["fvu_32768"] for _, f, _ in arms
    ]

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    xs = list(range(len(arms)))
    ax.bar(xs, values, 0.58, color=[c for _, _, c in arms],
           edgecolor="white", linewidth=1.0)
    for x, v in zip(xs, values):
        ax.text(x, v + 0.02, f"{v:.3f}", ha="center", va="bottom",
                fontsize=15, color=INK)

    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_ylabel("FVU", fontsize=15)
    ax.set_ylim(0, 0.85)
    ax.set_title(
        "Reconstruction quality of the two dictionaries",
        fontsize=16, pad=14,
    )
    _style_axes(ax)
    # _style_axes resets tick labels to 9pt; re-apply the sizes after it.
    # x labels sit just under the 16pt title size.
    ax.tick_params(axis="x", labelsize=15, length=0)
    ax.tick_params(axis="y", labelsize=13)

    fig.tight_layout()
    _save(fig, out_path)


# ==========================================================================
# Figure 10: top-10 predictions per method, one panel (top-40 pipeline)
# inputs: same as the top40 figures
# ==========================================================================


def fig_ledger_examples_top10(out_path: str) -> None:
    """Text panel: the top ten predictions of each method, with outcomes.

    Same data pipeline and trimming as the per-method top-40 pages; rows are
    straight by rank with no curation.

    Args:
        out_path: Destination PNG path.
    """
    groups = (
        ("gradient SAE", "grad", BLUE),
        ("activation SAE", "act", ORANGE),
        ("LLM predictor", "llm", VIOLET),
    )

    fig, ax = plt.subplots(figsize=(12.0, 11.3))
    ax.set_axis_off()
    ax.set_title(
        "Top-10 predictions of each method and how they scored",
        fontsize=17, loc="center", pad=14,
    )

    y = 0.99
    row_step = 0.0272
    for name, method, color in groups:
        ax.text(
            0.0, y, name, fontsize=15.5, fontweight="bold",
            color=color, va="top", transform=ax.transAxes,
        )
        y -= 0.031
        for rank, (text, hit) in enumerate(
            _top40_rows(method, clip=105)[:10], start=1
        ):
            word, chip = ("hit", STATUS_GOOD) if hit else ("miss", STATUS_BAD)
            ax.text(
                0.02, y, f"#{rank}", fontsize=11.5, color=INK,
                va="top", ha="left", transform=ax.transAxes,
            )
            ax.text(
                0.062, y, text, fontsize=11.5, color=INK,
                va="top", transform=ax.transAxes,
            )
            ax.text(
                0.995, y, word, fontsize=11.5, fontweight="bold", color=chip,
                va="top", ha="right", transform=ax.transAxes,
            )
            y -= row_step
        y -= 0.018

    fig.tight_layout()
    _save(fig, out_path)


# ==========================================================================
# Figures 11-13: overthinking case (E10, 19_overthinking.py)
# inputs: reports/label_examples/grad_v2_examples.part*.json.gz (atom 4141),
#         reports/E10_overthink.json (5 trivial questions x 3 seeds)
# ==========================================================================

OVERTHINK_ATOM = 4141  # top-mass gradient atom, "self-correction or hesitation"
OVERTHINK_QUESTIONS = {
    # E10_overthink.json stores q[:20]; full texts are in 19_overthinking.py
    "What is 17 times 23?": "17 × 23 = ?",
    "If 3x + 5 = 20, what": "3x + 5 = 20, x = ?",
    "Convert 0.375 to a f": "0.375 as a fraction",
    "What is the least co": "lcm(12, 18)",
    "What is the sum of i": "hexagon angle sum",
}


def _overthink_rows(model: str) -> list[dict]:
    """The 15 logged runs of one model ('sft' or 'base') from E10."""
    return _load(os.path.join(REPORTS, "E10_overthink.json"))[model]


def fig_overthink_examples(out_path: str) -> None:
    """Boxed panel: top firing examples of the flagship deliberation feature.

    Same visual family as the DeepSeek-case example boxes.

    Args:
        out_path: Destination PNG path.
    """
    shard = os.path.join(
        EXAMPLES, f"grad_v2_examples.part{OVERTHINK_ATOM // 4096:02d}.json.gz"
    )
    with gzip.open(shard, "rt", encoding="utf-8") as fh:
        entry = json.load(fh)[str(OVERTHINK_ATOM)]
    snippets = [" ".join(e["text"].split()) for e in entry["examples"][:3]]

    fig, ax = plt.subplots(figsize=(10.6, 4.1))
    ax.set_axis_off()
    ax.set_title(
        f"Top firing examples of gradient feature {OVERTHINK_ATOM}\n"
        "(the largest-mass feature: mid-reasoning self-correction)",
        fontsize=15, loc="center", pad=12,
    )
    slots = [0.84, 0.50, 0.16]
    for y, snippet in zip(slots, snippets):
        wrapped = "\n".join(textwrap.wrap(snippet, 84)[:2])
        ax.add_patch(
            FancyBboxPatch(
                (0.01, y - 0.145), 0.98, 0.29,
                boxstyle="round,pad=0.01", transform=ax.transAxes,
                facecolor="#f2f5fb", edgecolor=BAND, linewidth=1.0, zorder=1,
            )
        )
        ax.text(
            0.5, y, wrapped, transform=ax.transAxes, fontsize=14,
            va="center", ha="center", multialignment="left", color=INK, zorder=2,
        )
    fig.tight_layout()
    _save(fig, out_path)


def fig_overthink_accuracy(out_path: str) -> None:
    """Horizontal bars: correct answers on the trivial-arithmetic probe.

    Args:
        out_path: Destination PNG path.
    """
    n_sft = sum(r["correct"] for r in _overthink_rows("sft"))
    n_base = sum(r["correct"] for r in _overthink_rows("base"))
    total = len(_overthink_rows("sft"))

    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    rows = (
        ("OLMo 3\nmid-train", n_base, GREY_FILL),
        ("Think-SFT", n_sft, BLUE),
    )
    ys = [1, 0]
    for y, (label, value, color) in zip(ys, rows):
        ax.barh(y, value, 0.52, color=color, edgecolor="white",
                linewidth=1.0, zorder=2)
        ax.text(value + 0.25, y, f"{value} / {total}", va="center",
                fontsize=14, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=14)
    ax.set_xlim(0, total + 2.4)
    ax.set_xticks(range(0, total + 1, 5))
    ax.tick_params(axis="x", labelsize=13)
    ax.set_xlabel("correct answers", fontsize=14)
    ax.set_title(
        "Accuracy on 5 trivial arithmetic questions (3 seeds each)",
        fontsize=15, loc="center", pad=12,
    )
    ax.set_ylim(-0.55, 1.55)
    _style_axes(ax, grid_axis="x")
    fig.tight_layout()
    _save(fig, out_path)


def fig_overthink_tokens(out_path: str) -> None:
    """Horizontal bars: mean thinking length per trivial question (Think-SFT).

    The raw mid-train model is not shown: with a raw prompt it never emits
    an end-of-text token, so its length is the budget, not a behavior.

    Args:
        out_path: Destination PNG path.
    """
    rows = _overthink_rows("sft")
    means: dict[str, float] = {}
    for q, label in OVERTHINK_QUESTIONS.items():
        vals = [r["think_tok"] for r in rows if r["q"] == q]
        means[label] = sum(vals) / len(vals)
    order = sorted(means, key=lambda k: means[k])

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ys = range(len(order))
    ax.barh(list(ys), [means[k] for k in order], 0.58, color=BLUE,
            edgecolor="white", linewidth=1.0, zorder=2)
    for y, k in zip(ys, order):
        ax.text(means[k] + 25, y, f"{means[k]:.0f}", va="center",
                fontsize=13.5, color=INK)
    ax.axvline(2048, color=INK_MUTED, linewidth=1.2, linestyle="--")
    ax.text(2048, -0.55, " generation budget (2,048 tokens)",
            fontsize=12.5, color=INK_MUTED, ha="left", va="top")
    ax.set_yticks(list(ys))
    ax.set_yticklabels(order, fontsize=13.5)
    ax.set_xlim(0, 2650)
    ax.set_ylim(-0.85, len(order) - 0.45)
    ax.tick_params(axis="x", labelsize=13)
    ax.set_xlabel("mean thinking tokens (3 seeds)", fontsize=14)
    ax.set_title(
        "How long the Think-SFT model deliberates on trivial arithmetic",
        fontsize=15, loc="center", pad=12,
    )
    _style_axes(ax, grid_axis="x")
    fig.tight_layout()
    _save(fig, out_path)


# ==========================================================================
# Figure 14: one full overthinking specimen (regenerated Think-SFT run)
# input: reports/E10_overthink_texts.json (20_overthinking_transcripts.py; same
# prompts/seeds/sampling as E10, rerun on A6000 to capture the texts)
# ==========================================================================

OVERTHINK_CASE_Q = "What is 17 times 23?"
OVERTHINK_CASE_SEED = 1
OVERTHINK_CASE_ANCHOR = "Which again is 391."  # mid-run: the answer is found


def fig_overthink_case(out_path: str) -> None:
    """Boxed excerpts of one Think-SFT run on a trivial arithmetic question.

    Opening, middle (where the correct product is already obtained) and the
    point where the 2,048-token budget runs out, quoted verbatim from the
    regenerated run.

    Args:
        out_path: Destination PNG path.
    """
    runs = _load(os.path.join(REPORTS, "E10_overthink_texts.json"))
    run = next(
        r for r in runs
        if r["q"] == OVERTHINK_CASE_Q and r["seed"] == OVERTHINK_CASE_SEED
    )
    text = " ".join(run["text"].split())
    mid_at = text.find(OVERTHINK_CASE_ANCHOR)
    mid = text[mid_at:mid_at + 430] if mid_at >= 0 else text[900:1330]
    tail = text[-430:].split(" ", 1)[1]  # start the excerpt at a word boundary
    wrapped_excerpts = [
        textwrap.wrap(text[:450].rsplit(" ", 1)[0], 84)[:5],
        textwrap.wrap(mid.rsplit(" ", 1)[0], 84)[:5],
        textwrap.wrap(tail, 84)[-5:],  # keep the trailing repetition loop
    ]

    fig, ax = plt.subplots(figsize=(10.6, 5.9))
    ax.set_axis_off()
    ax.set_title(
        f'Think-SFT answering "{OVERTHINK_CASE_Q}"',
        fontsize=15, loc="center", pad=12,
    )
    # One box for the whole response; the excerpts are separated by "⋯" rows.
    line_h = 0.049
    pad = 0.045
    sep_h = 0.06
    seg_h = [line_h * len(lines) for lines in wrapped_excerpts]
    total = pad + sum(seg_h) + sep_h * 2 + pad
    y_top = 0.985
    ax.add_patch(
        FancyBboxPatch(
            (0.01, y_top - total), 0.98, total,
            boxstyle="round,pad=0.01", transform=ax.transAxes,
            facecolor="#f2f5fb", edgecolor=BAND, linewidth=1.0, zorder=1,
        )
    )
    y = y_top - pad
    for i, lines in enumerate(wrapped_excerpts):
        ax.text(
            0.09, y - seg_h[i] / 2, "\n".join(lines), transform=ax.transAxes,
            fontsize=13.5, va="center", ha="left", multialignment="left",
            color=INK, zorder=2, linespacing=1.35,
        )
        y -= seg_h[i]
        if i < 2:
            ax.text(0.5, y - sep_h / 2, "⋯", transform=ax.transAxes,
                    fontsize=16, va="center", ha="center", color=INK_MUTED,
                    zorder=2)
            y -= sep_h
    fig.tight_layout()
    _save(fig, out_path)


# ==========================================================================
# Figure 15: two thinking-feature firing examples, side by side
# input: reports/label_examples/grad_v2_examples.part*.json.gz
# ==========================================================================

# (atom, ledger label, indices into the shard's example list)
THINKING_EXAMPLE_PICKS = (
    (4141, "self-correction or hesitation", (0, 2, 3)),
    (5184, "error checking or reconsideration", (0, 1, 2)),
)


def fig_thinking_examples(out_path: str) -> None:
    """Two deliberation features side by side, three firing samples each.

    Args:
        out_path: Destination PNG path.
    """
    columns = []
    for atom, label, idxs in THINKING_EXAMPLE_PICKS:
        shard = os.path.join(
            EXAMPLES, f"grad_v2_examples.part{atom // 4096:02d}.json.gz"
        )
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            entry = json.load(fh)[str(atom)]
        texts = [
            " ".join(entry["examples"][i]["text"].split()) for i in idxs
        ]
        columns.append((atom, label, texts))

    fig, ax = plt.subplots(figsize=(12.6, 3.7))
    ax.set_axis_off()
    ax.set_title(
        "Top firing examples of thinking features",
        fontsize=15, loc="center", pad=14,
    )
    slots = [(0.0, 0.482), (0.518, 1.0)]
    box_h = 0.21
    box_gap = 0.058
    for (x0, x1), (atom, label, texts) in zip(slots, columns):
        ax.text(
            (x0 + x1) / 2, 1.0, f"feature {atom} · {label}",
            transform=ax.transAxes, fontsize=13, fontweight="bold",
            color=BLUE, va="top", ha="center",
        )
        y_top = 0.84
        for text in texts:
            ax.add_patch(
                FancyBboxPatch(
                    (x0 + 0.006, y_top - box_h), (x1 - x0) - 0.012, box_h,
                    boxstyle="round,pad=0.01", transform=ax.transAxes,
                    facecolor="#f2f5fb", edgecolor=BAND, linewidth=1.0,
                    zorder=1,
                )
            )
            wrapped = "\n".join(textwrap.wrap(text, 46)[:3])
            ax.text(
                (x0 + x1) / 2, y_top - box_h / 2, wrapped,
                transform=ax.transAxes, fontsize=12, va="center", ha="center",
                multialignment="left", color=INK, zorder=2, linespacing=1.3,
            )
            y_top -= box_h + box_gap
    fig.tight_layout()
    _save(fig, out_path)


# ==========================================================================
# Figure 16: appendix D — seeded random firing samples of the body's features
# input: reports/label_examples/grad_v2_examples.part*.json.gz
# ==========================================================================

RANDOM_EXAMPLE_FEATURES = (
    (4141, "self-correction or hesitation"),
    (5184, "error checking or reconsideration"),
    (8163, "DeepSeek persona"),
)
RANDOM_EXAMPLE_SEED = 0
RANDOM_EXAMPLE_N = 5


def _displayable(text: str) -> str:
    """Replace mostly-non-Latin snippets that the figure font cannot render."""
    ascii_frac = sum(ch.isascii() for ch in text) / max(len(text), 1)
    if ascii_frac < 0.5:
        return "(a chunk of Telugu script from a multilingual document)"
    return text


def fig_random_examples(out_path: str) -> None:
    """Five seeded random firing samples for each feature shown in the body.

    Samples are drawn uniformly (fixed seed) from each feature's 40 stored
    top-activating chunks and quoted verbatim, shown alongside the curated
    example figures.

    Args:
        out_path: Destination PNG path.
    """
    import random as _random

    groups = []
    for atom, label in RANDOM_EXAMPLE_FEATURES:
        shard = os.path.join(
            EXAMPLES, f"grad_v2_examples.part{atom // 4096:02d}.json.gz"
        )
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            entry = json.load(fh)[str(atom)]
        rng = _random.Random(RANDOM_EXAMPLE_SEED)
        picks = sorted(rng.sample(range(len(entry["examples"])), RANDOM_EXAMPLE_N))
        texts = [
            _displayable(" ".join(entry["examples"][i]["text"].split()))
            for i in picks
        ]
        groups.append((atom, label, texts))

    fig, ax = plt.subplots(figsize=(12.0, 10.6))
    ax.set_axis_off()
    ax.set_title(
        "Randomly sampled firing examples of the features shown in the main text",
        fontsize=16, loc="center", pad=14,
    )
    line_h = 0.017
    box_pad = 0.016
    box_gap = 0.014
    y = 0.995
    for atom, label, texts in groups:
        ax.text(
            0.0, y, f"feature {atom} · {label}",
            fontsize=14, fontweight="bold", color=BLUE,
            va="top", transform=ax.transAxes,
        )
        y -= 0.033
        for text in texts:
            lines = textwrap.wrap(text, 100)[:2]
            box_h = line_h * len(lines) + box_pad
            ax.add_patch(
                FancyBboxPatch(
                    (0.005, y - box_h), 0.99, box_h,
                    boxstyle="round,pad=0.008", transform=ax.transAxes,
                    facecolor="#f2f5fb", edgecolor=BAND, linewidth=1.0,
                    zorder=1,
                )
            )
            ax.text(
                0.03, y - box_h / 2, "\n".join(lines),
                fontsize=12, va="center", ha="left", multialignment="left",
                color=INK, zorder=2, linespacing=1.25, transform=ax.transAxes,
            )
            y -= box_h + box_gap
        y -= 0.022

    fig.tight_layout()
    _save(fig, out_path)


# ==========================================================================
# entry point
# ==========================================================================

FIGURES: dict[str, tuple[Callable[[str], None], str]] = {
    "forecast_comparison": (
        fig_forecast_comparison,
        os.path.join(OUT_DIR, "fig_forecast_comparison.png"),
    ),
    "exec_forecast": (fig_exec_forecast, os.path.join(OUT_DIR, "fig_exec_forecast.png")),
    "top100": (fig_top100, os.path.join(OUT_DIR, "fig_top100.png")),
    "deepseek_case": (
        fig_deepseek_case_v3,
        os.path.join(OUT_DIR, "fig_deepseek_case.png"),
    ),
    "sinhala": (fig_sinhala, os.path.join(OUT_DIR, "fig_sinhala.png")),
    "intervention": (
        fig_intervention_v2,
        os.path.join(OUT_DIR, "fig_intervention.png"),
    ),
    "ledger_examples": (
        fig_ledger_examples_v2,
        os.path.join(OUT_DIR, "fig_ledger_examples.png"),
    ),
    "top40_grad": (
        functools.partial(fig_top40, "grad"),
        os.path.join(OUT_DIR, "fig_top40_grad.png"),
    ),
    "top40_act": (
        functools.partial(fig_top40, "act"),
        os.path.join(OUT_DIR, "fig_top40_act.png"),
    ),
    "top40_llm": (
        functools.partial(fig_top40, "llm"),
        os.path.join(OUT_DIR, "fig_top40_llm.png"),
    ),
    "label_window": (
        fig_label_window,
        os.path.join(OUT_DIR, "fig_label_window.png"),
    ),
    "fvu_bars": (
        fig_fvu_bars,
        os.path.join(OUT_DIR, "fig_fvu_bars.png"),
    ),
    "ledger_examples_top10": (
        fig_ledger_examples_top10,
        os.path.join(OUT_DIR, "fig_ledger_examples_top10.png"),
    ),
    "overthink_examples": (
        fig_overthink_examples,
        os.path.join(OUT_DIR, "fig_overthink_examples.png"),
    ),
    "overthink_accuracy": (
        fig_overthink_accuracy,
        os.path.join(OUT_DIR, "fig_overthink_accuracy.png"),
    ),
    "overthink_tokens": (
        fig_overthink_tokens,
        os.path.join(OUT_DIR, "fig_overthink_tokens.png"),
    ),
    "overthink_case": (
        fig_overthink_case,
        os.path.join(OUT_DIR, "fig_overthink_case.png"),
    ),
    "thinking_examples": (
        fig_thinking_examples,
        os.path.join(OUT_DIR, "fig_thinking_examples.png"),
    ),
    "random_examples": (
        fig_random_examples,
        os.path.join(OUT_DIR, "fig_random_examples.png"),
    ),
}


def main() -> None:
    """Parse arguments and render the requested figures."""
    parser = argparse.ArgumentParser(
        description="Render the MATS write-up figures from git-tracked inputs only."
    )
    parser.add_argument(
        "--only",
        choices=sorted(FIGURES),
        default=None,
        help="render a single figure instead of all of them",
    )
    parser.add_argument(
        "--list", action="store_true", help="print figure names and exit"
    )
    args = parser.parse_args()

    if args.list:
        for name, (_, path) in FIGURES.items():
            print(f"{name}\t{os.path.relpath(path, BASE)}")
        return

    for name in [args.only] if args.only else list(FIGURES):
        render, path = FIGURES[name]
        render(path)


if __name__ == "__main__":
    main()
