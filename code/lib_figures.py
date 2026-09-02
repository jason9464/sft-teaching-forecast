#!/usr/bin/env python
"""Shared plotting style, plus the earlier diagnostic figures.

``26_make_figures.py`` imports the palette, the axis styling and the save and
load helpers from this module, so the two stay in lockstep. Running this module
directly renders the diagnostic figures listed below; several of them read
result files that are not part of this release, so only the style import is
exercised by the write-up figures.

No input is read from ``data/`` (3.1 TB, gitignored in the working repository).
Every input is either a tracked JSON, Markdown or log file or a constant
hardcoded below with its source cited in a comment.

Run commands:

    python3 code/lib_figures.py

    # single figure
    python3 code/lib_figures.py --only rq2_channelB

    # list the figure names
    python3 code/lib_figures.py --list

Outputs
-------
Figures written next to the JSON they summarise:
    reports/rq2_channelB.png
    reports/feedback_checks/alpha_dose.png

Regenerated versions of the four figures that were committed as PNG without
their source code. These go to ``reports/figures_regen/`` and never overwrite
the originals:
    reports/figures_regen/E11_dose_response.png
    reports/figures_regen/gradient_usefulness_summary.png
    reports/figures_regen/pareto_width_l0.png
    reports/figures_regen/fvu_l0_frontier.png

Colours come from the validated categorical palette in the ``dataviz`` skill
(light mode, fixed slot order). Slots 1-3 are validated for all-pairs forms;
figures that need more than three series also vary marker shape and line style,
so identity is never carried by colour alone.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(BASE, "reports")
FEEDBACK = os.path.join(REPORTS, "feedback_checks")
REGEN = os.path.join(REPORTS, "figures_regen")

# --------------------------------------------------------------------------
# style: validated categorical palette (dataviz skill, references/palette.md,
# light mode). Fixed slot order, never cycled.
# --------------------------------------------------------------------------

BLUE = "#2a78d6"  # slot 1
ORANGE = "#eb6834"  # slot 2
AQUA = "#1baf7a"  # slot 3
GREEN = "#008300"  # slot 6
VIOLET = "#4a3aa7"  # slot 7

INK = "#0b0b0b"  # text-primary
INK_MUTED = "#52514e"  # text-secondary
GRID = "#d8d7d2"
BAND = "#e6e5e0"

DPI = 150
LINE_WIDTH = 2.0
MARKER_SIZE = 8.0
BAR_GAP = 0.02  # surface gap between adjacent bars, in x-axis units


def _style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    """Apply the recessive grid / ink conventions to one axes.

    Args:
        ax: Axes to restyle in place.
        grid_axis: Which axis carries the grid ("x", "y" or "both").
    """
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_MUTED)
    ax.yaxis.label.set_color(INK_MUTED)


def _save(fig: plt.Figure, path: str) -> None:
    """Write a figure to disk, creating the parent directory if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path} ({os.path.getsize(path)} bytes)")


def _load(path: str) -> dict:
    """Read one JSON file."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _sem(values: Sequence[float]) -> float:
    """Standard error of the mean; 0.0 when fewer than two observations."""
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


# ==========================================================================
# Figure 1 (new): RQ2 channel B hit rates
# input: reports/rq2_channelB.json  (produced by 11_judge_hits.py)
# ==========================================================================

CHANNEL_B_ARMS = ("grad", "act", "err")
ARM_TITLE = {
    "grad": "gradient dict",
    "act": "activation dict",
    "err": "output-error dict",
}
# The two comparisons named in rq2_channelB.json["comparisons"]. Both score the
# Think-SFT model against the same base model; they differ in whether the
# generation prompt used the raw or the chat template (11_judge_hits.py, main()).
CHANNEL_B_CONDITIONS = (
    ("raw", "raw prompts (sft_raw vs base_raw)", BLUE),
    ("chat", "chat template (sft_chat vs base_raw)", ORANGE),
)


def fig_rq2_channel_b(out_path: str) -> None:
    """Bar chart of channel-B hit rate per dictionary arm, raw vs chat.

    A "hit" is a pre-registered per-atom prediction whose measured effect is
    positive with permutation p < 0.05 (11_judge_hits.py, ``effect``). The right
    panel is the specificity control: the same test on 52 discourse markers,
    split into markers the dictionaries predicted and reserved control markers.

    Args:
        out_path: Destination PNG path.
    """
    data = _load(os.path.join(REPORTS, "rq2_channelB.json"))
    comps = data["comparisons"]

    fig, (ax_arm, ax_mark) = plt.subplots(
        1, 2, figsize=(11.5, 5.0), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )

    # ---- left: hit rate per arm -----------------------------------------
    n_cond = len(CHANNEL_B_CONDITIONS)
    width = (1.0 - 0.34) / n_cond - BAR_GAP
    xs = list(range(len(CHANNEL_B_ARMS)))
    for j, (cond, label, color) in enumerate(CHANNEL_B_CONDITIONS):
        summary = comps[cond]["summary"]
        offs = (j - (n_cond - 1) / 2) * (width + BAR_GAP)
        vals = [summary[a]["hit_rate"] for a in CHANNEL_B_ARMS]
        ax_arm.bar(
            [x + offs for x in xs],
            vals,
            width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=1.0,
        )
        for x, v in zip(xs, vals):
            ax_arm.text(
                x + offs,
                v + 0.02,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK,
            )

    n_items = comps["raw"]["summary"]["grad"]["n_items"]
    ax_arm.set_xticks(xs)
    ax_arm.set_xticklabels(
        [
            f"{ARM_TITLE[a]}\n(n={comps['raw']['summary'][a]['n_items']} items)"
            for a in CHANNEL_B_ARMS
        ]
    )
    ax_arm.set_ylabel("channel B hit rate")
    ax_arm.set_ylim(0, 1.0)
    ax_arm.set_title("Predictions that survive the SFT test", fontsize=12, pad=22)
    ax_arm.text(
        0.0,
        1.02,
        f"hit = effect > 0 and permutation p < 0.05, {n_items} items per arm",
        transform=ax_arm.transAxes,
        fontsize=9,
        color=INK_MUTED,
    )
    _style_axes(ax_arm)

    # ---- right: marker predicted vs control ------------------------------
    groups = (
        ("pred_hit_rate", "predicted markers"),
        ("ctrl_hit_rate", "control markers"),
    )
    xs_m = list(range(len(groups)))
    for j, (cond, label, color) in enumerate(CHANNEL_B_CONDITIONS):
        summary = comps[cond]["summary"]["markers"]
        offs = (j - (n_cond - 1) / 2) * (width + BAR_GAP)
        vals = [summary[key] for key, _ in groups]
        ax_mark.bar(
            [x + offs for x in xs_m],
            vals,
            width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=1.0,
        )
        for x, v in zip(xs_m, vals):
            ax_mark.text(
                x + offs,
                v + 0.02,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK,
            )

    p_raw = comps["raw"]["summary"]["markers"]["mannwhitney_p"]
    p_chat = comps["chat"]["summary"]["markers"]["mannwhitney_p"]
    ax_mark.set_xticks(xs_m)
    ax_mark.set_xticklabels([label for _, label in groups])
    ax_mark.set_ylabel("channel B hit rate")
    ax_mark.set_ylim(0, 1.0)
    ax_mark.set_title("Specificity control: 52 discourse markers", fontsize=12, pad=38)
    ax_mark.text(
        0.0,
        1.02,
        "Mann-Whitney on effect size, predicted vs control:\n"
        f"p={p_raw:g} (raw), p={p_chat:g} (chat)",
        transform=ax_mark.transAxes,
        fontsize=9,
        color=INK_MUTED,
        va="bottom",
    )
    _style_axes(ax_mark)

    handles, labels = ax_arm.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=10,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, out_path)


# ==========================================================================
# Figure 2 (new): alpha dose-response
# input: reports/feedback_checks/alpha_dose.json (produced by the alpha dose-response script, not included)
# ==========================================================================

# Group tags written by the alpha dose-response script. "meta" = 20 metacognition-labelled
# gradient atoms, "anchor" = 10 lexical-anchor atoms, "random" = 15 random unit
# directions (the null).
ALPHA_GROUPS = (
    ("meta", "metacognition atoms (n=20)", BLUE, "o", "-"),
    ("anchor", "lexical-anchor atoms (n=10)", ORANGE, "s", "-"),
    ("random", "random directions (n=15)", AQUA, "^", "--"),
)


def _group_curve(
    runs: list[dict], group: str, metric: str
) -> tuple[list[float], list[float], list[float]]:
    """Mean and standard error of one metric per alpha, for one group.

    Args:
        runs: The ``runs`` list of alpha_dose.json.
        group: Group tag ("meta", "anchor" or "random").
        metric: Field name to aggregate ("ireg", "distinct2" or "anchor").

    Returns:
        Sorted alphas, per-alpha means, per-alpha standard errors.
    """
    by_alpha: dict[float, list[float]] = {}
    for run in runs:
        if run["group"] != group or run.get(metric) is None:
            continue
        by_alpha.setdefault(run["alpha"], []).append(float(run[metric]))
    alphas = sorted(by_alpha)
    means = [statistics.fmean(by_alpha[a]) for a in alphas]
    errs = [_sem(by_alpha[a]) for a in alphas]
    return alphas, means, errs


def fig_alpha_dose(out_path: str) -> None:
    """Dose-response of steering strength alpha, atoms vs random directions.

    Each direction is added to the layer-15 residual stream at strength
    alpha * 16.4 and four probe prompts are generated (alpha dose-response script, not included).
    Panels: I-register rate (the shared metacognitive register), distinct-2
    (a collapse guard) and, for the anchored atoms only, the rate of that
    atom's own anchor word (the atom-specific signal).

    Args:
        out_path: Destination PNG path.
    """
    data = _load(os.path.join(FEEDBACK, "alpha_dose.json"))
    runs = data["runs"]
    base = data["base"]

    fig, (ax_ireg, ax_d2, ax_anchor) = plt.subplots(1, 3, figsize=(13.0, 4.4))

    # ---- panel 1: I-register rate ---------------------------------------
    for group, label, color, marker, ls in ALPHA_GROUPS:
        alphas, means, errs = _group_curve(runs, group, "ireg")
        ax_ireg.errorbar(
            alphas,
            means,
            yerr=errs,
            label=label,
            color=color,
            marker=marker,
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            linestyle=ls,
            capsize=3,
            markeredgecolor="white",
            markeredgewidth=1.0,
        )
    ax_ireg.axhline(base["ireg"], color=INK_MUTED, linestyle=":", linewidth=1.4)
    ax_ireg.text(
        0.052,
        base["ireg"] + 1.0,
        f"unsteered base ({base['ireg']:.1f})",
        fontsize=9,
        color=INK_MUTED,
    )
    ax_ireg.set_xlabel("steering strength alpha")
    ax_ireg.set_ylabel("I-register markers per 1000 words")
    ax_ireg.set_title(
        "I-register rate: every dose sits above base,\n"
        "but atoms do not separate from random",
        fontsize=11,
    )
    ax_ireg.legend(frameon=False, fontsize=9, loc="upper left")
    _style_axes(ax_ireg)

    # ---- panel 2: distinct-2 (collapse guard) ----------------------------
    for group, label, color, marker, ls in ALPHA_GROUPS:
        alphas, means, errs = _group_curve(runs, group, "distinct2")
        ax_d2.errorbar(
            alphas,
            means,
            yerr=errs,
            label=label,
            color=color,
            marker=marker,
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            linestyle=ls,
            capsize=3,
            markeredgecolor="white",
            markeredgewidth=1.0,
        )
    ax_d2.axhline(base["distinct2"], color=INK_MUTED, linestyle=":", linewidth=1.4)
    ax_d2.text(
        0.052,
        base["distinct2"] + 0.008,
        f"unsteered base ({base['distinct2']:.3f})",
        fontsize=9,
        color=INK_MUTED,
    )
    ax_d2.set_xlabel("steering strength alpha")
    ax_d2.set_ylabel("distinct-2 (bigram diversity)")
    ax_d2.set_title(
        "Bigram diversity: only anchor atoms\ndegrade with dose",
        fontsize=11,
    )
    _style_axes(ax_d2)

    # ---- panel 3: own-anchor rate, anchored atoms only -------------------
    alphas, means, errs = _group_curve(runs, "anchor", "anchor")
    ax_anchor.errorbar(
        alphas,
        means,
        yerr=errs,
        color=ORANGE,
        marker="s",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        capsize=3,
        markeredgecolor="white",
        markeredgewidth=1.0,
    )
    base_anchor = statistics.fmean(
        [v for k, v in base.items() if k.startswith("anchor_") and v is not None]
    )
    ax_anchor.axhline(base_anchor, color=INK_MUTED, linestyle=":", linewidth=1.4)
    ax_anchor.text(
        0.30,
        base_anchor + 0.6,
        f"unsteered base ({base_anchor:.1f})",
        fontsize=9,
        color=INK_MUTED,
        ha="right",
    )
    ax_anchor.set_xlabel("steering strength alpha")
    ax_anchor.set_ylabel("own anchor word per 1000 words")
    ax_anchor.set_title(
        "Atom-specific signal: anchor atoms push\ntheir own word, monotonically in dose",
        fontsize=11,
    )
    _style_axes(ax_anchor)

    fig.tight_layout()
    _save(fig, out_path)


# ==========================================================================
# Figure 3 (regen): E11 identity dose-response
# inputs: reports/E11_caft_*.json  +  coverage percentages tabulated in
#         reports/E11_caft_identity.md (dose-response table, sweep over number of atoms)
# ==========================================================================

E11_PROBE_N = 18  # identity probe questions, see E11_caft_identity.md

# (coverage % of identity-doc chunks that are loss-masked, report tag, label).
# Coverage values are from the dose-response table in E11_caft_identity.md;
# the DeepSeek adoption counts are read from the JSON files.
E11_DOSE = (
    (0.0, "E11_caft_plain", "plain"),
    (26.8, "E11_caft_lossmask", "top-20"),
    (41.1, "E11_caft_lossmask_40", "top-40"),
    (50.4, "E11_caft_lossmask_60", "top-60"),
    (60.3, "E11_caft_lossmask_105", "top-105"),
    (100.0, "E11_caft_dropdocs", "drop docs"),
)
# Seed-2 replications (the seed-2 replication table of E11_caft_identity.md).
E11_SEED2 = ((0.0, "E11_caft_plain_s2"), (60.3, "E11_caft_lossmask_105_s2"))
# Same masked volume, random chunks instead of atom-selected ones.
E11_RANDOM = ((60.3, "E11_caft_lossmask_rand60"), (60.3, "E11_caft_lossmask_rand60_s2"))
# The three activation-projection conditions tabulated in E11_caft_identity.md.
E11_CAFT = ("E11_caft_caft", "E11_caft_caft_all", "E11_caft_randproj")
# Base model (no fine-tuning) probe range, stated in E11_caft_identity.md.
E11_BASE_LO, E11_BASE_HI = 2, 4


def _e11_deepseek(tag: str) -> int:
    """DeepSeek identity adoptions out of 18 probes for one E11 condition."""
    return int(_load(os.path.join(REPORTS, f"{tag}.json"))["probe"]["deepseek"])


def fig_e11_dose_response(out_path: str) -> None:
    """Identity adoption vs how much of the identity corpus is loss-masked.

    Args:
        out_path: Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    caft_vals = [_e11_deepseek(t) for t in E11_CAFT]
    ax.axhspan(min(caft_vals), max(caft_vals), color=BAND, zorder=0)

    xs = [c for c, _, _ in E11_DOSE]
    ys = [_e11_deepseek(t) for _, t, _ in E11_DOSE]
    ax.plot(
        xs,
        ys,
        color=BLUE,
        marker="o",
        markersize=MARKER_SIZE + 2,
        linewidth=LINE_WIDTH,
        markeredgecolor="white",
        markeredgewidth=1.2,
        label="gradient-atom targeted masking",
    )
    for (x, _, name), y in zip(E11_DOSE, ys):
        ax.annotate(
            name,
            (x, y),
            textcoords="offset points",
            xytext=(6, 8),
            fontsize=9,
            color=INK,
        )

    ax.plot(
        [x for x, _ in E11_SEED2],
        [_e11_deepseek(t) for _, t in E11_SEED2],
        linestyle="none",
        marker="o",
        markersize=MARKER_SIZE + 2,
        markerfacecolor="none",
        markeredgecolor=BLUE,
        markeredgewidth=1.8,
        label="replication (seed 2)",
    )
    ax.plot(
        [x for x, _ in E11_RANDOM],
        [_e11_deepseek(t) for _, t in E11_RANDOM],
        linestyle="none",
        marker="s",
        markersize=MARKER_SIZE + 2,
        color=ORANGE,
        markeredgecolor="white",
        markeredgewidth=1.0,
        label="random masking (same volume)",
    )

    ax.axhline(E11_BASE_HI - 1, color=GREEN, linestyle=":", linewidth=1.6)
    ax.text(
        2,
        E11_BASE_HI - 0.5,
        f"base model level ({E11_BASE_LO}-{E11_BASE_HI}/{E11_PROBE_N})",
        fontsize=9,
        color=GREEN,
    )

    ax.set_xlabel("% of identity-doc chunks loss-masked")
    ax.set_ylabel(f"DeepSeek identity adoptions (out of {E11_PROBE_N} probes)")
    ax.set_ylim(0, E11_PROBE_N + 1)
    ax.set_xlim(-4, 106)
    ax.set_title(
        "Blocking identity learning: activation projection fails,\n"
        "gradient-guided loss masking blocks by 41% coverage; random masking does not",
        fontsize=12,
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor=BAND, edgecolor="none"))
    labels.append(
        f"CAFT-style projection, any depth ({min(caft_vals)}-{max(caft_vals)})"
    )
    ax.legend(handles, labels, frameon=False, fontsize=9, loc="center right")
    _style_axes(ax)

    fig.tight_layout()
    _save(fig, out_path)


# ==========================================================================
# Figure 4 (regen): gradient usefulness summary
# inputs: reports/labels/grad_v2_category_stats.json, ledger/grad_v2_categories.json,
#         reports/rq2_channelB.json, reports/rq1_reduction.json,
#         reports/E1_act_shift.json  (+ one constant from gradient_usefulness.md)
# ==========================================================================

# Categories counted as content/form rather than reasoning moves. This split is
# the one used in reports/gradient_usefulness.md (section E4/E4b): it reproduces
# the reported 65% reasoning-move mass share and the move n=37 / content n=3
# channel-B item split.
CONTENT_CATEGORIES = frozenset(
    {
        "MATH_STEPS",
        "CODE_REASONING",
        "TECHNICAL_NOTATION",
        "DOMAIN_CONTENT",
        "FORMATTING_MARKUP",
        "MULTILINGUAL_CONTENT",
        "CREATIVE_NARRATIVE",
        "COMMUNICATION_STYLE",
        "SAFETY_REFUSAL",
    }
)
# cos(mean delta-h, -mean delta) over the aggregate. Reported in
# reports/gradient_usefulness.md, section E1 (aggregate bullet); no tracked JSON
# stores this aggregate, so it is hardcoded here.
COS_AGGREGATE = -0.02
CATEGORY_LABEL_CHARS = 22  # truncate long category names on the y axis


def fig_gradient_usefulness_summary(out_path: str) -> None:
    """Three-panel summary: what the gradient dictionary contains and predicts.

    Args:
        out_path: Destination PNG path.
    """
    stats = _load(os.path.join(REPORTS, "labels", "grad_v2_category_stats.json"))
    channel_b = _load(os.path.join(REPORTS, "rq2_channelB.json"))["comparisons"]["raw"]
    reduction = _load(os.path.join(REPORTS, "rq1_reduction.json"))
    act_shift = _load(os.path.join(REPORTS, "E1_act_shift.json"))
    categories = _load(os.path.join(BASE, "ledger", "grad_v2_categories.json"))

    fig, (ax_mass, ax_fore, ax_geo) = plt.subplots(1, 3, figsize=(13.2, 5.0))

    # ---- panel 1: dictionary mass by category ---------------------------
    rows = sorted(stats["categories"], key=lambda c: c["mass_pct"])
    colors = [ORANGE if r["id"] in CONTENT_CATEGORIES else BLUE for r in rows]
    ys = list(range(len(rows)))
    ax_mass.barh(
        ys,
        [r["mass_pct"] for r in rows],
        color=colors,
        height=0.72,
        edgecolor="white",
        linewidth=1.0,
    )
    ax_mass.set_yticks(ys)
    ax_mass.set_yticklabels([r["id"][:CATEGORY_LABEL_CHARS] for r in rows], fontsize=8)
    move_mass = sum(r["mass_pct"] for r in rows if r["id"] not in CONTENT_CATEGORIES)
    ax_mass.set_xlabel("share of dictionary mass (%)")
    ax_mass.set_title(
        f"Gradient dictionary: mass by category\nreasoning moves hold {move_mass:.0f}% of the mass",
        fontsize=11,
    )
    ax_mass.legend(
        handles=[
            Patch(color=BLUE, label="reasoning move"),
            Patch(color=ORANGE, label="content / form"),
        ],
        frameon=False,
        fontsize=9,
        loc="lower right",
    )
    _style_axes(ax_mass, grid_axis="x")

    # ---- panel 2: SFT forecast accuracy, moves vs content -----------------
    primary = {int(atom): value[0] for atom, value in categories.items()}
    hits: dict[str, list[bool]] = {"move": [], "content": []}
    for record in channel_b["items"]:
        if record["arm"] != "grad":
            continue
        measure = record.get("rubric") or record.get("regex")
        if not measure:
            continue
        cat = primary.get(record["atom"], "?")
        hits["content" if cat in CONTENT_CATEGORIES else "move"].append(measure["hit"])

    keys = ("move", "content")
    vals = [statistics.fmean([float(h) for h in hits[k]]) for k in keys]
    ax_fore.bar(
        [0, 1],
        vals,
        width=0.55,
        color=[BLUE, ORANGE],
        edgecolor="white",
        linewidth=1.0,
    )
    for x, v in zip([0, 1], vals):
        ax_fore.text(
            x, v + 0.02, f"{v:.0%}", ha="center", va="bottom", fontsize=12, color=INK
        )
    ax_fore.set_xticks([0, 1])
    ax_fore.set_xticklabels(
        [
            f"move categories\n(n={len(hits['move'])})",
            f"content / form categories\n(n={len(hits['content'])})",
        ]
    )
    ax_fore.set_ylabel("channel B hit rate")
    ax_fore.set_ylim(0, 1.0)
    ax_fore.set_title(
        "SFT forecast accuracy on OOD generations\n"
        f"moves {vals[0]:.0%} vs content {vals[1]:.0%}",
        fontsize=11,
    )
    _style_axes(ax_fore)

    # ---- panel 3: behaviour is predicted, geometry is not -----------------
    summary = channel_b["summary"]
    bars = [
        ("act to grad\nlinear R2", reduction["r2_act_to_grad"], INK_MUTED),
        ("cos per\nchunk", act_shift["cos_Dh_negdelta_mean"], INK_MUTED),
        ("cos of\nmeans", COS_AGGREGATE, INK_MUTED),
        ("hit rate\ngrad", summary["grad"]["hit_rate"], BLUE),
        ("hit rate\nact", summary["act"]["hit_rate"], ORANGE),
    ]
    xs = list(range(len(bars)))
    ax_geo.bar(
        xs,
        [b[1] for b in bars],
        width=0.6,
        color=[b[2] for b in bars],
        edgecolor="white",
        linewidth=1.0,
    )
    for x, (_, value, _) in zip(xs, bars):
        offset = 0.02 if value >= 0 else -0.06
        ax_geo.text(
            x,
            value + offset,
            f"{value:.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=10,
            color=INK,
        )
    ax_geo.axhline(0, color=GRID, linewidth=1.0)
    ax_geo.set_xticks(xs)
    ax_geo.set_xticklabels([b[0] for b in bars], fontsize=8.5)
    ax_geo.set_ylabel("value (dimensionless, bounded by -1 and 1)")
    ax_geo.set_ylim(-0.12, 0.95)
    ax_geo.set_title(
        "Gradient predicts BEHAVIOUR, not GEOMETRY\n"
        "geometry near zero; behaviour: grad > act",
        fontsize=11,
    )
    _style_axes(ax_geo)

    fig.tight_layout()
    _save(fig, out_path)


# ==========================================================================
# Figure 5 (regen): Pareto, width axis and L0 axis
# ==========================================================================

# Width sweep, chunk SAE, layer 15, v1 config, 25K steps. Values are the
# full-width train FVU and the holdout L0 printed by the sweep, read from the
# tracked log logs/v2stats_then_chain6.log (runs w4096/w16384/w65536,
# tags from sweeps/r8_width.txt). The 8192 and 32768 points come from the
# tracked run summaries reports/grad_8192_control.json and reports/grad_main.json;
# the 32768 L0*=64 point is grad_l064 in reports/tuning_log.md.
# The same numbers are tabulated in reports/token_vs_chunk.md, width-axis bullet.
WIDTH_SWEEP_L0_32 = (
    (4096, 0.8403, 32.6),
    (8192, 0.8376, 32.0),
    (16384, 0.8068, 32.3),
    (32768, 0.7988, 32.6),
    (65536, 0.7663, 34.5),
)
WIDTH_SWEEP_L0_64 = (
    (4096, 0.7911, 64.6),
    (16384, 0.7504, 64.3),
    (32768, 0.7447, 64.6),
    (65536, 0.7070, 67.4),
)

# L0 axis, chunk SAE v1 config, width 32768. Source: reports/tuning_log.md
# rows grad_l08 / grad_main / grad_l064 / grad_l0128 (L0 and train FVU columns).
CHUNK_V1_L0_CURVE = (
    (8.8, 0.8738),
    (32.6, 0.7988),
    (64.6, 0.7447),
    (129.1, 0.6729),
)
# Tuned config at L0 36: reports/tuning_log.md row r2_g3 (L0 35.6, train 0.7648).
CHUNK_TUNED_L0_36 = (35.6, 0.7648)
# Token SAE lambda sweep, 1 epoch. Source: reports/token_vs_chunk.md section 3
# table (the token-lambda rows: measured L0 and train FVU).
TOKEN_LAM_SWEEP = (
    (41.2, 0.739),
    (45.0, 0.711),
    (56.5, 0.675),
    (83.8, 0.651),
    (89.9, 0.622),
    (104.0, 0.596),
)


def fig_pareto_width_l0(out_path: str) -> None:
    """Two Pareto views of the chunk SAE: dictionary width and sparsity.

    Args:
        out_path: Destination PNG path.
    """
    grad_v2 = _load(os.path.join(REPORTS, "grad_v2.json"))

    fig, (ax_w, ax_l0) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # ---- left: width axis ------------------------------------------------
    for series, label, color, marker in (
        (WIDTH_SWEEP_L0_32, "target L0 = 32 (actual 32-35)", BLUE, "o"),
        (WIDTH_SWEEP_L0_64, "target L0 = 64 (actual 64-67)", ORANGE, "s"),
    ):
        widths = [w for w, _, _ in series]
        fvus = [f for _, f, _ in series]
        ax_w.plot(
            widths,
            fvus,
            color=color,
            marker=marker,
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            label=label,
            markeredgecolor="white",
            markeredgewidth=1.0,
        )
        for w, f, l0 in series:
            ax_w.annotate(
                f"{int(l0 + 0.5)}",
                (w, f),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=8,
                color=INK_MUTED,
            )
    ax_w.set_xscale("log", base=2)
    ax_w.set_xticks([w for w, _, _ in WIDTH_SWEEP_L0_32])
    ax_w.set_xticklabels(
        [f"$2^{{{int(math.log2(w))}}}$" for w, _, _ in WIDTH_SWEEP_L0_32]
    )
    ax_w.set_xlabel("dictionary width (atoms)")
    ax_w.set_ylabel("train FVU (in-sample)")
    ax_w.set_title(
        "Width axis: no saturation up to 64K atoms\n(chunk SAE, layer 15, 25K steps; labels = actual L0)",
        fontsize=11,
    )
    ax_w.legend(frameon=False, fontsize=9, loc="lower left")
    _style_axes(ax_w, grid_axis="both")

    # ---- right: L0 axis --------------------------------------------------
    ax_l0.plot(
        [l0 for l0, _ in CHUNK_V1_L0_CURVE],
        [f for _, f in CHUNK_V1_L0_CURVE],
        color=AQUA,
        marker="o",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        label="chunk 32K, v1 config",
        markeredgecolor="white",
        markeredgewidth=1.0,
    )
    ax_l0.plot(
        [grad_v2["hold"]["l0"]],
        [grad_v2["train"]["fvu_32768"]],
        linestyle="none",
        marker="*",
        markersize=MARKER_SIZE + 10,
        color=AQUA,
        markeredgecolor="white",
        markeredgewidth=1.0,
        label="chunk 32K, v2 tuned",
    )
    ax_l0.plot(
        [CHUNK_TUNED_L0_36[0]],
        [CHUNK_TUNED_L0_36[1]],
        linestyle="none",
        marker="P",
        markersize=MARKER_SIZE + 4,
        color=AQUA,
        markeredgecolor="white",
        markeredgewidth=1.0,
        label="chunk 32K, tuned at L0 36",
    )
    ax_l0.plot(
        [l0 for l0, _ in TOKEN_LAM_SWEEP],
        [f for _, f in TOKEN_LAM_SWEEP],
        color=VIOLET,
        marker="D",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        linestyle="--",
        label="token 32K (lambda sweep, 1 epoch)",
        markeredgecolor="white",
        markeredgewidth=1.0,
    )
    ax_l0.set_xscale("log")
    ax_l0.set_xlabel("L0 (actual mean active atoms)")
    ax_l0.set_ylabel("train FVU (in-sample)")
    ax_l0.set_title(
        "Sparsity axis: the token dictionary reconstructs better\nat matched L0 (layer 15)",
        fontsize=11,
    )
    ax_l0.legend(frameon=False, fontsize=9)
    _style_axes(ax_l0, grid_axis="both")

    fig.tight_layout()
    _save(fig, out_path)


# ==========================================================================
# Figure 6 (regen): FVU-L0 frontier across arms
# ==========================================================================

# grad arm, width 32768, v1 config. Columns L0 / train FVU / holdout FVU /
# train FVU at the 8192 prefix, from reports/tuning_log.md rows
# grad_l08, grad_main, grad_l064, grad_l0128.
GRAD_FRONTIER = (
    (8.8, 0.8738, 0.9160, 0.8949),
    (32.6, 0.7988, 0.8418, 0.8169),
    (64.6, 0.7447, 0.7865, 0.7604),
    (129.1, 0.6729, 0.7101, 0.6851),
)
# Soft reference line. This annotation is carried over from the original
# committed figure; no tracked JSON records the estimate, so it is reproduced
# here as a labelled constant rather than recomputed. The cross-fit machinery
# behind it is the cross-fit significance routine, which is not included.
SOFT_CEILING_FVU = 0.84


def fig_fvu_l0_frontier(out_path: str) -> None:
    """FVU vs L0 for every dictionary arm at layer 15.

    More than three series share this scatter, so each arm also carries its own
    marker shape and line style; colour is never the only cue.

    Args:
        out_path: Destination PNG path.
    """
    arms = {
        name: _load(os.path.join(REPORTS, f"{name}.json"))
        for name in (
            "rawgrad_main",
            "act_main",
            "err_main",
            "grad_8192_control",
            "unitnorm_8192",
        )
    }

    fig, ax = plt.subplots(figsize=(8.8, 5.6))

    l0s = [row[0] for row in GRAD_FRONTIER]
    ax.plot(
        l0s,
        [row[2] for row in GRAD_FRONTIER],
        color=BLUE,
        marker="s",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        markeredgecolor="white",
        markeredgewidth=1.0,
        label="grad (whitened), 32K, holdout",
    )
    ax.plot(
        l0s,
        [row[1] for row in GRAD_FRONTIER],
        color=BLUE,
        marker="o",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        linestyle="--",
        alpha=0.75,
        markeredgecolor="white",
        markeredgewidth=1.0,
        label="grad (whitened), 32K, train",
    )
    ax.plot(
        l0s,
        [row[3] for row in GRAD_FRONTIER],
        color=BLUE,
        marker="^",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        linestyle=":",
        alpha=0.6,
        markeredgecolor="white",
        markeredgewidth=1.0,
        label="grad, train at the 8192 prefix",
    )

    singles = (
        ("rawgrad_main", "fvu_32768", ORANGE, "^", "rawgrad (no whitening), holdout"),
        ("act_main", "fvu_32768", AQUA, "D", "activation, holdout"),
        ("err_main", "fvu_32768", VIOLET, "v", "output-error, holdout"),
        (
            "grad_8192_control",
            "fvu_8192",
            INK_MUTED,
            "X",
            "grad 8192 (independent run), holdout",
        ),
        ("unitnorm_8192", "fvu_8192", INK_MUTED, "P", "unit-norm grad 8192, holdout"),
    )
    for name, key, color, marker, label in singles:
        hold = arms[name]["hold"]
        ax.plot(
            [hold["l0"]],
            [hold[key]],
            linestyle="none",
            marker=marker,
            markersize=MARKER_SIZE + 3,
            color=color,
            markeredgecolor="white",
            markeredgewidth=1.0,
            label=label,
        )

    ax.axhline(SOFT_CEILING_FVU, color=INK_MUTED, linestyle="--", linewidth=1.2)
    ax.text(
        0.985,
        0.855,
        f"cross-fit reproducible-structure estimate ({SOFT_CEILING_FVU}, soft ceiling)",
        transform=ax.transAxes,
        fontsize=8.5,
        color=INK_MUTED,
        ha="right",
    )

    ax.set_xscale("log")
    ax.set_xlabel("L0 (holdout mean active atoms)")
    ax.set_ylabel("FVU (full width)")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "FVU-L0 frontier, layer 15, doc-split holdout\n"
        "activation and output-error reconstruct far more easily than gradients",
        fontsize=12,
    )
    ax.legend(frameon=False, fontsize=8.5, loc="center right")
    _style_axes(ax, grid_axis="both")

    fig.tight_layout()
    _save(fig, out_path)


# ==========================================================================
# registry / CLI
# ==========================================================================

FIGURES: dict[str, tuple[Callable[[str], None], str]] = {
    "rq2_channelB": (fig_rq2_channel_b, os.path.join(REPORTS, "rq2_channelB.png")),
    "alpha_dose": (fig_alpha_dose, os.path.join(FEEDBACK, "alpha_dose.png")),
    "E11_dose_response": (
        fig_e11_dose_response,
        os.path.join(REGEN, "E11_dose_response.png"),
    ),
    "gradient_usefulness_summary": (
        fig_gradient_usefulness_summary,
        os.path.join(REGEN, "gradient_usefulness_summary.png"),
    ),
    "pareto_width_l0": (
        fig_pareto_width_l0,
        os.path.join(REGEN, "pareto_width_l0.png"),
    ),
    "fvu_l0_frontier": (
        fig_fvu_l0_frontier,
        os.path.join(REGEN, "fvu_l0_frontier.png"),
    ),
}


def main() -> None:
    """Parse arguments and render the requested figures."""
    parser = argparse.ArgumentParser(
        description="Regenerate write-up figures from git-tracked inputs only."
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

    names = [args.only] if args.only else list(FIGURES)
    for name in names:
        render, path = FIGURES[name]
        render(path)


if __name__ == "__main__":
    main()
