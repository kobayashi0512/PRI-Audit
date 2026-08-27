"""Reference-guided PRI-Audit technical route and module architecture.

The figures follow the supplied visual language: large stage containers,
nested functional modules, compact biological/data icons, semantic blue/green/
orange accents, and explicit left-to-right information flow. Outputs are fully
editable SVG/PDF plus high-resolution PNG.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/paper_v18/main"
CACHE = ROOT / "results/paper_v18"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

PAPER = "#FBFCFC"
INK = "#20282C"
MUTED = "#617078"
EDGE = "#AABBC0"
ARROW = "#69777C"
BLUE = "#6EA5B7"
BLUE_DARK = "#397A91"
BLUE_FILL = "#E5F1F5"
GREEN = "#75A98F"
GREEN_FILL = "#E8F3EC"
ORANGE = "#D69B5F"
ORANGE_DARK = "#A96E35"
ORANGE_FILL = "#FFF0E1"
LILAC = "#8F87AF"
LILAC_FILL = "#EEEAF5"
GRAY_FILL = "#F1F4F4"
WHITE = "#FFFFFF"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7,
    "text.color": INK,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def save(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=500, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def rr(ax, x, y, w, h, *, fc=WHITE, ec=EDGE, lw=1.0, radius=0.018, shadow=False):
    if shadow:
        ax.add_patch(FancyBboxPatch((x + 0.004, y - 0.006), w, h,
            boxstyle=f"round,pad=0.009,rounding_size={radius}", transform=ax.transAxes,
            facecolor="#D7DFE1", edgecolor="none", alpha=0.45, clip_on=False))
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.009,rounding_size={radius}",
        transform=ax.transAxes, facecolor=fc, edgecolor=ec, linewidth=lw, clip_on=False)
    ax.add_patch(p)
    return p


def arr(ax, start, end, *, color=ARROW, lw=1.2, curve=0.0, ms=10, style="-|>"):
    ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes,
        arrowstyle=style, mutation_scale=ms, linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={curve}", clip_on=False))


def stage_title(ax, x, y, label, subtitle, color):
    stage, name = label.split(" · ", 1)
    ax.text(x, y, stage, transform=ax.transAxes, fontsize=5.4, fontweight="bold",
            ha="left", va="top", color=color)
    ax.text(x, y - 0.028, name, transform=ax.transAxes, fontsize=7.4, fontweight="bold",
            ha="left", va="top", color=INK)
    ax.text(x, y - 0.063, subtitle, transform=ax.transAxes, fontsize=4.6,
            ha="left", va="top", color=color)


def pill(ax, x, y, w, text, fc, color=INK):
    rr(ax, x, y, w, 0.034, fc=fc, ec=fc, lw=0, radius=0.012)
    ax.text(x + w / 2, y + 0.017, text, transform=ax.transAxes, fontsize=4.7,
            ha="center", va="center", color=color, fontweight="bold")


def patient_icon(ax, cx, cy, color):
    ax.add_patch(Circle((cx, cy + 0.018), 0.012, transform=ax.transAxes,
                        facecolor=WHITE, edgecolor=color, linewidth=1.0))
    ax.add_patch(FancyBboxPatch((cx - 0.022, cy - 0.024), 0.044, 0.027,
        boxstyle="round,pad=0.002,rounding_size=0.012", transform=ax.transAxes,
        facecolor=WHITE, edgecolor=color, linewidth=1.0))


def matrix_icon(ax, x, y, color, rows=4, cols=5):
    ax.add_patch(Rectangle((x, y), 0.058, 0.048, transform=ax.transAxes,
                           facecolor=WHITE, edgecolor=color, linewidth=0.8))
    for i in range(1, cols):
        ax.plot([x + i * 0.058 / cols] * 2, [y, y + 0.048], transform=ax.transAxes,
                color=color, lw=0.35)
    for j in range(1, rows):
        ax.plot([x, x + 0.058], [y + j * 0.048 / rows] * 2, transform=ax.transAxes,
                color=color, lw=0.35)


def cell_cloud(ax, cx, cy, color):
    pts = [(-0.026, 0.004), (-0.012, 0.022), (0.009, 0.018), (0.026, -0.002),
           (0.006, -0.021), (-0.020, -0.018), (0.000, 0.001)]
    for dx, dy in pts:
        ax.add_patch(Circle((cx + dx, cy + dy), 0.0075, transform=ax.transAxes,
                            facecolor=color, edgecolor=WHITE, linewidth=0.5))


def tissue_icon(ax, x, y, color):
    rng = np.random.default_rng(18)
    xx = np.linspace(x, x + 0.070, 30)
    yy = y + 0.013 + 0.008 * np.sin(np.linspace(0, 2.5 * np.pi, 30))
    ax.fill_between(xx, yy - 0.012, yy + 0.014, transform=ax.transAxes,
                    color=color, alpha=0.40, linewidth=0)
    ax.plot(xx, yy + 0.014, transform=ax.transAxes, color=color, lw=0.8)
    ax.scatter(rng.uniform(x + 0.005, x + 0.065, 12), rng.uniform(y + 0.003, y + 0.032, 12),
               transform=ax.transAxes, s=3.5, color=color, alpha=0.8, linewidths=0)


def null_hist_icon(ax, x, y, color):
    heights = [0.010, 0.022, 0.035, 0.050, 0.038, 0.024, 0.013]
    for i, h in enumerate(heights):
        ax.add_patch(Rectangle((x + i * 0.009, y), 0.007, h, transform=ax.transAxes,
                               facecolor=color, edgecolor="none", alpha=0.72))
    ax.plot([x + 0.071, x + 0.071], [y, y + 0.055], transform=ax.transAxes,
            color=ORANGE_DARK, lw=1.0)


def mini_box(ax, x, y, w, h, title, lines, color, fill, icon=None):
    rr(ax, x, y, w, h, fc=fill, ec=color, lw=0.9, radius=0.014, shadow=True)
    tx = x + 0.018
    if icon:
        icon(ax, x + 0.040, y + h / 2, color)
        line_x = x + 0.083
    else:
        line_x = x + 0.018
    ax.text(tx, y + h - 0.030, title, transform=ax.transAxes, fontsize=5.15,
            fontweight="bold", ha="left", va="center")
    for i, line in enumerate(lines):
        ax.text(line_x, y + h - 0.067 - i * 0.025, line, transform=ax.transAxes,
                fontsize=3.95, color=MUTED, ha="left", va="center")


def plot_technical_route():
    fig, ax = plt.subplots(figsize=(8.4, 5.15), facecolor=PAPER)
    ax.axis("off")
    fig.text(0.045, 0.965, "PRI-Audit technical route for public psoriasis transcriptomes",
             fontsize=13.0, fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.045, 0.920,
             "Public data → patient-isolated representations → matched testing → falsification-bounded evidence",
             fontsize=7.0, ha="left", va="top", color=MUTED)

    # Four stage containers.
    stages = [
        (0.035, 0.135, 0.205, 0.710, BLUE_FILL, BLUE, "STAGE A · DATA FOUNDATION", "11 public datasets · fixed analytical roles"),
        (0.270, 0.135, 0.245, 0.710, LILAC_FILL, LILAC, "STAGE B · PRI-AUDIT CONSTRUCTION", "fold-local candidate and reference paths"),
        (0.545, 0.135, 0.205, 0.710, GREEN_FILL, GREEN, "STAGE C · STRICT VALIDATION", "independent groups · paired uncertainty"),
        (0.780, 0.135, 0.185, 0.710, ORANGE_FILL, ORANGE, "STAGE D · EVIDENCE OUTPUT", "allowed claims, not score chasing"),
    ]
    for x, y, w, h, fc, ec, title, subtitle in stages:
        rr(ax, x, y, w, h, fc=fc, ec=ec, lw=1.0, radius=0.020, shadow=True)
        stage_title(ax, x + 0.014, y + h - 0.020, title, subtitle, ec)

    # Stage A.
    mini_box(ax, 0.052, 0.615, 0.171, 0.128, "Paired bulk · 7 cohorts",
             ["191 paired patients", "direction transfer"], BLUE, WHITE, patient_icon)
    mini_box(ax, 0.052, 0.430, 0.171, 0.128, "Single-cell · 2 cohorts",
             ["development + external", "cells remain nested"], BLUE, WHITE, cell_cloud)
    mini_box(ax, 0.052, 0.245, 0.171, 0.128, "Spatial + perturbation",
             ["12 paired patients", "IL-17 / TNF prior"], BLUE, WHITE, None)
    tissue_icon(ax, 0.065, 0.268, BLUE)
    pill(ax, 0.058, 0.165, 0.073, "patient / donor", WHITE, BLUE_DARK)
    pill(ax, 0.138, 0.165, 0.074, "fixed roles", WHITE, BLUE_DARK)

    # Stage B.
    mini_box(ax, 0.290, 0.620, 0.205, 0.113, "M1–M2 · Information boundary",
             ["lock group folds", "refit adaptive objects in training"], LILAC, WHITE, None)
    mini_box(ax, 0.290, 0.458, 0.205, 0.113, "M3 · Resolution encoders",
             ["expression · PCA/NMF", "cell neighborhoods · spatial graph"], LILAC, WHITE, None)
    mini_box(ax, 0.290, 0.296, 0.205, 0.113, "M4 · Matched comparator",
             ["same groups, folds and gene budget", "candidate − expression"], LILAC, WHITE, None)
    ax.text(0.392, 0.228, "ΔAUROC", transform=ax.transAxes, ha="center", va="center",
            fontsize=10.0, color=LILAC, fontweight="bold")
    ax.text(0.392, 0.190, "paired out-of-fold predictions", transform=ax.transAxes,
            ha="center", fontsize=4.8, color=MUTED)
    arr(ax, (0.392, 0.620), (0.392, 0.576), color=LILAC)
    arr(ax, (0.392, 0.458), (0.392, 0.414), color=LILAC)
    arr(ax, (0.392, 0.296), (0.392, 0.250), color=LILAC)

    # Stage C.
    mini_box(ax, 0.563, 0.625, 0.170, 0.105, "Patient-level resampling",
             ["paired prediction bootstrap", "CI for ΔAUROC"], GREEN, WHITE, patient_icon)
    mini_box(ax, 0.563, 0.475, 0.170, 0.105, "External transfer",
             ["frozen GSE151177 → GSE173706", "calibration + Brier"], GREEN, WHITE, None)
    mini_box(ax, 0.563, 0.325, 0.170, 0.105, "Spatial strictness",
             ["patient LODO", "coordinate shuffle"], GREEN, WHITE, None)
    mini_box(ax, 0.563, 0.175, 0.170, 0.105, "Resolution-specific nulls",
             ["labels · cells · coordinates", "complete distributions"], GREEN, WHITE, None)
    null_hist_icon(ax, 0.650, 0.188, GREEN)
    for y in (0.625, 0.475, 0.325):
        arr(ax, (0.648, y), (0.648, y - 0.040), color=GREEN)

    # Stage D.
    rr(ax, 0.800, 0.605, 0.145, 0.132, fc=WHITE, ec=ORANGE, lw=1.0, radius=0.014, shadow=True)
    ax.text(0.872, 0.695, "Evidence gate", transform=ax.transAxes, ha="center",
            fontsize=7.0, fontweight="bold")
    for i, text in enumerate(["patient isolated", "outer-fit strict", "stable CI", "null passed"]):
        pill(ax, 0.812, 0.646 - i * 0.027, 0.121, text, ORANGE_FILL, ORANGE_DARK)
    rr(ax, 0.800, 0.405, 0.145, 0.135, fc=WHITE, ec=ORANGE, lw=1.0, radius=0.014, shadow=True)
    ax.text(0.872, 0.500, "Bounded conclusions", transform=ax.transAxes, ha="center",
            fontsize=5.9, fontweight="bold")
    ax.text(0.872, 0.463, "disease signal ≠\nresolution-specific gain", transform=ax.transAxes,
            ha="center", va="center", fontsize=5.1, color=MUTED, linespacing=1.25)
    rr(ax, 0.800, 0.205, 0.145, 0.135, fc=WHITE, ec=ORANGE, lw=1.0, radius=0.014, shadow=True)
    ax.text(0.872, 0.300, "Reproducible outputs", transform=ax.transAxes, ha="center",
            fontsize=5.9, fontweight="bold")
    ax.text(0.872, 0.257, "predictions · bootstrap arrays\nnulls · claim registry", transform=ax.transAxes,
            ha="center", va="center", fontsize=4.8, color=MUTED, linespacing=1.3)
    arr(ax, (0.872, 0.605), (0.872, 0.545), color=ORANGE)
    arr(ax, (0.872, 0.405), (0.872, 0.345), color=ORANGE)

    # Cross-stage arrows and footer.
    arr(ax, (0.240, 0.500), (0.267, 0.500), color=ARROW, lw=1.8, ms=12)
    arr(ax, (0.515, 0.500), (0.542, 0.500), color=ARROW, lw=1.8, ms=12)
    arr(ax, (0.750, 0.500), (0.777, 0.500), color=ARROW, lw=1.8, ms=12)
    ax.text(0.035, 0.075,
            "Frozen claim: added cellular or spatial resolution is valuable only when it improves the same held-out patients beyond matched expression and survives a targeted falsification.",
            transform=ax.transAxes, fontsize=5.2, color=MUTED, ha="left")
    save(fig, "fig1_pri_audit_technical_route")


def module_card(ax, x, y, w, h, code, title, subtitle, rows, color, fill, icon_kind):
    rr(ax, x, y, w, h, fc=fill, ec=color, lw=1.0, radius=0.017, shadow=True)
    ax.text(x + 0.016, y + h - 0.031, code, transform=ax.transAxes,
            fontsize=4.7, fontweight="bold", ha="left", va="center", color=color)
    ax.text(x + 0.050, y + h - 0.031, title, transform=ax.transAxes,
            fontsize=5.25, fontweight="bold", ha="left", va="center", linespacing=0.95)
    ax.text(x + 0.018, y + h - 0.064, subtitle, transform=ax.transAxes,
            fontsize=3.45 if w < 0.16 else 3.85, color=MUTED, ha="left", va="center")
    compact = w < 0.16
    if not compact:
        if icon_kind == "patient":
            patient_icon(ax, x + 0.050, y + 0.060, color)
        elif icon_kind == "matrix":
            matrix_icon(ax, x + 0.022, y + 0.038, color)
        elif icon_kind == "cells":
            cell_cloud(ax, x + 0.050, y + 0.060, color)
        elif icon_kind == "null":
            null_hist_icon(ax, x + 0.020, y + 0.033, color)
    for index, (label, value) in enumerate(zip(("IN", "DO", "OUT"), rows)):
        yy = y + 0.082 - index * 0.026
        label_x = x + (0.018 if compact else 0.086)
        value_x = x + (0.048 if compact else 0.112)
        ax.text(label_x, yy, label, transform=ax.transAxes, fontsize=3.55,
                color=color, fontweight="bold", ha="left", va="center")
        ax.text(value_x, yy, value, transform=ax.transAxes, fontsize=3.25 if compact else 3.45,
                color=MUTED, ha="left", va="center")


def plot_module_architecture():
    fig, ax = plt.subplots(figsize=(8.4, 5.25), facecolor=PAPER)
    ax.axis("off")
    fig.text(0.045, 0.965, "PRI-Audit algorithm architecture and module functions",
             fontsize=13.0, fontweight="bold", ha="left", va="top")
    fig.text(0.045, 0.920,
             "Auditable artifacts move left to right; the held-out group bypasses all fitting operations",
             fontsize=7.0, color=MUTED, ha="left", va="top")

    # Patient boundary strip.
    rr(ax, 0.040, 0.760, 0.920, 0.105, fc=BLUE_FILL, ec=BLUE, lw=1.0, radius=0.018, shadow=True)
    patient_icon(ax, 0.075, 0.812, BLUE_DARK)
    ax.text(0.110, 0.825, "Independent unit", transform=ax.transAxes, fontsize=6.4,
            fontweight="bold", ha="left")
    ax.text(0.110, 0.790, "patient / donor controls the outer fold, bootstrap and permissible inference",
            transform=ax.transAxes, fontsize=5.0, color=MUTED, ha="left")
    pill(ax, 0.730, 0.795, 0.098, "training groups", WHITE, BLUE_DARK)
    pill(ax, 0.838, 0.795, 0.100, "held-out group", WHITE, GREEN)

    # Central cards.
    module_card(ax, 0.040, 0.500, 0.165, 0.185, "M1", "Unit Guard",
                "metadata → fold registry", ("metadata", "lock groups", "fold registry"), BLUE, BLUE_FILL, "patient")
    module_card(ax, 0.230, 0.500, 0.165, 0.185, "M2", "Fold-local Builder",
                "training matrices → fitted objects", ("train matrices", "fit HVG/basis", "fitted objects"), BLUE, BLUE_FILL, "matrix")
    module_card(ax, 0.420, 0.500, 0.200, 0.185, "M3", "Resolution Encoders",
                "fold-local data → sample vectors", ("fitted data", "encode views", "sample vectors"), LILAC, LILAC_FILL, "cells")
    module_card(ax, 0.645, 0.500, 0.145, 0.185, "M4", "Matched\nComparator",
                "paired predictions → ΔAUROC", ("paired vectors", "paired CV", "Δ + CI"), GREEN, GREEN_FILL, "matrix")
    module_card(ax, 0.815, 0.500, 0.145, 0.185, "M5", "Falsification\nEngine",
                "fixed pipeline → null distributions", ("fixed model", "shuffle source", "null array"), ORANGE, ORANGE_FILL, "null")
    for a, b in [((0.205, 0.592), (0.227, 0.592)), ((0.395, 0.592), (0.417, 0.592)),
                 ((0.620, 0.592), (0.642, 0.592)), ((0.790, 0.592), (0.812, 0.592))]:
        arr(ax, a, b, color=ARROW, lw=1.3)

    # Encoder library: visual miniatures, not a text-only rail.
    rr(ax, 0.230, 0.215, 0.560, 0.205, fc=WHITE, ec=LILAC, lw=1.0, radius=0.018, shadow=True)
    ax.text(0.250, 0.386, "M3 encoder library", transform=ax.transAxes, fontsize=6.2,
            fontweight="bold")
    encoders = [
        (0.250, "Expression", "log-CPM", BLUE, "matrix"),
        (0.355, "Low-rank", "PCA / NMF", BLUE, "matrix"),
        (0.460, "Cell states", "equal mass", LILAC, "cells"),
        (0.565, "Spatial graph", "agreement", GREEN, "tissue"),
        (0.670, "Trigger OT", "IL-17/TNF", ORANGE, "cells"),
    ]
    for x, name, detail, color, kind in encoders:
        rr(ax, x, 0.285, 0.085, 0.078, fc=GRAY_FILL, ec=color, lw=0.65, radius=0.010)
        if kind == "matrix": matrix_icon(ax, x + 0.011, 0.307, color, rows=3, cols=4)
        elif kind == "cells": cell_cloud(ax, x + 0.040, 0.326, color)
        else: tissue_icon(ax, x + 0.008, 0.304, color)
        ax.text(x + 0.0425, 0.264, name, transform=ax.transAxes, ha="center",
                fontsize=4.5, fontweight="bold", color=color)
        ax.text(x + 0.0425, 0.238, detail, transform=ax.transAxes, ha="center",
                fontsize=4.1, color=MUTED)
    arr(ax, (0.520, 0.500), (0.520, 0.423), color=LILAC)

    # Evidence gate.
    rr(ax, 0.040, 0.090, 0.920, 0.105, fc=GREEN_FILL, ec=GREEN, lw=1.0, radius=0.018, shadow=True)
    ax.text(0.065, 0.155, "M6 · Evidence Gate", transform=ax.transAxes, fontsize=6.7,
            fontweight="bold")
    ax.text(0.065, 0.118, "combines paired gain, group-level uncertainty, calibration and targeted null tests",
            transform=ax.transAxes, fontsize=4.8, color=MUTED)
    for x, text, color in [(0.520, "SUPPORTED", GREEN), (0.650, "BOUNDED", BLUE_DARK),
                           (0.780, "REFUTED", ORANGE_DARK)]:
        pill(ax, x, 0.120, 0.110, text, WHITE, color)
    arr(ax, (0.718, 0.500), (0.718, 0.198), color=GREEN, curve=0.07)
    arr(ax, (0.888, 0.500), (0.888, 0.198), color=ORANGE, curve=-0.05)
    ax.text(0.040, 0.035,
            "Auditable handoff: fold registry → fitted transforms → sample vectors → paired predictions → bootstrap/null arrays → claim registry",
            transform=ax.transAxes, fontsize=5.2, color=MUTED)
    save(fig, "fig2_pri_audit_module_architecture")


def main():
    plot_technical_route()
    plot_module_architecture()
    brief = {
        "reference_logic": "example analysis → content extraction → technical route → self-check",
        "reference_style": ["stage containers", "nested modules", "micro-icons", "semantic pastel palette", "thick directional arrows"],
        "figure_1": "data foundation → PRI-Audit construction → strict validation → evidence output",
        "figure_2": "M1–M6 module architecture with held-out bypass and encoder library",
        "checks": ["clear reading order", "no held-out flow into fitting", "editable SVG", "no invented measurements", "single-column readable"],
    }
    (CACHE / "v18_reference_guided_figure_brief.json").write_text(json.dumps(brief, indent=2), encoding="utf-8")
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
