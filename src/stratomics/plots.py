"""Publication-style plotting helpers (generic, journal-neutral defaults).

Provides a small, reusable NPG-flavoured palette and theme plus three plot
types used across the pipeline: a consensus heatmap, grouped score boxplots,
and Kaplan-Meier curves. All figures are saved as both PNG and PDF.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# A 10-colour qualitative palette in the Nature Publishing Group style.
NPG_PALETTE = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
    "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
]


def apply_theme() -> None:
    """Apply a clean, publication-friendly matplotlib theme."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
    })


def palette_for(labels) -> dict:
    """Map a sequence of category labels to stable palette colours."""
    uniq = list(dict.fromkeys(labels))
    return {lab: NPG_PALETTE[i % len(NPG_PALETTE)] for i, lab in enumerate(uniq)}


def _save(fig, out_prefix: str | Path) -> list[Path]:
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out_prefix.with_suffix(f".{ext}")
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def consensus_heatmap(consensus: pd.DataFrame, labels: pd.Series, out_prefix: str | Path) -> list[Path]:
    """Heatmap of the consensus matrix, samples ordered by subtype."""
    apply_theme()
    order = labels.sort_values().index
    mat = consensus.loc[order, order]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(mat.to_numpy(), cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Consensus matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="co-clustering frequency")
    return _save(fig, out_prefix)


def group_boxplots(scores: pd.DataFrame, groups: pd.Series, out_prefix: str | Path, *, max_panels: int = 12) -> list[Path]:
    """Grouped boxplots of signature scores across subtypes."""
    apply_theme()
    common = scores.index.intersection(groups.index)
    scores = scores.loc[common]
    groups = groups.loc[common]
    sigs = list(scores.columns)[:max_panels]
    levels = list(dict.fromkeys(groups.tolist()))
    colors = palette_for(levels)

    ncol = min(4, len(sigs)) or 1
    nrow = int(np.ceil(len(sigs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.6 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for i, sig in enumerate(sigs):
        ax = axes.flat[i]
        ax.set_visible(True)
        data = [scores.loc[groups.index[groups == lvl], sig].to_numpy() for lvl in levels]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
        for patch, lvl in zip(bp["boxes"], levels):
            patch.set_facecolor(colors[lvl])
            patch.set_alpha(0.8)
        for med in bp["medians"]:
            med.set_color("black")
        ax.set_xticks(range(1, len(levels) + 1))
        ax.set_xticklabels(levels, rotation=0)
        ax.set_title(sig, fontsize=9)
    fig.tight_layout()
    return _save(fig, out_prefix)


def km_curves(curves: dict[str, pd.DataFrame], pvalue: float, out_prefix: str | Path) -> list[Path]:
    """Kaplan-Meier curves with the log-rank p-value annotated."""
    apply_theme()
    colors = palette_for(list(curves.keys()))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for lvl, sf in curves.items():
        surv_col = [c for c in sf.columns if "lower" not in c and "upper" not in c][0]
        ax.step(sf.index, sf[surv_col], where="post", label=lvl, color=colors[lvl], lw=1.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.set_title("Kaplan-Meier by subtype")
    ax.legend(title=None, loc="best", fontsize=8)
    ax.text(0.04, 0.08, f"log-rank p = {pvalue:.3g}", transform=ax.transAxes, fontsize=9)
    return _save(fig, out_prefix)
