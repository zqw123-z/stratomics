"""Between-group comparison and landscape characterization.

Two generic operations:

- :func:`compare_features` tests every feature for differences across groups
  (Kruskal-Wallis for >2 groups, Mann-Whitney for 2), with Benjamini-Hochberg
  FDR control and an effect-size column.
- :func:`landscape` scores a second collection of gene sets (e.g. immune cell
  signatures supplied by the user) per sample and summarizes them per group.

Nothing here is specific to any tissue or signature; the "immune landscape" is
just whatever gene sets the caller passes in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .score import score_signatures


def _align(values: pd.DataFrame, groups: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    common = values.columns.intersection(groups.index)
    if len(common) < 3:
        raise ValueError("fewer than 3 samples shared between matrix and group labels")
    return values[common], groups.loc[common]


def compare_features(
    expr: pd.DataFrame,
    groups: pd.Series,
    *,
    fdr: float = 0.05,
    min_group: int = 2,
) -> pd.DataFrame:
    """Differential test of each feature across sample groups.

    ``expr`` is features-by-samples; ``groups`` maps sample -> group label.
    Returns a per-feature table sorted by adjusted p-value.
    """
    expr, groups = _align(expr, groups)
    levels = [g for g, n in groups.value_counts().items() if n >= min_group]
    if len(levels) < 2:
        raise ValueError("need at least two groups with sufficient samples")

    sample_lists = {lvl: groups.index[groups == lvl] for lvl in levels}
    mat = expr.loc[:, groups.index]
    records = []
    for feat in mat.index:
        row = mat.loc[feat]
        arrays = [row[sample_lists[lvl]].to_numpy(dtype=float) for lvl in levels]
        means = {f"mean_{lvl}": float(np.mean(a)) for lvl, a in zip(levels, arrays)}
        try:
            if len(levels) == 2:
                stat, p = stats.mannwhitneyu(arrays[0], arrays[1], alternative="two-sided")
                effect = means[f"mean_{levels[0]}"] - means[f"mean_{levels[1]}"]
            else:
                stat, p = stats.kruskal(*arrays)
                effect = float(np.max([np.mean(a) for a in arrays]) - np.min([np.mean(a) for a in arrays]))
        except ValueError:
            stat, p, effect = np.nan, 1.0, 0.0
        records.append({"feature": feat, "stat": stat, "pvalue": p, "effect": effect, **means})

    res = pd.DataFrame.from_records(records).set_index("feature")
    valid = res["pvalue"].notna()
    res["padj"] = 1.0
    if valid.any():
        res.loc[valid, "padj"] = multipletests(res.loc[valid, "pvalue"], method="fdr_bh")[1]
    res["significant"] = res["padj"] < fdr
    return res.sort_values("padj")


def landscape(
    expr: pd.DataFrame,
    groups: pd.Series,
    genesets: dict[str, list[str]],
    *,
    method: str = "ssgsea",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score landscape gene sets per sample and summarize per group.

    Returns ``(per_sample_scores, per_group_summary)`` where the summary is a
    signatures-by-groups table of group-mean scores plus a Kruskal/Mann-Whitney
    p-value column across groups.
    """
    scores = score_signatures(expr, genesets, method=method)
    scores, groups = _align(scores.T, groups)
    scores = scores.T  # samples x signatures, aligned to grouped samples

    levels = list(dict.fromkeys(groups.tolist()))
    summary = {}
    for lvl in levels:
        summary[f"mean_{lvl}"] = scores.loc[groups.index[groups == lvl]].mean(axis=0)
    summ = pd.DataFrame(summary)

    pvals = []
    for sig in scores.columns:
        arrays = [scores.loc[groups.index[groups == lvl], sig].to_numpy() for lvl in levels]
        arrays = [a for a in arrays if len(a) >= 2]
        if len(arrays) < 2:
            pvals.append(np.nan)
        elif len(arrays) == 2:
            pvals.append(stats.mannwhitneyu(arrays[0], arrays[1], alternative="two-sided")[1])
        else:
            pvals.append(stats.kruskal(*arrays)[1])
    summ["pvalue"] = pvals
    return scores, summ.sort_values("pvalue")
