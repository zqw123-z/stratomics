"""Signature scoring.

Reduce a features-by-samples matrix to a samples-by-signatures score matrix
using one of three transparent, dependency-light methods:

- ``mean_z``  : per-feature z-score across samples, averaged over set members
- ``rankmean``: per-sample rank of each feature (0..1), averaged over set members
- ``ssgsea``  : single-sample GSEA enrichment score (Barbie et al., 2009)

The choice of method, the gene sets, and any normalization are all caller
supplied. No set membership or feature identity is assumed here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METHODS = ("mean_z", "rankmean", "ssgsea")


def _members(expr: pd.DataFrame, genes: list[str]) -> list[str]:
    present = [g for g in genes if g in expr.index]
    return present


def _mean_z(expr: pd.DataFrame, genes: list[str]) -> pd.Series:
    sub = expr.loc[genes]
    z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1).replace(0, np.nan), axis=0)
    return z.mean(axis=0, skipna=True)


def _rankmean(expr: pd.DataFrame, genes: list[str]) -> pd.Series:
    ranks = expr.rank(axis=0, method="average")  # rank features within each sample
    ranks = ranks / float(expr.shape[0])  # normalize to (0, 1]
    return ranks.loc[genes].mean(axis=0)


def _ssgsea_sample(values: np.ndarray, in_set: np.ndarray, alpha: float = 0.25) -> float:
    """Single-sample GSEA enrichment score for one sample.

    ``values`` are the feature expression values for the sample and ``in_set``
    is a boolean membership mask aligned to ``values``.
    """
    order = np.argsort(values, kind="mergesort")[::-1]  # high -> low expression
    in_ordered = in_set[order]
    ranks = np.abs(values[order]) ** alpha

    hits = ranks * in_ordered
    sum_hits = hits.sum()
    n_miss = np.count_nonzero(~in_ordered)
    if sum_hits == 0 or n_miss == 0:
        return 0.0
    p_in = np.cumsum(hits) / sum_hits
    p_out = np.cumsum((~in_ordered).astype(float)) / n_miss
    return float(np.sum(p_in - p_out))


def _ssgsea(expr: pd.DataFrame, genes: list[str], alpha: float = 0.25) -> pd.Series:
    membership = expr.index.isin(set(genes))
    mat = expr.to_numpy(dtype=float)
    scores = [_ssgsea_sample(mat[:, j], membership, alpha) for j in range(mat.shape[1])]
    return pd.Series(scores, index=expr.columns)


def score_signatures(
    expr: pd.DataFrame,
    genesets: dict[str, list[str]],
    method: str = "ssgsea",
    *,
    alpha: float = 0.25,
    min_genes: int = 1,
    zscore_output: bool = False,
) -> pd.DataFrame:
    """Score every signature in ``genesets`` for every sample in ``expr``.

    Parameters
    ----------
    expr:
        Features-by-samples expression matrix.
    genesets:
        Mapping of signature name -> list of feature identifiers.
    method:
        One of :data:`METHODS`.
    alpha:
        Exponent used by ``ssgsea`` only.
    min_genes:
        Skip a signature if fewer than this many members are present.
    zscore_output:
        If True, z-score each signature column across samples (comparable scale).

    Returns
    -------
    DataFrame
        Samples (rows) by signatures (columns).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    cols: dict[str, pd.Series] = {}
    skipped: list[str] = []
    for name, genes in genesets.items():
        present = _members(expr, genes)
        if len(present) < max(1, min_genes):
            skipped.append(name)
            continue
        if method == "mean_z":
            cols[name] = _mean_z(expr, present)
        elif method == "rankmean":
            cols[name] = _rankmean(expr, present)
        else:
            cols[name] = _ssgsea(expr, present, alpha=alpha)

    if not cols:
        raise ValueError(
            "no signature had enough members present in the expression matrix; "
            "check that feature identifiers match between --expr and --genesets"
        )
    out = pd.DataFrame(cols)
    if zscore_output:
        out = (out - out.mean(axis=0)) / out.std(axis=0).replace(0, np.nan)
        out = out.fillna(0.0)
    out.index.name = expr.columns.name or "sample"
    return out
