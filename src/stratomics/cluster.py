"""Consensus clustering for subtype discovery.

Repeatedly subsample the samples, cluster each subsample, and accumulate a
consensus matrix recording how often each pair lands together. A final
clustering of the consensus matrix yields stable subtype labels. The number of
subtypes ``k`` is chosen automatically over a caller-defined range using the
average silhouette on the consensus distance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score


def _agglomerative(n_clusters: int):
    """Construct AgglomerativeClustering with a precomputed-distance metric.

    sklearn renamed ``affinity`` to ``metric`` in 1.2; support both.
    """
    try:
        return AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage="average"
        )
    except TypeError:  # pragma: no cover - older sklearn
        return AgglomerativeClustering(
            n_clusters=n_clusters, affinity="precomputed", linkage="average"
        )


@dataclass
class ConsensusResult:
    """Outcome of :func:`consensus_cluster` for a single chosen ``k``."""

    k: int
    labels: pd.Series
    consensus: pd.DataFrame
    silhouette: float
    scores_by_k: dict[int, float] = field(default_factory=dict)


def _one_k(data: np.ndarray, k: int, n_resample: int, resample_frac: float, seed: int):
    n = data.shape[0]
    consensus = np.zeros((n, n))
    counts = np.zeros((n, n))
    rng = np.random.RandomState(seed)
    size = max(k + 1, int(round(resample_frac * n)))
    size = min(size, n)
    for _ in range(n_resample):
        idx = rng.choice(n, size=size, replace=False)
        sub = data[idx]
        km = KMeans(n_clusters=k, n_init=10, random_state=rng.randint(0, 2**31 - 1))
        labels = km.fit_predict(sub)
        eq = (labels[:, None] == labels[None, :]).astype(float)
        grid = np.ix_(idx, idx)
        consensus[grid] += eq
        counts[grid] += 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.divide(consensus, counts, out=np.zeros_like(consensus), where=counts > 0)
    np.fill_diagonal(m, 1.0)
    distance = 1.0 - m
    final = _agglomerative(k).fit_predict(distance)
    sil = float(silhouette_score(distance, final, metric="precomputed"))
    return final, m, sil


def consensus_cluster(
    data: pd.DataFrame,
    *,
    k_min: int = 2,
    max_k: int = 6,
    n_resample: int = 50,
    resample_frac: float = 0.8,
    seed: int = 0,
) -> ConsensusResult:
    """Discover subtypes from a samples-by-features table.

    ``data`` is typically the signature score matrix from :mod:`stratomics.score`
    but any numeric samples-by-features frame works.
    """
    if data.shape[0] < k_min + 1:
        raise ValueError("not enough samples for clustering")
    max_k = min(max_k, data.shape[0] - 1)
    mat = np.nan_to_num(data.to_numpy(dtype=float))

    best: tuple[int, np.ndarray, np.ndarray, float] | None = None
    scores_by_k: dict[int, float] = {}
    for k in range(k_min, max_k + 1):
        labels, m, sil = _one_k(mat, k, n_resample, resample_frac, seed)
        scores_by_k[k] = sil
        if best is None or sil > best[3]:
            best = (k, labels, m, sil)

    assert best is not None
    k, labels, m, sil = best
    # Relabel subtypes 1..k by descending size for stable, human-friendly names.
    order = pd.Series(labels).value_counts().index.tolist()
    remap = {old: f"S{i + 1}" for i, old in enumerate(order)}
    named = pd.Series([remap[v] for v in labels], index=data.index, name="subtype")
    consensus = pd.DataFrame(m, index=data.index, columns=data.index)
    return ConsensusResult(k=k, labels=named, consensus=consensus, silhouette=sil, scores_by_k=scores_by_k)
