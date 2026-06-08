"""Tests for consensus clustering."""

from __future__ import annotations

from stratomics import cluster, score


def test_recovers_planted_groups(synthetic):
    scores = score.score_signatures(synthetic["expr"], synthetic["genesets"], method="ssgsea")
    res = cluster.consensus_cluster(scores, max_k=6, n_resample=30, seed=0)
    # Three latent groups were planted; consensus should pick a small k.
    assert 2 <= res.k <= 5
    assert res.labels.shape[0] == scores.shape[0]
    assert res.labels.nunique() == res.k
    # Labels are stable, human-friendly names.
    assert all(lbl.startswith("S") for lbl in res.labels.unique())
    assert 0.0 <= res.silhouette <= 1.0


def test_consensus_matrix_is_square_and_bounded(synthetic):
    scores = score.score_signatures(synthetic["expr"], synthetic["genesets"], method="mean_z")
    res = cluster.consensus_cluster(scores, max_k=4, n_resample=20, seed=1)
    m = res.consensus
    assert m.shape[0] == m.shape[1] == scores.shape[0]
    assert float(m.to_numpy().min()) >= 0.0
    assert float(m.to_numpy().max()) <= 1.0
