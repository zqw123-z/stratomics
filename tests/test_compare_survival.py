"""Tests for comparison, landscape, and survival stages."""

from __future__ import annotations

import pandas as pd

from stratomics import cluster, compare, score, survival


def _subtypes(synthetic):
    scores = score.score_signatures(synthetic["expr"], synthetic["genesets"], method="ssgsea")
    return scores, cluster.consensus_cluster(scores, max_k=5, n_resample=30, seed=0).labels


def test_compare_features_flags_program_genes(synthetic):
    _, labels = _subtypes(synthetic)
    res = compare.compare_features(synthetic["expr"], labels)
    assert {"pvalue", "padj", "significant", "effect"}.issubset(res.columns)
    assert res["significant"].sum() > 0  # planted programs should differ


def test_landscape_returns_summary(synthetic):
    _, labels = _subtypes(synthetic)
    scores, summary = compare.landscape(
        synthetic["expr"], labels, synthetic["genesets"], method="mean_z"
    )
    assert "pvalue" in summary.columns
    assert summary.shape[0] == len(synthetic["genesets"])


def test_logrank_runs_and_returns_curves(synthetic):
    _, labels = _subtypes(synthetic)
    res = survival.logrank_by_group(
        labels, synthetic["clinical"], time_col="OS_time", event_col="OS_event"
    )
    assert 0.0 <= res.pvalue <= 1.0
    assert len(res.curves) >= 2
    for curve in res.curves.values():
        assert isinstance(curve, pd.DataFrame)


def test_cox_regression_summary(synthetic):
    _, labels = _subtypes(synthetic)
    summary = survival.cox_regression(
        labels, synthetic["clinical"], time_col="OS_time", event_col="OS_event"
    )
    assert "coef" in summary.columns
    assert summary.shape[0] >= 1
