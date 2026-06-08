"""Tests for signature scoring."""

from __future__ import annotations

import pytest

from stratomics import score


@pytest.mark.parametrize("method", list(score.METHODS))
def test_score_shape_and_index(synthetic, method):
    scores = score.score_signatures(synthetic["expr"], synthetic["genesets"], method=method)
    assert scores.shape[0] == synthetic["expr"].shape[1]  # one row per sample
    assert list(scores.columns) == list(synthetic["genesets"].keys())
    assert not scores.isna().any().any()


def test_planted_signal_separates_groups(synthetic):
    """PROGRAM_1 should score higher in samples where it was elevated."""
    scores = score.score_signatures(synthetic["expr"], synthetic["genesets"], method="ssgsea")
    # The synthetic generator boosts PROGRAM_1 members in latent group 0; that
    # subset of samples should have a clearly higher mean than the rest.
    top = scores["PROGRAM_1"].sort_values(ascending=False)
    high = top.head(40).mean()
    low = top.tail(40).mean()
    assert high > low


def test_missing_features_are_tolerated(synthetic):
    sets = {"mostly_absent": ["NOT_A_GENE_1", "NOT_A_GENE_2", "GENE_0001"]}
    scores = score.score_signatures(synthetic["expr"], sets, method="mean_z")
    assert scores.shape == (synthetic["expr"].shape[1], 1)


def test_all_absent_raises(synthetic):
    with pytest.raises(ValueError):
        score.score_signatures(synthetic["expr"], {"ghost": ["NOPE_1", "NOPE_2"]}, method="mean_z")


def test_invalid_method(synthetic):
    with pytest.raises(ValueError):
        score.score_signatures(synthetic["expr"], synthetic["genesets"], method="bogus")
