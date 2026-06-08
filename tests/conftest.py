"""Shared fixtures: a small deterministic synthetic dataset in memory."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the example generator importable without installing it.
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

from make_synthetic import build  # noqa: E402


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    """Generate the synthetic dataset once and load the three inputs."""
    import pandas as pd

    from stratomics import io

    out = tmp_path_factory.mktemp("synthetic")
    paths = build(out)
    return {
        "paths": paths,
        "expr": io.read_expr(paths["expr"]),
        "genesets": io.read_genesets(paths["genesets"]),
        "clinical": io.read_clinical(paths["clinical"]),
    }
