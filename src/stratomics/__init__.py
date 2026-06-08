"""stratomics: a generic, signature-driven sample stratification framework.

The package exposes four composable analysis stages that operate on a user
supplied expression matrix, gene-set definitions, and (optionally) a clinical
table. No biological prior, gene panel, threshold, or dataset is hard-coded:
everything is provided by the user through configuration or the CLI.

Stages
------
score    : reduce an expression matrix to per-sample signature scores
cluster  : discover sample subtypes by consensus clustering
compare  : characterize differences between groups (features + enrichment)
survival : relate subtypes / scores to time-to-event outcomes
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import cluster, compare, io, plots, score, survival

__all__ = ["io", "score", "cluster", "compare", "survival", "plots", "__version__"]
