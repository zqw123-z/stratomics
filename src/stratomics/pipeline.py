"""End-to-end pipeline: score -> cluster -> compare -> survival.

Driven entirely by a YAML config so that no dataset, gene set, or threshold is
embedded in the code. Each stage writes its outputs (tables + figures) under
the configured output directory and a small ``summary.json`` records the run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from . import cluster, compare, io, plots, score, survival


@dataclass
class Config:
    expr: str
    genesets: str
    output: str = "results_demo"
    clinical: str | None = None
    immune_sets: str | None = None
    score: dict[str, Any] = None  # type: ignore[assignment]
    cluster: dict[str, Any] = None  # type: ignore[assignment]
    survival: dict[str, Any] = None  # type: ignore[assignment]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        base = Path(path).resolve().parent

        def _resolve(p: str | None) -> str | None:
            if p is None:
                return None
            q = Path(p)
            return str(q if q.is_absolute() else (base / q))

        return cls(
            expr=_resolve(raw["expr"]),
            genesets=_resolve(raw["genesets"]),
            output=_resolve(raw.get("output", "results_demo")),
            clinical=_resolve(raw.get("clinical")),
            immune_sets=_resolve(raw.get("immune_sets")),
            score=raw.get("score") or {},
            cluster=raw.get("cluster") or {},
            survival=raw.get("survival") or {},
        )


def run_pipeline(cfg: Config) -> dict[str, Any]:
    """Run all configured stages and return a JSON-serializable summary."""
    out = Path(cfg.output)
    fig_dir = out / "figures"
    tab_dir = out / "tables"
    summary: dict[str, Any] = {"output": str(out)}

    expr = io.read_expr(cfg.expr)
    genesets = io.read_genesets(cfg.genesets)

    # Stage 1 — signature scoring
    method = (cfg.score or {}).get("method", "ssgsea")
    scores = score.score_signatures(expr, genesets, method=method)
    io.write_table(scores, tab_dir / "signature_scores.csv")
    summary["n_samples"] = int(scores.shape[0])
    summary["n_signatures"] = int(scores.shape[1])
    summary["score_method"] = method

    # Stage 2 — consensus clustering
    ccfg = cfg.cluster or {}
    res = cluster.consensus_cluster(
        scores,
        max_k=ccfg.get("max_k", 6),
        n_resample=ccfg.get("n_resample", 50),
        resample_frac=ccfg.get("resample_frac", 0.8),
        seed=ccfg.get("seed", 0),
    )
    io.write_table(res.labels.to_frame(), tab_dir / "subtypes.csv")
    plots.consensus_heatmap(res.consensus, res.labels, fig_dir / "consensus_heatmap")
    plots.group_boxplots(scores, res.labels, fig_dir / "signature_boxplots")
    summary["k"] = res.k
    summary["silhouette"] = round(res.silhouette, 4)
    summary["subtype_sizes"] = res.labels.value_counts().to_dict()

    # Stage 3 — between-group comparison + optional landscape
    diff = compare.compare_features(expr, res.labels)
    io.write_table(diff, tab_dir / "feature_comparison.csv")
    summary["n_significant_features"] = int(diff["significant"].sum())

    if cfg.immune_sets:
        land_sets = io.read_genesets(cfg.immune_sets)
        land_scores, land_summary = compare.landscape(expr, res.labels, land_sets, method=method)
        io.write_table(land_summary, tab_dir / "landscape_summary.csv")
        plots.group_boxplots(land_scores, res.labels, fig_dir / "landscape_boxplots")
        summary["n_landscape_sets"] = int(land_summary.shape[0])

    # Stage 4 — survival (optional, needs clinical table)
    scfg = cfg.survival or {}
    if cfg.clinical and scfg.get("time") and scfg.get("event"):
        clinical = io.read_clinical(cfg.clinical)
        sres = survival.logrank_by_group(
            res.labels, clinical, time_col=scfg["time"], event_col=scfg["event"]
        )
        plots.km_curves(sres.curves, sres.pvalue, fig_dir / "km_curves")
        cox = survival.cox_regression(
            res.labels, clinical, time_col=scfg["time"], event_col=scfg["event"]
        )
        io.write_table(cox, tab_dir / "cox_summary.csv")
        summary["logrank_p"] = round(sres.pvalue, 6)
        summary["survival_n_by_group"] = sres.n_by_group

    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
