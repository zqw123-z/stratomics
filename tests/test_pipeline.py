"""End-to-end pipeline test using a generated config + synthetic data."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from stratomics.pipeline import Config, run_pipeline


def test_full_pipeline(synthetic, tmp_path):
    paths = synthetic["paths"]
    out = tmp_path / "out"
    cfg_dict = {
        "expr": str(paths["expr"]),
        "genesets": str(paths["genesets"]),
        "immune_sets": str(paths["genesets"]),
        "clinical": str(paths["clinical"]),
        "output": str(out),
        "score": {"method": "ssgsea"},
        "cluster": {"max_k": 5, "n_resample": 20, "seed": 0},
        "survival": {"time": "OS_time", "event": "OS_event"},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict))

    summary = run_pipeline(Config.from_yaml(cfg_path))

    # Tables
    for name in ("signature_scores.csv", "subtypes.csv", "feature_comparison.csv",
                 "landscape_summary.csv", "cox_summary.csv"):
        assert (out / "tables" / name).exists(), f"missing table {name}"
    # Figures (png + pdf each)
    for stem in ("consensus_heatmap", "signature_boxplots", "km_curves"):
        assert (out / "figures" / f"{stem}.png").exists()
        assert (out / "figures" / f"{stem}.pdf").exists()
    # Summary
    saved = json.loads((out / "summary.json").read_text())
    assert saved["n_samples"] == summary["n_samples"]
    assert "logrank_p" in saved
    assert saved["k"] >= 2


def test_config_resolves_relative_paths(tmp_path):
    (tmp_path / "sub").mkdir()
    cfg = {"expr": "sub/e.csv", "genesets": "sub/g.yaml", "output": "out"}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    resolved = Config.from_yaml(p)
    assert resolved.expr == str(tmp_path / "sub" / "e.csv")
    assert resolved.output == str(tmp_path / "out")
