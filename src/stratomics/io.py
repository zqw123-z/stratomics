"""Input/output helpers.

All readers are format-agnostic about biology: an "expression matrix" is any
numeric features-by-samples table, a "gene set" is any named list of feature
identifiers, and a "clinical table" is any per-sample metadata frame. The
package never ships or assumes specific feature names.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def _sep_for(path: Path) -> str:
    return "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","


def read_expr(path: str | Path) -> pd.DataFrame:
    """Read a features-by-samples expression matrix.

    The first column is treated as the feature identifier and becomes the
    index; remaining columns are samples. Returns a ``float`` DataFrame.
    """
    path = Path(path)
    df = pd.read_csv(path, sep=_sep_for(path), index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df.astype(float)


def read_genesets(path: str | Path) -> dict[str, list[str]]:
    """Read named gene sets from YAML (``name: [g1, g2, ...]``) or GMT.

    GMT lines are ``name<TAB>description<TAB>g1<TAB>g2...``. Identifiers are
    returned verbatim, deduplicated, preserving order.
    """
    path = Path(path)
    sets: dict[str, list[str]] = {}
    if path.suffix.lower() in {".gmt"}:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name, _desc, *genes = parts
            sets[name] = list(dict.fromkeys(g for g in genes if g))
    else:
        raw = yaml.safe_load(path.read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a mapping of set name -> gene list")
        for name, genes in raw.items():
            if not isinstance(genes, (list, tuple)):
                raise ValueError(f"gene set '{name}' must be a list of identifiers")
            sets[str(name)] = list(dict.fromkeys(str(g) for g in genes))
    if not sets:
        raise ValueError(f"{path}: no gene sets found")
    return sets


def read_clinical(path: str | Path) -> pd.DataFrame:
    """Read a per-sample clinical/metadata table (samples as the first column)."""
    path = Path(path)
    df = pd.read_csv(path, sep=_sep_for(path), index_col=0)
    df.index = df.index.astype(str)
    return df


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a generic table produced by another stage (samples as index)."""
    path = Path(path)
    df = pd.read_csv(path, sep=_sep_for(path), index_col=0)
    df.index = df.index.astype(str)
    return df


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a table next to its destination, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep=_sep_for(path))
    return path
