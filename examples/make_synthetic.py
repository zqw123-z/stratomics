"""Generate a small, deterministic synthetic dataset for demos and tests.

The data has NO biological meaning. It plants three latent sample groups with
differing activity of a few abstract gene sets and group-dependent survival, so
that the pipeline produces a non-trivial, reproducible result. Feature names are
generic placeholders (``GENE_0001`` ...), and the gene sets reference only those
placeholders — nothing from any real study is used.

Run:  python examples/make_synthetic.py   ->   examples/synthetic/{expr,clinical,genesets}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SEED = 0
N_GENES = 300
N_SAMPLES = 150
N_GROUPS = 3
SET_SIZE = 25
N_SETS = 4


def build(out_dir: Path) -> dict[str, Path]:
    rng = np.random.RandomState(SEED)
    genes = [f"GENE_{i:04d}" for i in range(N_GENES)]
    samples = [f"S{j:03d}" for j in range(N_SAMPLES)]

    # Assign each sample to a latent group (roughly balanced).
    group = rng.randint(0, N_GROUPS, size=N_SAMPLES)

    # Define disjoint marker sets; each set is up-regulated in one group.
    sets: dict[str, list[str]] = {}
    set_to_group: dict[str, int] = {}
    cursor = 0
    for s in range(N_SETS):
        members = genes[cursor : cursor + SET_SIZE]
        cursor += SET_SIZE
        name = f"PROGRAM_{s + 1}"
        sets[name] = members
        set_to_group[name] = s % N_GROUPS

    # Baseline expression + group-specific elevation of each program's members.
    expr = rng.normal(loc=6.0, scale=1.0, size=(N_GENES, N_SAMPLES))
    gene_index = {g: i for i, g in enumerate(genes)}
    for name, members in sets.items():
        g = set_to_group[name]
        rows = [gene_index[m] for m in members]
        boost = (group == g).astype(float) * 2.5
        expr[np.ix_(rows, range(N_SAMPLES))] += boost[None, :]

    expr_df = pd.DataFrame(expr, index=genes, columns=samples)
    expr_df.index.name = "gene"

    # Survival: hazard depends on latent group, with right-censoring.
    base_hazard = np.array([0.03, 0.06, 0.12])[group]
    time = rng.exponential(1.0 / base_hazard)
    censor = rng.exponential(scale=30.0, size=N_SAMPLES)
    os_time = np.minimum(time, censor)
    os_event = (time <= censor).astype(int)
    clinical = pd.DataFrame(
        {"OS_time": np.round(os_time, 3), "OS_event": os_event}, index=samples
    )
    clinical.index.name = "sample"

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "expr": out_dir / "expr.csv",
        "clinical": out_dir / "clinical.csv",
        "genesets": out_dir / "genesets.yaml",
    }
    expr_df.to_csv(paths["expr"])
    clinical.to_csv(paths["clinical"])
    paths["genesets"].write_text(yaml.safe_dump({k: v for k, v in sets.items()}, sort_keys=False))
    return paths


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "synthetic"
    written = build(out)
    for kind, path in written.items():
        print(f"{kind:10s} -> {path}")
