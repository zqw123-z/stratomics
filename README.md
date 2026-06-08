# stratomics

A generic, signature-driven framework for **sample stratification** and
**multi-omics characterization**. Given an expression matrix, your own gene-set
definitions, and (optionally) a clinical table, `stratomics` runs a transparent
four-stage pipeline:

1. **Score** — reduce features×samples to per-sample signature scores
   (ssGSEA / mean-z / rank-mean).
2. **Cluster** — discover sample subtypes by consensus clustering, with the
   number of subtypes chosen automatically by silhouette.
3. **Compare** — test features between subtypes (FDR-controlled) and summarize a
   user-supplied "landscape" of gene sets per subtype.
4. **Survive** — relate subtypes/scores to time-to-event outcomes
   (Kaplan-Meier log-rank + Cox).

`stratomics` is **method, not data**: it ships no gene panels, thresholds, or
datasets. You bring the signatures and the matrix; the framework does the
analysis. The repository contains **no data files at all** — the demo/test set
is generated on demand by `examples/make_synthetic.py` and is pure random noise
with a planted structure, carrying no biological meaning.

## Install

```bash
git clone https://github.com/zqw123-z/stratomics.git
cd stratomics
pip install -e ".[dev]"
```

Requires Python ≥ 3.10. Dependencies: numpy, pandas, scipy, scikit-learn,
statsmodels, lifelines, matplotlib, seaborn, typer, pyyaml.

## Quick start (synthetic demo)

```bash
python examples/make_synthetic.py          # writes examples/synthetic/{expr,clinical,genesets}
stratomics run --config configs/example.yaml
```

Outputs land in `results_demo/` (tables + PNG/PDF figures + `summary.json`).

## Input formats

| Input | Format | Shape |
|-------|--------|-------|
| Expression matrix | CSV/TSV | features (rows) × samples (columns), first column = feature ID |
| Gene sets | YAML (`name: [g1, g2, ...]`) or GMT | named lists of feature IDs |
| Clinical table | CSV/TSV | samples (rows), first column = sample ID; includes a time and an event column |

Feature identifiers in the matrix and gene sets must use the same namespace
(e.g. all HGNC symbols, or all Ensembl IDs).

## CLI

Run stages individually or the whole pipeline:

```bash
# Stage 1: signature scores
stratomics score --expr expr.csv --genesets sets.yaml --method ssgsea -o scores.csv

# Stage 2: consensus subtypes
stratomics cluster --scores scores.csv --max-k 6 -o subtypes.csv

# Stage 3: between-subtype comparison (+ optional immune/landscape sets)
stratomics compare --expr expr.csv --groups subtypes.csv --immune-sets immune.yaml -o compare/

# Stage 4: survival by subtype
stratomics survival --groups subtypes.csv --clinical clin.csv \
    --time OS_time --event OS_event -o survival/

# Or everything at once, config-driven:
stratomics run --config configs/example.yaml
```

`stratomics --help` lists all commands; `stratomics <command> --help` shows options.

## Library API

```python
from stratomics import io, score, cluster, compare, survival

expr = io.read_expr("expr.csv")
sets = io.read_genesets("sets.yaml")

scores = score.score_signatures(expr, sets, method="ssgsea")
result = cluster.consensus_cluster(scores, max_k=6)
diff = compare.compare_features(expr, result.labels)
surv = survival.logrank_by_group(result.labels, io.read_clinical("clin.csv"),
                                 time_col="OS_time", event_col="OS_event")
print(result.k, surv.pvalue)
```

## Methods notes

- **ssGSEA** follows the single-sample enrichment score of Barbie et al. (2009).
- **Consensus clustering** resamples samples, clusters each subsample with
  k-means, accumulates a co-clustering (consensus) matrix, and clusters that
  matrix; `k` is selected by the best average silhouette over the consensus
  distance.
- **Differential testing** uses Mann-Whitney (2 groups) or Kruskal-Wallis
  (>2 groups) with Benjamini-Hochberg FDR.
- **Survival** uses lifelines for the multivariate log-rank test and Cox PH.

## Tests

```bash
pytest -q
```

The suite generates the synthetic dataset on the fly and exercises every stage
plus the full pipeline.

## Citation

If `stratomics` is useful in your work, please cite this repository. A
`CITATION.cff` will be added on first release.

## License

MIT — see [LICENSE](LICENSE).
