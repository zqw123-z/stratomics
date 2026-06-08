"""Typer command-line interface.

Each analysis stage is exposed as a subcommand, plus a ``run`` command that
executes the whole config-driven pipeline. Run ``stratomics --help`` for usage.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__, cluster, compare, io, plots, score, survival
from .pipeline import Config, run_pipeline

app = typer.Typer(add_completion=False, help="Signature-driven sample stratification framework.")
console = Console()


def _version_callback(value: bool):
    if value:
        console.print(f"stratomics {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
):
    """stratomics — generic stratification toolkit."""


def score_cmd(
    expr: Path = typer.Option(..., "--expr", help="Features-by-samples expression matrix (csv/tsv)."),
    genesets: Path = typer.Option(..., "--genesets", help="Gene sets (yaml/gmt)."),
    method: str = typer.Option("ssgsea", "--method", help="ssgsea | mean_z | rankmean"),
    output: Path = typer.Option("scores.csv", "--output", "-o", help="Output score table."),
):
    """Stage 1 — compute per-sample signature scores."""
    expr_df = io.read_expr(expr)
    sets = io.read_genesets(genesets)
    scores = score.score_signatures(expr_df, sets, method=method)
    io.write_table(scores, output)
    console.print(f"[green]Wrote[/] {output}  ({scores.shape[0]} samples x {scores.shape[1]} signatures)")


def cluster_cmd(
    scores: Path = typer.Option(..., "--scores", help="Samples-by-signatures score table."),
    max_k: int = typer.Option(6, "--max-k", help="Maximum number of subtypes to consider."),
    n_resample: int = typer.Option(50, "--n-resample", help="Resampling iterations."),
    seed: int = typer.Option(0, "--seed"),
    output: Path = typer.Option("subtypes.csv", "--output", "-o"),
):
    """Stage 2 — discover subtypes by consensus clustering."""
    data = io.read_table(scores)
    res = cluster.consensus_cluster(data, max_k=max_k, n_resample=n_resample, seed=seed)
    io.write_table(res.labels.to_frame(), output)
    console.print(f"[green]Selected k={res.k}[/] (silhouette={res.silhouette:.3f}); wrote {output}")


def compare_cmd(
    expr: Path = typer.Option(..., "--expr"),
    groups: Path = typer.Option(..., "--groups", help="Sample-to-subtype table from `cluster`."),
    immune_sets: Path = typer.Option(None, "--immune-sets", help="Optional landscape gene sets."),
    method: str = typer.Option("ssgsea", "--method"),
    output: Path = typer.Option("compare", "--output", "-o", help="Output directory."),
):
    """Stage 3 — between-group feature comparison (+ optional landscape)."""
    expr_df = io.read_expr(expr)
    grp = io.read_table(groups).iloc[:, 0]
    diff = compare.compare_features(expr_df, grp)
    io.write_table(diff, Path(output) / "feature_comparison.csv")
    msg = f"{int(diff['significant'].sum())} significant features"
    if immune_sets:
        sets = io.read_genesets(immune_sets)
        land_scores, land_summary = compare.landscape(expr_df, grp, sets, method=method)
        io.write_table(land_summary, Path(output) / "landscape_summary.csv")
        plots.group_boxplots(land_scores, grp, Path(output) / "landscape_boxplots")
        msg += f"; {land_summary.shape[0]} landscape sets"
    console.print(f"[green]Done[/] — {msg}; outputs in {output}/")


def survival_cmd(
    groups: Path = typer.Option(..., "--groups"),
    clinical: Path = typer.Option(..., "--clinical"),
    time: str = typer.Option(..., "--time", help="Survival time column name."),
    event: str = typer.Option(..., "--event", help="Event indicator column name (1=event)."),
    output: Path = typer.Option("survival", "--output", "-o", help="Output directory."),
):
    """Stage 4 — Kaplan-Meier log-rank + Cox by subtype."""
    grp = io.read_table(groups).iloc[:, 0]
    clin = io.read_clinical(clinical)
    res = survival.logrank_by_group(grp, clin, time_col=time, event_col=event)
    plots.km_curves(res.curves, res.pvalue, Path(output) / "km_curves")
    cox = survival.cox_regression(grp, clin, time_col=time, event_col=event)
    io.write_table(cox, Path(output) / "cox_summary.csv")
    console.print(f"[green]log-rank p = {res.pvalue:.4g}[/]; outputs in {output}/")


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Pipeline YAML config."),
):
    """Run the full config-driven pipeline (all four stages)."""
    cfg = Config.from_yaml(config)
    summary = run_pipeline(cfg)
    table = Table(title="stratomics run summary")
    table.add_column("metric")
    table.add_column("value")
    for key, val in summary.items():
        table.add_row(str(key), json.dumps(val) if isinstance(val, (dict, list)) else str(val))
    console.print(table)


# Subcommand names use underscores in the function but hyphens on the CLI.
app.command(name="score")(score_cmd)
app.command(name="cluster")(cluster_cmd)
app.command(name="compare")(compare_cmd)
app.command(name="survival")(survival_cmd)


if __name__ == "__main__":
    app()
