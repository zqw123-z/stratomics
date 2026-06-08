"""Survival analysis.

Relate subtype labels or a continuous score to a time-to-event outcome:

- :func:`logrank_by_group` runs a (multivariate) log-rank test across groups
  and returns per-group Kaplan-Meier curves for plotting.
- :func:`cox_regression` fits a Cox proportional-hazards model on group dummies
  or a continuous covariate.

The time and event column names are always provided by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test


def _clean(clinical: pd.DataFrame, time_col: str, event_col: str) -> pd.DataFrame:
    for col in (time_col, event_col):
        if col not in clinical.columns:
            raise KeyError(f"clinical table has no column {col!r}")
    df = clinical[[time_col, event_col]].apply(pd.to_numeric, errors="coerce").dropna()
    df = df[df[time_col] > 0]
    return df


@dataclass
class SurvivalResult:
    """Log-rank result plus per-group KM curves (for plotting)."""

    pvalue: float
    test_statistic: float
    curves: dict[str, pd.DataFrame]
    n_by_group: dict[str, int]


def logrank_by_group(
    groups: pd.Series,
    clinical: pd.DataFrame,
    *,
    time_col: str,
    event_col: str,
) -> SurvivalResult:
    """Multivariate log-rank test across subtype groups."""
    surv = _clean(clinical, time_col, event_col)
    common = surv.index.intersection(groups.index)
    if len(common) < 4:
        raise ValueError("too few samples with both survival data and group labels")
    surv = surv.loc[common]
    grp = groups.loc[common].astype(str)

    test = multivariate_logrank_test(surv[time_col], grp, surv[event_col])
    curves: dict[str, pd.DataFrame] = {}
    n_by_group: dict[str, int] = {}
    for lvl in sorted(grp.unique()):
        mask = grp == lvl
        n_by_group[lvl] = int(mask.sum())
        if mask.sum() < 2:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(surv.loc[mask, time_col], surv.loc[mask, event_col], label=lvl)
        curves[lvl] = kmf.survival_function_.join(kmf.confidence_interval_)
    return SurvivalResult(
        pvalue=float(test.p_value),
        test_statistic=float(test.test_statistic),
        curves=curves,
        n_by_group=n_by_group,
    )


def cox_regression(
    covariate: pd.Series,
    clinical: pd.DataFrame,
    *,
    time_col: str,
    event_col: str,
) -> pd.DataFrame:
    """Fit a Cox model for one covariate (continuous score or group labels).

    A categorical ``covariate`` is one-hot encoded (first level dropped as the
    reference). Returns the lifelines summary table (hazard ratios etc.).
    """
    surv = _clean(clinical, time_col, event_col)
    common = surv.index.intersection(covariate.index)
    if len(common) < 5:
        raise ValueError("too few samples for Cox regression")
    surv = surv.loc[common]
    cov = covariate.loc[common]

    if cov.dtype == object or str(cov.dtype).startswith("category"):
        design = pd.get_dummies(cov.astype(str), prefix="grp", drop_first=True).astype(float)
    else:
        design = pd.to_numeric(cov, errors="coerce").to_frame(name=cov.name or "score")

    data = design.join(surv).dropna()
    cph = CoxPHFitter()
    cph.fit(data, duration_col=time_col, event_col=event_col)
    return cph.summary
