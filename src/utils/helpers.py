"""
src/utils/helpers.py
────────────────────────────────────────────────────────────────
Shared utility functions used across loaders and estimators.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


# ── Logging ────────────────────────────────────────────────────


def setup_logging(level: str = "INFO", log_file: str | None = None):
    """Configure root logger with console + optional file handler."""
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=handlers,
        force=True,
    )


# ── Config loading ─────────────────────────────────────────────


def load_config(path: str | Path) -> dict:
    """Load a YAML config file and return as dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


# ── Normalization ──────────────────────────────────────────────


def minmax_normalize(series: pd.Series, clip: bool = True) -> pd.Series:
    """
    Normalize a series to [0, 1] using min-max scaling.
    Handles NaN by leaving them as NaN.
    """
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(np.zeros(len(series)), index=series.index)
    result = (series - mn) / (mx - mn)
    if clip:
        result = result.clip(0.0, 1.0)
    return result


def percentile_normalize(series: pd.Series) -> pd.Series:
    """
    Normalize using percentile rank (robust to outliers).
    Useful for structural estimators with skewed distributions.
    """
    return series.rank(pct=True)


def sigmoid_normalize(series: pd.Series, center: float | None = None) -> pd.Series:
    """
    Apply sigmoid to map any real-valued series to (0, 1).
    center: if provided, shifts sigmoid so that center maps to 0.5.
    """
    x = series.copy().astype(float)
    if center is not None:
        x = x - center
    return 1 / (1 + np.exp(-x))


# ── Text preprocessing ─────────────────────────────────────────


def truncate_text(text: str, max_chars: int = 2000) -> str:
    """Truncate long text to max_chars. Used before tokenizing."""
    if len(text) > max_chars:
        return text[:max_chars] + " [TRUNCATED]"
    return text


def count_words(text: str) -> int:
    """Simple whitespace-based word count."""
    return len(str(text).split())


def count_sentences(text: str) -> int:
    """Approximate sentence count by splitting on sentence-ending punctuation."""
    import re

    sentences = re.split(r"[.!?]+", str(text))
    return max(1, len([s for s in sentences if s.strip()]))


# ── Evaluation helpers ─────────────────────────────────────────


def spearman_rho(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Spearman rank correlation between predicted and true complexity."""
    from scipy.stats import spearmanr

    mask = y_true.notna() & y_pred.notna()
    if mask.sum() < 5:
        return float("nan")
    rho, _ = spearmanr(y_true[mask], y_pred[mask])
    return round(float(rho), 4)


def auroc_binary(
    y_true: pd.Series, y_score: pd.Series, threshold: float = 0.5
) -> float:
    """
    AUROC treating samples above `threshold` as 'hard' (positive class).
    y_true: ground truth complexity [0,1]
    y_score: estimated complexity [0,1]
    """
    from sklearn.metrics import roc_auc_score

    labels = (y_true >= threshold).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")  # only one class present
    mask = y_score.notna()
    return round(roc_auc_score(labels[mask], y_score[mask]), 4)


def summarize_results(results: dict[str, dict]) -> pd.DataFrame:
    """
    Convert a nested results dict to a summary DataFrame.
    results = {dataset_name: {estimator_name: {metric: value}}}
    """
    rows = []
    for dataset, estimators in results.items():
        for estimator, metrics in estimators.items():
            row = {"dataset": dataset, "estimator": estimator}
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)
