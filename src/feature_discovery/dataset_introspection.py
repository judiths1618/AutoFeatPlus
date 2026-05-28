"""
dataset_introspection.py
========================
Lightweight schema/dtype/value inspection used by:
  - auto_pipeline.py pre-flight checks
  - scripts/diagnose.py CLI
  - the dashboard's "Run on your data" tab

The goal is to make the pipeline work on *arbitrary* tabular datasets, not just
the 6G/KUL benchmarks. Most users handing us a new CSV don't know whether their
target is regression vs classification, what column is the temporal key, or
whether their timestamps are seconds, milliseconds, or strings. This module
figures that out so the pipeline doesn't crash with cryptic errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─── Vocabulary ──────────────────────────────────────────────────────────────
PROBLEM_TYPES = {"regression", "binary", "multiclass"}

# Heuristic vocabulary used to recognise timestamp-like columns by name.
_TIME_NAME_HINTS = ("time", "timestamp", "ts", "datetime", "date",
                    "epoch", "captured_at", "event_time", "created_at")


# ─── Data classes ────────────────────────────────────────────────────────────
@dataclass
class ColumnSpec:
    name: str
    dtype: str
    n_unique: int
    n_null: int
    n_rows: int
    is_numeric: bool
    is_datetime: bool
    is_constant: bool
    cardinality_ratio: float  # n_unique / n_rows


@dataclass
class TableSpec:
    path: Path
    n_rows: int
    columns: List[ColumnSpec]

    def col(self, name: str) -> Optional[ColumnSpec]:
        return next((c for c in self.columns if c.name == name), None)

    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]


@dataclass
class Diagnosis:
    """Aggregated pre-flight report for a base table + target choice."""
    spec: TableSpec
    target: str
    inferred_problem_type: Optional[str]
    inferred_temporal_key: Optional[str]
    timestamp_unit: Optional[str]              # "s"/"ms"/"us"/"ns"/"datetime64"/None
    candidate_join_keys: List[str]             # likely FK columns (low cardinality)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def ok(self) -> bool:
        return not self.errors


# ─── Loading ─────────────────────────────────────────────────────────────────
def describe_table(path: Path, sample_rows: int = 50_000) -> TableSpec:
    """Read a CSV and produce a column-level summary. Reads up to `sample_rows`
    rows for speed; passes `low_memory=False` for accurate dtype inference."""
    df = pd.read_csv(path, low_memory=False, nrows=sample_rows)
    cols: List[ColumnSpec] = []
    for name in df.columns:
        s = df[name]
        n_unique = int(s.dropna().nunique())
        n_null = int(s.isna().sum())
        is_dt = bool(pd.api.types.is_datetime64_any_dtype(s)) or _looks_like_datetime(s)
        is_num = bool(pd.api.types.is_numeric_dtype(s)) and not is_dt
        cols.append(ColumnSpec(
            name=name,
            dtype=str(s.dtype),
            n_unique=n_unique,
            n_null=n_null,
            n_rows=len(s),
            is_numeric=is_num,
            is_datetime=is_dt,
            is_constant=(n_unique <= 1),
            cardinality_ratio=(n_unique / max(len(s), 1)),
        ))
    return TableSpec(path=Path(path), n_rows=len(df), columns=cols)


def _looks_like_datetime(s: pd.Series) -> bool:
    """Best-effort sniff: parse the first non-null value as a date if string.
    Cheap; avoids the cost of converting the full column."""
    if s.dtype != object:
        return False
    sample = s.dropna().head(3)
    if sample.empty:
        return False
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # dateutil fallback warning is noisy
        try:
            pd.to_datetime(sample, errors="raise")
            return True
        except (ValueError, TypeError):
            return False


# ─── Inference ───────────────────────────────────────────────────────────────
def suggest_problem_type(target: pd.Series, max_classes: int = 20) -> str:
    """Pick the AutoGluon problem_type that fits a target column.

      - bool / object with ≤2 levels       → binary
      - object / int with 3..max_classes   → multiclass
      - everything else (continuous float) → regression

    Edge case: integer targets with few unique values (e.g. {0,1}) are
    classification, not regression — fixed by checking unique count first.
    """
    s = target.dropna()
    if s.empty:
        return "regression"
    nuniq = s.nunique()
    if nuniq <= 2:
        return "binary"
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_bool_dtype(s):
        return "multiclass" if nuniq <= max_classes else "regression"
    if pd.api.types.is_integer_dtype(s) and nuniq <= max_classes:
        return "multiclass"
    return "regression"


def suggest_temporal_key(spec: TableSpec) -> Optional[str]:
    """Return the first column that looks like a timestamp by dtype or name."""
    for c in spec.columns:
        if c.is_datetime:
            return c.name
    for c in spec.columns:
        if any(h in c.name.lower() for h in _TIME_NAME_HINTS) and c.is_numeric:
            return c.name
    return None


def detect_timestamp_unit(series: pd.Series) -> Optional[str]:
    """Guess the unit of a numeric timestamp column. Heuristic ranges (Unix
    epoch since 1970):

      ~1.5e9     seconds        (typical 2020s)
      ~1.5e12    milliseconds
      ~1.5e15    microseconds
      ~1.5e18    nanoseconds
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime64"
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    sample = float(s.iloc[len(s) // 2])
    if abs(sample) < 1e10:
        return "s"
    if abs(sample) < 1e13:
        return "ms"
    if abs(sample) < 1e16:
        return "us"
    return "ns"


def candidate_join_keys(spec: TableSpec, top_k: int = 5) -> List[str]:
    """Columns most likely to be join keys: numeric/integer with low cardinality
    ratio (= many repeated values, FK-like)."""
    scored = [
        (c.name, c.cardinality_ratio)
        for c in spec.columns
        if not c.is_constant and (c.is_numeric or c.dtype == "object")
    ]
    scored.sort(key=lambda x: x[1])  # low cardinality first
    return [name for name, _ in scored[:top_k]]


# ─── Aggregated diagnosis ────────────────────────────────────────────────────
def diagnose(path: Path, target: str,
             dataset_type: Optional[str] = None,
             temporal_key: Optional[str] = None) -> Diagnosis:
    """Pre-flight check: introspect a base table, infer missing parameters,
    and surface warnings/errors before the pipeline starts."""
    spec = describe_table(path)
    diag = Diagnosis(
        spec=spec, target=target,
        inferred_problem_type=None,
        inferred_temporal_key=None,
        timestamp_unit=None,
        candidate_join_keys=[],
    )

    # ---- Target column checks ----
    tcol = spec.col(target)
    if tcol is None:
        diag.errors.append(
            f"Target column '{target}' not found in {path.name}. "
            f"Available: {spec.column_names()[:10]}{'...' if len(spec.columns)>10 else ''}"
        )
        return diag
    if tcol.is_constant:
        diag.errors.append(f"Target column '{target}' has only 1 unique value — nothing to predict.")
    if tcol.n_null == tcol.n_rows:
        diag.errors.append(f"Target column '{target}' is all-null.")
    null_ratio = tcol.n_null / max(tcol.n_rows, 1)
    if 0 < null_ratio < 1:
        diag.warnings.append(f"Target '{target}' has {null_ratio:.1%} null values; rows will be dropped before training.")

    # ---- Problem type ----
    df = pd.read_csv(path, usecols=[target], nrows=50_000)
    inferred = suggest_problem_type(df[target])
    diag.inferred_problem_type = inferred
    if dataset_type and dataset_type not in PROBLEM_TYPES:
        diag.errors.append(f"--dataset-type '{dataset_type}' not in {sorted(PROBLEM_TYPES)}.")
    if dataset_type and dataset_type != inferred:
        diag.warnings.append(
            f"You specified --dataset-type={dataset_type} but the target looks like {inferred} "
            f"({tcol.n_unique} unique values across {tcol.n_rows} rows)."
        )

    # ---- Temporal key ----
    if temporal_key:
        tk_col = spec.col(temporal_key)
        if tk_col is None:
            diag.errors.append(f"--temporal-key '{temporal_key}' not in {path.name}.")
        else:
            diag.inferred_temporal_key = temporal_key
            ts_series = pd.read_csv(path, usecols=[temporal_key], nrows=50_000)[temporal_key]
            diag.timestamp_unit = detect_timestamp_unit(ts_series)
    else:
        diag.inferred_temporal_key = suggest_temporal_key(spec)
        if diag.inferred_temporal_key:
            ts_series = pd.read_csv(path, usecols=[diag.inferred_temporal_key], nrows=50_000)[diag.inferred_temporal_key]
            diag.timestamp_unit = detect_timestamp_unit(ts_series)

    # ---- Candidate join keys ----
    diag.candidate_join_keys = [
        c for c in candidate_join_keys(spec) if c != target
    ]

    # ---- Feature sanity ----
    feature_cols = [c for c in spec.columns if c.name != target and not c.is_constant]
    if not feature_cols:
        diag.errors.append("No usable feature columns — every non-target column is constant.")
    if len(feature_cols) > 200:
        diag.warnings.append(
            f"{len(feature_cols)} feature columns. Consider --no-transformer-discovery + "
            f"an explicit connections.csv to avoid slow all-pairs schema matching."
        )

    return diag


# ─── Tolerance parsing (for temporal joins) ──────────────────────────────────
def parse_tolerance(value, default_unit: str = "s") -> Tuple[float, str]:
    """Accept tolerance as ``int`` (seconds, legacy), ``float``, or a string
    like ``'60s'`` / ``'5min'`` / ``'1h'`` / ``'200ms'`` / ``'1d'``.

    Returns ``(amount, unit)`` where unit is one of {s, ms, us, ns, min, h, d}.

    Examples:
        >>> parse_tolerance(60)
        (60.0, 's')
        >>> parse_tolerance("5min")
        (5.0, 'min')
        >>> parse_tolerance("200ms")
        (200.0, 'ms')
    """
    if isinstance(value, (int, float)):
        return float(value), default_unit
    s = str(value).strip().lower()
    # Order matters: longer suffixes first so "min" doesn't match before "ms".
    for suffix in ("min", "ms", "us", "ns", "s", "h", "d"):
        if s.endswith(suffix):
            amount = s[:-len(suffix)].strip()
            try:
                return float(amount), suffix
            except ValueError:
                raise ValueError(f"Could not parse tolerance '{value}'.")
    # Bare number
    try:
        return float(s), default_unit
    except ValueError:
        raise ValueError(
            f"Tolerance '{value}' not recognised. Use a number (seconds) or "
            f"a string like '60s', '5min', '1h', '200ms'."
        )


def tolerance_to_seconds(amount: float, unit: str) -> float:
    """Convert ``(amount, unit)`` to seconds for numeric (Unix-epoch) keys."""
    factor = {
        "ns": 1e-9, "us": 1e-6, "ms": 1e-3,
        "s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0,
    }
    if unit not in factor:
        raise ValueError(f"Unknown unit '{unit}'.")
    return amount * factor[unit]
