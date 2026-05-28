from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_SENSITIVE_PATTERNS = (
    "user_id",
    "sample_id",
    "sample_key",
    "target_y",
    "target_z",
    "position",
    "location",
    "trajectory",
    "device_id",
    "node_id",
    "container_id",
    "pod_id",
    "ip",
)


@dataclass
class AutoFeatPlusSelection:
    dataframe: pd.DataFrame
    selected_features: list[str]
    blocked_features: list[str]
    sensitive_features: list[str]
    feature_scores: pd.DataFrame
    privacy_risk_score: float


def _matches_sensitive_pattern(feature: str, sensitive_patterns: Iterable[str]) -> bool:
    lower_feature = feature.lower()
    column_name = lower_feature.rsplit(".", 1)[-1]
    for pattern in sensitive_patterns:
        lower_pattern = pattern.lower()
        if lower_pattern.startswith("column:"):
            if column_name == lower_pattern.removeprefix("column:"):
                return True
            continue
        if lower_pattern.startswith("re:"):
            if re.search(pattern[3:], feature, flags=re.IGNORECASE):
                return True
            continue
        if lower_pattern in lower_feature:
            return True
    return False


def is_sensitive_feature(feature: str, sensitive_patterns: Iterable[str] = DEFAULT_SENSITIVE_PATTERNS) -> bool:
    return _matches_sensitive_pattern(feature, sensitive_patterns)


def _safe_abs_correlation(feature: pd.Series, target: pd.Series) -> float:
    numeric_feature = pd.to_numeric(feature, errors="coerce")
    numeric_target = pd.to_numeric(target, errors="coerce")
    if numeric_feature.notna().sum() < 3 or numeric_feature.nunique(dropna=True) <= 1:
        return 0.0
    corr = numeric_feature.corr(numeric_target, method="spearman")
    if pd.isna(corr):
        return 0.0
    return float(abs(corr))


def score_features(
    dataframe: pd.DataFrame,
    target_column: str,
    sensitive_patterns: Iterable[str] = DEFAULT_SENSITIVE_PATTERNS,
    proxy_correlation_threshold: float = 0.999,
) -> pd.DataFrame:
    target = dataframe[target_column]
    rows = []
    for feature in dataframe.columns:
        if feature == target_column:
            continue

        series = dataframe[feature]
        utility = _safe_abs_correlation(series, target)
        pattern_risk = int(_matches_sensitive_pattern(feature, sensitive_patterns))
        proxy_risk = int(utility >= proxy_correlation_threshold)
        # Treat high-cardinality non-numeric columns as identifier-like. Do not
        # penalize continuous CSI/telemetry signals for having many unique values.
        uniqueness_ratio = series.nunique(dropna=True) / max(len(series), 1)
        identifier_risk = int(not pd.api.types.is_numeric_dtype(series) and uniqueness_ratio > 0.9)
        missing_ratio = float(series.isna().mean())
        privacy_risk = pattern_risk + proxy_risk + identifier_risk

        rows.append(
            {
                "feature": feature,
                "utility": utility,
                "privacy_risk": privacy_risk,
                "pattern_risk": pattern_risk,
                "proxy_risk": proxy_risk,
                "identifier_risk": identifier_risk,
                "missing_ratio": missing_ratio,
            }
        )

    return pd.DataFrame(rows)


def select_autofeat_plus_features(
    dataframe: pd.DataFrame,
    target_column: str,
    top_k: int = 50,
    sensitive_patterns: Iterable[str] = DEFAULT_SENSITIVE_PATTERNS,
    privacy_penalty: float = 0.25,
    missing_penalty: float = 0.10,
    cost_penalty: float = 0.001,
    block_sensitive: bool = True,
    proxy_correlation_threshold: float = 0.999,
    max_missing_ratio: float = 0.95,
) -> AutoFeatPlusSelection:
    feature_scores = score_features(
        dataframe=dataframe,
        target_column=target_column,
        sensitive_patterns=sensitive_patterns,
        proxy_correlation_threshold=proxy_correlation_threshold,
    )
    if feature_scores.empty:
        return AutoFeatPlusSelection(
            dataframe=dataframe[[target_column]].copy(),
            selected_features=[],
            blocked_features=[],
            sensitive_features=[],
            feature_scores=feature_scores,
            privacy_risk_score=0.0,
        )

    feature_scores["score_plus"] = (
        feature_scores["utility"]
        - privacy_penalty * feature_scores["privacy_risk"]
        - missing_penalty * feature_scores["missing_ratio"]
        - cost_penalty
    )

    sensitive = feature_scores.loc[feature_scores["privacy_risk"] > 0, "feature"].tolist()
    # Only hard-block features matched by name patterns (identifiers/PII).
    # proxy_risk and identifier_risk are penalized via score_plus but not blocked,
    # because high correlation with the target is often legitimate (e.g. lat75 vs lat99).
    if block_sensitive:
        blocked = feature_scores.loc[feature_scores["pattern_risk"] > 0, "feature"].tolist()
    else:
        blocked = []
    candidates = feature_scores[feature_scores["missing_ratio"] <= max_missing_ratio]
    if block_sensitive:
        candidates = candidates[candidates["pattern_risk"] == 0]

    candidates = candidates.sort_values(["score_plus", "utility"], ascending=[False, False])
    selected_features = candidates.head(top_k)["feature"].tolist()

    # Fallback: if blocking removed all candidates, relax to missing-ratio-only filter so
    # the caller always receives at least some features rather than an empty selection.
    if not selected_features and block_sensitive:
        logging.warning(
            "select_autofeat_plus_features: all candidates blocked by policy — "
            "falling back to least-risky available features (missing_ratio filter only)."
        )
        fallback = (
            feature_scores[feature_scores["missing_ratio"] <= max_missing_ratio]
            .sort_values(["privacy_risk", "missing_ratio", "utility"], ascending=[True, True, False])
        )
        selected_features = fallback.head(top_k)["feature"].tolist()

    selected_columns = selected_features + [target_column]
    privacy_risk_score = float(feature_scores[feature_scores["feature"].isin(selected_features)]["privacy_risk"].sum())

    return AutoFeatPlusSelection(
        dataframe=dataframe[selected_columns].copy(),
        selected_features=selected_features,
        blocked_features=blocked,
        sensitive_features=sensitive,
        feature_scores=feature_scores.sort_values(["score_plus", "utility"], ascending=[False, False]),
        privacy_risk_score=privacy_risk_score,
    )
