from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import itertools
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from feature_discovery.config import rel


TIME_TOLERANCE_GRID = (1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600)
CONFIG_KEY_NAMES = {
    "n",
    "c",
    "cpu_limit",
    "ram_limit",
    "ram_limit_mb",
    "cpu_limit_mb",
}
MEASUREMENT_NAMES = {
    "cpu_usage",
    "ram_usage",
    "ram_usage_mb",
    "lat50",
    "lat66",
    "lat75",
    "lat80",
    "lat90",
    "lat95",
    "lat98",
    "lat99",
    "lat100",
    "min",
    "mean",
}


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    normalized_name: str
    dtype: str
    non_null_ratio: float
    unique_ratio: float
    sample_values: tuple[str, ...]
    metadata_description: str


def normalize_column_name(name: str) -> str:
    normalized = name.strip().lower()
    normalized = normalized.replace("_ms", "")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def tokenize_text(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token}


def parse_metadata_file(metadata_path: Path) -> dict[str, dict[str, Any]]:
    if not metadata_path.exists():
        return {}

    entries: dict[str, dict[str, Any]] = {}
    current_files: list[str] = []
    current_group = "unknown"

    for raw_line in metadata_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.endswith(":") and "file" not in line.lower():
            current_group = line[:-1].strip()
            continue

        if line.lower().startswith("file:") or line.lower().startswith("files:"):
            _, value = line.split(":", 1)
            current_files = [part.strip() for part in value.split(",") if part.strip()]
            for filename in current_files:
                entries.setdefault(filename, {"group": current_group, "columns": {}, "notes": []})
            continue

        if ":" in line and current_files:
            key, value = [part.strip() for part in line.split(":", 1)]
            if re.fullmatch(r"[A-Za-z0-9_]+", key):
                for filename in current_files:
                    entries.setdefault(filename, {"group": current_group, "columns": {}, "notes": []})
                    entries[filename]["columns"][key] = value
                continue

        if current_files:
            for filename in current_files:
                entries.setdefault(filename, {"group": current_group, "columns": {}, "notes": []})
                entries[filename]["notes"].append(line)

    return entries


def _clean_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace(r"[^0-9eE+\-\.]", "", regex=True)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _sample_values(series: pd.Series, limit: int = 5) -> tuple[str, ...]:
    values = series.dropna().astype(str).head(limit).tolist()
    return tuple(values)


def profile_table(table_path: Path, metadata_entry: dict[str, Any] | None = None, sample_rows: int = 5000) -> dict[str, ColumnProfile]:
    dataframe = pd.read_csv(table_path, nrows=sample_rows)
    metadata_columns = (metadata_entry or {}).get("columns", {})
    profiles: dict[str, ColumnProfile] = {}
    n_rows = max(len(dataframe), 1)

    for column in dataframe.columns:
        series = dataframe[column]
        profiles[column] = ColumnProfile(
            name=column,
            normalized_name=normalize_column_name(column),
            dtype=str(series.dtype),
            non_null_ratio=float(series.notna().mean()),
            unique_ratio=float(series.nunique(dropna=True) / n_rows),
            sample_values=_sample_values(series),
            metadata_description=str(metadata_columns.get(column, "")),
        )

    return profiles


def _description_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    ta = tokenize_text(a)
    tb = tokenize_text(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _overlap_similarity(left: pd.Series, right: pd.Series) -> float:
    left_values = left.dropna()
    right_values = right.dropna()
    if left_values.empty or right_values.empty:
        return 0.0

    left_num = _clean_numeric_series(left_values)
    right_num = _clean_numeric_series(right_values)
    if left_num.notna().mean() > 0.8 and right_num.notna().mean() > 0.8:
        left_set = set(left_num.dropna().round(6).unique().tolist())
        right_set = set(right_num.dropna().round(6).unique().tolist())
    else:
        left_set = set(left_values.astype(str).unique().tolist())
        right_set = set(right_values.astype(str).unique().tolist())
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _time_alignment_stats(left: pd.Series, right: pd.Series) -> dict[str, float]:
    left_num = np.sort(_clean_numeric_series(left).dropna().unique())
    right_num = np.sort(_clean_numeric_series(right).dropna().unique())
    if len(left_num) == 0 or len(right_num) == 0:
        return {
            "match_ratio_best": 0.0,
            "recommended_tolerance_seconds": float("nan"),
            "median_nearest_delta": float("nan"),
        }

    if len(left_num) > len(right_num):
        left_num, right_num = right_num, left_num

    positions = np.searchsorted(right_num, left_num)
    nearest_deltas = []
    for idx, value in enumerate(left_num):
        candidates = []
        pos = positions[idx]
        if pos < len(right_num):
            candidates.append(abs(right_num[pos] - value))
        if pos > 0:
            candidates.append(abs(right_num[pos - 1] - value))
        if candidates:
            nearest_deltas.append(min(candidates))
    if not nearest_deltas:
        return {
            "match_ratio_best": 0.0,
            "recommended_tolerance_seconds": float("nan"),
            "median_nearest_delta": float("nan"),
        }

    nearest = np.array(nearest_deltas, dtype=float)
    tolerance_scores = {tol: float((nearest <= tol).mean()) for tol in TIME_TOLERANCE_GRID}
    recommended_tolerance = min(
        TIME_TOLERANCE_GRID,
        key=lambda tol: (abs(tolerance_scores[tol] - 0.9), tol),
    )
    best_tolerance = max(tolerance_scores, key=tolerance_scores.get)
    return {
        "match_ratio_best": tolerance_scores[best_tolerance],
        "recommended_tolerance_seconds": float(recommended_tolerance),
        "median_nearest_delta": float(np.median(nearest)),
    }


def infer_pair_relationships(
    *,
    left_path: Path,
    right_path: Path,
    metadata: dict[str, dict[str, Any]],
    sample_rows: int = 5000,
) -> pd.DataFrame:
    left_df = pd.read_csv(left_path, nrows=sample_rows)
    right_df = pd.read_csv(right_path, nrows=sample_rows)
    left_profiles = profile_table(left_path, metadata.get(left_path.name), sample_rows=sample_rows)
    right_profiles = profile_table(right_path, metadata.get(right_path.name), sample_rows=sample_rows)

    rows: list[dict[str, Any]] = []
    for left_col, right_col in itertools.product(left_df.columns, right_df.columns):
        lp = left_profiles[left_col]
        rp = right_profiles[right_col]

        name_exact = float(lp.normalized_name == rp.normalized_name)
        name_partial = float(
            lp.normalized_name in rp.normalized_name or rp.normalized_name in lp.normalized_name
        )
        desc_sim = _description_similarity(lp.metadata_description, rp.metadata_description)

        relation_type = "exact"
        join_role = "semantic_match"
        content_score = _overlap_similarity(left_df[left_col], right_df[right_col])
        recommended_tolerance = float("nan")
        median_nearest_delta = float("nan")
        if lp.normalized_name == "time" and rp.normalized_name == "time":
            relation_type = "temporal_asof"
            join_role = "temporal_key"
            time_stats = _time_alignment_stats(left_df[left_col], right_df[right_col])
            content_score = time_stats["match_ratio_best"]
            recommended_tolerance = time_stats["recommended_tolerance_seconds"]
            median_nearest_delta = time_stats["median_nearest_delta"]
        elif lp.normalized_name == rp.normalized_name and lp.normalized_name in CONFIG_KEY_NAMES:
            join_role = "config_key"
        elif (
            lp.normalized_name == rp.normalized_name
            and content_score >= 0.8
            and ((lp.unique_ratio + rp.unique_ratio) / 2.0) >= 0.05
        ):
            join_role = "exact_key"
        elif lp.normalized_name == rp.normalized_name and lp.normalized_name in MEASUREMENT_NAMES:
            join_role = "semantic_match"

        confidence = (
            0.45 * max(name_exact, 0.7 * name_partial)
            + 0.20 * desc_sim
            + 0.35 * content_score
        )

        reason_parts = []
        if name_exact:
            reason_parts.append("same column name")
        elif name_partial:
            reason_parts.append("similar column name")
        if desc_sim > 0:
            reason_parts.append(f"metadata similarity={desc_sim:.2f}")
        if relation_type == "temporal_asof":
            reason_parts.append(f"time alignment match={content_score:.2f}")
            if not math.isnan(recommended_tolerance):
                reason_parts.append(f"recommended tolerance={int(recommended_tolerance)}s")
        elif content_score > 0:
            reason_parts.append(f"value overlap={content_score:.2f}")

        rows.append(
            {
                "left_table": left_path.name,
                "right_table": right_path.name,
                "left_column": left_col,
                "right_column": right_col,
                "relation_type": relation_type,
                "join_role": join_role,
                "confidence": round(float(confidence), 6),
                "name_exact": name_exact,
                "name_partial": name_partial,
                "metadata_similarity": round(float(desc_sim), 6),
                "content_similarity": round(float(content_score), 6),
                "left_unique_ratio": round(lp.unique_ratio, 6),
                "right_unique_ratio": round(rp.unique_ratio, 6),
                "recommended_tolerance_seconds": recommended_tolerance,
                "median_nearest_delta": median_nearest_delta,
                "left_group": metadata.get(left_path.name, {}).get("group", ""),
                "right_group": metadata.get(right_path.name, {}).get("group", ""),
                "reason": "; ".join(reason_parts),
            }
        )

    return pd.DataFrame(rows).sort_values("confidence", ascending=False).reset_index(drop=True)


def infer_dataset_relationships(
    *,
    data_dir: Path,
    metadata_path: Path | None = None,
    sample_rows: int = 5000,
) -> pd.DataFrame:
    files = sorted(path for path in data_dir.glob("*.csv") if path.name != "connections.csv")
    metadata = parse_metadata_file(metadata_path) if metadata_path else {}
    all_rows = []
    for left_path, right_path in itertools.combinations(files, 2):
        pair_df = infer_pair_relationships(
            left_path=left_path,
            right_path=right_path,
            metadata=metadata,
            sample_rows=sample_rows,
        )
        all_rows.append(pair_df)
    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def recommend_connections(relationships: pd.DataFrame, confidence_threshold: float = 0.7) -> pd.DataFrame:
    if relationships.empty:
        return pd.DataFrame(columns=["pk_table", "pk_column", "fk_table", "fk_column", "relation_type", "join_role", "confidence", "recommended_tolerance_seconds", "reason"])

    candidates = relationships.copy()
    candidates = candidates[candidates["confidence"] >= confidence_threshold]
    candidates = candidates[candidates["join_role"].isin(["temporal_key", "config_key", "exact_key"])]

    chosen_rows: list[dict[str, Any]] = []
    used_pairs: set[tuple[str, str, str, str]] = set()
    role_priority = {"temporal_key": 0, "exact_key": 1, "config_key": 2}
    candidates = candidates.assign(
        _role_priority=candidates["join_role"].map(role_priority).fillna(99)
    )
    for _, row in candidates.sort_values(["_role_priority", "confidence"], ascending=[True, False]).iterrows():
        pair_key = (row["left_table"], row["right_table"], row["relation_type"], row["join_role"])
        if pair_key in used_pairs:
            continue
        chosen_rows.append(
            {
                "pk_table": row["left_table"],
                "pk_column": row["left_column"],
                "fk_table": row["right_table"],
                "fk_column": row["right_column"],
                "relation_type": row["relation_type"],
                "join_role": row["join_role"],
                "confidence": row["confidence"],
                "recommended_tolerance_seconds": row["recommended_tolerance_seconds"],
                "reason": row["reason"],
            }
        )
        used_pairs.add(pair_key)

    return pd.DataFrame(chosen_rows)


def build_relationship_report(
    *,
    data_dir: Path,
    metadata_path: Path | None,
    relationships: pd.DataFrame,
    recommended: pd.DataFrame,
) -> str:
    lines = ["# Hybrid Relationship Discovery Report", ""]
    lines.append(f"Data directory: {rel(data_dir)}")
    lines.append(f"Metadata file: {rel(metadata_path) if metadata_path else 'None'}")
    lines.append("")
    lines.append("## Recommended Connections")
    if recommended.empty:
        lines.append("No recommended connections exceeded the confidence threshold.")
    else:
        lines.append("```text")
        lines.append(recommended.to_string(index=False))
        lines.append("```")
    lines.append("")
    lines.append("## Top Candidate Relationships")
    if relationships.empty:
        lines.append("No candidate relationships found.")
    else:
        display = relationships[
            [
                "left_table",
                "right_table",
                "left_column",
                "right_column",
                "relation_type",
                "join_role",
                "confidence",
                "content_similarity",
                "recommended_tolerance_seconds",
                "reason",
            ]
        ].head(30)
        lines.append("```text")
        lines.append(display.to_string(index=False))
        lines.append("```")
    return "\n".join(lines)


def export_connections_csv(recommended: pd.DataFrame) -> pd.DataFrame:
    if recommended.empty:
        return pd.DataFrame(columns=["pk_table", "pk_column", "fk_table", "fk_column"])
    return recommended[["pk_table", "pk_column", "fk_table", "fk_column"]].copy()


def build_benchmark_plan(
    *,
    recommended: pd.DataFrame,
    base_table: str,
    temporal_confidence_threshold: float = 0.6,
) -> dict[str, Any]:
    if recommended.empty:
        return {
            "base_table": base_table,
            "join_mode": "asof",
            "join_key": "time",
            "join_tables": [],
            "time_tolerance_seconds": 120,
        }

    touching_base = recommended[
        (recommended["pk_table"] == base_table) | (recommended["fk_table"] == base_table)
    ].copy()

    temporal = touching_base[
        (touching_base["join_role"] == "temporal_key")
        & (touching_base["confidence"] >= temporal_confidence_threshold)
    ].copy()

    join_tables: list[str] = []
    if not temporal.empty:
        for _, row in temporal.sort_values("confidence", ascending=False).iterrows():
            other_table = row["fk_table"] if row["pk_table"] == base_table else row["pk_table"]
            if other_table not in join_tables:
                join_tables.append(other_table)

        tolerances = temporal["recommended_tolerance_seconds"].dropna().tolist()
        tolerance = int(max(tolerances)) if tolerances else 120
        return {
            "base_table": base_table,
            "join_mode": "asof",
            "join_key": "time",
            "join_tables": join_tables,
            "time_tolerance_seconds": tolerance,
        }

    config = touching_base[touching_base["join_role"] == "config_key"].copy()
    for _, row in config.sort_values("confidence", ascending=False).iterrows():
        other_table = row["fk_table"] if row["pk_table"] == base_table else row["pk_table"]
        if other_table not in join_tables:
            join_tables.append(other_table)

    return {
        "base_table": base_table,
        "join_mode": "exact",
        "join_key": "cpu_limit",
        "join_tables": join_tables,
        "time_tolerance_seconds": 120,
    }
