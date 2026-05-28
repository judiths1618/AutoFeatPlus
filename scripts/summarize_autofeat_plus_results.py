from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PATTERNS = [
    "results/6g_data/kul_autofeat_plus*.csv",
    "results/6g_data/EUR/*autofeat_plus*.csv",
]


def parse_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.startswith("["):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def infer_dataset_family(path: Path, row: pd.Series) -> str:
    data_label = str(row.get("data_label", ""))
    if data_label.startswith("EUR/") or "/EUR/" in str(path):
        return "EUR"
    if data_label.startswith("kul_") or "kul_" in str(path):
        return "KUL"
    return "unknown"


def infer_privacy_policy(path: Path, row: pd.Series) -> str:
    join_name = row.get("join_name", "")
    if isinstance(join_name, str) and join_name and join_name != "nan":
        return join_name.split(";", 1)[0]

    test_groups = row.get("test_groups", "")
    if isinstance(test_groups, str) and ";" in test_groups:
        return test_groups.split(";", 1)[0]

    name = path.stem
    if "all_private" in name:
        return "all"
    if "with_metadata" in name:
        return "with-metadata"
    if "user_holdout" in name:
        return "privacy-cleaned"
    if "position_holdout" in name:
        return "privacy-cleaned"
    if "random" in name or "local" in name:
        return "privacy-cleaned"
    return "unknown"


def infer_privacy_mode(row: pd.Series) -> str:
    for column in ["join_name", "test_groups"]:
        value = row.get(column, "")
        if not isinstance(value, str):
            continue
        for part in value.split(";"):
            if part.startswith("privacy_mode="):
                return part.removeprefix("privacy_mode=")
    return "hard"


def infer_split(path: Path, row: pd.Series) -> str:
    split_mode = row.get("split_mode", "")
    if isinstance(split_mode, str) and split_mode and split_mode != "nan":
        return split_mode
    name = path.stem
    if "user_holdout" in name:
        return "user-holdout"
    if "position_holdout" in name:
        return "position-holdout"
    return "random"


def is_result_file(path: Path) -> bool:
    name = path.name
    return name.endswith(".csv") and "scores" not in name and "feature_scores" not in name


def collect_files(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(Path().glob(pattern))
    return sorted({path for path in files if is_result_file(path)})


def normalize_results(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "approach" not in frame.columns or "accuracy" not in frame.columns:
        return pd.DataFrame()

    rows = []
    for _, row in frame.iterrows():
        join_path_features = parse_list(row.get("join_path_features", ""))
        blocked_features = parse_list(row.get("blocked_features", ""))
        sensitive_features = parse_list(row.get("sensitive_features", ""))
        n_features = row.get("n_features", 0)
        if pd.isna(n_features) or int(n_features) == 0:
            n_features = len(join_path_features)

        rows.append(
            {
                "source_file": str(path),
                "dataset_family": infer_dataset_family(path, row),
                "data_label": row.get("data_label", ""),
                "target_column": "lat99" if "EUR" in str(path) else "target_x",
                "privacy_policy": infer_privacy_policy(path, row),
                "privacy_mode": infer_privacy_mode(row),
                "split_mode": infer_split(path, row),
                "test_groups": row.get("test_groups", ""),
                "approach": row.get("approach", ""),
                "algorithm": row.get("algorithm", ""),
                "r2": row.get("accuracy", 0.0),
                "error_1_minus_r2": 1 - row.get("accuracy", 0.0),
                "rmse": row.get("rmse", pd.NA),
                "mae": row.get("mae", pd.NA),
                "total_time": row.get("total_time", 0.0),
                "train_time": row.get("train_time", 0.0),
                "feature_selection_time": row.get("feature_selection_time", 0.0),
                "n_features": int(n_features),
                "privacy_risk_score": row.get("privacy_risk_score", 0.0),
                "n_sensitive_features": row.get("n_sensitive_features", len(sensitive_features)),
                "n_blocked_features": len(blocked_features),
                "blocked_features": blocked_features,
                "selected_features": join_path_features,
            }
        )
    return pd.DataFrame(rows)


def add_comparison_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    comparison_keys = [
        "source_file",
        "dataset_family",
        "data_label",
        "privacy_policy",
        "privacy_mode",
        "split_mode",
        "algorithm",
    ]
    frame = frame.copy()
    for column in [
        "base_r2",
        "base_error_1_minus_r2",
        "base_rmse",
        "join_all_r2",
        "join_all_error_1_minus_r2",
        "join_all_rmse",
        "join_all_n_features",
        "join_all_total_time",
        "delta_r2_vs_base",
        "delta_r2_vs_join_all",
    ]:
        frame[column] = pd.NA
    frame["feature_reduction_vs_join_all"] = pd.NA
    frame["speedup_vs_join_all"] = pd.NA
    frame["error_reduction_vs_base"] = pd.NA
    frame["error_reduction_vs_join_all"] = pd.NA
    frame["rmse_reduction_vs_base"] = pd.NA
    frame["rmse_reduction_vs_join_all"] = pd.NA

    for _, group in frame.groupby(comparison_keys, dropna=False):
        base = group[group["approach"] == "BASE"]
        join_all = group[group["approach"] == "Join_All_BFS"]
        idx = group.index

        if not base.empty:
            baseline = base.iloc[0]
            frame.loc[idx, "base_r2"] = baseline["r2"]
            frame.loc[idx, "base_error_1_minus_r2"] = baseline["error_1_minus_r2"]
            frame.loc[idx, "base_rmse"] = baseline["rmse"]
            frame.loc[idx, "delta_r2_vs_base"] = frame.loc[idx, "r2"] - baseline["r2"]
            if baseline["error_1_minus_r2"] and pd.notna(baseline["error_1_minus_r2"]):
                frame.loc[idx, "error_reduction_vs_base"] = (
                    baseline["error_1_minus_r2"] - frame.loc[idx, "error_1_minus_r2"]
                ) / baseline["error_1_minus_r2"]
            if pd.notna(baseline["rmse"]) and baseline["rmse"]:
                frame.loc[idx, "rmse_reduction_vs_base"] = (
                    baseline["rmse"] - frame.loc[idx, "rmse"]
                ) / baseline["rmse"]

        if join_all.empty:
            continue
        baseline = join_all.iloc[0]
        frame.loc[idx, "join_all_r2"] = baseline["r2"]
        frame.loc[idx, "join_all_error_1_minus_r2"] = baseline["error_1_minus_r2"]
        frame.loc[idx, "join_all_rmse"] = baseline["rmse"]
        frame.loc[idx, "join_all_n_features"] = baseline["n_features"]
        frame.loc[idx, "join_all_total_time"] = baseline["total_time"]
        frame.loc[idx, "delta_r2_vs_join_all"] = frame.loc[idx, "r2"] - baseline["r2"]
        if baseline["error_1_minus_r2"] and pd.notna(baseline["error_1_minus_r2"]):
            frame.loc[idx, "error_reduction_vs_join_all"] = (
                baseline["error_1_minus_r2"] - frame.loc[idx, "error_1_minus_r2"]
            ) / baseline["error_1_minus_r2"]
        if pd.notna(baseline["rmse"]) and baseline["rmse"]:
            frame.loc[idx, "rmse_reduction_vs_join_all"] = (
                baseline["rmse"] - frame.loc[idx, "rmse"]
            ) / baseline["rmse"]
        if baseline["n_features"]:
            frame.loc[idx, "feature_reduction_vs_join_all"] = 1 - frame.loc[idx, "n_features"] / baseline["n_features"]
        if baseline["total_time"]:
            frame.loc[idx, "speedup_vs_join_all"] = baseline["total_time"] / frame.loc[idx, "total_time"].replace(0, pd.NA)

    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize AutoFeatPlus benchmark CSVs for analysis.")
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_PATTERNS)
    parser.add_argument("--output", type=Path, default=Path("results/6g_data/autofeat_plus_comparison_summary.csv"))
    args = parser.parse_args()

    frames = [normalize_results(path) for path in collect_files(args.inputs)]
    summary = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if frames else pd.DataFrame()
    summary = add_comparison_columns(summary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"Saved {len(summary)} summary rows to {args.output}")
    if not summary.empty:
        display_columns = [
            "dataset_family",
            "privacy_policy",
            "privacy_mode",
            "split_mode",
            "approach",
            "algorithm",
            "r2",
            "rmse",
            "mae",
            "n_features",
            "n_sensitive_features",
            "n_blocked_features",
            "delta_r2_vs_base",
            "delta_r2_vs_join_all",
            "error_reduction_vs_base",
            "feature_reduction_vs_join_all",
            "speedup_vs_join_all",
        ]
        print(summary[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
