from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feature_discovery.experiments.benchmark_scenarios import (
    SCENARIOS,
    infer_benchmark_scenarios,
    scenario_markdown,
    scenario_titles,
)


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


def normalize_6g_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    rows = []
    for _, row in frame.iterrows():
        scenarios = infer_benchmark_scenarios(
            data_label=str(row.get("scenario", "")),
            source_file=str(path),
            split_mode="random",
            target_column="lat99",
        )
        rows.append(
            {
                "source_file": str(path),
                "dataset_label": row.get("scenario", ""),
                "data_variant": "raw",
                "benchmark_family": "6g_benchmark",
                "method": row.get("approach", ""),
                "model": row.get("algorithm", ""),
                "primary_metric": "r2",
                "primary_value": row.get("mean_score", pd.NA),
                "rmse": pd.NA,
                "mae": pd.NA,
                "n_features": pd.NA,
                "split_mode": "random",
                "test_groups": "",
                "time_tolerance_seconds": pd.NA,
                "joined_feature_count": pd.NA,
                "all_null_joined_features": pd.NA,
                "mean_missing_ratio_joined": pd.NA,
                "n_blocked_features": pd.NA,
                "privacy_policy": "",
                "scenario_slugs": "|".join(scenarios),
                "scenario_titles": scenario_titles(scenarios),
            }
        )
    return pd.DataFrame(rows)


def normalize_approach_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "approach" not in frame.columns or "accuracy" not in frame.columns:
        return pd.DataFrame()

    rows = []
    for _, row in frame.iterrows():
        data_label = str(row.get("data_label", ""))
        scenarios = infer_benchmark_scenarios(
            data_label=data_label,
            source_file=str(path),
            split_mode=str(row.get("split_mode", "")),
            target_column=str(row.get("target_column", "")),
        )
        rows.append(
            {
                "source_file": str(path),
                "dataset_label": data_label,
                "data_variant": "cleaned" if "cleaned" in str(path).lower() else "raw",
                "benchmark_family": "autofeatplus_local",
                "method": row.get("approach", ""),
                "model": row.get("algorithm", ""),
                "primary_metric": "r2",
                "primary_value": row.get("accuracy", pd.NA),
                "rmse": row.get("rmse", pd.NA),
                "mae": row.get("mae", pd.NA),
                "n_features": row.get("n_features", pd.NA),
                "split_mode": row.get("split_mode", ""),
                "test_groups": row.get("test_groups", ""),
                "time_tolerance_seconds": pd.NA,
                "joined_feature_count": pd.NA,
                "all_null_joined_features": pd.NA,
                "mean_missing_ratio_joined": pd.NA,
                "n_blocked_features": len(parse_list(row.get("blocked_features", "[]"))),
                "privacy_policy": row.get("join_name", ""),
                "scenario_slugs": "|".join(scenarios),
                "scenario_titles": scenario_titles(scenarios),
            }
        )
    return pd.DataFrame(rows)


def normalize_downstream_csv(path: Path, primary_metric: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    rows = []
    for _, row in frame.iterrows():
        data_label = str(row.get("data_label", ""))
        scenarios = infer_benchmark_scenarios(
            data_label=data_label,
            source_file=str(path),
            split_mode=str(row.get("split_mode", "")),
            target_column=str(row.get("target_column", "")),
        )
        rows.append(
            {
                "source_file": str(path),
                "dataset_label": data_label,
                "data_variant": "cleaned" if "cleaned" in str(path).lower() else "raw",
                "benchmark_family": "downstream",
                "method": row.get("variant", ""),
                "model": row.get("model", ""),
                "primary_metric": primary_metric,
                "primary_value": row.get(primary_metric, pd.NA),
                "rmse": row.get("rmse", pd.NA),
                "mae": row.get("mae", pd.NA),
                "n_features": row.get("n_features", pd.NA),
                "split_mode": row.get("split_mode", ""),
                "test_groups": row.get("test_groups", ""),
                "time_tolerance_seconds": row.get("time_tolerance_seconds", pd.NA),
                "joined_feature_count": row.get("joined_feature_count", pd.NA),
                "all_null_joined_features": row.get("all_null_joined_features", pd.NA),
                "mean_missing_ratio_joined": row.get("mean_missing_ratio_joined", pd.NA),
                "n_blocked_features": row.get("n_blocked_features", pd.NA),
                "privacy_policy": row.get("privacy_policy", ""),
                "scenario_slugs": "|".join(scenarios),
                "scenario_titles": scenario_titles(scenarios),
            }
        )
    return pd.DataFrame(rows)


def collect_rows() -> pd.DataFrame:
    files = [
        normalize_6g_summary(Path("results/6g_data/benchmark_6g_summary.csv")),
        normalize_approach_csv(Path("results/6g_data/kul_autofeat_plus_random.csv")),
        normalize_approach_csv(Path("results/6g_data/kul_autofeat_plus_user_holdout.csv")),
        normalize_approach_csv(Path("results/6g_data/kul_autofeat_plus_position_holdout.csv")),
        normalize_approach_csv(Path("results/6g_data/kul_autofeat_plus_local_with_metadata.csv")),
        normalize_downstream_csv(Path("results/6g_data/kul_downstream_models_smoke.csv"), "accuracy"),
        normalize_downstream_csv(Path("results/6g_data/EUR/6907619_downstream_models_tolerance_grid.csv"), "r2"),
        normalize_downstream_csv(Path("results/6g_data/EUR/cleaned_downstream_models_recompute.csv"), "r2"),
    ]
    frames = [frame for frame in files if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_markdown(summary: pd.DataFrame) -> str:
    lines = [scenario_markdown(), "", "# Current Evidence", ""]
    if summary.empty:
        lines.append("No benchmark rows were found.")
        return "\n".join(lines)

    for slug, scenario in SCENARIOS.items():
        lines.append(f"## Evidence: {scenario.title}")
        subset = summary[summary["scenario_slugs"].str.contains(slug, na=False)].copy()
        if subset.empty:
            lines.append("No benchmark evidence collected yet.")
            lines.append("")
            continue
        display = (
            subset.sort_values("primary_value", ascending=False)
            .head(12)[
                [
                    "dataset_label",
                    "data_variant",
                    "benchmark_family",
                    "method",
                    "model",
                    "primary_metric",
                    "primary_value",
                    "rmse",
                    "mae",
                    "n_features",
                    "split_mode",
                    "time_tolerance_seconds",
                    "joined_feature_count",
                    "all_null_joined_features",
                    "mean_missing_ratio_joined",
                    "n_blocked_features",
                ]
            ]
        )
        lines.append("```text")
        lines.append(display.to_string(index=False))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a scenario-centric benchmark report from existing CSVs.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/6g_data/benchmark_scenarios_summary.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/6g_data/benchmark_scenarios_report.md"),
    )
    args = parser.parse_args()

    summary = collect_rows()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)

    report = build_markdown(summary)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")

    print(f"Saved {len(summary)} rows to {args.output_csv}")
    print(f"Saved scenario report to {args.output_md}")


if __name__ == "__main__":
    main()
