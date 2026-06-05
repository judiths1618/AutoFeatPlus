from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feature_discovery.pipelines.base_table_pipeline import run_base_table_pipeline
from feature_discovery.config import rel


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the base-table-centric augmentation pipeline.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--base-table", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--sample-rows", type=int, default=5000)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--temporal-confidence-threshold", type=float, default=0.6)
    parser.add_argument("--output-dir", type=Path, default=Path("results/6g_data/base_table_pipeline"))
    parser.add_argument("--write-connections-to-data-dir", action="store_true")
    parser.add_argument("--run-benchmark", action="store_true")
    parser.add_argument("--run-graph-mode", action="store_true")
    parser.add_argument("--benchmark-models", nargs="+", default=["ridge", "rf", "xt", "gbr"])
    parser.add_argument("--feature-source", choices=["csv", "recompute"], default="recompute")
    parser.add_argument(
        "--autofeatplus-results-csv",
        type=Path,
        default=Path("results/6g_data/EUR/6907619_autofeat_plus_local.csv"),
    )
    parser.add_argument("--autofeatplus-algorithm", default="XGBoost")
    parser.add_argument("--graph-algorithm", default="XGB")
    parser.add_argument("--graph-top-k", type=int, default=15)
    parser.add_argument("--graph-value-ratio", type=float, default=0.65)
    parser.add_argument("--dataset-type", default="regression")
    parser.add_argument(
        "--python-bin",
        default=os.getenv("AUTOFEAT_PYTHON", "python"),
    )
    args = parser.parse_args()

    artifacts = run_base_table_pipeline(
        data_dir=args.data_dir,
        metadata_path=args.metadata,
        base_table=args.base_table,
        target_column=args.target_column,
        output_dir=args.output_dir,
        sample_rows=args.sample_rows,
        confidence_threshold=args.confidence_threshold,
        temporal_confidence_threshold=args.temporal_confidence_threshold,
        write_connections_to_data_dir=args.write_connections_to_data_dir,
        run_benchmark=args.run_benchmark,
        run_graph_mode=args.run_graph_mode,
        benchmark_models=args.benchmark_models,
        feature_source=args.feature_source,
        autofeatplus_results_csv=args.autofeatplus_results_csv,
        autofeatplus_algorithm=args.autofeatplus_algorithm,
        graph_algorithm=args.graph_algorithm,
        graph_top_k=args.graph_top_k,
        graph_value_ratio=args.graph_value_ratio,
        dataset_type=args.dataset_type,
        python_bin=args.python_bin,
    )

    print(f"Run directory: {artifacts.run_dir}")
    print(f"Candidate relationships: {rel(artifacts.candidate_relationships_path)}")
    print(f"Recommended connections: {rel(artifacts.recommended_connections_path)}")
    print(f"Connections CSV: {rel(artifacts.connections_path)}")
    print(f"Relationship report: {rel(artifacts.relationship_report_path)}")
    print(f"Benchmark plan: {rel(artifacts.benchmark_plan_path)}")
    print(f"Use cases: {rel(artifacts.use_cases_path)}")
    print(f"Pipeline summary: {rel(artifacts.pipeline_summary_path)}")
    if artifacts.benchmark_report_path is not None:
        print(f"Benchmark report: {rel(artifacts.benchmark_report_path)}")
    if artifacts.benchmark_results_path is not None:
        print(f"Benchmark results: {rel(artifacts.benchmark_results_path)}")
    if artifacts.graph_report_path is not None:
        print(f"Graph-mode report: {rel(artifacts.graph_report_path)}")
    if artifacts.graph_results_path is not None:
        print(f"Graph-mode results: {rel(artifacts.graph_results_path)}")


if __name__ == "__main__":
    main()
