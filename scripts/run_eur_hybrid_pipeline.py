from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feature_discovery.dataset_relation_graph.hybrid_discovery import (
    build_benchmark_plan,
    build_relationship_report,
    export_connections_csv,
    infer_dataset_relationships,
    recommend_connections,
)


def display_path(path: Path | str | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return resolved.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full EUR metadata+content relation discovery pipeline.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--base-table", default="rabbitmq-performance.csv")
    parser.add_argument("--target-column", default="lat99")
    parser.add_argument("--sample-rows", type=int, default=5000)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--temporal-confidence-threshold", type=float, default=0.6)
    parser.add_argument("--output-dir", type=Path, default=Path("results/6g_data/relation_discovery"))
    parser.add_argument("--write-connections-to-data-dir", action="store_true")
    parser.add_argument("--run-benchmark", action="store_true")
    parser.add_argument("--benchmark-models", nargs="+", default=["ridge", "rf", "xt", "gbr"])
    parser.add_argument("--feature-source", choices=["csv", "recompute"], default="csv")
    parser.add_argument(
        "--autofeatplus-results-csv",
        type=Path,
        default=Path("results/6g_data/EUR/6907619_autofeat_plus_local.csv"),
    )
    parser.add_argument("--autofeatplus-algorithm", default="XGBoost")
    parser.add_argument(
        "--python-bin",
        default=os.getenv("AUTOFEAT_PYTHON", "python"),
    )
    args = parser.parse_args()

    relationships = infer_dataset_relationships(
        data_dir=args.data_dir,
        metadata_path=args.metadata,
        sample_rows=args.sample_rows,
    )
    recommended = recommend_connections(
        relationships,
        confidence_threshold=args.confidence_threshold,
    )

    run_dir = args.output_dir / args.data_dir.name
    run_dir.mkdir(parents=True, exist_ok=True)

    relationships_path = run_dir / "candidate_relationships.csv"
    recommended_path = run_dir / "recommended_connections.csv"
    plain_connections_path = run_dir / "connections.csv"
    report_path = run_dir / "relationship_report.md"
    plan_path = run_dir / "benchmark_plan.txt"

    relationships.to_csv(relationships_path, index=False)
    recommended.to_csv(recommended_path, index=False)
    export_connections_csv(recommended).to_csv(plain_connections_path, index=False)
    report = build_relationship_report(
        data_dir=args.data_dir,
        metadata_path=args.metadata,
        relationships=relationships,
        recommended=recommended,
    )
    report_path.write_text(report, encoding="utf-8")

    if args.write_connections_to_data_dir:
        export_connections_csv(recommended).to_csv(args.data_dir / "connections.csv", index=False)

    plan = build_benchmark_plan(
        recommended=recommended,
        base_table=args.base_table,
        temporal_confidence_threshold=args.temporal_confidence_threshold,
    )
    plan_lines = [
        f"base_table={plan['base_table']}",
        f"join_mode={plan['join_mode']}",
        f"join_key={plan['join_key']}",
        f"time_tolerance_seconds={plan['time_tolerance_seconds']}",
        f"join_tables={','.join(plan['join_tables'])}",
    ]
    plan_path.write_text("\n".join(plan_lines), encoding="utf-8")

    print(f"Saved candidate relationships to {display_path(relationships_path)}")
    print(f"Saved recommended connections to {display_path(recommended_path)}")
    print(f"Saved plain connections.csv to {display_path(plain_connections_path)}")
    print(f"Saved report to {display_path(report_path)}")
    print(f"Saved benchmark plan to {display_path(plan_path)}")

    if not args.run_benchmark:
        return

    if not plan["join_tables"]:
        print("No join tables selected by the benchmark plan; skipping benchmark run.")
        return

    benchmark_output = run_dir / "benchmark_results.csv"
    command = [
        args.python_bin,
        str(Path("downstream ML") / "benchmark_eur_augmented.py"),
        "--data-dir",
        display_path(args.data_dir),
        "--base-table",
        plan["base_table"],
        "--join-tables",
        *plan["join_tables"],
        "--join-key",
        plan["join_key"],
        "--join-mode",
        plan["join_mode"],
        "--time-tolerance-seconds",
        str(plan["time_tolerance_seconds"]),
        "--target-column",
        args.target_column,
        "--split-mode",
        "time",
        "--models",
        *args.benchmark_models,
        "--feature-source",
        args.feature_source,
        "--autofeatplus-results-csv",
        display_path(args.autofeatplus_results_csv),
        "--autofeatplus-algorithm",
        args.autofeatplus_algorithm,
        "--output",
        display_path(benchmark_output),
    ]
    print("Running benchmark command:")
    print(" ".join(command))
    subprocess.run(command, check=True, cwd=ROOT)
    print(f"Saved benchmark results to {display_path(benchmark_output)}")


if __name__ == "__main__":
    main()
