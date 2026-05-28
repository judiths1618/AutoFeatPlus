from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd

from feature_discovery.dataset_relation_graph.hybrid_discovery import (
    build_benchmark_plan,
    build_relationship_report,
    export_connections_csv,
    infer_dataset_relationships,
    recommend_connections,
)
from feature_discovery.experiments.benchmark_scenarios import (
    base_table_use_case_markdown,
    infer_base_table_use_cases,
    use_case_titles,
)


@dataclass
class BaseTablePipelineArtifacts:
    run_dir: Path
    candidate_relationships_path: Path
    recommended_connections_path: Path
    connections_path: Path
    relationship_report_path: Path
    benchmark_plan_path: Path
    use_cases_path: Path
    pipeline_summary_path: Path
    benchmark_report_path: Path | None
    benchmark_results_path: Path | None
    graph_results_path: Path | None
    graph_report_path: Path | None


def _filter_relationships_for_base_table(relationships: pd.DataFrame, base_table: str) -> pd.DataFrame:
    if relationships.empty:
        return relationships
    return relationships[
        (relationships["left_table"] == base_table) | (relationships["right_table"] == base_table)
    ].copy()


def _load_columns(data_dir: Path, base_table: str, sample_rows: int = 1000) -> list[str]:
    dataframe = pd.read_csv(data_dir / base_table, nrows=sample_rows)
    return list(dataframe.columns)


def _load_metadata_text(metadata_path: Path | None) -> str:
    if metadata_path is None or not metadata_path.exists():
        return ""
    return metadata_path.read_text(encoding="utf-8")


def _infer_privacy_policy(columns: list[str], target_column: str) -> list[str]:
    normalized = {column.lower() for column in columns}
    target = target_column.lower()
    policies: list[str] = []
    if "time" in normalized or any("time_" in column for column in normalized):
        policies.append("time-private")
    if {"n", "c"} & normalized:
        policies.append("workload-private")
    if target.startswith("lat") or target in {"target_x", "target_y", "target_z"}:
        policies.append("target-proxy-private")
    if not policies:
        policies.append("none")
    return policies


def _write_pipeline_summary(
    path: Path,
    *,
    base_table: str,
    target_column: str,
    columns: list[str],
    use_case_slugs: list[str],
    plan: dict[str, Any],
    privacy_policy: list[str],
    benchmark_results_path: Path | None,
) -> None:
    payload = {
        "base_table": base_table,
        "target_column": target_column,
        "columns": columns,
        "use_case_slugs": use_case_slugs,
        "join_plan": plan,
        "privacy_policy": privacy_policy,
        "benchmark_results_path": str(benchmark_results_path) if benchmark_results_path else None,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_benchmark_report(
    benchmark_results_path: Path,
    *,
    base_table: str,
    target_column: str,
    plan: dict[str, Any],
    use_case_slugs: list[str],
    privacy_policy: list[str],
) -> str:
    frame = pd.read_csv(benchmark_results_path)
    best_rows = (
        frame.sort_values("r2", ascending=False)
        .groupby("variant", as_index=False)
        .first()[["variant", "model", "r2", "rmse", "mae", "n_features"]]
    )
    lines = [
        "# Base-Table Pipeline Benchmark Report",
        "",
        f"Base table: {base_table}",
        f"Target column: {target_column}",
        f"Join mode: {plan['join_mode']}",
        f"Join key: {plan['join_key']}",
        f"Time tolerance seconds: {plan['time_tolerance_seconds']}",
        f"Join tables: {', '.join(plan['join_tables']) if plan['join_tables'] else '(none)'}",
        f"Use cases: {use_case_titles(use_case_slugs)}",
        f"Privacy policy: {', '.join(privacy_policy)}",
        "",
        "## Best Result Per Variant",
        "```text",
        best_rows.to_string(index=False),
        "```",
        "",
        "## Full Benchmark Results",
        "```text",
        frame.to_string(index=False),
        "```",
    ]
    return "\n".join(lines)


def _build_graph_report(
    graph_results_path: Path,
    *,
    base_table: str,
    target_column: str,
    plan: dict[str, Any],
) -> str:
    frame = pd.read_csv(graph_results_path)
    summary = (
        frame.groupby(["approach", "algorithm"], as_index=False)["accuracy"]
        .mean()
        .sort_values("accuracy", ascending=False)
    )
    lines = [
        "# Base-Table Pipeline Graph-Mode Report",
        "",
        f"Base table: {base_table}",
        f"Target column: {target_column}",
        f"Join mode: {plan['join_mode']}",
        f"Join key: {plan['join_key']}",
        f"Join tables: {', '.join(plan['join_tables']) if plan['join_tables'] else '(none)'}",
        "",
        "## Mean Accuracy / R2 By Approach",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Full Graph-Mode Results",
        "```text",
        frame.to_string(index=False),
        "```",
    ]
    return "\n".join(lines)


def _build_dataset_for_graph_mode(
    *,
    data_dir: Path,
    base_table: str,
    target_column: str,
    plan: dict[str, Any],
    dataset_type: str,
) -> Dataset:
    from feature_discovery.config import DATA_FOLDER
    from feature_discovery.experiments.dataset_object import Dataset

    relative_base_path = data_dir.resolve().relative_to(DATA_FOLDER.resolve())
    return Dataset(
        base_table_path=relative_base_path,
        base_table_name=base_table,
        base_table_label=f"base_table_pipeline::{base_table.removesuffix('.csv')}",
        target_column=target_column,
        dataset_type=dataset_type,
        temporal_key=plan["join_key"] if plan["join_mode"] == "asof" else None,
        temporal_tolerance=int(plan["time_tolerance_seconds"]),
    )


def run_base_table_pipeline(
    *,
    data_dir: Path,
    metadata_path: Path | None,
    base_table: str,
    target_column: str,
    output_dir: Path,
    sample_rows: int = 5000,
    confidence_threshold: float = 0.7,
    temporal_confidence_threshold: float = 0.6,
    write_connections_to_data_dir: bool = False,
    run_benchmark: bool = False,
    run_graph_mode: bool = False,
    benchmark_models: list[str] | None = None,
    feature_source: str = "recompute",
    autofeatplus_results_csv: Path | None = None,
    autofeatplus_algorithm: str = "XGBoost",
    graph_algorithm: str = "XGB",
    graph_top_k: int = 15,
    graph_value_ratio: float = 0.65,
    dataset_type: str = "regression",
    python_bin: str | None = None,
) -> BaseTablePipelineArtifacts:
    benchmark_models = benchmark_models or ["ridge", "rf", "xt", "gbr"]
    python_bin = python_bin or str(Path.home() / "miniconda3" / "envs" / "autofeat-py3.10" / "bin" / "python")
    run_dir = output_dir / data_dir.name / base_table.removesuffix(".csv")
    run_dir.mkdir(parents=True, exist_ok=True)

    relationships = infer_dataset_relationships(
        data_dir=data_dir,
        metadata_path=metadata_path,
        sample_rows=sample_rows,
    )
    base_relationships = _filter_relationships_for_base_table(relationships, base_table)
    recommended = recommend_connections(
        base_relationships,
        confidence_threshold=confidence_threshold,
    )

    candidate_relationships_path = run_dir / "candidate_relationships.csv"
    recommended_connections_path = run_dir / "recommended_connections.csv"
    connections_path = run_dir / "connections.csv"
    relationship_report_path = run_dir / "relationship_report.md"
    benchmark_plan_path = run_dir / "benchmark_plan.txt"
    use_cases_path = run_dir / "use_cases.md"
    pipeline_summary_path = run_dir / "pipeline_summary.json"
    benchmark_report_path: Path | None = None
    benchmark_results_path: Path | None = None
    graph_results_path: Path | None = None
    graph_report_path: Path | None = None

    base_relationships.to_csv(candidate_relationships_path, index=False)
    recommended.to_csv(recommended_connections_path, index=False)
    export_connections_csv(recommended).to_csv(connections_path, index=False)
    if write_connections_to_data_dir or run_graph_mode:
        export_connections_csv(recommended).to_csv(data_dir / "connections.csv", index=False)

    relationship_report = build_relationship_report(
        data_dir=data_dir,
        metadata_path=metadata_path,
        relationships=base_relationships,
        recommended=recommended,
    )
    relationship_report_path.write_text(relationship_report, encoding="utf-8")

    plan = build_benchmark_plan(
        recommended=recommended,
        base_table=base_table,
        temporal_confidence_threshold=temporal_confidence_threshold,
    )
    benchmark_plan_path.write_text(
        "\n".join(
            [
                f"base_table={plan['base_table']}",
                f"join_mode={plan['join_mode']}",
                f"join_key={plan['join_key']}",
                f"time_tolerance_seconds={plan['time_tolerance_seconds']}",
                f"join_tables={','.join(plan['join_tables'])}",
            ]
        ),
        encoding="utf-8",
    )

    columns = _load_columns(data_dir, base_table, sample_rows=min(sample_rows, 1000))
    metadata_text = _load_metadata_text(metadata_path)
    use_case_slugs = infer_base_table_use_cases(
        base_table=base_table,
        columns=columns,
        target_column=target_column,
        metadata_text=metadata_text,
    )
    use_case_lines = [
        base_table_use_case_markdown(),
        "",
        "# Current Base Table",
        "",
        f"Base table: {base_table}",
        f"Target column: {target_column}",
        f"Columns: {', '.join(columns)}",
        "",
        f"Inferred use cases: {use_case_titles(use_case_slugs)}",
        f"Use case slugs: {'|'.join(use_case_slugs)}",
        "",
    ]
    use_cases_path.write_text("\n".join(use_case_lines), encoding="utf-8")
    privacy_policy = _infer_privacy_policy(columns, target_column)

    if run_benchmark and plan["join_tables"]:
        benchmark_results_path = run_dir / "benchmark_results.csv"
        root = Path(__file__).resolve().parents[3]
        benchmark_script = root / "downstream ML" / "benchmark_eur_augmented.py"
        command = [
            python_bin,
            str(benchmark_script),
            "--data-dir",
            str(data_dir),
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
            target_column,
            "--split-mode",
            "time",
            "--models",
            *benchmark_models,
            "--feature-source",
            feature_source,
            "--policy",
            *privacy_policy,
            "--output",
            str(benchmark_results_path),
        ]
        if autofeatplus_results_csv is not None:
            command.extend(["--autofeatplus-results-csv", str(autofeatplus_results_csv)])
        if autofeatplus_algorithm:
            command.extend(["--autofeatplus-algorithm", autofeatplus_algorithm])
        subprocess.run(command, check=True)
        benchmark_report_path = run_dir / "benchmark_report.md"
        benchmark_report_path.write_text(
            _build_benchmark_report(
                benchmark_results_path,
                base_table=base_table,
                target_column=target_column,
                plan=plan,
                use_case_slugs=use_case_slugs,
                privacy_policy=privacy_policy,
            ),
            encoding="utf-8",
        )

    if run_graph_mode and plan["join_tables"]:
        try:
            from feature_discovery.dataset_relation_graph.ingest_data import ingest_data_with_pk_fk
            from feature_discovery.experiments.ablation import autofeat, autofeat_plus
            from feature_discovery.experiments.baselines import join_all_bfs, non_augmented
            from feature_discovery.graph_processing.neo4j_transactions import clear_graph

            dataset = _build_dataset_for_graph_mode(
                data_dir=data_dir,
                base_table=base_table,
                target_column=target_column,
                plan=plan,
                dataset_type=dataset_type,
            )
            clear_graph()
            ingest_data_with_pk_fk(dataset=dataset, profile_valentine=False)

            base_df = pd.read_csv(data_dir / base_table)
            graph_rows = []
            graph_rows.extend(non_augmented(dataframe=base_df, dataset=dataset, algorithm=graph_algorithm))
            graph_rows.extend(join_all_bfs(dataset=dataset, algorithm=graph_algorithm))
            autofeat_rows, _ = autofeat(
                dataset=dataset,
                value_ratio=graph_value_ratio,
                top_k=graph_top_k,
                algorithm=graph_algorithm,
                store_augmented_data=False,
            )
            graph_rows.extend(autofeat_rows)
            autofeat_plus_rows, _ = autofeat_plus(
                dataset=dataset,
                value_ratio=graph_value_ratio,
                top_k=graph_top_k,
                algorithm=graph_algorithm,
                store_augmented_data=False,
            )
            graph_rows.extend(autofeat_plus_rows)

            graph_results_path = run_dir / "graph_mode_results.csv"
            pd.DataFrame([vars(row) for row in graph_rows]).to_csv(graph_results_path, index=False)
            graph_report_path = run_dir / "graph_mode_report.md"
            graph_report_path.write_text(
                _build_graph_report(
                    graph_results_path,
                    base_table=base_table,
                    target_column=target_column,
                    plan=plan,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            graph_report_path = run_dir / "graph_mode_report.md"
            graph_report_path.write_text(
                "# Base-Table Pipeline Graph-Mode Report\n\n"
                f"Graph-mode execution failed: {exc}\n",
                encoding="utf-8",
            )

    _write_pipeline_summary(
        pipeline_summary_path,
        base_table=base_table,
        target_column=target_column,
        columns=columns,
        use_case_slugs=use_case_slugs,
        plan=plan,
        privacy_policy=privacy_policy,
        benchmark_results_path=benchmark_results_path,
    )

    return BaseTablePipelineArtifacts(
        run_dir=run_dir,
        candidate_relationships_path=candidate_relationships_path,
        recommended_connections_path=recommended_connections_path,
        connections_path=connections_path,
        relationship_report_path=relationship_report_path,
        benchmark_plan_path=benchmark_plan_path,
        use_cases_path=use_cases_path,
        pipeline_summary_path=pipeline_summary_path,
        benchmark_report_path=benchmark_report_path,
        benchmark_results_path=benchmark_results_path,
        graph_results_path=graph_results_path,
        graph_report_path=graph_report_path,
    )
