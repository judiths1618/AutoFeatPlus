"""
benchmark_6g.py
===============
Runs all three 6G benchmark scenarios and writes a consolidated comparison CSV.

Scenarios
---------
  1  Intra-lake temporal   rabbitmq (lat99, asof join Δ=60s)
  2C Synthetic projection  rabbitmq-reduced (lat99, exact join to full rabbitmq)
  3  Segment-level         amf seg01 (lat99, config join on `n`)

For each scenario the following methods are evaluated:
  BASE        — train on base table alone (lower bound)
  JOIN_ALL    — join all reachable tables, no feature selection
  AutoFeat    — BFS + mRMR feature selection

Results are saved to:
  results/6g_data/benchmark_6g_<scenario_label>.csv   (per-scenario)
  results/6g_data/benchmark_6g_summary.csv            (all scenarios merged)

Usage (from repo root, conda env autofeat-py3.10):
  python -m feature_discovery.experiments.benchmark_6g
  python -m feature_discovery.experiments.benchmark_6g --scenario scenario1_rabbitmq
  python -m feature_discovery.experiments.benchmark_6g --algorithm XGB
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from feature_discovery.config import DATA_FOLDER, RESULTS_FOLDER
from feature_discovery.dataset_relation_graph.ingest_data import ingest_data_with_pk_fk
from feature_discovery.graph_processing.neo4j_transactions import clear_graph
from feature_discovery.experiments.ablation import autofeat, autofeat_plus
from feature_discovery.experiments.baselines import non_augmented, join_all_bfs
from feature_discovery.experiments.dataset_object import Dataset
from feature_discovery.experiments.init_datasets import init_datasets, ALL_DATASETS
from feature_discovery.experiments.result_object import Result
from feature_discovery.experiments.utils_dataset import filter_datasets

logging.getLogger().setLevel(logging.WARNING)

# ─── Constants ────────────────────────────────────────────────────────────────
SCENARIO_LABELS = [
    "scenario1_rabbitmq",
    "scenario2c_rabbitmq_reduced",
    "scenario3_amf_seg01",
    "kul_nomadic_ula_static",
]

DEFAULT_ALGORITHM = "XGB"
DEFAULT_TOP_K = 15
DEFAULT_VALUE_RATIO = 0.65


# ─── Per-scenario ingestion ───────────────────────────────────────────────────
def _ingest_scenario(dataset: Dataset) -> None:
    """Load the dataset's tables + connections into Neo4j."""
    print(f"  Ingesting Neo4j graph for '{dataset.base_table_label}' ...")
    ingest_data_with_pk_fk(dataset=dataset, profile_valentine=False)


# ─── Run one scenario ─────────────────────────────────────────────────────────
def run_scenario(
    dataset: Dataset,
    algorithm: str = DEFAULT_ALGORITHM,
    top_k: int = DEFAULT_TOP_K,
    value_ratio: float = DEFAULT_VALUE_RATIO,
    ingest: bool = True,
    store_augmented_data: bool = True,
    include_autofeat_plus: bool = False,
) -> List[Result]:
    label = dataset.base_table_label
    print(f"\n{'='*60}")
    print(f"Scenario : {label}")
    print(f"Base     : {dataset.base_table_id}")
    print(f"Target   : {dataset.target_column}  ({dataset.dataset_type})")
    print(f"Temporal : key={dataset.temporal_key}, tolerance={dataset.temporal_tolerance}s")
    print(f"Algorithm: {algorithm}")
    print(f"{'='*60}")

    if ingest:
        print("  Clearing Neo4j graph ...")
        clear_graph()
        _ingest_scenario(dataset)

    all_results: List[Result] = []

    # 1. Baseline — no augmentation
    print("\n[1/3] BASE ...")
    base_df = pd.read_csv(
        DATA_FOLDER / dataset.base_table_id,
        header=0, engine="python", encoding="utf8", quotechar='"', escapechar="\\",
    )
    base_results = non_augmented(dataframe=base_df, dataset=dataset, algorithm=algorithm)
    all_results.extend(base_results)

    # 2. Join-All (upper bound on data, no selection)
    print("\n[2/3] JOIN_ALL ...")
    try:
        join_all_results = join_all_bfs(dataset=dataset, algorithm=algorithm)
        all_results.extend(join_all_results)
    except Exception as e:
        import traceback
        print(f"  JOIN_ALL failed: {e}")
        traceback.print_exc()

    # 3. AutoFeat (BFS + mRMR)
    print("\n[3/3] AutoFeat ...")
    try:
        autofeat_results, top_k_paths = autofeat(
            dataset=dataset,
            value_ratio=value_ratio,
            top_k=top_k,
            algorithm=algorithm,
            store_augmented_data=store_augmented_data,
        )
        all_results.extend(autofeat_results)
    except Exception as e:
        import traceback
        print(f"  AutoFeat failed: {e}")
        traceback.print_exc()

    if include_autofeat_plus:
        print("\n[4/4] AutoFeatPlus ...")
        try:
            autofeat_plus_results, _ = autofeat_plus(
                dataset=dataset,
                value_ratio=value_ratio,
                top_k=top_k,
                algorithm=algorithm,
                store_augmented_data=store_augmented_data,
            )
            all_results.extend(autofeat_plus_results)
        except Exception as e:
            import traceback
            print(f"  AutoFeatPlus failed: {e}")
            traceback.print_exc()

    # Save per-scenario CSV
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_FOLDER / f"benchmark_6g_{label}.csv"
    pd.DataFrame([vars(r) for r in all_results]).to_csv(out_path, index=False)
    print(f"\nSaved {len(all_results)} results → {out_path}")

    return all_results


# ─── Summary table ────────────────────────────────────────────────────────────
def summarise(all_results: List[Result]) -> pd.DataFrame:
    """Return a pivot table: scenario × approach → mean score.

    For regression tasks the score is R² (higher is better, 1.0 = perfect).
    For classification tasks the score is accuracy.
    """
    df = pd.DataFrame([vars(r) for r in all_results])
    if df.empty:
        return df

    summary = (
        df.groupby(["data_label", "approach", "algorithm"])["accuracy"]
        .mean()
        .reset_index()
        .rename(columns={"accuracy": "mean_score", "data_label": "scenario"})
        .sort_values(["scenario", "mean_score"], ascending=[True, False])
    )
    return summary


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Run 6G benchmark scenarios.")
    parser.add_argument(
        "--scenario", "-s",
        nargs="+",
        default=SCENARIO_LABELS,
        help="Scenario label(s) to run (default: all three).",
    )
    parser.add_argument(
        "--algorithm", "-a",
        default=DEFAULT_ALGORITHM,
        help="AutoGluon algorithm key, e.g. XGB, RF, KNN (default: XGB).",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int, default=DEFAULT_TOP_K,
        help="Top-k join paths for AutoFeat (default: 15).",
    )
    parser.add_argument(
        "--value-ratio", "-v",
        type=float, default=DEFAULT_VALUE_RATIO,
        help="Null-value pruning threshold (default: 0.65).",
    )
    parser.add_argument(
        "--no-ingest",
        action="store_true",
        help="Skip Neo4j ingestion (use when graph is already populated).",
    )
    parser.add_argument(
        "--no-store",
        action="store_true",
        help="Do not save augmented datasets to disk.",
    )
    parser.add_argument(
        "--include-autofeat-plus",
        action="store_true",
        help="Also run the privacy/cost-aware AutoFeatPlus prototype.",
    )
    args = parser.parse_args()

    init_datasets()

    target_labels = args.scenario
    datasets = [d for d in ALL_DATASETS if d.base_table_label in target_labels]
    missing = set(target_labels) - {d.base_table_label for d in datasets}
    if missing:
        print(f"Warning: scenario(s) not found in datasets.csv: {missing}", file=sys.stderr)

    all_results: List[Result] = []
    for dataset in datasets:
        results = run_scenario(
            dataset=dataset,
            algorithm=args.algorithm,
            top_k=args.top_k,
            value_ratio=args.value_ratio,
            ingest=not args.no_ingest,
            store_augmented_data=not args.no_store,
            include_autofeat_plus=args.include_autofeat_plus,
        )
        all_results.extend(results)

    if not all_results:
        print("No results produced.")
        return

    summary = summarise(all_results)
    summary_path = RESULTS_FOLDER / "benchmark_6g_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY  (mean_score: R² for regression, accuracy for classification)")
    print(f"{'='*60}")
    print(summary.to_string(index=False))
    print(f"\nSaved → {summary_path}")


if __name__ == "__main__":
    main()
