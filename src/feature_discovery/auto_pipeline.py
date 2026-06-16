"""
auto_pipeline.py
================
End-to-end AutoFeat pipeline driven by a (base_table, target, data_dir) query.

Replaces Valentine schema-matching with a sentence-transformer over column
descriptions parsed from ``metadata.txt``, then runs ingest → discovery →
BASE / JOIN_ALL / AutoFeat and writes results.

Usage
-----
    python -m feature_discovery.auto_pipeline \\
        --base-table datasets/EUR/6907619/rabbitmq-performance.csv \\
        --target lat99 \\
        --data-dir  datasets/EUR/6907619 \\
        --dataset-type regression \\
        --temporal-key time --temporal-tolerance 60

Defaults
--------
  metadata     <data-dir>/../metadata.txt (fallback: <data-dir>/metadata.txt)
  model        sentence-transformers/all-mpnet-base-v2
  threshold    0.6 (schema cosine), 0.2 (value Jaccard)
  algorithm    XGB, top_k=15, value_ratio=0.65

Requires Neo4j to be reachable (same as benchmark_6g.py).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# DATA_FOLDER must be set *before* importing feature_discovery modules,
# since config.py reads it once at import time.
def _bootstrap_data_folder(data_dir: Path) -> Path:
    """Point config.DATA_FOLDER at the parent of ``data_dir`` and return the
    leaf folder name that the rest of the pipeline uses as ``dataset_folder_name``.

    The existing pipeline expects ``DATA_FOLDER / <dataset_folder_name> / *.csv``,
    so we split the user-supplied folder into (parent, leaf).
    """
    data_dir = data_dir.resolve()
    os.environ["DATA_FOLDER"] = str(data_dir.parent)
    return data_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated AutoFeat pipeline with transformer-based discovery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-table", required=True, type=Path,
                        help="Path to the base-table CSV.")
    parser.add_argument("--target", required=True,
                        help="Target column name in the base table.")
    parser.add_argument("--data-dir", required=True, type=Path,
                        help="Folder containing the base table and lake CSVs.")
    parser.add_argument("--metadata", type=Path, default=None,
                        help="Path to metadata.txt (default: <data-dir>/../metadata.txt).")
    parser.add_argument("--dataset-type",
                        choices=["regression", "binary", "multiclass", "auto"],
                        default="auto",
                        help="Problem type. 'auto' infers from the target column.")
    parser.add_argument("--label", default=None,
                        help="Scenario label for result files (default: base-table stem).")
    parser.add_argument("--temporal-key", default=None,
                        help="Column used as the temporal join key. If omitted, "
                             "auto-detected from a datetime / *_time* / *_timestamp column.")
    parser.add_argument("--temporal-tolerance", default=60,
                        help="Tolerance for asof joins. Accepts an int (seconds) "
                             "or a string like '60s', '5min', '1h', '200ms'. "
                             "Default: 60.")
    parser.add_argument("--temporal-direction", choices=["nearest", "backward", "forward"],
                        default="nearest",
                        help="merge_asof direction. Use 'backward' for forecasting "
                             "(no look-ahead leakage); 'nearest' (default) for static "
                             "analysis.")
    parser.add_argument("--skip-diagnose", action="store_true",
                        help="Skip pre-flight introspection of the base table.")
    parser.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2",
                        help="Sentence-transformers model name.")
    parser.add_argument("--schema-threshold", type=float, default=0.6,
                        help="Cosine threshold for the embedding match.")
    parser.add_argument("--value-threshold", type=float, default=0.2,
                        help="Jaccard threshold for value-overlap confirmation.")
    parser.add_argument("--algorithm", "--algorithms", dest="algorithms", default="XGB",
                        help="Comma-separated AutoGluon algorithm keys to evaluate per "
                             "approach. Choose from XGB, RF, GBM, XT, KNN, LR. "
                             "Example: --algorithms XGB,RF,KNN (default: XGB).")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--value-ratio", type=float, default=0.65)
    parser.add_argument(
        "--autofeat-plus-policy", nargs="*", default=["target-proxy-private"],
        help="One or more policy preset keys layered on top of DEFAULT_SENSITIVE_PATTERNS "
             "for the AutoFeatPlus pass. Valid: time-private, resource-private, "
             "workload-private, target-proxy-private, all, none. "
             "Default ['target-proxy-private'] strips sibling-percentile leakers "
             "without blocking the temporal key.")
    parser.add_argument(
        "--autofeat-plus-top-k", type=int, default=15,
        help="Top-k for the AutoFeatPlus pass; defaults to --top-k for symmetry "
             "with the AutoFeat pass.")
    parser.add_argument("--seed", type=int, default=int(os.getenv("AUTOFEAT_SEED", "42")),
                        help="Global random seed for reproducible runs (default: 42). "
                             "Pins PYTHONHASHSEED, NumPy/Python/Torch RNGs, and the "
                             "train/test split.")
    parser.add_argument("--no-ingest", action="store_true",
                        help="Skip Neo4j clear + ingest (graph already populated).")
    parser.add_argument("--no-transformer-discovery", action="store_true",
                        help="Ingest connections.csv only; skip transformer-based edge discovery. "
                             "Use when the data ships with explicit PK/FK edges and a sentence-"
                             "transformer scan would be wasted work (e.g. KUL CSI with 200+ cols).")
    parser.add_argument("--no-store", action="store_true",
                        help="Do not save augmented datasets to disk.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def _ensure_reproducible(seed: int) -> None:
    """Make the whole run deterministic.

    PYTHONHASHSEED governs set/dict iteration order and must be fixed *before* the
    interpreter starts — feature-selection tie-breaks depend on it — so if it is
    not already pinned we set it and re-exec this module once. Then seed the
    Python/NumPy/Torch RNGs for the rest of the process.
    """
    # Make the seed visible to config.SEED (read at import) so every module that
    # routes its RNG through it — train/test split, group sampling — uses --seed.
    # NOTE: do NOT import feature_discovery here. config.py freezes DATA_FOLDER at
    # import time, and that env var is only set later by _bootstrap_data_folder();
    # importing config before the bootstrap would pin the wrong data directory.
    os.environ["AUTOFEAT_SEED"] = str(seed)
    if os.environ.get("PYTHONHASHSEED") != str(seed):
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.execv(sys.executable,
                 [sys.executable, "-m", "feature_discovery.auto_pipeline", *sys.argv[1:]])
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def main() -> None:
    args = _parse_args()
    _ensure_reproducible(args.seed)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    data_dir = _bootstrap_data_folder(args.data_dir)
    base_table = args.base_table.resolve()
    if not base_table.is_file():
        sys.exit(f"Base table not found: {base_table}")
    if base_table.parent.resolve() != data_dir:
        sys.exit(f"--base-table must live inside --data-dir ({data_dir}); got {base_table}")

    metadata_path = args.metadata.resolve() if args.metadata else None
    if metadata_path is None:
        for candidate in (data_dir / "metadata.txt", data_dir.parent / "metadata.txt"):
            if candidate.exists():
                metadata_path = candidate
                break
    if metadata_path is None:
        logging.warning("No metadata.txt found near %s; matching on column names only.", data_dir)

    # Imports deferred until after DATA_FOLDER is set.
    from feature_discovery.config import RESULTS_FOLDER
    from feature_discovery.dataset_relation_graph.ingest_data import ingest_data_with_pk_fk
    from feature_discovery.experiments.ablation import autofeat
    from feature_discovery.experiments.baselines import join_all_bfs, non_augmented
    from feature_discovery.experiments.dataset_object import Dataset
    from feature_discovery.experiments.result_object import Result
    from feature_discovery.graph_processing.neo4j_transactions import clear_graph
    import pandas as pd

    label = args.label or base_table.stem

    # ─── Pre-flight: introspect the base table ──────────────────────────────
    dataset_type = args.dataset_type
    temporal_key = args.temporal_key
    if not args.skip_diagnose:
        from feature_discovery.dataset_introspection import diagnose
        diag = diagnose(base_table, args.target,
                        dataset_type=None if dataset_type == "auto" else dataset_type,
                        temporal_key=temporal_key)
        print("Pre-flight:")
        if diag.inferred_problem_type:
            print(f"  inferred problem type   : {diag.inferred_problem_type}")
        if diag.inferred_temporal_key:
            print(f"  inferred temporal key   : {diag.inferred_temporal_key} (unit: {diag.timestamp_unit})")
        if diag.candidate_join_keys:
            print(f"  candidate join keys     : {diag.candidate_join_keys}")
        for w in diag.warnings:
            print(f"  ⚠  {w}")
        for e in diag.errors:
            print(f"  ✗  {e}")
        if diag.errors:
            sys.exit("\nAborted: fix the errors above or pass --skip-diagnose to force.")
        if dataset_type == "auto":
            dataset_type = diag.inferred_problem_type or "regression"
        if temporal_key is None and diag.inferred_temporal_key:
            temporal_key = diag.inferred_temporal_key
            print(f"  → using --temporal-key {temporal_key}")
    elif dataset_type == "auto":
        dataset_type = "regression"
        print("--skip-diagnose set and --dataset-type auto: defaulting to 'regression'.")

    dataset = Dataset(
        base_table_path=Path(data_dir.name),                       # leaf name, relative to DATA_FOLDER
        base_table_name=base_table.name,
        base_table_label=label,
        target_column=args.target,
        dataset_type=dataset_type,
        temporal_key=temporal_key,
        temporal_tolerance=args.temporal_tolerance,
        temporal_direction=args.temporal_direction,
    )

    print(f"\n{'='*60}")
    print(f"Auto pipeline")
    from feature_discovery.config import rel
    print(f"  base   : {rel(base_table)}")
    print(f"  target : {args.target}  ({args.dataset_type})")
    print(f"  lake   : {rel(data_dir)}")
    print(f"  meta   : {rel(metadata_path) if metadata_path else None}")
    print(f"  label  : {label}")
    print(f"  model  : {args.model}")
    print(f"{'='*60}\n")

    # 1) Ingest into Neo4j + (optional) transformer-based edge discovery
    if not args.no_ingest:
        print("Clearing Neo4j graph ...")
        clear_graph()
        run_transformer = not args.no_transformer_discovery
        connections_dump = data_dir / "connections_transformer.csv" if run_transformer else None
        ingest_data_with_pk_fk(
            dataset=dataset,
            profile_transformer=run_transformer,
            transformer_schema_threshold=args.schema_threshold,
            transformer_value_threshold=args.value_threshold,
            transformer_model=args.model,
            metadata_path=metadata_path,
            transformer_connections_csv=connections_dump,
        )
        if run_transformer:
            print(f"  Edge dump → {rel(connections_dump)}\n")
        else:
            print("  Transformer discovery skipped — using explicit connections.csv only.\n")

    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    print(f"Algorithms to evaluate: {algorithms}\n")

    all_results: List[Result] = []
    base_df = pd.read_csv(base_table, header=0, engine="python",
                          encoding="utf8", quotechar='"', escapechar="\\")

    for ai, alg in enumerate(algorithms, 1):
        print(f"\n══════ Algorithm {ai}/{len(algorithms)}: {alg} ══════")

        # 2) BASE (no augmentation)
        print(f"[1/3 · {alg}] BASE ...")
        all_results.extend(
            non_augmented(dataframe=base_df, dataset=dataset, algorithm=alg)
        )

        # 3) JOIN_ALL
        print(f"\n[2/3 · {alg}] JOIN_ALL ...")
        try:
            all_results.extend(join_all_bfs(
                dataset=dataset, algorithm=alg,
                autofeat_plus_policies=args.autofeat_plus_policy,
                autofeat_plus_top_k=args.autofeat_plus_top_k,
            ))
        except Exception as e:
            import traceback
            print(f"  JOIN_ALL failed: {e}")
            traceback.print_exc()

        # 4) AutoFeat
        print(f"\n[3/3 · {alg}] AutoFeat ...")
        try:
            autofeat_results, _ = autofeat(
                dataset=dataset,
                value_ratio=args.value_ratio,
                top_k=args.top_k,
                algorithm=alg,
                store_augmented_data=not args.no_store,
            )
            all_results.extend(autofeat_results)
        except Exception as e:
            import traceback
            print(f"  AutoFeat failed: {e}")
            traceback.print_exc()

    if not all_results:
        print("No results produced.")
        return

    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    _write_clean_outputs(all_results, label, RESULTS_FOLDER)


def _write_clean_outputs(results: List, label: str, out_dir) -> None:
    """Write three output files alongside each pipeline run:

      auto_pipeline_<label>_raw.csv      original wide format (all 25 columns)
      auto_pipeline_<label>_summary.csv  clean metrics (scenario, approach, algorithm, accuracy, ...)
      auto_pipeline_<label>_features.csv long-format feature importance (one row per feature)

    The legacy `auto_pipeline_<label>.csv` is kept as an alias for `_raw` so existing
    consumers (dashboards, summary scripts) keep working.
    """
    import ast
    import pandas as pd
    from pathlib import Path

    from feature_discovery.config import rel

    raw = pd.DataFrame([vars(r) for r in results])
    raw_path = Path(out_dir) / f"auto_pipeline_{label}.csv"
    raw.to_csv(raw_path, index=False)

    # Clean summary: one row per (approach, algorithm) — drop the heavy dict columns
    summary_cols = ["data_label", "approach", "algorithm", "accuracy",
                    "rmse", "mae", "n_features", "train_time", "total_time",
                    "feature_selection_time", "rank", "top_k", "join_name"]
    summary = (
        raw[[c for c in summary_cols if c in raw.columns]]
        .drop_duplicates(subset=["approach", "algorithm"], keep="first")
        .rename(columns={"data_label": "scenario"})
        .reset_index(drop=True)
    )
    summary_path = Path(out_dir) / f"auto_pipeline_{label}_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Long-format features
    rows = []
    for r in results:
        imp = r.feature_importance
        if isinstance(imp, str):
            try:
                imp = ast.literal_eval(imp)
            except (ValueError, SyntaxError):
                imp = {}
        if not isinstance(imp, dict):
            continue
        for feature, value in imp.items():
            # Pull the source table out of the feature name.
            if "." in feature and "/" in feature:
                source = feature.rsplit(".", 1)[0].split("/")[-1]
            elif "." in feature:
                source = feature.rsplit(".", 1)[0]
            else:
                source = "base"
            rows.append({
                "scenario": r.data_label,
                "approach": r.approach,
                "algorithm": r.algorithm,
                "feature": feature,
                "source_table": source,
                "importance": float(value),
            })
    features = pd.DataFrame(rows)
    features_path = Path(out_dir) / f"auto_pipeline_{label}_features.csv"
    features.to_csv(features_path, index=False)

    print(f"\nSaved {len(results)} raw results → {rel(raw_path)}")
    print(f"      {len(summary)} summary rows → {rel(summary_path)}")
    print(f"      {len(features)} feature rows → {rel(features_path)}")


if __name__ == "__main__":
    main()
