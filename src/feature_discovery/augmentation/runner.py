"""
AutoFeatRunner — programmatic driver that wraps the CLI pipeline so it can be
called from inside an Airflow / Dagster / Prefect task (or any plain Python
script).

The runner:
  1. Stages every input (DataFrame *or* path) into a temporary working folder.
  2. Optionally writes a `connections.csv` from the config.
  3. Sets DATA_FOLDER, clears Neo4j, and invokes the pipeline in-process.
  4. Collects results into an `AugmentationResult` and (optionally) writes the
     augmented DataFrame to disk for downstream tasks to pick up.
  5. Cleans the workdir unless `keep_workdir=True`.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from feature_discovery.augmentation.config import AugmentationConfig, TableSource
from feature_discovery.augmentation.result import AugmentationResult

log = logging.getLogger(__name__)


class AutoFeatRunner:
    """One-shot runner. Construct with a config, then call ``.run()``."""

    def __init__(self, config: AugmentationConfig) -> None:
        self.config = config
        self.workdir: Optional[Path] = None

    # ─── Public ──────────────────────────────────────────────────────────
    def run(self) -> AugmentationResult:
        cfg = self.config
        self.workdir = self._stage_inputs()

        # Pre-flight introspection (also auto-detects dataset_type / temporal_key).
        diagnose_report = self._preflight(self.workdir / self._base_name())

        dataset_type = cfg.dataset_type or diagnose_report.get("inferred_problem_type") or "regression"
        temporal_key = cfg.temporal_key or diagnose_report.get("inferred_temporal_key")

        # Configure DATA_FOLDER before importing pipeline modules so config.py
        # resolves to our workdir's parent.
        os.environ["DATA_FOLDER"] = str(self.workdir.parent)

        # Deferred imports — config.py reads DATA_FOLDER at import time.
        from feature_discovery.config import RESULTS_FOLDER
        from feature_discovery.dataset_relation_graph.ingest_data import ingest_data_with_pk_fk
        from feature_discovery.experiments.ablation import autofeat
        from feature_discovery.experiments.baselines import join_all_bfs, non_augmented
        from feature_discovery.experiments.dataset_object import Dataset
        from feature_discovery.graph_processing.neo4j_transactions import clear_graph

        label = cfg.label or f"run_{uuid.uuid4().hex[:8]}"
        dataset = Dataset(
            base_table_path=Path(self.workdir.name),
            base_table_name=self._base_name(),
            base_table_label=label,
            target_column=cfg.target,
            dataset_type=dataset_type,
            temporal_key=temporal_key,
            temporal_tolerance=cfg.temporal_tolerance,
            temporal_direction=cfg.temporal_direction,
        )

        # Ingest into Neo4j (+ optional transformer discovery).
        log.info("Clearing Neo4j graph for run '%s'", label)
        clear_graph()
        ingest_data_with_pk_fk(
            dataset=dataset,
            profile_transformer=cfg.use_transformer_discovery,
            transformer_schema_threshold=cfg.schema_threshold,
            transformer_value_threshold=cfg.value_threshold,
            transformer_model=cfg.model,
            metadata_path=self.workdir / "metadata.txt" if (self.workdir / "metadata.txt").exists() else None,
            transformer_connections_csv=None,
        )

        # Run each algorithm through BASE / JOIN_ALL / AutoFeat.
        all_results: list = []
        base_df = pd.read_csv(self.workdir / self._base_name(), header=0, engine="python",
                              encoding="utf8", quotechar='"', escapechar="\\")

        best_join_path = ""
        best_features: List[str] = []
        for alg in cfg.algorithms:
            log.info("Algorithm %s: BASE", alg)
            all_results.extend(non_augmented(dataframe=base_df, dataset=dataset, algorithm=alg))
            log.info("Algorithm %s: Join_All_BFS", alg)
            try:
                all_results.extend(join_all_bfs(dataset=dataset, algorithm=alg))
            except Exception as exc:
                log.warning("Join_All_BFS failed for %s: %s", alg, exc)
            log.info("Algorithm %s: AutoFeat", alg)
            try:
                af_results, top_paths = autofeat(
                    dataset=dataset,
                    value_ratio=cfg.value_ratio,
                    top_k=cfg.top_k,
                    algorithm=alg,
                    store_augmented_data=False,
                )
                all_results.extend(af_results)
                # Remember the best AutoFeat path for materialising the augmented df.
                if not best_join_path:
                    af = [r for r in af_results if r.approach == "AutoFeat"]
                    if af:
                        best_join_path = af[0].join_name or ""
                        best_features = list(af[0].join_path_features or [])
            except Exception as exc:
                log.warning("AutoFeat failed for %s: %s", alg, exc)

        # Build the AugmentationResult.
        result = self._build_result(all_results, base_df, best_join_path, best_features,
                                    label, dataset_type, diagnose_report,
                                    Path(RESULTS_FOLDER) if cfg.output_dir is None else cfg.output_dir)

        # Cleanup
        if not cfg.keep_workdir:
            shutil.rmtree(self.workdir, ignore_errors=True)
            self.workdir = None
        result.workdir = self.workdir
        return result

    # ─── Internal ────────────────────────────────────────────────────────
    def _base_name(self) -> str:
        return _source_filename(self.config.base, default="base.csv")

    def _stage_inputs(self) -> Path:
        """Materialise base + lake into a unique tempdir under datasets/ so the
        pipeline's DATA_FOLDER convention resolves correctly."""
        cfg = self.config
        from feature_discovery.config import DATA_FOLDER
        # If DATA_FOLDER isn't writable (e.g. external mount), fall back to /tmp.
        parent = DATA_FOLDER if Path(DATA_FOLDER).exists() else Path(tempfile.gettempdir())
        workdir = Path(parent) / f"autofeat_run_{uuid.uuid4().hex[:8]}"
        workdir.mkdir(parents=True, exist_ok=True)

        # Base
        _stage_one(cfg.base, workdir / self._base_name())

        # Lake
        lake = cfg.lake
        if isinstance(lake, dict):
            for name, src in lake.items():
                fname = name if name.endswith(".csv") else f"{name}.csv"
                _stage_one(src, workdir / fname)
        else:
            for src in lake:
                _stage_one(src, workdir / _source_filename(src))

        # connections.csv (optional)
        if cfg.connections is not None:
            target = workdir / "connections.csv"
            if isinstance(cfg.connections, (str, Path)):
                shutil.copy(cfg.connections, target)
            else:
                pd.DataFrame(cfg.connections).to_csv(target, index=False)

        return workdir

    def _preflight(self, base_path: Path) -> Dict[str, object]:
        if self.config.skip_diagnose:
            return {}
        from feature_discovery.dataset_introspection import diagnose
        diag = diagnose(base_path, self.config.target,
                        dataset_type=self.config.dataset_type,
                        temporal_key=self.config.temporal_key)
        if diag.errors:
            raise ValueError("Pre-flight failed:\n  " + "\n  ".join(diag.errors))
        return {
            "inferred_problem_type": diag.inferred_problem_type,
            "inferred_temporal_key": diag.inferred_temporal_key,
            "timestamp_unit": diag.timestamp_unit,
            "candidate_join_keys": diag.candidate_join_keys,
            "warnings": diag.warnings,
        }

    def _build_result(self, raw_results, base_df, best_join_path, best_features,
                      label, dataset_type, diagnose_report, output_dir) -> AugmentationResult:
        cfg = self.config
        # Summary (one row per approach/algorithm)
        summary = (
            pd.DataFrame([vars(r) for r in raw_results])
            [["approach", "algorithm", "accuracy", "n_features", "train_time"]]
            .drop_duplicates(subset=["approach", "algorithm"], keep="first")
            .reset_index(drop=True)
        ) if raw_results else pd.DataFrame()

        # Feature-importance long table
        rows = []
        for r in raw_results:
            imp = r.feature_importance or {}
            for feature, value in imp.items():
                source = (feature.rsplit(".", 1)[0].split("/")[-1]
                          if ("/" in feature and "." in feature)
                          else (feature.rsplit(".", 1)[0] if "." in feature else "base"))
                rows.append({
                    "approach": r.approach,
                    "algorithm": r.algorithm,
                    "feature": feature,
                    "source_table": source,
                    "importance": float(value),
                })
        feature_importance = pd.DataFrame(rows)

        # Augmented DataFrame: the base table by default. If AutoFeat
        # selected joined features, materialise the joined df by reading
        # the lake tables and joining via the connections we know about.
        augmented = base_df.copy()
        if best_features:
            try:
                augmented = self._materialise_augmented(best_join_path, best_features, cfg.target)
            except Exception as exc:
                log.warning("Could not materialise augmented df: %s", exc)

        # Persist outputs.
        summary_path = augmented_path = None
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            summary_path = output_dir / f"auto_pipeline_{label}_summary.csv"
            summary.to_csv(summary_path, index=False)
            if not augmented.empty:
                augmented_path = output_dir / f"auto_pipeline_{label}_augmented.csv"
                augmented.to_csv(augmented_path, index=False)

        return AugmentationResult(
            augmented_df=augmented,
            selected_features=best_features,
            selected_join_path=best_join_path,
            summary=summary,
            feature_importance=feature_importance,
            summary_path=summary_path,
            augmented_path=augmented_path,
            diagnose=diagnose_report,
        )

    def _materialise_augmented(self, join_name: str, selected: List[str],
                               target: str) -> pd.DataFrame:
        """Reconstruct the augmented DataFrame using the evaluated join path.

        This mirrors ``evaluate_paths`` so DataOps exports match the dataframe
        used for scoring instead of doing a separate best-effort re-join.
        """
        if self.workdir is None or not selected or not join_name:
            return pd.read_csv(self.workdir / self._base_name()) if self.workdir else pd.DataFrame()

        from feature_discovery.experiments.evaluate_join_paths import materialize_augmented_dataframe

        base_node = str(Path(self.workdir.name) / self._base_name())
        return materialize_augmented_dataframe(
            join_name=join_name,
            selected_features=selected,
            target_column=target,
            base_node=base_node,
            temporal_key=self.config.temporal_key,
            temporal_tolerance=self.config.temporal_tolerance,
            temporal_direction=self.config.temporal_direction,
        )


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _source_filename(src: TableSource, default: str = "table.csv") -> str:
    if isinstance(src, pd.DataFrame):
        return default
    return Path(str(src)).name


def _stage_one(src: TableSource, dst: Path) -> None:
    if isinstance(src, pd.DataFrame):
        src.to_csv(dst, index=False)
    else:
        shutil.copy(str(src), dst)


# ─── Convenience function ───────────────────────────────────────────────────
def augment_features(
    base: TableSource,
    lake: Union[List[TableSource], Dict[str, TableSource]],
    target: str,
    **kwargs,
) -> AugmentationResult:
    """One-shot wrapper. Equivalent to::

        AutoFeatRunner(AugmentationConfig(base=base, lake=lake, target=target,
                                          **kwargs)).run()

    Supports the same kwargs as `AugmentationConfig` — `dataset_type`,
    `temporal_key`, `temporal_tolerance`, `temporal_direction`, `algorithms`,
    `top_k`, `value_ratio`, `use_transformer_discovery`, `skip_diagnose`,
    `label`, `output_dir`, `keep_workdir`, `connections`.
    """
    config = AugmentationConfig(base=base, lake=lake, target=target, **kwargs)
    return AutoFeatRunner(config).run()
