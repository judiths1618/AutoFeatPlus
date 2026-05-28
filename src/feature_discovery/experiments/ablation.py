import logging
import time
from pathlib import Path
from typing import Tuple, List

import pandas as pd

from feature_discovery.autofeat_pipeline.autofeat import AutoFeat
from feature_discovery.autofeat_pipeline.join_path_utils import get_path_length
from feature_discovery.config import RESULTS_FOLDER
from feature_discovery.experiments.autofeat_plus import select_autofeat_plus_features
from feature_discovery.experiments.dataset_object import Dataset
from feature_discovery.experiments.evaluate_join_paths import evaluate_paths
from feature_discovery.experiments.init_datasets import init_datasets
from feature_discovery.experiments.result_object import Result
from feature_discovery.experiments.utils_dataset import filter_datasets
from feature_discovery.helpers.optional_polars import POLARS_AVAILABLE


def autofeat(
    dataset: Dataset,
    value_ratio: float,
    top_k: int,
    algorithm: str,
    approach: str = Result.TFD,
    pearson: bool = False,
    jmi: bool = False,
    no_relevance: bool = False,
    no_redundancy: bool = False,
    save_joins_to_disk: bool = True,
    store_augmented_data: bool = True,
    use_polars: bool = POLARS_AVAILABLE,
) -> Tuple[List[Result], List[Tuple]]:
    logging.debug(f"Running on TFD (Transitive Feature Discovery) result with AutoGluon")

    start = time.time()
    bfs_traversal = AutoFeat(
        base_table_id=str(dataset.base_table_id),
        base_table_label=dataset.base_table_label,
        save_joins_to_disk=save_joins_to_disk,
        use_polars=use_polars,
        target_column=dataset.target_column,
        value_ratio=value_ratio,
        top_k=top_k,
        task=dataset.dataset_type,
        pearson=pearson,
        jmi=jmi,
        no_redundancy=no_redundancy,
        no_relevance=no_relevance,
        temporal_key=dataset.temporal_key,
        temporal_tolerance=dataset.temporal_tolerance,
    )
    bfs_traversal.streaming_feature_selection(queue={str(dataset.base_table_id)})
    end = time.time()

    logging.debug(f"FINISHED {approach}")

    all_results, top_k_paths = evaluate_paths(
        bfs_result=bfs_traversal,
        problem_type=dataset.dataset_type,
        algorithm=algorithm,
        store_augmented_data=store_augmented_data,
    )
    for result in all_results:
        result.approach = approach
        result.feature_selection_time = end - start
        result.total_time += result.feature_selection_time
        result.top_k = top_k
        result.data_label = dataset.base_table_label
        result.cutoff_threshold = value_ratio
        result.n_features = len(result.join_path_features)

    logging.debug("Save results ... ")
    print(f'Save results ... {all_results, top_k_paths}')
    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / f"{dataset.base_table_label}_{approach}.csv", index=False)

    return all_results, top_k_paths


def autofeat_plus(
    dataset: Dataset,
    value_ratio: float,
    top_k: int,
    algorithm: str,
    approach: str = Result.AUTOFEAT_PLUS,
    privacy_penalty: float = 0.25,
    path_length_penalty: float = 0.01,
    save_joins_to_disk: bool = True,
    store_augmented_data: bool = True,
    use_polars: bool = POLARS_AVAILABLE,
) -> Tuple[List[Result], List[Tuple]]:
    """Run AutoFeatPlus for 6G-style augmentation.

    Reuses AutoFeat's BFS graph exploration, then applies data-driven
    privacy-aware feature scoring (utility vs. privacy risk via
    ``select_autofeat_plus_features``) and cost-aware path reranking
    before evaluating the selected join paths.

    Privacy penalty is normalized by the number of features in each path so
    that the adjustment is on the same scale as the mRMR ranking score.
    """
    start = time.time()
    bfs_traversal = AutoFeat(
        base_table_id=str(dataset.base_table_id),
        base_table_label=dataset.base_table_label,
        save_joins_to_disk=save_joins_to_disk,
        use_polars=use_polars,
        target_column=dataset.target_column,
        value_ratio=value_ratio,
        top_k=top_k,
        task=dataset.dataset_type,
        temporal_key=dataset.temporal_key,
        temporal_tolerance=dataset.temporal_tolerance,
    )
    bfs_traversal.streaming_feature_selection(queue={str(dataset.base_table_id)})
    end = time.time()

    # ── Privacy-aware re-scoring of each join path ──────────────────────────
    # For each path we load the corresponding intermediate joined dataframe
    # (sampled, saved to temp dir during BFS) and run select_autofeat_plus_features
    # which computes data-driven utility (Spearman corr) and three privacy risk
    # dimensions (name pattern, proxy correlation, identifier cardinality).
    blocked_features_by_path: dict = {}
    sensitive_features_by_path: dict = {}

    for join_name, features in list(bfs_traversal.partial_join_selected_features.items()):
        if join_name == bfs_traversal.base_table_id:
            # Base table: apply privacy filter but keep ranking score unchanged.
            selection = select_autofeat_plus_features(
                dataframe=bfs_traversal.partial_join[
                    [f for f in features if f in bfs_traversal.partial_join.columns]
                    + [dataset.target_column]
                ],
                target_column=dataset.target_column,
                top_k=len(features),
                privacy_penalty=privacy_penalty,
                block_sensitive=True,
            )
            bfs_traversal.partial_join_selected_features[join_name] = selection.selected_features
            blocked_features_by_path[join_name] = selection.blocked_features
            sensitive_features_by_path[join_name] = selection.sensitive_features
            continue

        filename_key = bfs_traversal.join_name_mapping.get(join_name)
        if filename_key is None:
            blocked_features_by_path[join_name] = []
            sensitive_features_by_path[join_name] = []
            continue

        try:
            if bfs_traversal.save_joins_to_disk:
                joined_df = pd.read_parquet(Path(bfs_traversal.temp_dir.name) / filename_key)
            else:
                joined_df = bfs_traversal.joins_to_df.get(filename_key)
                if joined_df is None:
                    blocked_features_by_path[join_name] = []
                    sensitive_features_by_path[join_name] = []
                    continue
        except Exception as exc:
            logging.warning("AutoFeatPlus: could not load temp join '%s': %s", filename_key, exc)
            blocked_features_by_path[join_name] = []
            sensitive_features_by_path[join_name] = []
            continue

        available = [f for f in features if f in joined_df.columns]
        if dataset.target_column not in joined_df.columns or not available:
            blocked_features_by_path[join_name] = []
            sensitive_features_by_path[join_name] = []
            continue

        selection = select_autofeat_plus_features(
            dataframe=joined_df[available + [dataset.target_column]],
            target_column=dataset.target_column,
            top_k=len(available),
            privacy_penalty=privacy_penalty,
            block_sensitive=True,
        )

        bfs_traversal.partial_join_selected_features[join_name] = selection.selected_features
        blocked_features_by_path[join_name] = selection.blocked_features
        sensitive_features_by_path[join_name] = selection.sensitive_features

        # Adjust the path ranking score.
        # privacy_risk_score is the raw count of risky features; normalise by
        # total features so the penalty is on the same [0,1] scale as the mRMR score.
        n_features = max(len(available), 1)
        normalised_privacy = selection.privacy_risk_score / n_features
        bfs_traversal.ranking[join_name] = (
            bfs_traversal.ranking[join_name]
            - privacy_penalty * normalised_privacy
            - path_length_penalty * get_path_length(join_name)
        )

    all_results, top_k_paths = evaluate_paths(
        bfs_result=bfs_traversal,
        problem_type=dataset.dataset_type,
        algorithm=algorithm,
        store_augmented_data=store_augmented_data,
    )
    for result in all_results:
        result.approach = approach
        result.feature_selection_time = end - start
        result.total_time += result.feature_selection_time
        result.top_k = top_k
        result.data_label = dataset.base_table_label
        result.cutoff_threshold = value_ratio
        result.n_features = len(result.join_path_features)

        # sensitive_features = all features flagged as privacy-risky (detected)
        # blocked_features   = features actually removed from selection
        # (identical when block_sensitive=True, may differ when False)
        sensitive = sensitive_features_by_path.get(result.join_name, [])
        blocked = blocked_features_by_path.get(result.join_name, [])
        result.sensitive_features = sensitive
        result.blocked_features = blocked
        result.n_sensitive_features = len(sensitive)
        result.privacy_risk_score = float(len(sensitive))

    pd.DataFrame([vars(result) for result in all_results]).to_csv(
        RESULTS_FOLDER / f"{dataset.base_table_label}_{approach}.csv",
        index=False,
    )

    return all_results, top_k_paths


if __name__ == "__main__":
    init_datasets()
    dataset = filter_datasets(["credit"])[0]
    autofeat(dataset, value_ratio=0.65, top_k=15)
