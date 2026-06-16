import hashlib
import logging
import re
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import tqdm

from feature_discovery.autofeat_pipeline.autofeat import AutoFeat
from feature_discovery.autofeat_pipeline.join_path_utils import get_path_length, parse_join_step
from feature_discovery.config import RESULTS_FOLDER, SEED
from feature_discovery.dataset_introspection import (
    detect_timestamp_unit,
    parse_tolerance,
    tolerance_to_seconds,
)
from feature_discovery.experiments.evaluation_algorithms import evaluate_all_algorithms
from feature_discovery.experiments.init_datasets import ALL_DATASETS
from feature_discovery.experiments.result_object import Result
from feature_discovery.experiments.utils_dataset import filter_datasets
from feature_discovery.helpers.read_data import get_df_with_prefix


def _get_augmented_dataset_path(output_dir: Path, dataset_label: str, join_name: str) -> Path:
    """Return a deterministic file path for an augmented dataset."""

    safe_dataset_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_label)
    dataset_hash = hashlib.sha1(join_name.encode("utf8")).hexdigest()[:12]
    filename = f"{safe_dataset_label}_{dataset_hash}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir / filename


def evaluate_paths(
    bfs_result: AutoFeat,
    problem_type: str,
    algorithm: str,
    top_k_paths: int = 15,
    store_augmented_data: bool = True,
) -> Tuple[List[Result], List[Tuple]]:
    """Evaluate the top-k join paths discovered by AutoFeat.

    Parameters
    ----------
    bfs_result
        AutoFeat run containing the join ranking and selected features.
    problem_type
        Task type (classification/regression) used when evaluating downstream models.
    algorithm
        Identifier of the evaluation algorithm to execute.
    top_k_paths
        Maximum number of join paths to evaluate.
    store_augmented_data
        When ``True`` the augmented datasets are persisted to ``RESULTS_FOLDER`` for
        offline inspection; otherwise the evaluation happens in-memory only.
    """
    logging.debug(f"Evaluate top-{top_k_paths} paths ... ")
    sorted_paths = sorted(
        bfs_result.ranking.items(),
        key=lambda r: (-r[1], get_path_length(r[0]), r[0]),
    )
    top_k_path_list = sorted_paths if len(sorted_paths) < top_k_paths else sorted_paths[:top_k_paths]
    base_features = bfs_result.partial_join_selected_features[bfs_result.base_table_id]

    augmented_dir = None
    if store_augmented_data:
        augmented_dir = RESULTS_FOLDER / "augmented_datasets"
    all_results = []
    for path in tqdm.tqdm(top_k_path_list):
        join_name, rank = path
        if join_name == bfs_result.base_table_id:
            continue

        features = list(bfs_result.partial_join_selected_features[join_name])
        features.extend(base_features)
        augmented_dataframe, path_list = materialize_augmented_dataframe(
            join_name=join_name,
            selected_features=features,
            target_column=bfs_result.target_column,
            base_node=bfs_result.base_table_id,
            fallback_features=base_features,
            temporal_key=bfs_result.temporal_key,
            temporal_tolerance=bfs_result.temporal_tolerance,
            temporal_direction=bfs_result.temporal_direction,
            return_path_list=True,
        )
        if store_augmented_data:
            augmented_path = _get_augmented_dataset_path(
                augmented_dir,
                bfs_result.base_table_label,
                join_name,
            )
            augmented_path.parent.mkdir(parents=True, exist_ok=True)
            augmented_dataframe.to_csv(augmented_path, index=False)
            logging.info("Saved augmented dataset for join '%s' to %s", join_name, augmented_path)

        # Detect the temporal column in the augmented dataframe for time-aware train/test split.
        time_column = None
        if bfs_result.temporal_key:
            matches = [
                c for c in augmented_dataframe.columns
                if c == bfs_result.temporal_key or c.endswith(f".{bfs_result.temporal_key}")
            ]
            time_column = matches[0] if matches else None

        results, _ = evaluate_all_algorithms(
            dataframe=augmented_dataframe,
            target_column=bfs_result.target_column,
            problem_type=problem_type,
            algorithm=algorithm,
            time_column=time_column,
        )
        for result in results:
            result.rank = rank
            result.data_path = path_list
            result.join_name = join_name
        all_results.extend(results)

        dataframe = None

    # Fallback: if no valid join path produced results, evaluate on base features only.
    if not all_results:
        logging.warning("evaluate_paths: no valid join paths found — falling back to base-table evaluation.")
        try:
            base_df, _ = get_df_with_prefix(bfs_result.base_table_id, bfs_result.target_column)
            fallback_features = [f for f in base_features if f in base_df.columns]
            if bfs_result.target_column not in fallback_features:
                fallback_features.append(bfs_result.target_column)
            if len(fallback_features) >= 2:
                results, _ = evaluate_all_algorithms(
                    dataframe=base_df[fallback_features],
                    target_column=bfs_result.target_column,
                    problem_type=problem_type,
                    algorithm=algorithm,
                )
                for result in results:
                    result.rank = 0
                    result.data_path = [bfs_result.base_table_id]
                    result.join_name = bfs_result.base_table_id
                    result.approach = "BASE_FALLBACK"
                all_results.extend(results)
        except Exception as exc:
            logging.error("evaluate_paths: base-table fallback also failed: %s", exc)

    return all_results, top_k_path_list


def evaluate_paths_from_file(filename: str, algorithm: str, top_k_paths: int = 15) -> List[Result]:
    logging.debug(f"Evaluate top-{top_k_paths} paths ... ")

    data = pd.read_csv(RESULTS_FOLDER / filename)

    data_paths = data[~data['data_label'].isin(['air', 'yprop', 'superconduct'])]
    data_paths = data_paths.loc[data_paths.groupby(by=['data_path'])['accuracy'].idxmax()]

    all_results = []
    for index, row in data_paths.iterrows():
        dataset = filter_datasets([row['data_label']])[0]
        path_list = pd.eval(row['data_path'])
        rank = pd.eval(row['rank'])
        features = pd.eval(row['join_path_features'])
        logging.debug(f"Feature before join_key removal:\n{features}")

        dataframe = join_from_path(path_list, dataset.target_column, dataset.base_table_id)
        features = [f for f in features if f in dataframe.columns]
        target = f"{dataset.base_table_label}/{dataset.base_table_name}.{dataset.target_column}"
        features.append(target)

        results, _ = evaluate_all_algorithms(dataframe=dataframe[features],
                                             target_column=target,
                                             algorithm=algorithm)
        for result in results:
            result.rank = rank
            result.data_path = path_list
            result.approach = Result.TFD
            result.feature_selection_time = pd.eval(row['feature_selection_time'])
            result.total_time += pd.eval(row['total_time'])
            result.top_k = pd.eval(row['top_k'])
            result.data_label = dataset.base_table_label
            result.cutoff_threshold = pd.eval(row['cutoff_threshold'])
        all_results.extend(results)

        dataframe = None

    pd.DataFrame(all_results).to_csv(RESULTS_FOLDER / f"results_autofeat_from_path_{algorithm}.csv", index=False)

    return all_results


def join_from_path(
    path,
    target,
    base_node,
    temporal_key: Optional[str] = None,
    temporal_tolerance: int = 60,
    temporal_direction: str = "nearest",
):
    """Reconstruct the full joined dataframe from a join-path list.

    Uses ``pd.merge_asof`` (nearest-timestamp) when the join column matches
    *temporal_key*, matching the strategy used during feature discovery.
    Falls back to an exact left-join for all other columns.
    """
    if temporal_direction not in ("nearest", "backward", "forward"):
        raise ValueError(f"temporal_direction must be nearest|backward|forward, got {temporal_direction!r}")

    join_path = ''
    step = 3
    joined_df = None
    for p in path:
        if ".".join(p) in join_path:
            continue
        for i, el in enumerate(p[::step]):
            if (i * step) + 1 == len(p):
                continue

            current_idx = i * step
            next_idx = (i + 1) * step
            aux = ".".join([p[current_idx], p[current_idx + 1], p[current_idx + 2], p[next_idx]])
            if aux in join_path:
                continue

            if p[current_idx] not in join_path:
                if p[current_idx] == base_node:
                    left_table, _ = get_df_with_prefix(p[current_idx], target)
                else:
                    left_table, _ = get_df_with_prefix(p[current_idx])
            else:
                left_table = joined_df

            left_on = f"{p[current_idx]}.{p[current_idx + 1]}"
            right_on = f"{p[next_idx]}.{p[current_idx + 2]}"
            bare_from_col = p[current_idx + 1]
            right_table, _ = get_df_with_prefix(p[next_idx])

            if temporal_key and bare_from_col == temporal_key:
                # Nearest-timestamp join — mirrors the discovery phase
                order_column = "__autofeat_left_order__"
                while order_column in left_table.columns or order_column in right_table.columns:
                    order_column = f"_{order_column}"
                left_with_order = left_table.copy()
                left_with_order[order_column] = range(len(left_with_order))
                left_clean = left_with_order.dropna(subset=[left_on])
                left_null = left_with_order[left_with_order[left_on].isna()]
                right_clean = right_table.dropna(subset=[right_on])
                if left_clean.empty or right_clean.empty:
                    joined_df = left_with_order.sort_values(order_column).drop(columns=[order_column])
                    join_path += aux
                    continue

                left_dtype = left_clean[left_on].dtype
                right_dtype = right_clean[right_on].dtype
                if left_dtype != right_dtype and (
                    pd.api.types.is_numeric_dtype(left_dtype)
                    and pd.api.types.is_numeric_dtype(right_dtype)
                ):
                    left_clean = left_clean.copy()
                    right_clean = right_clean.copy()
                    left_clean[left_on] = pd.to_numeric(left_clean[left_on], errors="coerce").astype("float64")
                    right_clean[right_on] = pd.to_numeric(right_clean[right_on], errors="coerce").astype("float64")

                left_sorted = left_clean.sort_values(left_on).reset_index(drop=True)
                right_sorted = right_clean.sort_values(right_on).reset_index(drop=True)
                # Resolve tolerance for merge_asof, mirroring the discovery phase.
                # temporal_tolerance may arrive as an int or a string like "60s";
                # tolerance=0 means exact match, a negative amount means no limit.
                try:
                    amount, unit = parse_tolerance(temporal_tolerance)
                except (ValueError, TypeError):
                    amount, unit = -1, "s"
                key_series = left_sorted[left_on]
                if amount < 0:
                    tolerance = None
                elif pd.api.types.is_datetime64_any_dtype(key_series):
                    tolerance = pd.Timedelta(seconds=tolerance_to_seconds(amount, unit))
                else:
                    col_unit = detect_timestamp_unit(key_series) or "s"
                    tol_seconds = tolerance_to_seconds(amount, unit)
                    tolerance = tol_seconds * {"s": 1, "ms": 1e3, "us": 1e6, "ns": 1e9}.get(col_unit, 1)
                    # merge_asof rejects a float tolerance against an integer key.
                    if pd.api.types.is_integer_dtype(key_series):
                        tolerance = int(round(tolerance))
                joined_df = pd.merge_asof(
                    left_sorted,
                    right_sorted,
                    left_on=left_on,
                    right_on=right_on,
                    tolerance=tolerance,
                    direction=temporal_direction,
                )
                if not left_null.empty:
                    joined_df = pd.concat([joined_df, left_null], ignore_index=True, sort=False)
                joined_df = joined_df.sort_values(order_column).drop(columns=[order_column]).reset_index(drop=True)
            else:
                to_column = f"{p[next_idx]}.{p[current_idx + 2]}"
                right_table = right_table.groupby(to_column).sample(n=1, random_state=SEED)
                joined_df = pd.merge(
                    left_table,
                    right_table,
                    how="left",
                    left_on=left_on,
                    right_on=right_on,
                )

            join_path += aux

    return joined_df


def materialize_augmented_dataframe(
    join_name: str,
    selected_features: List[str],
    target_column: str,
    base_node,
    fallback_features: Optional[List[str]] = None,
    temporal_key: Optional[str] = None,
    temporal_tolerance: int = 60,
    temporal_direction: str = "nearest",
    return_path_list: bool = False,
):
    """Materialize the selected augmented dataframe for a discovered join path."""
    features = list(selected_features)
    if target_column not in features:
        features.append(target_column)
    features = list(dict.fromkeys(features))
    logging.debug("Feature before join_key removal:\n%s", features)

    features_tables = list(dict.fromkeys(sorted(f"{feat.split('.csv')[0]}.csv" for feat in features)))

    path_tables = {}
    for p in join_name.split("--"):
        if "|" not in p:
            continue
        try:
            from_table, from_col, to_col, to_table = parse_join_step(p)
            path_tables[to_table] = (from_table, from_col, to_col, to_table)
        except ValueError:
            continue

    path_list = []
    for table in features_tables:
        if table in path_list:
            continue
        path_aux = create_join_tree(table, path_tables)
        if not (type(path_aux) is list) and (path_aux not in path_tables.keys()):
            continue
        path_list.append(path_aux)

    if path_list:
        dataframe = join_from_path(
            path_list,
            target_column,
            base_node,
            temporal_key=temporal_key,
            temporal_tolerance=temporal_tolerance,
            temporal_direction=temporal_direction,
        )
    else:
        dataframe, _ = get_df_with_prefix(base_node, target_column)
    features = list(dict.fromkeys([f for f in features if f in dataframe.columns]))

    if len(features) < 2 and fallback_features:
        features = [f for f in fallback_features if f in dataframe.columns]
        if target_column in dataframe.columns and target_column not in features:
            features.append(target_column)

    augmented = dataframe[features]
    if return_path_list:
        return augmented, path_list
    return augmented


def create_join_tree(table, path_tables):
    if table in path_tables.keys():
        value = path_tables[table]
        result = create_join_tree(value[0], path_tables)
        if type(result) is not list:
            result = [result]

        result.append(value[1])
        result.append(value[2])
        result.append(value[3])
        return result

    return table
