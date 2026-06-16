import logging
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from autogluon.features.generators import AutoMLPipelineFeatureGenerator


from feature_discovery.config import SEED
from feature_discovery.autofeat_pipeline.join_data import join_and_save, temporal_join_and_save
from feature_discovery.autofeat_pipeline.join_path_feature_selection import RelevanceRedundancy
from feature_discovery.autofeat_pipeline.join_path_utils import compute_join_name
from feature_discovery.experiments.dataset_object import CLASSIFICATION
from feature_discovery.graph_processing.neo4j_transactions import (
    get_adjacent_nodes,
    get_relation_properties_node_name,
)
from feature_discovery.helpers.read_data import get_df_with_prefix
from feature_discovery.helpers.optional_polars import POLARS_AVAILABLE, pl

logging.getLogger().setLevel(logging.WARNING)


def _pop_sorted(values: Set[str]) -> str:
    value = sorted(values)[0]
    values.remove(value)
    return value


class AutoFeat:
    def __init__(
        self,
        base_table_label: str,
        base_table_id: str,
        target_column: str,
        save_joins_to_disk: bool,
        use_polars: bool,
        task: str = CLASSIFICATION,
        value_ratio: float = 0.65,
        top_k: int = 5,
        sample_size: int = 3000,
        pearson: bool = False,
        jmi: bool = False,
        no_relevance: bool = False,
        no_redundancy: bool = False,
        temporal_key: Optional[str] = None,
        temporal_tolerance: int = 60,
        temporal_direction: str = "nearest",
    ):
        """

        :param base_table_label: The name (label) of the base table to be used for saving data.
        :param target_column: Target column containing the class labels for training.
        :param value_ratio: Pruning threshold. It represents the ration between the number of non-null values in a column and the total number of values.
        :param temporal_key: Column name used as a Unix-timestamp join key.  When set,
            step_join uses pd.merge_asof (nearest-timestamp) instead of exact merge.
        :param temporal_tolerance: Maximum allowed time difference in seconds for
            temporal joins (0 = exact match, ignored when temporal_key is None).
        """
        self.base_table_label: str = base_table_label
        self.target_column: str = target_column
        self.value_ratio: float = value_ratio
        self.top_k: int = top_k
        self.sample_size: int = sample_size
        self.base_table_id: str = base_table_id
        self.task: str = task
        # Mapping with the name of the join and the corresponding name of the file containing the join result.
        self.join_name_mapping: Dict[str, str] = {}
        # Set used to track the visited nodes.
        self.discovered: Set[str] = set()
        # Save the selected features of the previous join path (used for conditional redundancy)
        self.partial_join_selected_features: Dict[str, List] = {}

        self.ranking: Dict[str, float] = {}
        self.join_keys: Dict[str, list] = {}
        self.temporal_key: Optional[str] = temporal_key
        self.temporal_tolerance: int = temporal_tolerance
        if temporal_direction not in ("nearest", "backward", "forward"):
            raise ValueError(f"temporal_direction must be nearest|backward|forward, got {temporal_direction!r}")
        self.temporal_direction: str = temporal_direction
        self.rel_red = RelevanceRedundancy(target_column, jmi=jmi, pearson=pearson)
        self.temp_dir = tempfile.TemporaryDirectory()
        if use_polars and not POLARS_AVAILABLE:
            logging.warning(
                "Polars is not installed; defaulting to pandas for join operations. "
                "Install the optional 'polars' dependency to re-enable the faster path."
            )

        self.use_polars = use_polars and POLARS_AVAILABLE
        self.partial_join = self.initialisation()

        # Ablation study parameters
        self.sample_data_step = True
        self.no_relevance = no_relevance
        self.no_redundancy = no_redundancy

        # Whether to save the joins to disk or not
        self.save_joins_to_disk = save_joins_to_disk
        if self.save_joins_to_disk is not True:
            self.joins_to_df: Dict[str, pd.DataFrame] = {}
        else:
            logging.warn(f"Saving intermediate joins to disk: {self.temp_dir.name}")

    def initialisation(self):
        from sklearn.model_selection import train_test_split

        # Read dataframe
        base_table_df, partial_join_name = get_df_with_prefix(
            self.base_table_id, self.target_column, use_polars=self.use_polars
        )

        # Stratified sampling
        if self.sample_size < base_table_df.shape[0]:
            if self.task == CLASSIFICATION:
                X_train, X_test = train_test_split(
                    base_table_df,
                    train_size=self.sample_size,
                    stratify=base_table_df[self.target_column],
                    random_state=SEED,
                )
            else:
                X_train, X_test = train_test_split(base_table_df, train_size=self.sample_size, random_state=SEED)
        else:
            X_train = base_table_df

        # Base table features are the selected features
        features = list(X_train.columns)
        if self.target_column in features:
            features.remove(self.target_column)

        self.partial_join_selected_features[partial_join_name] = features
        self.ranking[partial_join_name] = 0
        self.join_keys[partial_join_name] = []

        return X_train

    def streaming_feature_selection(self, queue: set, previous_queue: set = None):
        if len(queue) == 0:
            return

        if previous_queue is None:
            previous_queue = queue.copy()

        # Iterate through all the elements of the queue:
        # 1) in the first iteration: queue = base_node_id
        # 2) in all the other iterations: queue = neighbours of the previous node
        all_neighbours = set()

        while len(queue) > 0:
            # Get the current/base node
            base_node_id = _pop_sorted(queue)
            self.discovered.add(base_node_id)
            logging.debug(f"New iteration with base node: {base_node_id}")

            # Determine the neighbours (unvisited)
            neighbours = sorted(set(get_adjacent_nodes(base_node_id)) - set(self.discovered))
            if len(neighbours) == 0:
                continue

            all_neighbours.update(neighbours)

            # Process every neighbour - join, determine quality, get features
            for node in neighbours:
                self.discovered.add(node)
                logging.debug(f"Adjacent node: {node}")

                # Get the join keys with the highest score
                join_keys = get_relation_properties_node_name(from_id=base_node_id, to_id=node)
                if len(join_keys) == 1:
                    highest_ranked_join_keys = join_keys
                else:
                    highest_ranked_join_keys = []
                    for jk in join_keys:
                        if jk[0]['weight'] == join_keys[0][0]['weight']:
                            highest_ranked_join_keys.append(jk)
                        else:
                            break

                # Read the neighbour node
                right_df, right_label = get_df_with_prefix(node, use_polars=self.use_polars)
                logging.debug(f"\tRight table shape: {right_df.shape}")

                current_queue = set()
                while len(previous_queue) > 0:
                    previous_join_name = _pop_sorted(previous_queue)

                    previous_join = None
                    if previous_join_name == self.base_table_id:
                        previous_join_name = self.base_table_id
                        if POLARS_AVAILABLE and isinstance(self.partial_join, pl.DataFrame):
                            previous_join = self.partial_join.clone()
                        else:
                            previous_join = self.partial_join.copy()
                    else:
                        filename_key = self.join_name_mapping[previous_join_name]
                        if self.save_joins_to_disk:
                            previous_join = pd.read_parquet(
                                Path(self.temp_dir.name) / filename_key,
                            )
                        else:
                            previous_join = self.joins_to_df[filename_key]

                    # The current node can only be joined through the base node.
                    # If the base node doesn't exist in the previous join path, the join can't be performed

                    if base_node_id not in previous_join_name:
                        logging.debug(f"\tBase node {base_node_id} not in partial join {previous_join_name}")
                        continue

                    for prop in highest_ranked_join_keys:
                        join_prop, from_table, to_table = prop
                        if join_prop['from_label'] != from_table:
                            continue

                        if join_prop['from_column'] == self.target_column:
                            current_queue.add(previous_join_name)
                            continue

                        logging.debug(f"\t\tJoin properties: {join_prop}")

                        # Step - Explore all possible join paths based on the join keys - Compute the name of the join
                        join_name = compute_join_name(join_key_property=prop, partial_join_name=previous_join_name)
                        logging.debug(f"\tJoin name: {join_name}")

                        # Step - Join
                        joined_df, join_filename, join_columns = self.step_join(
                            join_key_properties=prop, left_df=previous_join, right_df=right_df, right_label=right_label
                        )

                        if joined_df is None:
                            current_queue.add(previous_join_name)
                            continue

                        data_quality = self.step_data_quality(join_key_properties=prop, joined_df=joined_df)
                        if not data_quality:
                            current_queue.add(previous_join_name)
                            continue

                        result = self.streaming_relevance_redundancy(
                            dataframe=joined_df.copy(),
                            new_features=list(right_df.columns),
                            selected_features=self.partial_join_selected_features[previous_join_name],
                        )
                        if result is not None:
                            self.ranking[join_name] = result[0]
                            all_selected_features = list(self.partial_join_selected_features[previous_join_name])
                            all_selected_features.extend(result[1])
                            # Keep feature state isolated per join path.
                            self.partial_join_selected_features[join_name] = list(dict.fromkeys(all_selected_features))
                        else:
                            self.partial_join_selected_features[join_name] = list(
                                self.partial_join_selected_features[previous_join_name]
                            )

                        join_columns.extend(self.join_keys[previous_join_name])
                        self.join_keys[join_name] = join_columns
                        self.join_name_mapping[join_name] = join_filename

                        current_queue.add(join_name)
                        if not self.save_joins_to_disk:
                            self.joins_to_df[join_filename] = joined_df
                # Initialise the queue with the new paths (current_queue)
                previous_queue.update(current_queue)

        self.streaming_feature_selection(all_neighbours, previous_queue)

    def streaming_relevance_redundancy(
        self, dataframe: pd.DataFrame, new_features: List[str], selected_features: List[str]
    ) -> Optional[Tuple[float, List[dict]]]:
        df = AutoMLPipelineFeatureGenerator(
            enable_text_special_features=False, enable_text_ngram_features=False
        ).fit_transform(X=dataframe, random_state=SEED, random_seed=SEED)

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        requested = set(new_features)
        features = [col for col in X.columns if col in requested]
        top_feat = len(features) if len(features) < self.top_k else self.top_k

        relevant_features = new_features
        sum_m = 0
        m = 1
        if not self.no_relevance:
            feature_score_relevance = self.rel_red.measure_relevance(
                dataframe=X, new_features=features, target_column=y
            )[:top_feat]
            if len(feature_score_relevance) == 0:
                return None
            relevant_features = list(dict(feature_score_relevance).keys())
            m = len(feature_score_relevance) if len(feature_score_relevance) > 0 else m
            sum_m = sum(list(map(lambda x: x[1], feature_score_relevance)))

        final_features = relevant_features
        sum_o = 0
        o = 1
        if not self.no_redundancy:
            # Only compute redundancy against features that survived preprocessing.
            # If no prior features exist in the transformed space (e.g. the base
            # table contained only identifier columns that were dropped), skip
            # the redundancy step entirely — all relevant features are accepted.
            active_selected = [f for f in selected_features if f in X.columns]
            if not active_selected:
                sum_o = sum_m
                o = m
            else:
                feature_score_redundancy = self.rel_red.measure_redundancy(
                    dataframe=X, selected_features=active_selected, relevant_features=relevant_features, target_column=y
                )

                if len(feature_score_redundancy) == 0:
                    return None

                o = len(feature_score_redundancy) if feature_score_redundancy else o
                sum_o = sum(list(map(lambda x: x[1], feature_score_redundancy)))
                final_features = list(dict(feature_score_redundancy).keys())

        score = (o * sum_m + m * sum_o) / (m * o)

        return score, final_features

    def step_join(
        self,
        join_key_properties: tuple,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        right_label: str,
    ) -> Tuple[Optional[pd.DataFrame], str, list]:
        logging.debug("\tSTEP Join ... ")
        join_prop, from_table, to_table = join_key_properties

        left_on = f"{from_table}.{join_prop['from_column']}"
        right_on = f"{to_table}.{join_prop['to_column']}"

        # The graph stores bare column names in join_prop['from_column']; we
        # match it directly against the user-supplied temporal_key.
        is_temporal = (
            self.temporal_key is not None
            and join_prop['from_column'] == self.temporal_key
        )

        # Step - Sample neighbour data - Transform to 1:1 or M:1.
        # Skip for temporal joins: merge_asof already picks the nearest row,
        # and random sampling within a timestamp group can discard the actual
        # nearest neighbour.
        sampled_right_df = right_df
        if self.sample_data_step and not is_temporal:
            if self.use_polars:
                right_df_pl = pl.from_pandas(right_df)
                sampled_right_df = right_df_pl.filter(
                    pl.int_range(0, pl.count()).shuffle(seed=SEED).over(f"{right_label}.{join_prop['to_column']}") < 1
                )
            else:
                sampled_right_df = right_df.groupby(f"{right_label}.{join_prop['to_column']}").sample(
                    n=1, random_state=SEED
                )

        # File naming convention as the filename can be gigantic
        join_filename = f"{self.base_table_label}_join_BFS_{self.value_ratio}_{str(uuid.uuid4())}.parquet"
        join_path = Path(self.temp_dir.name) / join_filename

        if is_temporal:
            # Temporal nearest-neighbour join — Polars not yet supported here
            left_pd = left_df.to_pandas() if POLARS_AVAILABLE and isinstance(left_df, pl.DataFrame) else left_df
            right_pd = sampled_right_df.to_pandas() if POLARS_AVAILABLE and isinstance(sampled_right_df, pl.DataFrame) else sampled_right_df
            joined_df = temporal_join_and_save(
                left_df=left_pd,
                right_df=right_pd,
                left_column_name=left_on,
                right_column_name=right_on,
                join_path=join_path,
                tolerance_s=self.temporal_tolerance,
                csv=False,
                save_to_disk=self.save_joins_to_disk,
                direction=self.temporal_direction,
            )
        else:
            # Normalize left_df to polars when use_polars is enabled. left_df
            # may arrive as pandas (loaded from parquet) or polars (first
            # iteration from self.partial_join), so guard both cases.
            if self.use_polars and POLARS_AVAILABLE and isinstance(left_df, pd.DataFrame):
                left_df = pl.from_pandas(left_df)
            joined_df = join_and_save(
                left_df=left_df,
                right_df=sampled_right_df,
                left_column_name=left_on,
                right_column_name=right_on,
                join_path=join_path,
                csv=False,
                save_to_disk=self.save_joins_to_disk,
            )

        if joined_df is None:
            return None, join_filename, []

        return joined_df, join_filename, [left_on, right_on]

    def step_data_quality(self, join_key_properties: tuple, joined_df: pd.DataFrame) -> bool:
        logging.debug("\tSTEP data quality ...")
        join_prop, _, to_table = join_key_properties

        # Data Quality check - Prune the joins with high null values ratio
        if joined_df[f"{to_table}.{join_prop['to_column']}"].count() / joined_df.shape[0] < self.value_ratio:
            logging.debug(f"\t\tRight column value ration below {self.value_ratio}.\nSKIPPED Join")
            return False

        return True
