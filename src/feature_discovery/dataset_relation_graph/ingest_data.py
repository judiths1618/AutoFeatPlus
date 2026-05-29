import glob
from pathlib import Path
from typing import Optional

import pandas as pd

from feature_discovery.config import CONNECTIONS, DATA_FOLDER
from feature_discovery.dataset_relation_graph.dataset_discovery import profile_valentine_all, profile_valentine_dataset, profile_LSH_all
from feature_discovery.experiments.dataset_object import Dataset
from feature_discovery.graph_processing.neo4j_transactions import merge_nodes_relation_tables, create_node


def ingest_unprocessed_data(dataset_folder_name: str = None):
    print("Process the tables ...")

    if dataset_folder_name:
        files = glob.glob(f"{DATA_FOLDER / dataset_folder_name}/**/*.csv", recursive=True)
    else:
        files = glob.glob(f"{DATA_FOLDER}/**/*.csv", recursive=True)

    # Filter out connections.csv file
    files = [f for f in files if CONNECTIONS not in f and f.endswith("csv")]
    mapping = {}

    for f in files:
        table_path = f.partition(f"{DATA_FOLDER}/")[2]
        table_name = table_path.split("/")[-1]

        mapping[table_name] = table_path
        # Always create a node per table so downstream lookups can resolve the
        # base-table id even when no connections.csv exists and transformer
        # discovery finds nothing to link (e.g. heterogeneous-lake refusal
        # scenarios). Cypher MERGE makes this safe to re-run.
        create_node(table_path, table_name)

    print("Add the ground-truth ... ")
    if dataset_folder_name:
        connection_filename = glob.glob(f"{DATA_FOLDER / dataset_folder_name}/**/{CONNECTIONS}", recursive=True)
    else:
        connection_filename = glob.glob(f"{DATA_FOLDER}/**/{CONNECTIONS}", recursive=True)

    for connection_file in connection_filename:
        connections = pd.read_csv(connection_file)
        for index, row in connections.iterrows():
            merge_nodes_relation_tables(a_table_name=row["fk_table"], a_table_path=mapping[row["fk_table"]],
                                        b_table_name=row["pk_table"], b_table_path=mapping[row["pk_table"]],
                                        a_col=row["fk_column"], b_col=row["pk_column"], weight=1)
            # merge_nodes_relation_tables(a_table_name=row["pk_table"], a_table_path=mapping[row["pk_table"]],
            #                             b_table_name=row["fk_table"], b_table_path=mapping[row["fk_table"]],
            #                             a_col=row["pk_column"], b_col=row["fk_column"], weight=1)

    return mapping


def ingest_nodes(dataset_folder_name: str = None) -> None:
    print("Process the tables ...")
    # datafolder is the name of the dlake
    if dataset_folder_name:
        files = glob.glob(f"{DATA_FOLDER / dataset_folder_name}/**/*.csv", recursive=True)
    else:
        files = glob.glob(f"{DATA_FOLDER}/**/*.csv", recursive=True)

    for f in files:
        if "datasets.csv" in f:
            continue
        table_path = f.partition(f"{DATA_FOLDER}/")[2]
        table_name = table_path.split("/")[-1]
        create_node(table_path, table_name)


def ingest_data_with_pk_fk(
    dataset: Dataset,
    profile_valentine: bool = False,
    profile_LSH: bool = False,
    profile_transformer: bool = False,
    mix_datasets: bool = False,
    transformer_schema_threshold: float = 0.6,
    transformer_value_threshold: float = 0.2,
    transformer_model: str = "sentence-transformers/all-mpnet-base-v2",
    metadata_path: Optional[Path] = None,
    transformer_connections_csv: Optional[Path] = None,
):
    mapping = ingest_unprocessed_data(str(dataset.base_table_path))

    if profile_valentine and mix_datasets:
        profile_valentine_all()
    elif profile_valentine and not mix_datasets:
        profile_valentine_dataset(dataset.base_table_label)

    if profile_transformer:
        # Delayed import: keeps sentence-transformers an optional dependency.
        from feature_discovery.dataset_relation_graph.transformer_discovery import (
            profile_transformer_all,
            profile_transformer_dataset,
        )
        common_kwargs = dict(
            schema_threshold=transformer_schema_threshold,
            value_threshold=transformer_value_threshold,
            model_name=transformer_model,
            metadata_path=metadata_path,
            write_connections_csv=transformer_connections_csv,
        )
        if mix_datasets:
            profile_transformer_all(**common_kwargs)
        else:
            profile_transformer_dataset(str(dataset.base_table_path), **common_kwargs)

    if profile_LSH and mix_datasets:
        profile_LSH_all()
    elif profile_LSH and not mix_datasets:
        pass

