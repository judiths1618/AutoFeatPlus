from pathlib import Path

import pandas as pd

from feature_discovery.graph_processing.neo4j_transactions import (
    get_pk_fk_nodes,
)
from feature_discovery.helpers.dict_utils import transform_node_to_dict
from feature_discovery.helpers.optional_polars import POLARS_AVAILABLE, pl


def join_directly_connected(base_table_id: str):
    nodes = get_pk_fk_nodes(base_table_id)
    partial_join = None
    for pk, fk in nodes:
        pk_node = transform_node_to_dict(pk)
        fk_node = transform_node_to_dict(fk)

        left_table = pd.read_csv(pk_node["source_path"])
        right_table = pd.read_csv(fk_node["source_path"])
        if partial_join is not None:
            left_table = partial_join

        partial_join = pd.merge(
            left_table,
            right_table,
            how="left",
            left_on=pk_node["name"],
            right_on=fk_node["name"],
            suffixes=("", "_b"),
        )
        columns_to_drop = [c for c in list(partial_join.columns) if c.endswith("_b")]
        partial_join.drop(columns=columns_to_drop, inplace=True)

    return partial_join


# def pl_outer_join(df1: pl.DataFrame, df2: pl.DataFrame, how, left_on, right_on):
#     new_names = [f"{x}_tmp" for x in [left_on, right_on]]

#     # replicate join columns with new names
#     df1 = df1.with_columns(col(left_on).alias(new_names[0]))
#     df2 = df2.with_columns(col(right_on).alias(new_names[1]))

#     # perform join and drop columns
#     return df1.join(df2, left_on=new_names[0], right_on=new_names[1], how=how).drop(new_names)

def pl_outer_join(
    df1,
    df2,
    left_on: str,
    right_on: str,
    how: str = "left",
) -> "pl.DataFrame":
    """
    Perform a safe join on two Polars DataFrames with temporary aliasing,
    avoiding column conflicts and missing-column crashes.
    """
    if not POLARS_AVAILABLE:
        raise ModuleNotFoundError("Polars is required for 'pl_outer_join' but is not installed.")

    left_tmp = f"{left_on}_tmp"
    right_tmp = f"{right_on}_tmp"

    # Debug logging (optional)
    print(f"[pl_outer_join] df1.columns: {df1.columns}")
    print(f"[pl_outer_join] df2.columns: {df2.columns}")
    print(f"[pl_outer_join] Join on: {left_on} == {right_on}")

    # --- Check that join keys exist ---
    if left_on not in df1.columns:
        raise ValueError(f"[pl_outer_join] Column '{left_on}' not found in left_df.")
    if right_on not in df2.columns:
        raise ValueError(f"[pl_outer_join] Column '{right_on}' not found in right_df.")

    # --- Create alias to avoid column name conflicts ---
    df1 = df1.with_columns(pl.col(left_on).alias(left_tmp))
    df2 = df2.with_columns(pl.col(right_on).alias(right_tmp))

    # --- Perform join on temporary keys ---
    df_joined = df1.join(df2, left_on=left_tmp, right_on=right_tmp, how=how)

    # --- Drop temporary keys if they exist ---
    drop_cols = [col for col in [left_tmp, right_tmp] if col in df_joined.columns]
    df_joined = df_joined.drop(drop_cols)

    return df_joined



def join_and_save(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_column_name: str,
    right_column_name: str,
    join_path: Path,
    csv: bool = True,
    save_to_disk: bool = True,
) -> pd.DataFrame or None: # type: ignore
    """
    Join two dataframes and save the result on disk.

    :param left_df: Left side of the join
    :param right_df: Right side of the join
    :param left_column_name: The left join column
    :param right_column_name: The right join column
    :param join_path: The path to save the join result.
    :param csv: Whether to save as CSV or not.
    :return: The join result.
    """
    if left_df[left_column_name].dtype != right_df[right_column_name].dtype:
        return None

    if POLARS_AVAILABLE and isinstance(left_df, pl.DataFrame):
        partial_join = pl_outer_join(
            left_df,
            right_df,
            how="left",
            left_on=left_column_name,
            right_on=right_column_name,
        ).to_pandas()
    elif isinstance(left_df, pd.DataFrame):
        partial_join = pd.merge(
            left_df,
            right_df,
            how="left",
            left_on=left_column_name,
            right_on=right_column_name,
        )
    else:
        raise Exception("Unknown dataframe type")

    if save_to_disk:
        join_path.parent.mkdir(parents=True, exist_ok=True)
        if csv:
            partial_join.to_csv(join_path, index=False)
        else:
            partial_join.to_parquet(join_path, index=False)

    return partial_join
