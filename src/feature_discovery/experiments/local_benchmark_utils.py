from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


def read_table(data_dir: Path, filename: str) -> pd.DataFrame:
    dataframe = pd.read_csv(data_dir / filename)
    if "time" in dataframe.columns:
        dataframe["time"] = pd.to_numeric(dataframe["time"], errors="coerce")
        dataframe = dataframe.dropna(subset=["time"]).copy()
        dataframe["time"] = dataframe["time"].astype("int64")
    return dataframe


def prefix_columns(dataframe: pd.DataFrame, table_name: str, join_key: str) -> pd.DataFrame:
    prefix = table_name.removesuffix(".csv")
    return dataframe.rename(columns={column: f"{prefix}.{column}" for column in dataframe.columns if column != join_key})


def join_tables(
    data_dir: Path,
    base_table: str,
    join_tables_to_use: list[str],
    join_key: str,
    join_mode: str,
    tolerance_seconds: int,
) -> pd.DataFrame:
    joined = read_table(data_dir, base_table)
    if join_key not in joined.columns:
        raise ValueError(f"Base table {base_table} does not contain join key {join_key}")

    for table_name in join_tables_to_use:
        right = read_table(data_dir, table_name)
        if join_key not in right.columns:
            raise ValueError(f"Join table {table_name} does not contain join key {join_key}")
        right = prefix_columns(right, table_name, join_key)

        if join_mode == "exact":
            joined = pd.merge(joined, right, how="left", on=join_key)
        elif join_mode == "asof":
            left = joined.copy()
            left["__row_id"] = np.arange(len(left))
            left = left.sort_values(join_key)
            right = right.sort_values(join_key)
            joined = pd.merge_asof(
                left,
                right,
                on=join_key,
                direction="nearest",
                tolerance=tolerance_seconds,
            ).sort_values("__row_id")
            joined = joined.drop(columns=["__row_id"]).reset_index(drop=True)
        else:
            raise ValueError(f"Unsupported join mode: {join_mode}")

    return joined


def join_antenna_tables(data_dir: Path, base: pd.DataFrame, antennas: list[int]) -> pd.DataFrame:
    joined = base.copy()
    for antenna in antennas:
        features = pd.read_csv(data_dir / f"antenna_{antenna}_features.csv")
        rename_columns = {
            column: f"antenna_{antenna}_{column}"
            for column in features.columns
            if column not in {"sample_key", "user_id", "sample_id"}
        }
        features = features.rename(columns=rename_columns)
        joined = pd.merge(joined, features, how="left", on=["sample_key", "user_id", "sample_id"])
    return joined


def parse_feature_list(raw_value: str) -> list[str]:
    if raw_value is None or (isinstance(raw_value, float) and math.isnan(raw_value)):
        return []
    raw = str(raw_value).strip()
    if not raw:
        return []
    if raw.startswith("["):
        import ast

        value = ast.literal_eval(raw)
        return [str(v) for v in value]
    return [part.strip() for part in raw.split(",") if part.strip()]


def make_random_split(index: pd.Index | np.ndarray, test_size: float, random_state: int = 10) -> tuple[np.ndarray, np.ndarray]:
    indices = np.array(index)
    rng = np.random.default_rng(random_state)
    rng.shuffle(indices)
    test_count = max(1, int(round(len(indices) * test_size)))
    test_index = np.sort(indices[:test_count])
    train_index = np.sort(indices[test_count:])
    return train_index, test_index


def make_kul_split(
    metadata: pd.DataFrame,
    split_mode: str,
    test_size: float,
    holdout_user: int | None,
    holdout_position: str | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    if split_mode == "random":
        train_index, test_index = make_random_split(metadata.index.to_numpy(), test_size=test_size, random_state=10)
        return train_index, test_index, f"random_test_size={test_size}"

    if split_mode == "user-holdout":
        user = holdout_user if holdout_user is not None else int(metadata["user_id"].max())
        test_mask = metadata["user_id"].to_numpy() == user
        return np.where(~test_mask)[0], np.where(test_mask)[0], f"user_id={user}"

    if split_mode == "position-holdout":
        positions = metadata[["target_x", "target_y", "target_z"]].drop_duplicates().sort_values(
            ["target_x", "target_y", "target_z"]
        )
        if holdout_position:
            values = [int(value.strip()) for value in holdout_position.split(",")]
            if len(values) != 3:
                raise ValueError("--holdout-position must use the format target_x,target_y,target_z")
            position = tuple(values)
        else:
            position = tuple(positions.iloc[-1].tolist())
        test_mask = (
            (metadata["target_x"].to_numpy() == position[0])
            & (metadata["target_y"].to_numpy() == position[1])
            & (metadata["target_z"].to_numpy() == position[2])
        )
        return np.where(~test_mask)[0], np.where(test_mask)[0], f"position={position}"

    raise ValueError(f"Unsupported split mode: {split_mode}")


def make_tabular_split(
    dataframe: pd.DataFrame,
    split_mode: str,
    test_size: float,
    time_column: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if split_mode == "time":
        if time_column is None or time_column not in dataframe.columns:
            raise ValueError("Time split requested but time column is unavailable.")
        ordered = dataframe.sort_values(time_column).reset_index(drop=True)
        test_count = max(1, int(round(len(ordered) * test_size)))
        split_at = len(ordered) - test_count
        train_df = ordered.iloc[:split_at].copy()
        test_df = ordered.iloc[split_at:].copy()
        return train_df, test_df, f"time_last_fraction={test_size}"

    if split_mode == "random":
        train_index, test_index = make_random_split(dataframe.index.to_numpy(), test_size=test_size, random_state=10)
        return dataframe.loc[train_index].copy(), dataframe.loc[test_index].copy(), f"random_test_size={test_size}"

    raise ValueError(f"Unsupported split mode: {split_mode}")
