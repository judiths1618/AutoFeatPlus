from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def display_path(path: Path | str | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return resolved.name

from feature_discovery.experiments.autofeat_plus import select_autofeat_plus_features
from feature_discovery.experiments.local_benchmark_utils import join_tables, read_table


DATASETS = {
    "rabbitmq": "rabbitmq-performance.csv",
    "amf": "amf-performance.csv",
    "golang_web": "golang-web-server-performance.csv",
    "python_web": "python-web-server-performance.csv",
}

DEFAULT_JOIN_TABLES = {
    "rabbitmq": ["golang-web-server-performance.csv", "python-web-server-performance.csv", "amf-performance.csv"],
    "amf": ["golang-web-server-performance.csv", "rabbitmq-performance.csv", "python-web-server-performance.csv"],
    "golang_web": ["rabbitmq-performance.csv", "python-web-server-performance.csv", "amf-performance.csv"],
    "python_web": ["rabbitmq-performance.csv", "golang-web-server-performance.csv", "amf-performance.csv"],
}

EUR_CLEANED_POLICY_PATTERNS = {
    "time-private": [
        "column:time",
        "column:t_norm",
        "column:sin_day",
        "column:cos_day",
        "column:sin_hour",
        "column:cos_hour",
        "column:time_since_last_obs",
        "column:time_to_next_obs",
    ],
    "resource-private": [
        "column:cpu_limit",
        "column:cpu_usage",
        "column:ram_limit",
        "column:ram_usage",
        "column:ram_limit_mb",
        "column:ram_usage_mb",
    ],
    "workload-private": ["column:n", "column:c"],
    "target-proxy-private": [r"re:(^|\.)lat\d+(_ms)?$", "column:min", "column:min_ms", "column:mean", "column:mean_ms"],
}

RESULT_COLUMNS = [
    "run_id",
    "dataset",
    "target_column",
    "variant",
    "model",
    "augmentation_method",
    "augmentation_ratio",
    "augmentation_intensity",
    "split_mode",
    "n_train_rows",
    "n_synthetic_rows",
    "n_test_rows",
    "n_features",
    "r2",
    "rmse",
    "mae",
    "train_time_seconds",
    "feature_selection_time_seconds",
    "privacy_policy",
    "n_blocked_features",
    "blocked_features",
    "join_mode",
    "join_key",
    "time_tolerance_seconds",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate classic train-only time-series augmentations as a downstream "
            "tabular utility experiment on EUR 6G data."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/EUR/6907619"))
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="rabbitmq")
    parser.add_argument("--base-table", default=None)
    parser.add_argument("--join-tables", nargs="+", default=None)
    parser.add_argument("--join-key", default="time")
    parser.add_argument("--join-mode", choices=["asof", "exact"], default="asof")
    parser.add_argument("--time-tolerance-seconds", type=int, default=120)
    parser.add_argument("--target-column", default="lat99")
    parser.add_argument("--split-mode", choices=["time", "random"], default="time")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["BASE", "Join_All_Local", "AutoFeatPlus_Local"],
        default=["BASE", "AutoFeatPlus_Local"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ridge", "rf"],
        choices=["ridge", "knn", "rf", "xt", "gbr"],
    )
    parser.add_argument(
        "--augmentation-method",
        nargs="+",
        default=["none", "scaling", "magnitude_mask"],
        choices=["none", "jitter", "scaling", "time_mask", "magnitude_mask"],
    )
    parser.add_argument("--augmentation-ratio", type=float, default=0.5)
    parser.add_argument("--augmentation-intensity", type=float, default=0.05)
    parser.add_argument(
        "--policy",
        nargs="+",
        default=["time-private", "target-proxy-private"],
        choices=["none", "all", *EUR_CLEANED_POLICY_PATTERNS.keys()],
    )
    parser.add_argument("--autofeatplus-top-k", type=int, default=20)
    parser.add_argument("--privacy-penalty", type=float, default=0.25)
    parser.add_argument("--missing-penalty", type=float, default=0.10)
    parser.add_argument("--cost-penalty", type=float, default=0.001)
    parser.add_argument("--proxy-correlation-threshold", type=float, default=0.98)
    parser.add_argument("--max-missing-ratio", type=float, default=0.95)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/6g_data/downstream/ts_augmented_downstream.csv"),
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("results/6g_data/downstream/metadata"),
    )
    return parser.parse_args()


def policy_patterns(policy_names: list[str]) -> list[str]:
    patterns: list[str] = []
    for policy in policy_names:
        if policy == "none":
            continue
        if policy == "all":
            for values in EUR_CLEANED_POLICY_PATTERNS.values():
                patterns.extend(values)
            continue
        patterns.extend(EUR_CLEANED_POLICY_PATTERNS[policy])
    return list(dict.fromkeys(patterns))


def coerce_numeric_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    converted = dataframe.copy()
    for column in converted.columns:
        if pd.api.types.is_numeric_dtype(converted[column]):
            numeric = pd.to_numeric(converted[column], errors="coerce").astype("float64")
        else:
            cleaned = (
                converted[column]
                .astype(str)
                .str.strip()
                .str.replace(r"(?i)(mb|m)$", "000000", regex=True)
                .str.replace(r"(?i)(gb|g)$", "000000000", regex=True)
                .str.replace(r"[^0-9eE+\-\.]", "", regex=True)
                .replace({"": np.nan, "nan": np.nan, "None": np.nan})
            )
            numeric = pd.to_numeric(cleaned, errors="coerce").astype("float64")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        converted[column] = numeric.mask(numeric.abs() > 1e18, np.nan)
    return converted


def make_split_indices(
    dataframe: pd.DataFrame,
    split_mode: str,
    test_size: float,
    time_column: str | None,
    seed: int,
) -> tuple[pd.Index, pd.Index]:
    if split_mode == "time" and time_column and time_column in dataframe.columns:
        ordered = dataframe.sort_values(time_column)
        test_count = max(1, int(round(len(ordered) * test_size)))
        train = ordered.iloc[:-test_count]
        test = ordered.iloc[-test_count:]
        return pd.Index(train.index), pd.Index(test.index)

    rng = np.random.default_rng(seed)
    indices = np.array(dataframe.index)
    rng.shuffle(indices)
    test_count = max(1, int(round(len(indices) * test_size)))
    test_index = pd.Index(indices[:test_count])
    train_index = pd.Index(indices[test_count:])
    return train_index, test_index


def get_regressor(name: str, seed: int):
    try:
        from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError:
        if name == "ridge":
            return NumpyRidgeRegressor(alpha=10.0)
        raise

    if name == "ridge":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        )
    if name == "knn":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=5)),
            ]
        )
    if name == "rf":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)),
            ]
        )
    if name == "xt":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", ExtraTreesRegressor(n_estimators=200, random_state=seed, n_jobs=-1)),
            ]
        )
    if name == "gbr":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingRegressor(random_state=seed)),
            ]
        )
    raise ValueError(f"Unsupported model: {name}")


class NumpyRidgeRegressor:
    """Small sklearn-free fallback for smoke runs in minimal Python envs."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.medians: np.ndarray | None = None
        self.means: np.ndarray | None = None
        self.stds: np.ndarray | None = None
        self.coef_: np.ndarray | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "NumpyRidgeRegressor":
        x_values = x.to_numpy(dtype="float64")
        y_values = y.to_numpy(dtype="float64")
        self.medians = np.nanmedian(x_values, axis=0)
        self.medians = np.where(np.isfinite(self.medians), self.medians, 0.0)
        missing = ~np.isfinite(x_values)
        if missing.any():
            x_values[missing] = np.take(self.medians, np.where(missing)[1])

        self.means = np.mean(x_values, axis=0)
        self.stds = np.std(x_values, axis=0)
        self.stds = np.where(self.stds > 0, self.stds, 1.0)
        x_scaled = (x_values - self.means) / self.stds
        design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        self.coef_ = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y_values
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None or self.medians is None or self.means is None or self.stds is None:
            raise RuntimeError("Model must be fitted before prediction.")
        x_values = x.to_numpy(dtype="float64")
        missing = ~np.isfinite(x_values)
        if missing.any():
            x_values[missing] = np.take(self.medians, np.where(missing)[1])
        x_scaled = (x_values - self.means) / self.stds
        design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
        return design @ self.coef_


def augment_training_rows(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    method: str,
    ratio: float,
    intensity: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series, int]:
    if method == "none" or ratio <= 0 or x_train.empty:
        return x_train.copy(), y_train.copy(), 0

    rng = np.random.default_rng(seed)
    n_synthetic = max(1, int(round(len(x_train) * ratio)))
    sampled_positions = rng.integers(0, len(x_train), size=n_synthetic)
    synthetic = x_train.iloc[sampled_positions].copy().reset_index(drop=True)
    y_synthetic = y_train.iloc[sampled_positions].copy().reset_index(drop=True)
    values = synthetic.to_numpy(dtype="float64")
    observed = np.isfinite(values)

    if method == "jitter":
        scale = np.nanstd(x_train.to_numpy(dtype="float64"), axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        noise = rng.normal(0.0, max(intensity, 1e-8), size=values.shape) * scale
        values = np.where(observed, values + noise, values)
    elif method == "scaling":
        factors = rng.normal(1.0, max(intensity, 1e-8), size=(len(synthetic), values.shape[1]))
        values = np.where(observed, values * factors, values)
    elif method == "time_mask":
        mask_width = max(1, int(round(values.shape[1] * max(min(intensity, 1.0), 1e-8))))
        for row_idx in range(len(values)):
            start = int(rng.integers(0, max(1, values.shape[1] - mask_width + 1)))
            values[row_idx, start : start + mask_width] = np.nan
    elif method == "magnitude_mask":
        mask = rng.random(values.shape) < max(min(intensity, 1.0), 0.0)
        values[mask] = np.nan
    else:
        raise ValueError(f"Unsupported augmentation method: {method}")

    synthetic = pd.DataFrame(values, columns=x_train.columns)
    x_augmented = pd.concat([x_train.reset_index(drop=True), synthetic], ignore_index=True)
    y_augmented = pd.concat([y_train.reset_index(drop=True), y_synthetic], ignore_index=True)
    return x_augmented, y_augmented, n_synthetic


def regression_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(y_true.to_numpy(dtype="float64")) & np.isfinite(predictions)
    if not valid.any():
        return {"r2": np.nan, "rmse": np.nan, "mae": np.nan}
    y_valid = y_true.to_numpy(dtype="float64")[valid]
    pred_valid = predictions[valid]
    denominator = float(np.sum((y_valid - np.mean(y_valid)) ** 2))
    return {
        "r2": 1.0 - float(np.sum((y_valid - pred_valid) ** 2)) / denominator if denominator > 0 else np.nan,
        "rmse": float(math.sqrt(np.mean((y_valid - pred_valid) ** 2))),
        "mae": float(np.mean(np.abs(y_valid - pred_valid))),
    }


def run_id_for(parts: dict[str, Any]) -> str:
    raw = json.dumps(parts, sort_keys=True)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return "_".join(str(parts[key]) for key in ["dataset", "variant", "model", "augmentation_method"]) + f"_{digest}"


def evaluate_downstream(
    dataframe: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    train_index: pd.Index,
    test_index: pd.Index,
    model_name: str,
    augmentation_method: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    x_train = coerce_numeric_features(dataframe.loc[train_index, feature_columns])
    x_test = coerce_numeric_features(dataframe.loc[test_index, feature_columns])
    y_train = pd.to_numeric(dataframe.loc[train_index, target_column], errors="coerce")
    y_test = pd.to_numeric(dataframe.loc[test_index, target_column], errors="coerce")

    x_augmented, y_augmented, n_synthetic = augment_training_rows(
        x_train,
        y_train,
        augmentation_method,
        args.augmentation_ratio,
        args.augmentation_intensity,
        args.seed,
    )

    regressor = get_regressor(model_name, args.seed)
    start = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")
        regressor.fit(x_augmented, y_augmented)
        predictions = regressor.predict(x_test)
    train_time = time.time() - start

    metrics = regression_metrics(y_test, predictions)
    metrics.update(
        {
            "n_train_rows": len(x_augmented),
            "n_synthetic_rows": n_synthetic,
            "n_test_rows": len(x_test),
            "n_features": len(feature_columns),
            "train_time_seconds": train_time,
        }
    )
    return metrics


def append_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    for column in RESULT_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame[RESULT_COLUMNS]
    if path.exists() and os.getenv("AUTOFEAT_APPEND_RESULTS", "0") == "1":
        old = pd.read_csv(path)
        frame = pd.concat([old, frame], ignore_index=True)
        frame = frame.drop_duplicates(subset=["run_id"], keep="last")
    frame.to_csv(path, index=False)


def save_metadata(args: argparse.Namespace, rows: list[dict[str, Any]], feature_sets: dict[str, list[str]]) -> Path:
    args.metadata_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(pd.DataFrame(rows).to_csv(index=False).encode("utf-8")).hexdigest()[:12]
    path = args.metadata_dir / f"ts_augmented_downstream_{args.dataset}_{digest}.json"
    metadata = {
        "experiment": "ts_augmented_downstream",
        "description": "Train-only classic time-series-style augmentation evaluated by downstream tabular regressors.",
        "args": vars(args)
        | {
            "data_dir": display_path(args.data_dir),
            "output": display_path(args.output),
            "metadata_dir": display_path(args.metadata_dir),
        },
        "feature_sets": feature_sets,
        "run_ids": [row["run_id"] for row in rows],
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    base_table = args.base_table or DATASETS[args.dataset]
    join_tables_to_use = args.join_tables or DEFAULT_JOIN_TABLES[args.dataset]
    policy_name = ",".join(args.policy)

    base = read_table(args.data_dir, base_table)
    joined = join_tables(
        data_dir=args.data_dir,
        base_table=base_table,
        join_tables_to_use=join_tables_to_use,
        join_key=args.join_key,
        join_mode=args.join_mode,
        tolerance_seconds=args.time_tolerance_seconds,
    )

    if args.target_column not in base.columns:
        raise ValueError(f"Target column {args.target_column!r} not found in {base_table}")
    if args.target_column not in joined.columns:
        raise ValueError(f"Target column {args.target_column!r} not found after joining")

    time_column = args.join_key if args.join_key in joined.columns else None
    train_index, test_index = make_split_indices(
        joined,
        split_mode=args.split_mode,
        test_size=args.test_size,
        time_column=time_column,
        seed=args.seed,
    )

    selection_start = time.time()
    selection = select_autofeat_plus_features(
        dataframe=joined.loc[train_index],
        target_column=args.target_column,
        top_k=args.autofeatplus_top_k,
        sensitive_patterns=policy_patterns(args.policy),
        privacy_penalty=args.privacy_penalty,
        missing_penalty=args.missing_penalty,
        cost_penalty=args.cost_penalty,
        block_sensitive=True,
        proxy_correlation_threshold=args.proxy_correlation_threshold,
        max_missing_ratio=args.max_missing_ratio,
    )
    selection_time = time.time() - selection_start

    base_features = [column for column in base.columns if column != args.target_column]
    joined_features = [column for column in joined.columns if column != args.target_column]
    autofeatplus_features = [
        column for column in selection.selected_features if column in joined.columns and column != args.target_column
    ]
    variant_frames = {
        "BASE": joined[[*base_features, args.target_column]],
        "Join_All_Local": joined[[*joined_features, args.target_column]],
        "AutoFeatPlus_Local": joined[[*autofeatplus_features, args.target_column]],
    }
    variant_features = {
        "BASE": base_features,
        "Join_All_Local": joined_features,
        "AutoFeatPlus_Local": autofeatplus_features,
    }

    rows: list[dict[str, Any]] = []
    for variant in args.variants:
        dataframe = variant_frames[variant]
        feature_columns = variant_features[variant]
        if not feature_columns:
            continue
        for augmentation_method in args.augmentation_method:
            for model_name in args.models:
                run_parts = {
                    "dataset": args.dataset,
                    "variant": variant,
                    "model": model_name,
                    "augmentation_method": augmentation_method,
                    "target_column": args.target_column,
                    "seed": args.seed,
                    "policy": policy_name,
                }
                row = {
                    "run_id": run_id_for(run_parts),
                    "dataset": args.dataset,
                    "target_column": args.target_column,
                    "variant": variant,
                    "model": model_name,
                    "augmentation_method": augmentation_method,
                    "augmentation_ratio": args.augmentation_ratio if augmentation_method != "none" else 0.0,
                    "augmentation_intensity": args.augmentation_intensity,
                    "split_mode": args.split_mode,
                    "feature_selection_time_seconds": selection_time if variant == "AutoFeatPlus_Local" else 0.0,
                    "privacy_policy": policy_name,
                    "n_blocked_features": len(selection.blocked_features) if variant == "AutoFeatPlus_Local" else 0,
                    "blocked_features": str(selection.blocked_features) if variant == "AutoFeatPlus_Local" else "[]",
                    "join_mode": args.join_mode,
                    "join_key": args.join_key,
                    "time_tolerance_seconds": args.time_tolerance_seconds,
                    "status": "ok",
                    "error": "",
                }
                try:
                    row.update(
                        evaluate_downstream(
                            dataframe=dataframe,
                            target_column=args.target_column,
                            feature_columns=feature_columns,
                            train_index=train_index,
                            test_index=test_index,
                            model_name=model_name,
                            augmentation_method=augmentation_method,
                            args=args,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - keep grid runs inspectable.
                    row.update({"status": "failed", "error": repr(exc)})
                rows.append(row)

    append_results(args.output, rows)
    metadata_path = save_metadata(args, rows, variant_features)
    print(f"Saved downstream augmentation rows to {display_path(args.output)}")
    print(f"Saved metadata to {display_path(metadata_path)}")
    preview = pd.DataFrame(rows)
    for column in ["r2", "rmse", "mae"]:
        if column not in preview.columns:
            preview[column] = np.nan
    print(preview[["dataset", "variant", "model", "augmentation_method", "r2", "rmse", "mae", "status", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()
