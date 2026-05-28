from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")
try:
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ModuleNotFoundError:
    ConvergenceWarning = None

DATASETS = {
    "rabbitmq": "rabbitmq-performance.csv",
    "amf": "amf-performance.csv",
    "golang_web": "golang-web-server-performance.csv",
    "python_web": "python-web-server-performance.csv",
}

DARTS_BASELINE_MODELS = [
    "naive_mean",
    "naive_drift",
    "naive_seasonal",
    "naive_moving_average",
]

DARTS_CPU_MODELS = [
    "linear_regression",
    "random_forest",
    "tcn",
    "nbeats",
    "nhits",
    "rnn",
    "transformer",
]

RESULT_COLUMNS = [
    "run_id",
    "dataset",
    "task",
    "model",
    "model_family",
    "augmentation_method",
    "augmentation_ratio",
    "augmentation_intensity",
    "window_size",
    "stride",
    "horizon",
    "seed",
    "split_mode",
    "n_train_windows",
    "n_val_windows",
    "n_test_windows",
    "n_features",
    "target_column",
    "mae",
    "rmse",
    "mape",
    "r2",
    "accuracy",
    "macro_f1",
    "train_time_seconds",
    "predict_time_seconds",
    "similarity_ks_mean",
    "similarity_wasserstein_mean",
    "similarity_corr_delta_mean",
    "discriminative_accuracy",
    "output_path",
    "metadata_path",
    "status",
    "error",
]


@dataclass
class PreparedData:
    dataframe: pd.DataFrame
    feature_columns: list[str]
    target_column: str
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_medians: pd.Series
    train_stds: pd.Series


@dataclass
class WindowData:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train_times: list[dict[str, int]]
    val_times: list[dict[str, int]]
    test_times: list[dict[str, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Darts-first EUR time-series benchmark with train-only augmentation.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/EUR/6907619"))
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="rabbitmq")
    parser.add_argument("--task", choices=["forecasting"], default="forecasting")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--augmentation-method",
        nargs="+",
        default=["none"],
        choices=["none", "jitter", "scaling", "time_mask", "magnitude_mask"],
    )
    parser.add_argument("--augmentation-ratio", type=float, default=0.5)
    parser.add_argument("--augmentation-intensity", type=float, default=0.05)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--target-column", default="lat99")
    parser.add_argument(
        "--target-columns",
        nargs="+",
        default=None,
        help="One or more target columns. Use 'all_features' to forecast every selected non-time numeric feature.",
    )
    parser.add_argument("--feature-columns", nargs="+", default=None)
    parser.add_argument("--time-column", default="time")
    parser.add_argument("--split-mode", choices=["time", "random"], default="time")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--darts-epochs", type=int, default=5)
    parser.add_argument("--darts-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--nn-preset",
        choices=["cpu_tiny", "cpu_small"],
        default="cpu_tiny",
        help="Small Darts neural model preset for CPU-only facilities.",
    )
    parser.add_argument(
        "--list-darts-models",
        action="store_true",
        help="List Darts models wired into this runner and exit.",
    )
    parser.add_argument("--profile-output", type=Path, default=Path("results/6g_data/darts/profiles"))
    parser.add_argument("--output", type=Path, default=Path("results/6g_data/darts/evaluation_summary.csv"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("results/6g_data/darts/predictions"))
    parser.add_argument("--metadata-dir", type=Path, default=Path("results/6g_data/darts/metadata"))
    parser.add_argument("--augmented-dir", type=Path, default=Path("results/6g_data/darts/augmented_windows"))
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--save-augmented-windows", action="store_true")
    return parser.parse_args()


def default_models() -> list[str]:
    return ["naive_drift", "linear_regression"]


def expand_model_aliases(models: list[str]) -> list[str]:
    expanded: list[str] = []
    for model in models:
        if model == "darts_baselines":
            expanded.extend(DARTS_BASELINE_MODELS)
        elif model == "darts_cpu":
            expanded.extend(["linear_regression", "random_forest", "tcn", "nbeats", "nhits"])
        elif model == "all_darts":
            expanded.extend([*DARTS_BASELINE_MODELS, *DARTS_CPU_MODELS])
        else:
            expanded.append(model)
    return list(dict.fromkeys(expanded))


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)


def coerce_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    else:
        cleaned = (
            series.astype(str)
            .str.strip()
            .str.replace(r"(?i)(mb|m)$", "000000", regex=True)
            .str.replace(r"(?i)(gb|g)$", "000000000", regex=True)
            .str.replace(r"[^0-9eE+\-\.]", "", regex=True)
            .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        )
        numeric = pd.to_numeric(cleaned, errors="coerce").astype("float64")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return numeric.mask(numeric.abs() > 1e18, np.nan)


def load_eur_table(data_dir: Path, dataset: str, time_column: str) -> pd.DataFrame:
    path = data_dir / DATASETS[dataset]
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    dataframe = pd.read_csv(path)
    if time_column not in dataframe.columns:
        raise ValueError(f"Expected time column {time_column!r} in {path}")
    dataframe[time_column] = pd.to_numeric(dataframe[time_column], errors="coerce")
    dataframe = dataframe.dropna(subset=[time_column]).copy()
    dataframe[time_column] = dataframe[time_column].astype("int64")
    return dataframe.sort_values(time_column).reset_index(drop=True)


def select_feature_columns(
    dataframe: pd.DataFrame,
    requested: list[str] | None,
    target_column: str,
    time_column: str,
) -> list[str]:
    if requested:
        missing = [column for column in requested if column not in dataframe.columns]
        if missing:
            raise ValueError(f"Requested feature columns not found: {missing}")
        return requested

    candidates: list[str] = []
    for column in dataframe.columns:
        if column in {time_column, "dt", "datetime"}:
            continue
        numeric = coerce_numeric_series(dataframe[column])
        if numeric.notna().mean() >= 0.2:
            candidates.append(column)
    if target_column not in candidates and target_column in dataframe.columns:
        candidates.append(target_column)
    return candidates


def resolve_target_columns(dataframe: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    if not args.target_columns:
        targets = [args.target_column]
    elif "all_features" in args.target_columns:
        targets = select_feature_columns(dataframe, args.feature_columns, args.target_column, args.time_column)
    else:
        targets = args.target_columns

    missing = [column for column in targets if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Target columns not found: {missing}")

    numeric_targets: list[str] = []
    for column in targets:
        if column == args.time_column:
            continue
        numeric = coerce_numeric_series(dataframe[column])
        if numeric.notna().mean() >= 0.2:
            numeric_targets.append(column)
    if not numeric_targets:
        raise ValueError("No numeric target columns available.")
    return list(dict.fromkeys(numeric_targets))


def write_profile(dataframe: pd.DataFrame, dataset: str, time_column: str, target_column: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_rows: list[dict[str, Any]] = []
    time_values = pd.to_numeric(dataframe[time_column], errors="coerce").dropna().sort_values()
    time_deltas = time_values.diff().dropna()

    for column in dataframe.columns:
        numeric = coerce_numeric_series(dataframe[column]) if column != time_column else pd.to_numeric(dataframe[column])
        profile_rows.append(
            {
                "dataset": dataset,
                "column": column,
                "rows": len(dataframe),
                "missing_ratio": float(numeric.isna().mean()),
                "numeric_ratio": float(numeric.notna().mean()),
                "mean": float(numeric.mean()) if numeric.notna().any() else np.nan,
                "std": float(numeric.std()) if numeric.notna().any() else np.nan,
                "min": float(numeric.min()) if numeric.notna().any() else np.nan,
                "p50": float(numeric.quantile(0.50)) if numeric.notna().any() else np.nan,
                "p95": float(numeric.quantile(0.95)) if numeric.notna().any() else np.nan,
                "p99": float(numeric.quantile(0.99)) if numeric.notna().any() else np.nan,
                "max": float(numeric.max()) if numeric.notna().any() else np.nan,
                "is_target": column == target_column,
                "time_start": int(time_values.iloc[0]) if not time_values.empty else np.nan,
                "time_end": int(time_values.iloc[-1]) if not time_values.empty else np.nan,
                "time_delta_median": float(time_deltas.median()) if not time_deltas.empty else np.nan,
                "time_delta_p95": float(time_deltas.quantile(0.95)) if not time_deltas.empty else np.nan,
                "large_gap_count": int((time_deltas > time_deltas.quantile(0.95)).sum()) if not time_deltas.empty else 0,
            }
        )

    output_path = output_dir / f"{dataset}_profile.csv"
    pd.DataFrame(profile_rows).to_csv(output_path, index=False)
    return output_path


def split_dataframe(
    dataframe: pd.DataFrame,
    split_mode: str,
    val_size: float,
    test_size: float,
    seed: int,
    time_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_size < 0 or test_size <= 0 or val_size + test_size >= 0.9:
        raise ValueError("Require val_size >= 0, test_size > 0, and val_size + test_size < 0.9")

    if split_mode == "time":
        ordered = dataframe.sort_values(time_column).reset_index(drop=True)
        n_total = len(ordered)
        test_count = max(1, int(round(n_total * test_size)))
        val_count = max(1, int(round(n_total * val_size))) if val_size > 0 else 0
        train_end = n_total - test_count - val_count
        if train_end <= 0:
            raise ValueError("Split leaves no training rows.")
        return (
            ordered.iloc[:train_end].copy(),
            ordered.iloc[train_end : train_end + val_count].copy(),
            ordered.iloc[train_end + val_count :].copy(),
        )

    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataframe))
    rng.shuffle(indices)
    test_count = max(1, int(round(len(indices) * test_size)))
    val_count = max(1, int(round(len(indices) * val_size))) if val_size > 0 else 0
    test_idx = indices[:test_count]
    val_idx = indices[test_count : test_count + val_count]
    train_idx = indices[test_count + val_count :]
    return dataframe.iloc[np.sort(train_idx)].copy(), dataframe.iloc[np.sort(val_idx)].copy(), dataframe.iloc[np.sort(test_idx)].copy()


def prepare_data(args: argparse.Namespace) -> PreparedData:
    dataframe = load_eur_table(args.data_dir, args.dataset, args.time_column)
    if args.target_column not in dataframe.columns:
        raise ValueError(f"Target column {args.target_column!r} not found in {DATASETS[args.dataset]}")
    feature_columns = select_feature_columns(dataframe, args.feature_columns, args.target_column, args.time_column)

    converted = dataframe[[args.time_column, *feature_columns]].copy()
    for column in feature_columns:
        converted[column] = coerce_numeric_series(converted[column])
    train_df, val_df, test_df = split_dataframe(
        converted,
        split_mode=args.split_mode,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        time_column=args.time_column,
    )

    train_medians = train_df[feature_columns].median(numeric_only=True).fillna(0.0)
    train_stds = train_df[feature_columns].std(numeric_only=True).replace(0, 1.0).fillna(1.0)
    return PreparedData(
        dataframe=converted,
        feature_columns=feature_columns,
        target_column=args.target_column,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_medians=train_medians,
        train_stds=train_stds,
    )


def scale_frame(dataframe: pd.DataFrame, feature_columns: list[str], medians: pd.Series, stds: pd.Series) -> pd.DataFrame:
    scaled = dataframe.copy()
    scaled[feature_columns] = (scaled[feature_columns] - medians[feature_columns]) / stds[feature_columns]
    return scaled


def build_windows_for_frame(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    time_column: str,
    window_size: int,
    stride: int,
    horizon: int,
    max_windows: int | None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    values = dataframe[feature_columns].to_numpy(dtype="float64")
    target_values = dataframe[target_column].to_numpy(dtype="float64")
    times = dataframe[time_column].to_numpy(dtype="int64")
    x_windows: list[np.ndarray] = []
    y_values: list[float] = []
    mappings: list[dict[str, int]] = []
    max_start = len(dataframe) - window_size - horizon
    if max_start < 0:
        return np.empty((0, window_size, len(feature_columns))), np.empty((0,), dtype="float64"), []

    for start in range(0, max_start + 1, stride):
        end = start + window_size
        x_windows.append(values[start:end])
        y_values.append(float(target_values[end + horizon - 1]))
        mappings.append(
            {
                "start_row": int(dataframe.index[start]),
                "end_row": int(dataframe.index[end - 1]),
                "start_time": int(times[start]),
                "end_time": int(times[end - 1]),
            }
        )
        if max_windows is not None and len(x_windows) >= max_windows:
            break

    x = np.stack(x_windows) if x_windows else np.empty((0, window_size, len(feature_columns)))
    y = np.array(y_values, dtype="float64")
    return x, y, mappings


def prepare_windows(args: argparse.Namespace, prepared: PreparedData) -> WindowData:
    train_scaled = scale_frame(prepared.train_df, prepared.feature_columns, prepared.train_medians, prepared.train_stds)
    val_scaled = scale_frame(prepared.val_df, prepared.feature_columns, prepared.train_medians, prepared.train_stds)
    test_scaled = scale_frame(prepared.test_df, prepared.feature_columns, prepared.train_medians, prepared.train_stds)

    x_train, y_train, train_times = build_windows_for_frame(
        train_scaled,
        prepared.feature_columns,
        prepared.target_column,
        args.time_column,
        args.window_size,
        args.stride,
        args.horizon,
        args.max_windows,
    )
    x_val, y_val, val_times = build_windows_for_frame(
        val_scaled,
        prepared.feature_columns,
        prepared.target_column,
        args.time_column,
        args.window_size,
        args.stride,
        args.horizon,
        args.max_windows,
    )
    x_test, y_test, test_times = build_windows_for_frame(
        test_scaled,
        prepared.feature_columns,
        prepared.target_column,
        args.time_column,
        args.window_size,
        args.stride,
        args.horizon,
        args.max_windows,
    )
    return WindowData(x_train, x_val, x_test, y_train, y_val, y_test, train_times, val_times, test_times)


def sample_for_augmentation(x_train: np.ndarray, ratio: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if ratio <= 0 or len(x_train) == 0:
        return x_train[:0].copy(), np.empty((0,), dtype=int)
    n_augmented = max(1, int(round(len(x_train) * ratio)))
    indices = rng.integers(0, len(x_train), size=n_augmented)
    return x_train[indices].copy(), indices


def augment_windows(
    x_train: np.ndarray,
    y_train: np.ndarray,
    method: str,
    ratio: float,
    intensity: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if method == "none":
        return x_train.copy(), y_train.copy(), x_train[:0].copy()

    rng = np.random.default_rng(seed)
    sampled, sampled_indices = sample_for_augmentation(x_train, ratio, rng)
    if len(sampled) == 0:
        return x_train.copy(), y_train.copy(), sampled

    synthetic = sampled.copy()
    observed = np.isfinite(synthetic)

    if method == "jitter":
        noise = rng.normal(loc=0.0, scale=max(intensity, 1e-8), size=synthetic.shape)
        synthetic = np.where(observed, synthetic + noise, synthetic)
    elif method == "scaling":
        factors = rng.normal(loc=1.0, scale=max(intensity, 1e-8), size=(len(synthetic), 1, synthetic.shape[2]))
        synthetic = np.where(observed, synthetic * factors, synthetic)
    elif method == "time_mask":
        mask_len = max(1, int(round(synthetic.shape[1] * max(intensity, 1e-8))))
        for i in range(len(synthetic)):
            start = int(rng.integers(0, max(1, synthetic.shape[1] - mask_len + 1)))
            synthetic[i, start : start + mask_len, :] = np.nan
    elif method == "magnitude_mask":
        mask = rng.random(synthetic.shape) < max(min(intensity, 1.0), 0.0)
        synthetic[mask] = np.nan
    else:
        raise ValueError(f"Unsupported augmentation method: {method}")

    x_augmented = np.concatenate([x_train, synthetic], axis=0)
    y_augmented = np.concatenate([y_train, y_train[sampled_indices]], axis=0)
    return x_augmented, y_augmented, synthetic


def fill_nan_with_train_median(x: np.ndarray, train_x: np.ndarray) -> np.ndarray:
    feature_medians = np.nanmedian(train_x.reshape(-1, train_x.shape[-1]), axis=0)
    feature_medians = np.where(np.isfinite(feature_medians), feature_medians, 0.0)
    filled = x.copy()
    missing = ~np.isfinite(filled)
    if missing.any():
        filled[missing] = np.take(feature_medians, np.where(missing)[-1])
    return filled


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() == 0:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan}
    yt = y_true[valid]
    yp = y_pred[valid]
    mape_mask = np.abs(yt) > 1e-12
    denominator = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = 1.0 - float(np.sum((yt - yp) ** 2)) / denominator if denominator > 0 else np.nan
    return {
        "mae": float(np.mean(np.abs(yt - yp))),
        "rmse": float(math.sqrt(np.mean((yt - yp) ** 2))),
        "mape": float(np.mean(np.abs((yt[mape_mask] - yp[mape_mask]) / yt[mape_mask]))) if mape_mask.any() else np.nan,
        "r2": r2 if len(yt) >= 2 else np.nan,
    }


def target_series_from_window(window: np.ndarray, y_value: float, target_index: int):
    from darts import TimeSeries

    values = np.concatenate([window[:, target_index], np.array([y_value], dtype="float64")])
    values = pd.Series(values).interpolate(limit_direction="both").ffill().bfill().fillna(0.0).to_numpy()
    return TimeSeries.from_values(values)


def history_series_from_window(window: np.ndarray, target_index: int):
    from darts import TimeSeries

    values = pd.Series(window[:, target_index]).interpolate(limit_direction="both").ffill().bfill().fillna(0.0).to_numpy()
    return TimeSeries.from_values(values)


def window_series_list(x_train: np.ndarray, y_train: np.ndarray, target_index: int) -> list[Any]:
    return [target_series_from_window(window, y, target_index) for window, y in zip(x_train, y_train, strict=True)]


def predict_scalar(prediction: Any) -> float:
    values = np.asarray(prediction.values(copy=False), dtype="float64").reshape(-1)
    return float(values[-1])


def build_darts_model(model_name: str, args: argparse.Namespace):
    if model_name == "linear_regression":
        from darts.models import LinearRegressionModel

        return LinearRegressionModel(lags=args.window_size, output_chunk_length=args.horizon)
    if model_name == "random_forest":
        try:
            from darts.models import RandomForestModel
        except ImportError:
            from darts.models import RandomForest as RandomForestModel

        return RandomForestModel(
            lags=args.window_size,
            output_chunk_length=args.horizon,
            n_estimators=100 if args.nn_preset == "cpu_small" else 50,
            random_state=args.seed,
            n_jobs=-1,
        )

    trainer_kwargs = {
        "accelerator": "cpu" if args.device == "cpu" else "auto",
        "devices": 1,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    n_epochs = args.darts_epochs
    batch_size = args.darts_batch_size
    input_chunk_length = args.window_size
    output_chunk_length = args.horizon
    width = 32 if args.nn_preset == "cpu_tiny" else 64

    if model_name == "tcn":
        from darts.models import TCNModel

        return TCNModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            n_epochs=n_epochs,
            batch_size=batch_size,
            num_filters=width,
            kernel_size=3,
            dropout=0.1,
            random_state=args.seed,
            pl_trainer_kwargs=trainer_kwargs,
            force_reset=True,
            save_checkpoints=False,
        )
    if model_name == "nbeats":
        from darts.models import NBEATSModel

        return NBEATSModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            n_epochs=n_epochs,
            batch_size=batch_size,
            num_stacks=2,
            num_blocks=1,
            num_layers=2,
            layer_widths=width,
            random_state=args.seed,
            pl_trainer_kwargs=trainer_kwargs,
            force_reset=True,
            save_checkpoints=False,
        )
    if model_name == "nhits":
        from darts.models import NHiTSModel

        return NHiTSModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            n_epochs=n_epochs,
            batch_size=batch_size,
            num_stacks=2,
            num_blocks=1,
            num_layers=2,
            layer_widths=width,
            random_state=args.seed,
            pl_trainer_kwargs=trainer_kwargs,
            force_reset=True,
            save_checkpoints=False,
        )
    if model_name == "rnn":
        from darts.models import RNNModel

        return RNNModel(
            input_chunk_length=input_chunk_length,
            training_length=input_chunk_length + output_chunk_length,
            hidden_dim=width,
            n_rnn_layers=1,
            n_epochs=n_epochs,
            batch_size=batch_size,
            random_state=args.seed,
            pl_trainer_kwargs=trainer_kwargs,
            force_reset=True,
            save_checkpoints=False,
        )
    if model_name == "transformer":
        from darts.models import TransformerModel

        return TransformerModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            d_model=width,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=width * 2,
            dropout=0.1,
            n_epochs=n_epochs,
            batch_size=batch_size,
            random_state=args.seed,
            pl_trainer_kwargs=trainer_kwargs,
            force_reset=True,
            save_checkpoints=False,
        )

    raise ValueError(f"Unsupported Darts model: {model_name}")


def run_darts_forecasting_model(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    target_index: int,
    args: argparse.Namespace,
) -> tuple[dict[str, float], np.ndarray, float, float, str]:
    start_train = time.time()
    train_series = window_series_list(x_train, y_train, target_index)

    if model_name == "naive_mean":
        from darts.models import NaiveMean

        train_time = time.time() - start_train
        start_predict = time.time()
        preds = []
        for window in x_test:
            model = NaiveMean()
            model.fit(history_series_from_window(window, target_index))
            preds.append(predict_scalar(model.predict(args.horizon)))
        predict_time = time.time() - start_predict
        family = "darts_baseline"
    elif model_name == "naive_drift":
        from darts.models import NaiveDrift

        train_time = time.time() - start_train
        start_predict = time.time()
        preds = []
        for window in x_test:
            model = NaiveDrift()
            model.fit(history_series_from_window(window, target_index))
            preds.append(predict_scalar(model.predict(args.horizon)))
        predict_time = time.time() - start_predict
        family = "darts_baseline"
    elif model_name == "naive_seasonal":
        from darts.models import NaiveSeasonal

        train_time = time.time() - start_train
        start_predict = time.time()
        preds = []
        k = min(max(1, args.horizon), max(1, args.window_size // 2))
        for window in x_test:
            model = NaiveSeasonal(K=k)
            model.fit(history_series_from_window(window, target_index))
            preds.append(predict_scalar(model.predict(args.horizon)))
        predict_time = time.time() - start_predict
        family = "darts_baseline"
    elif model_name == "naive_moving_average":
        from darts.models import NaiveMovingAverage

        train_time = time.time() - start_train
        start_predict = time.time()
        preds = []
        input_chunk_length = min(args.window_size, 8)
        for window in x_test:
            model = NaiveMovingAverage(input_chunk_length=input_chunk_length)
            model.fit(history_series_from_window(window, target_index))
            preds.append(predict_scalar(model.predict(args.horizon)))
        predict_time = time.time() - start_predict
        family = "darts_baseline"
    else:
        model = build_darts_model(model_name, args)
        model.fit(train_series)
        train_time = time.time() - start_train
        start_predict = time.time()
        preds = []
        for window in x_test:
            history = history_series_from_window(window, target_index)
            preds.append(predict_scalar(model.predict(args.horizon, series=history)))
        predict_time = time.time() - start_predict
        family = "darts"

    y_pred = np.array(preds, dtype="float64")
    return regression_metrics(y_test, y_pred), y_pred, train_time, predict_time, family


def similarity_metrics(real: np.ndarray, synthetic: np.ndarray, seed: int) -> dict[str, float]:
    if len(synthetic) == 0:
        return {
            "similarity_ks_mean": np.nan,
            "similarity_wasserstein_mean": np.nan,
            "similarity_corr_delta_mean": np.nan,
            "discriminative_accuracy": np.nan,
        }

    real_flat_features = real.reshape(-1, real.shape[-1])
    syn_flat_features = synthetic.reshape(-1, synthetic.shape[-1])
    ks_values: list[float] = []
    wasserstein_values: list[float] = []
    try:
        from scipy.stats import ks_2samp, wasserstein_distance
    except ModuleNotFoundError:
        ks_2samp = None
        wasserstein_distance = None

    for feature_idx in range(real.shape[-1]):
        r = real_flat_features[:, feature_idx]
        s = syn_flat_features[:, feature_idx]
        r = r[np.isfinite(r)]
        s = s[np.isfinite(s)]
        if len(r) > 1 and len(s) > 1:
            if ks_2samp is not None and wasserstein_distance is not None:
                ks_values.append(float(ks_2samp(r, s).statistic))
                wasserstein_values.append(float(wasserstein_distance(r, s)))
            else:
                pooled_std = float(np.nanstd(r)) or 1.0
                ks_values.append(float(abs(np.nanmean(r) - np.nanmean(s)) / pooled_std))
                wasserstein_values.append(float(abs(np.nanmedian(r) - np.nanmedian(s))))

    real_corr = np.nan_to_num(np.corrcoef(np.nanmean(real, axis=1), rowvar=False), nan=0.0)
    syn_corr = np.nan_to_num(np.corrcoef(np.nanmean(synthetic, axis=1), rowvar=False), nan=0.0)
    corr_delta = float(np.mean(np.abs(real_corr - syn_corr))) if real_corr.shape == syn_corr.shape else np.nan

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        n = min(len(real), len(synthetic), 2000)
        rng = np.random.default_rng(seed)
        real_idx = rng.choice(len(real), size=n, replace=False)
        syn_idx = rng.choice(len(synthetic), size=n, replace=False)
        x_disc = np.concatenate([real[real_idx], synthetic[syn_idx]], axis=0)
        y_disc = np.array([0] * n + [1] * n)
        x_disc = fill_nan_with_train_median(x_disc, real).reshape(len(x_disc), -1)
        x_train, x_test, y_train, y_test = train_test_split(x_disc, y_disc, test_size=0.3, random_state=seed, stratify=y_disc)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300))
        clf.fit(x_train, y_train)
        disc_acc = float(accuracy_score(y_test, clf.predict(x_test)))
    except (ModuleNotFoundError, ValueError):
        disc_acc = np.nan

    return {
        "similarity_ks_mean": float(np.mean(ks_values)) if ks_values else np.nan,
        "similarity_wasserstein_mean": float(np.mean(wasserstein_values)) if wasserstein_values else np.nan,
        "similarity_corr_delta_mean": corr_delta,
        "discriminative_accuracy": disc_acc,
    }


def build_run_id(args: argparse.Namespace, model_name: str, augmentation_method: str) -> str:
    raw = json.dumps(
        {
            "backend": "darts",
            "dataset": args.dataset,
            "task": args.task,
            "model": model_name,
            "augmentation_method": augmentation_method,
            "augmentation_ratio": args.augmentation_ratio,
            "augmentation_intensity": args.augmentation_intensity,
            "window_size": args.window_size,
            "stride": args.stride,
            "horizon": args.horizon,
            "seed": args.seed,
            "target_column": args.target_column,
        },
        sort_keys=True,
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{args.dataset}_{args.task}_{safe_name(args.target_column)}_{model_name}_{augmentation_method}_{digest}"


def save_metadata(
    args: argparse.Namespace,
    prepared: PreparedData,
    windows: WindowData,
    run_id: str,
    model_name: str,
    augmentation_method: str,
    output_path: Path,
) -> Path:
    args.metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_id": run_id,
        "backend": "darts",
        "args": vars(args)
        | {
            "data_dir": str(args.data_dir),
            "output": str(args.output),
            "predictions_dir": str(args.predictions_dir),
            "metadata_dir": str(args.metadata_dir),
            "augmented_dir": str(args.augmented_dir),
            "profile_output": str(args.profile_output),
        },
        "model": model_name,
        "augmentation_method": augmentation_method,
        "dataset_file": DATASETS[args.dataset],
        "feature_columns": prepared.feature_columns,
        "target_column": prepared.target_column,
        "train_rows": len(prepared.train_df),
        "val_rows": len(prepared.val_df),
        "test_rows": len(prepared.test_df),
        "train_windows": windows.train_times,
        "val_windows": windows.val_times,
        "test_windows": windows.test_times,
        "output_path": str(output_path),
    }
    metadata_path = args.metadata_dir / f"{run_id}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def append_results(output: Path, rows: list[dict[str, Any]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    for column in RESULT_COLUMNS:
        if column not in new_df.columns:
            new_df[column] = np.nan
    new_df = new_df[RESULT_COLUMNS]
    if output.exists():
        old_df = pd.read_csv(output)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["run_id"], keep="last")
    else:
        combined = new_df
    combined.to_csv(output, index=False)


def save_predictions(predictions_dir: Path, run_id: str, predictions: np.ndarray) -> Path:
    predictions_dir.mkdir(parents=True, exist_ok=True)
    path = predictions_dir / f"{run_id}.npz"
    np.savez_compressed(path, predictions=predictions)
    return path


def save_augmented_windows(augmented_dir: Path, run_id: str, synthetic: np.ndarray) -> None:
    augmented_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(augmented_dir / f"{run_id}.npz", synthetic=synthetic)


def run_one(
    args: argparse.Namespace,
    prepared: PreparedData,
    windows: WindowData,
    model_name: str,
    augmentation_method: str,
) -> dict[str, Any]:
    run_id = build_run_id(args, model_name, augmentation_method)
    row: dict[str, Any] = {
        "run_id": run_id,
        "dataset": args.dataset,
        "task": args.task,
        "model": model_name,
        "augmentation_method": augmentation_method,
        "augmentation_ratio": args.augmentation_ratio if augmentation_method != "none" else 0.0,
        "augmentation_intensity": args.augmentation_intensity,
        "window_size": args.window_size,
        "stride": args.stride,
        "horizon": args.horizon,
        "seed": args.seed,
        "split_mode": args.split_mode,
        "n_train_windows": len(windows.x_train),
        "n_val_windows": len(windows.x_val),
        "n_test_windows": len(windows.x_test),
        "n_features": len(prepared.feature_columns),
        "target_column": prepared.target_column,
        "status": "ok",
        "error": "",
    }

    try:
        x_train_aug, y_train_aug, synthetic = augment_windows(
            windows.x_train,
            windows.y_train,
            augmentation_method,
            args.augmentation_ratio,
            args.augmentation_intensity,
            args.seed,
        )
        row.update(similarity_metrics(windows.x_train, synthetic, args.seed))
        if args.save_augmented_windows and len(synthetic):
            save_augmented_windows(args.augmented_dir, run_id, synthetic)

        target_index = prepared.feature_columns.index(prepared.target_column)
        metrics, predictions, train_time, predict_time, family = run_darts_forecasting_model(
            model_name,
            x_train_aug,
            y_train_aug,
            windows.x_test,
            windows.y_test,
            target_index,
            args,
        )
        row.update(metrics)
        output_path = save_predictions(args.predictions_dir, run_id, predictions)
        metadata_path = save_metadata(args, prepared, windows, run_id, model_name, augmentation_method, output_path)
        row.update(
            {
                "model_family": family,
                "train_time_seconds": train_time,
                "predict_time_seconds": predict_time,
                "output_path": str(output_path),
                "metadata_path": str(metadata_path),
                "n_train_windows": len(x_train_aug),
            }
        )
    except Exception as exc:  # noqa: BLE001 - result CSV should capture failed model rows.
        row.update({"status": "failed", "error": repr(exc)})

    return row


def validate_window_data(windows: WindowData) -> None:
    if len(windows.x_train) == 0 or len(windows.x_test) == 0:
        raise ValueError("Window construction produced no train or test windows.")
    if len(windows.y_train) == 0 or len(windows.y_test) == 0:
        raise ValueError("Forecasting requires train and test targets.")


def main() -> None:
    args = parse_args()
    if args.list_darts_models:
        print("Darts baseline models:")
        print(" ".join(DARTS_BASELINE_MODELS))
        print("\nDarts CPU candidate models:")
        print(" ".join(DARTS_CPU_MODELS))
        print("\nAliases:")
        print("darts_baselines darts_cpu all_darts")
        return

    args.models = default_models() if args.models is None else expand_model_aliases(args.models)

    raw_dataframe = load_eur_table(args.data_dir, args.dataset, args.time_column)
    target_columns = resolve_target_columns(raw_dataframe, args)
    profile_path = write_profile(raw_dataframe, args.dataset, args.time_column, target_columns[0], args.profile_output)
    print(f"Saved profile to {profile_path}")
    if args.profile_only:
        return

    rows: list[dict[str, Any]] = []
    original_target = args.target_column
    try:
        for target_column in target_columns:
            args.target_column = target_column
            prepared = prepare_data(args)
            windows = prepare_windows(args, prepared)
            validate_window_data(windows)
            for augmentation_method in args.augmentation_method:
                for model_name in args.models:
                    rows.append(run_one(args, prepared, windows, model_name, augmentation_method))
    finally:
        args.target_column = original_target

    append_results(args.output, rows)
    print(f"Saved evaluation rows to {args.output}")
    display_columns = [
        "dataset",
        "task",
        "target_column",
        "model",
        "augmentation_method",
        "model_family",
        "mae",
        "rmse",
        "r2",
        "status",
        "error",
    ]
    print(pd.DataFrame(rows)[display_columns])


if __name__ == "__main__":
    main()
