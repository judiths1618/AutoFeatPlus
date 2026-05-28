from __future__ import annotations

import argparse
import json
import math
import os
import time as wall_time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASETS = {
    "rabbitmq": "rabbitmq-performance.csv",
    "amf": "amf-performance.csv",
    "golang_web": "golang-web-server-performance.csv",
    "python_web": "python-web-server-performance.csv",
}

DEFAULT_METHODS = ["ffill_bfill", "linear_time", "mean", "imputegap_interpolation", "imputegap_mean"]


@dataclass
class PreparedTable:
    dataset: str
    path: Path
    raw: pd.DataFrame
    ordered: pd.DataFrame
    numeric: pd.DataFrame
    regular_numeric: pd.DataFrame
    expected_frequency: pd.Timedelta
    gap_rows: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate EUR CSVs, detect timestamp gaps, and compare time-series imputation methods. "
            "ImputeGAP is used when installed; deterministic local baselines remain available."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/EUR/6907619"))
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--time-column", default="time")
    parser.add_argument("--output-dir", type=Path, default=Path("results/6g_data/imputegap_quality"))
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--mask-rate", type=float, default=0.1)
    parser.add_argument("--gap-multiple", type=float, default=1.5)
    parser.add_argument("--max-regular-rows", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-imputed", action="store_true")
    return parser.parse_args()


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


def parse_time_column(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() >= 0.8:
        median_abs = float(numeric.dropna().abs().median()) if numeric.notna().any() else 0.0
        if median_abs > 1e17:
            unit = "ns"
        elif median_abs > 1e14:
            unit = "us"
        elif median_abs > 1e11:
            unit = "ms"
        else:
            unit = "s"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def format_timestamp(value: pd.Timestamp | pd.NaT) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def infer_frequency(times: pd.Series) -> pd.Timedelta:
    deltas = times.sort_values().diff().dropna()
    deltas = deltas[deltas > pd.Timedelta(0)]
    if deltas.empty:
        return pd.Timedelta(seconds=1)
    return deltas.median()


def run_great_expectations_checks(dataframe: pd.DataFrame, dataset: str, time_column: str) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    gx_available = False
    gx_error: str | None = None

    try:
        import great_expectations as gx  # type: ignore

        gx_available = True
        gx_results = run_gx_v1_batch_checks(gx, dataframe, dataset, time_column)
        if gx_results is None:
            gx_results = run_gx_legacy_pandas_checks(gx, dataframe, time_column)

        if gx_results is not None:
            checks.extend(gx_results)
        else:
            gx_error = "installed Great Expectations API does not expose a supported pandas validator"
    except Exception as exc:  # noqa: BLE001 - optional dependency should not stop the audit
        gx_error = f"{type(exc).__name__}: {exc}"

    parsed_time = parse_time_column(dataframe[time_column]) if time_column in dataframe else pd.Series(dtype="datetime64[ns, UTC]")
    native_checks = {
        "row_count_positive": len(dataframe) > 0,
        "column_count_gt_one": len(dataframe.columns) > 1,
        "time_column_exists": time_column in dataframe.columns,
        "time_parse_success_ratio_ge_0_99": parsed_time.notna().mean() >= 0.99 if len(dataframe) else False,
        "time_monotonic_increasing": parsed_time.dropna().is_monotonic_increasing,
        "time_unique": parsed_time.dropna().is_unique,
        "no_fully_null_columns": not dataframe.isna().all(axis=0).any(),
    }
    checks.extend(
        {"check": check, "success": bool(success), "source": "native_ge_fallback"}
        for check, success in native_checks.items()
    )

    return {
        "dataset": dataset,
        "great_expectations_available": gx_available,
        "great_expectations_error": gx_error,
        "success": all(check["success"] for check in checks),
        "checks": checks,
    }


def run_gx_v1_batch_checks(gx: Any, dataframe: pd.DataFrame, dataset: str, time_column: str) -> list[dict[str, Any]] | None:
    if not hasattr(gx, "get_context") or not hasattr(gx, "expectations"):
        return None

    context = gx.get_context(mode="ephemeral")
    data_source_name = f"eur_{dataset}_pandas"
    data_asset_name = f"{dataset}_dataframe"
    batch_definition_name = "whole_dataframe"

    data_source = context.data_sources.add_pandas(name=data_source_name)
    data_asset = data_source.add_dataframe_asset(name=data_asset_name)
    batch_definition = data_asset.add_batch_definition_whole_dataframe(batch_definition_name)
    batch = batch_definition.get_batch(batch_parameters={"dataframe": dataframe})

    expectations = [
        ("expect_table_row_count_to_be_between", gx.expectations.ExpectTableRowCountToBeBetween(min_value=1)),
        ("expect_column_to_exist:time", gx.expectations.ExpectColumnToExist(column=time_column)),
        ("expect_column_values_to_not_be_null:time", gx.expectations.ExpectColumnValuesToNotBeNull(column=time_column)),
        ("expect_column_values_to_be_unique:time", gx.expectations.ExpectColumnValuesToBeUnique(column=time_column)),
    ]
    return [
        {
            "check": name,
            "success": bool(batch.validate(expectation).success),
            "source": "great_expectations_v1",
        }
        for name, expectation in expectations
    ]


def run_gx_legacy_pandas_checks(gx: Any, dataframe: pd.DataFrame, time_column: str) -> list[dict[str, Any]] | None:
    validator = None
    try:
        if hasattr(gx, "from_pandas"):
            validator = gx.from_pandas(dataframe)
        elif hasattr(gx, "dataset") and hasattr(gx.dataset, "PandasDataset"):
            validator = gx.dataset.PandasDataset(dataframe)
    except Exception:
        return None
    if validator is None:
        return None

    expectations = [
        ("expect_table_row_count_to_be_between", validator.expect_table_row_count_to_be_between(min_value=1)),
        ("expect_column_to_exist:time", validator.expect_column_to_exist(time_column)),
        ("expect_column_values_to_not_be_null:time", validator.expect_column_values_to_not_be_null(time_column)),
        ("expect_column_values_to_be_unique:time", validator.expect_column_values_to_be_unique(time_column)),
    ]
    return [
        {
            "check": name,
            "success": bool(result.get("success", False)),
            "source": "great_expectations_legacy",
        }
        for name, result in expectations
    ]


def quality_summary(
    dataframe: pd.DataFrame,
    dataset: str,
    path: Path,
    time_column: str,
    validation: dict[str, Any],
) -> dict[str, Any]:
    parsed_time = parse_time_column(dataframe[time_column]) if time_column in dataframe else pd.Series(dtype="datetime64[ns, UTC]")
    numeric_columns = [
        column
        for column in dataframe.columns
        if column != time_column and coerce_numeric_series(dataframe[column]).notna().mean() >= 0.2
    ]
    missing_ratios = dataframe.isna().mean()
    numeric_parse_ratios = {
        column: float(coerce_numeric_series(dataframe[column]).notna().mean())
        for column in dataframe.columns
        if column != time_column
    }
    return {
        "dataset": dataset,
        "file": str(path),
        "row_count": len(dataframe),
        "column_count": len(dataframe.columns),
        "time_start": format_timestamp(parsed_time.min()),
        "time_end": format_timestamp(parsed_time.max()),
        "time_parse_success_ratio": float(parsed_time.notna().mean()) if len(dataframe) else 0.0,
        "time_is_monotonic": bool(parsed_time.dropna().is_monotonic_increasing),
        "time_is_unique": bool(parsed_time.dropna().is_unique),
        "max_missing_ratio": float(missing_ratios.max()) if not missing_ratios.empty else 0.0,
        "fully_null_columns": json.dumps(dataframe.columns[dataframe.isna().all(axis=0)].tolist()),
        "numeric_column_count": len(numeric_columns),
        "numeric_columns": json.dumps(numeric_columns),
        "numeric_parse_ratios": json.dumps(numeric_parse_ratios, sort_keys=True),
        "great_expectations_available": validation["great_expectations_available"],
        "great_expectations_success": validation["success"],
        "great_expectations_error": validation["great_expectations_error"],
    }


def prepare_table(args: argparse.Namespace, dataset: str) -> tuple[PreparedTable, dict[str, Any], dict[str, Any]]:
    path = args.data_dir / DATASETS[dataset]
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    raw = pd.read_csv(path)
    if args.time_column not in raw.columns:
        raise ValueError(f"Expected time column {args.time_column!r} in {path}")

    validation = run_great_expectations_checks(raw, dataset, args.time_column)
    summary = quality_summary(raw, dataset, path, args.time_column, validation)

    ordered = raw.copy()
    ordered[args.time_column] = parse_time_column(ordered[args.time_column])
    ordered = ordered.dropna(subset=[args.time_column]).sort_values(args.time_column).reset_index(drop=True)

    numeric_columns = [
        column
        for column in ordered.columns
        if column != args.time_column and coerce_numeric_series(ordered[column]).notna().mean() >= 0.2
    ]
    numeric = pd.DataFrame({column: coerce_numeric_series(ordered[column]) for column in numeric_columns})
    numeric.index = pd.DatetimeIndex(ordered[args.time_column], name=args.time_column)

    expected_frequency = infer_frequency(pd.Series(numeric.index))
    deltas = pd.Series(numeric.index).diff()
    gap_threshold = expected_frequency * args.gap_multiple
    gap_rows: list[dict[str, Any]] = []
    for position, delta in deltas[deltas > gap_threshold].items():
        start = ordered.loc[position - 1, args.time_column]
        end = ordered.loc[position, args.time_column]
        missing_slots = max(0, int(math.floor(delta / expected_frequency)) - 1)
        if missing_slots == 0:
            continue
        gap_rows.append(
            {
                "dataset": dataset,
                "file": str(path),
                "gap_start": format_timestamp(start),
                "gap_end": format_timestamp(end),
                "gap_seconds": float(delta.total_seconds()),
                "expected_frequency_seconds": float(expected_frequency.total_seconds()),
                "estimated_missing_rows": missing_slots,
            }
        )

    base_numeric = numeric.groupby(level=0).mean()
    regular_index = base_numeric.index.union(
        pd.date_range(base_numeric.index.min(), base_numeric.index.max(), freq=expected_frequency, tz="UTC")
    )
    if len(regular_index) > args.max_regular_rows:
        raise ValueError(
            f"{dataset} would expand to {len(regular_index)} rows at {expected_frequency}; "
            f"increase --max-regular-rows if this is expected."
        )
    regular_numeric = base_numeric.reindex(regular_index)
    regular_numeric.index.name = args.time_column

    return (
        PreparedTable(dataset, path, raw, ordered, numeric, regular_numeric, expected_frequency, gap_rows),
        validation,
        summary,
    )


def impute_with_pandas(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "ffill_bfill":
        return frame.ffill().bfill()
    if method == "linear_time":
        return frame.interpolate(method="time", limit_direction="both").ffill().bfill()
    if method == "mean":
        return frame.fillna(frame.mean()).ffill().bfill()
    raise ValueError(f"Unsupported pandas imputation method: {method}")


def recovered_imputegap_frame(frame: pd.DataFrame, imputer: Any) -> pd.DataFrame:
    imputed = imputer.impute()
    recovered = getattr(imputed, "recov_data", None)
    if recovered is None:
        recovered = getattr(imputer, "recov_data", None)
    if recovered is None:
        recovered = getattr(imputed, "imputed_matrix", None)
    if recovered is None:
        recovered = getattr(imputer, "imputed_matrix", None)
    if recovered is None:
        raise RuntimeError("ImputeGAP did not expose recov_data or imputed_matrix")
    return pd.DataFrame(np.asarray(recovered).T, index=frame.index, columns=frame.columns)


def impute_with_imputegap(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
    from imputegap.recovery.imputation import Imputation  # type: ignore

    matrix = frame.to_numpy(dtype="float64").T
    if method == "imputegap_cdrec":
        imputer = Imputation.MatrixCompletion.CDRec(matrix)
    elif method == "imputegap_interpolation":
        imputer = Imputation.Statistics.Interpolation(matrix)
    elif method == "imputegap_mean":
        imputer = Imputation.Statistics.MeanImpute(matrix)
    elif method == "imputegap_mean_by_series":
        imputer = Imputation.Statistics.MeanImputeBySeries(matrix)
    elif method == "imputegap_knn":
        imputer = Imputation.Statistics.KNNImpute(matrix)
    else:
        raise ValueError(f"Unsupported ImputeGAP method: {method}")
    return recovered_imputegap_frame(frame, imputer)


def impute_frame(frame: pd.DataFrame, method: str) -> tuple[pd.DataFrame, str, str | None]:
    try:
        if method.startswith("imputegap_"):
            return impute_with_imputegap(frame, method), "imputegap", None
        return impute_with_pandas(frame, method), "pandas", None
    except Exception as exc:  # noqa: BLE001 - one failed imputer should not hide other comparisons
        fallback = impute_with_pandas(frame, "linear_time")
        return fallback, "fallback_linear_time", f"{type(exc).__name__}: {exc}"


def mask_observed_values(frame: pd.DataFrame, rate: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    observed = frame.notna()
    random_mask = pd.DataFrame(rng.random(frame.shape) < rate, index=frame.index, columns=frame.columns)
    eval_mask = observed & random_mask
    masked = frame.mask(eval_mask)
    return masked, eval_mask


def score_imputation(original: pd.DataFrame, imputed: pd.DataFrame, eval_mask: pd.DataFrame) -> dict[str, float]:
    truth = original.where(eval_mask).to_numpy(dtype="float64")
    pred = imputed.where(eval_mask).to_numpy(dtype="float64")
    valid = np.isfinite(truth) & np.isfinite(pred)
    if not valid.any():
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan}
    errors = pred[valid] - truth[valid]
    denom = np.abs(truth[valid])
    mape_mask = denom > 1e-12
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mape": float(np.mean(np.abs(errors[mape_mask] / denom[mape_mask]))) if mape_mask.any() else np.nan,
    }


def save_imputed_csv(table: PreparedTable, imputed: pd.DataFrame, method: str, output_dir: Path) -> Path:
    imputed_dir = output_dir / "imputed_csv"
    imputed_dir.mkdir(parents=True, exist_ok=True)
    output_path = imputed_dir / f"{table.dataset}_{method}.csv"
    output = imputed.reset_index()
    output[table.regular_numeric.index.name or "time"] = output[table.regular_numeric.index.name or "time"].map(format_timestamp)
    output.to_csv(output_path, index=False)
    return output_path


def compare_imputers(args: argparse.Namespace, table: PreparedTable) -> list[dict[str, Any]]:
    masked, eval_mask = mask_observed_values(table.regular_numeric, args.mask_rate, args.seed)
    rows: list[dict[str, Any]] = []
    for method in args.methods:
        start = wall_time.time()
        imputed, backend, error = impute_frame(masked, method)
        runtime = wall_time.time() - start
        metrics = score_imputation(table.regular_numeric, imputed, eval_mask)
        output_path = None
        if args.save_imputed:
            output_path = str(save_imputed_csv(table, imputed, method, args.output_dir))
        rows.append(
            {
                "dataset": table.dataset,
                "method": method,
                "backend": backend,
                "status": "ok" if error is None else "fallback",
                "error": error,
                "mask_rate": args.mask_rate,
                "seed": args.seed,
                "n_regular_rows": len(table.regular_numeric),
                "n_features": len(table.regular_numeric.columns),
                "n_eval_cells": int(eval_mask.to_numpy().sum()),
                "real_missing_cells": int(table.regular_numeric.isna().to_numpy().sum()),
                "expected_frequency_seconds": float(table.expected_frequency.total_seconds()),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "runtime_seconds": runtime,
                "output_path": output_path,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validation_payloads: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for dataset in args.datasets:
        table, validation, summary = prepare_table(args, dataset)
        validation_payloads.append(validation)
        quality_rows.append(summary)
        gap_rows.extend(table.gap_rows)
        comparison_rows.extend(compare_imputers(args, table))

    quality_path = args.output_dir / "eur_quality_summary.csv"
    gaps_path = args.output_dir / "eur_time_gap_summary.csv"
    comparisons_path = args.output_dir / "eur_imputation_comparison.csv"
    validation_path = args.output_dir / "great_expectations_validation.json"

    pd.DataFrame(quality_rows).to_csv(quality_path, index=False)
    pd.DataFrame(gap_rows).to_csv(gaps_path, index=False)
    pd.DataFrame(comparison_rows).to_csv(comparisons_path, index=False)
    validation_path.write_text(json.dumps(validation_payloads, indent=2), encoding="utf-8")

    print(f"Wrote {quality_path}")
    print(f"Wrote {gaps_path}")
    print(f"Wrote {comparisons_path}")
    print(f"Wrote {validation_path}")


if __name__ == "__main__":
    main()
