from __future__ import annotations

import argparse
import ast
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feature_discovery.experiments.autofeat_plus import select_autofeat_plus_features
from feature_discovery.experiments.local_benchmark_utils import (
    join_tables,
    make_tabular_split,
    parse_feature_list,
    read_table,
)


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
        "column:ram_limit_mb",
        "column:ram_usage_mb",
    ],
    "workload-private": ["column:n", "column:c"],
    "target-proxy-private": [r"re:(^|\.)lat\d+_ms$", "column:min_ms", "column:mean_ms"],
}
def load_autofeatplus_features(results_csv: Path, algorithm: str) -> list[str]:
    if not results_csv.exists():
        raise FileNotFoundError(f"AutoFeatPlus results file not found: {results_csv}")

    results = pd.read_csv(results_csv)
    subset = results[results["approach"] == "AutoFeatPlus_Local"].copy()
    if subset.empty:
        raise ValueError(f"No AutoFeatPlus_Local rows found in {results_csv}")

    model_rows = subset[subset["algorithm"].astype(str) == algorithm]
    row_source = model_rows if not model_rows.empty else subset
    row = row_source.sort_values("accuracy", ascending=False).iloc[0]

    features = parse_feature_list(row.get("join_path_features", ""))
    if not features:
        features = parse_feature_list(row.get("data_path", ""))
    return features


def policy_patterns(policy_names: list[str]) -> list[str]:
    patterns: list[str] = []
    for policy in policy_names:
        if policy == "none":
            continue
        if policy == "all":
            for values in EUR_CLEANED_POLICY_PATTERNS.values():
                patterns.extend(values)
            continue
        if policy not in EUR_CLEANED_POLICY_PATTERNS:
            raise ValueError(f"Unsupported policy: {policy}")
        patterns.extend(EUR_CLEANED_POLICY_PATTERNS[policy])
    return list(dict.fromkeys(patterns))


def compute_join_diagnostics(base: pd.DataFrame, joined: pd.DataFrame, target_column: str) -> dict:
    joined_only = [c for c in joined.columns if c not in base.columns and c != target_column]
    if not joined_only:
        return {
            "joined_feature_count": 0,
            "all_null_joined_features": 0,
            "mean_missing_ratio_joined": 0.0,
        }
    missing = joined[joined_only].isna().mean()
    return {
        "joined_feature_count": len(joined_only),
        "all_null_joined_features": int((missing == 1.0).sum()),
        "mean_missing_ratio_joined": float(missing.mean()),
    }


def get_regressor(name: str):
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

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
                ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
            ]
        )
    if name == "xt":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
            ]
        )
    if name == "gbr":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingRegressor(random_state=42)),
            ]
        )

    raise ValueError(f"Unsupported model: {name}")


def coerce_numeric_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    converted = dataframe.copy()
    for column in converted.columns:
        if pd.api.types.is_numeric_dtype(converted[column]):
            numeric = pd.to_numeric(converted[column], errors="coerce").astype("float64")
        else:
            cleaned = (
                converted[column]
                .astype(str)
                .str.replace(r"[^0-9eE+\-\.]", "", regex=True)
                .replace({"": np.nan, "nan": np.nan, "None": np.nan})
            )
            numeric = pd.to_numeric(cleaned, errors="coerce").astype("float64")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        # Extremely large parsed values are usually unit/parsing artifacts and make
        # linear models numerically unstable without helping predictive signal.
        numeric = numeric.mask(numeric.abs() > 1e15, np.nan)
        converted[column] = numeric
    return converted


def evaluate_variant(
    dataframe: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    model_name: str,
    split_mode: str,
    test_size: float,
    time_column: str | None,
) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    train_df, test_df, split_desc = make_tabular_split(
        dataframe=dataframe,
        split_mode=split_mode,
        test_size=test_size,
        time_column=time_column,
    )

    x_train = train_df[feature_columns]
    y_train = train_df[target_column]
    x_test = test_df[feature_columns]
    y_test = test_df[target_column]
    x_train = coerce_numeric_features(x_train)
    x_test = coerce_numeric_features(x_test)
    y_train = pd.to_numeric(y_train, errors="coerce")
    y_test = pd.to_numeric(y_test, errors="coerce")

    regressor = get_regressor(model_name)
    start = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")
        regressor.fit(x_train, y_train)
        predictions = regressor.predict(x_test)
    runtime = time.time() - start

    return {
        "model": model_name,
        "r2": float(r2_score(y_test, predictions)),
        "rmse": float(math.sqrt(mean_squared_error(y_test, predictions))),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "train_time": runtime,
        "n_features": len(feature_columns),
        "split_mode": split_mode,
        "test_groups": split_desc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate EUR base/joined/augmented data with tabular regressors.")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/EUR/6907619"))
    parser.add_argument("--base-table", default="rabbitmq-performance.csv")
    parser.add_argument(
        "--join-tables",
        nargs="+",
        default=["golang-web-server-performance.csv", "python-web-server-performance.csv", "amf-performance.csv"],
    )
    parser.add_argument("--join-key", default="time")
    parser.add_argument("--join-mode", choices=["asof", "exact"], default="asof")
    parser.add_argument("--time-tolerance-seconds", type=int, default=120)
    parser.add_argument(
        "--tolerance-grid",
        type=int,
        nargs="+",
        default=None,
        help="If set, run the benchmark once per tolerance value and concatenate the results.",
    )
    parser.add_argument("--target-column", default="lat99")
    parser.add_argument("--split-mode", choices=["time", "random"], default="time")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ridge", "knn", "rf", "xt", "gbr"],
    )
    parser.add_argument(
        "--autofeatplus-results-csv",
        type=Path,
        default=Path("results/6g_data/EUR/6907619_autofeat_plus_local.csv"),
    )
    parser.add_argument(
        "--feature-source",
        choices=["csv", "recompute"],
        default="recompute",
        help="Use AutoFeatPlus features from an existing CSV or recompute them on the current joined dataframe.",
    )
    parser.add_argument(
        "--autofeatplus-algorithm",
        default="XGBoost",
        help="Which AutoFeatPlus result row to use when extracting selected features.",
    )
    parser.add_argument(
        "--policy",
        nargs="+",
        default=["time-private", "target-proxy-private"],
        choices=["none", "all", *EUR_CLEANED_POLICY_PATTERNS.keys()],
        help="Privacy policies used when recomputing AutoFeatPlus on cleaned EUR data.",
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
        default=Path("results/6g_data/EUR/6907619_downstream_models.csv"),
    )
    args = parser.parse_args()

    rows = []
    tolerances = args.tolerance_grid or [args.time_tolerance_seconds]

    for tolerance_seconds in tolerances:
        base = read_table(args.data_dir, args.base_table)
        joined = join_tables(
            data_dir=args.data_dir,
            base_table=args.base_table,
            join_tables_to_use=args.join_tables,
            join_key=args.join_key,
            join_mode=args.join_mode,
            tolerance_seconds=tolerance_seconds,
        )

        if args.target_column not in base.columns:
            raise ValueError(f"Target column {args.target_column} not found in {args.base_table}")
        if args.target_column not in joined.columns:
            raise ValueError(f"Target column {args.target_column} not found in joined dataframe")

        time_column = args.join_key if args.join_key in joined.columns else None
        diagnostics = compute_join_diagnostics(base, joined, args.target_column)

        base_features = [c for c in base.columns if c != args.target_column]
        join_all_features = [c for c in joined.columns if c != args.target_column]
        if args.feature_source == "csv":
            autofeatplus_features = [
                c for c in load_autofeatplus_features(args.autofeatplus_results_csv, args.autofeatplus_algorithm)
                if c in joined.columns and c != args.target_column
            ]
            selection_meta = {
                "selection_source": "csv",
                "privacy_policy": ",".join(args.policy),
                "n_selected_features": len(autofeatplus_features),
                "n_blocked_features": np.nan,
                "blocked_features": "[]",
            }
        else:
            selection = select_autofeat_plus_features(
                dataframe=joined,
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
            autofeatplus_features = [c for c in selection.selected_features if c in joined.columns and c != args.target_column]
            selection_meta = {
                "selection_source": "recompute",
                "privacy_policy": ",".join(args.policy),
                "n_selected_features": len(autofeatplus_features),
                "n_blocked_features": len(selection.blocked_features),
                "blocked_features": str(selection.blocked_features),
            }

        variants = [
            ("BASE", base, base_features),
            ("Join_All_Local", joined, join_all_features),
            ("AutoFeatPlus_Local", joined, autofeatplus_features),
        ]

        for variant_name, dataframe, feature_columns in variants:
            for model_name in args.models:
                result = evaluate_variant(
                    dataframe=dataframe,
                    target_column=args.target_column,
                    feature_columns=feature_columns,
                    model_name=model_name,
                    split_mode=args.split_mode,
                    test_size=args.test_size,
                    time_column=time_column,
                )
                result.update(
                    {
                        "variant": variant_name,
                        "data_label": str(args.data_dir),
                        "target_column": args.target_column,
                        "feature_columns": str(feature_columns),
                        "autofeatplus_source": str(args.autofeatplus_results_csv),
                        **diagnostics,
                        **selection_meta,
                        "join_mode": args.join_mode,
                        "time_tolerance_seconds": tolerance_seconds,
                    }
                )
                rows.append(result)

    output_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print(f"Saved EUR downstream benchmark results to {args.output}")


if __name__ == "__main__":
    main()
