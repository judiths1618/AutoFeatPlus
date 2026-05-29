from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import pandas as pd

from feature_discovery.config import AUTO_GLUON_FOLDER
from feature_discovery.experiments.autofeat_plus import (
    EUR_POLICY_PRESETS,
    score_features,
    select_autofeat_plus_features,
)
from feature_discovery.experiments.local_benchmark_utils import join_tables, read_table
from feature_discovery.experiments.result_object import Result


# Kept as a module-local alias so the existing references (`EUR_POLICY_PATTERNS`)
# in this script keep working — single source of truth lives in autofeat_plus.py.
EUR_POLICY_PATTERNS = EUR_POLICY_PRESETS


MODEL_SUITES = {
    "compact": ["XGB", "XT"],
    "robust": ["XGB", "XT", "RF", "GBM"],
    "diagnostic": ["XGB", "XT", "KNN", "LR"],
    "all": ["RF", "GBM", "XT", "XGB", "KNN", "LR"],
}


def resolve_algorithms(algorithm: str, model_suite: str) -> list[str]:
    if model_suite == "single":
        return [algorithm]
    return MODEL_SUITES[model_suite]


def get_hyperparameters(algorithm: str) -> list[dict]:
    if algorithm == "LR":
        return [{"LR": {"penalty": "L1"}}]
    supported = {"RF", "GBM", "XT", "XGB", "KNN"}
    if algorithm in supported:
        return [{algorithm: {}}]
    raise ValueError("Unsupported algorithm. Choose one from: RF, GBM, XT, XGB, KNN, LR.")


def policy_patterns(policy_names: list[str]) -> list[str]:
    patterns: list[str] = []
    for policy in policy_names:
        if policy == "none":
            continue
        if policy == "all":
            for values in EUR_POLICY_PATTERNS.values():
                patterns.extend(values)
            continue
        if policy not in EUR_POLICY_PATTERNS:
            raise ValueError(f"Unsupported policy: {policy}")
        patterns.extend(EUR_POLICY_PATTERNS[policy])
    return list(dict.fromkeys(patterns))


def policy_slug(policy_names: list[str]) -> str:
    return "__".join(policy_names).replace("-", "_")


def make_split(dataframe: pd.DataFrame, target_column: str, test_size: float) -> tuple[pd.Index, pd.Index, str]:
    from sklearn.model_selection import train_test_split

    train_index, test_index = train_test_split(
        dataframe.index,
        test_size=test_size,
        random_state=10,
        stratify=None,
    )
    return pd.Index(train_index), pd.Index(test_index), f"random_test_size={test_size}"


def evaluate_with_fixed_split(
    dataframe: pd.DataFrame,
    target_column: str,
    algorithm: str,
    train_index: pd.Index,
    test_index: pd.Index,
) -> list[Result]:
    from autogluon.features.generators import AutoMLPipelineFeatureGenerator
    from autogluon.tabular import TabularPredictor
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    feature_columns = [column for column in dataframe.columns if column != target_column]
    if not feature_columns:
        return [Result(algorithm=algorithm, accuracy=0.0, join_path_features=[])]

    start = time.time()
    generator = AutoMLPipelineFeatureGenerator(
        enable_text_special_features=False, enable_text_ngram_features=False
    )
    train_data = generator.fit_transform(X=dataframe.loc[train_index].copy())
    test_data = generator.transform(X=dataframe.loc[test_index].copy())
    join_path_features = [column for column in train_data.columns if column != target_column]

    results = []
    for hyperparameters in get_hyperparameters(algorithm):
        predictor = TabularPredictor(
            label=target_column,
            problem_type="regression",
            verbosity=0,
            path=AUTO_GLUON_FOLDER / "models",
        ).fit(train_data=train_data, hyperparameters=hyperparameters)

        for model in predictor.leaderboard(silent=True)["model"].tolist()[:-1]:
            score = predictor.evaluate(data=test_data, model=model)["r2"]
            predictions = predictor.predict(data=test_data, model=model)
            rmse = math.sqrt(mean_squared_error(test_data[target_column], predictions))
            mae = mean_absolute_error(test_data[target_column], predictions)
            feature_importance = predictor.feature_importance(
                data=test_data, model=model, feature_stage="original"
            )
            runtime = time.time() - start
            results.append(
                Result(
                    algorithm=model,
                    accuracy=score,
                    rmse=float(rmse),
                    mae=float(mae),
                    train_time=runtime,
                    feature_importance=dict(zip(list(feature_importance.index), feature_importance["importance"])),
                    join_path_features=join_path_features,
                )
            )
    return results


def evaluate_frame(
    dataframe: pd.DataFrame,
    target_column: str,
    algorithm: str,
    approach: str,
    data_path: str,
    data_label: str,
    train_index: pd.Index,
    test_index: pd.Index,
    policy: str,
    privacy_risk_score: float = 0.0,
    sensitive_features: list[str] | None = None,
    blocked_features: list[str] | None = None,
    feature_selection_time: float = 0.0,
    sensitive_patterns: list[str] | None = None,
    proxy_correlation_threshold: float = 0.98,
) -> list[Result]:
    results = evaluate_with_fixed_split(
        dataframe=dataframe,
        target_column=target_column,
        algorithm=algorithm,
        train_index=train_index,
        test_index=test_index,
    )
    if sensitive_patterns is not None and sensitive_features is None and blocked_features is None:
        feature_scores = score_features(
            dataframe=dataframe.loc[train_index],
            target_column=target_column,
            sensitive_patterns=sensitive_patterns,
            proxy_correlation_threshold=proxy_correlation_threshold,
        )
        sensitive_features = feature_scores.loc[feature_scores["privacy_risk"] > 0, "feature"].tolist()
        blocked_features = []
        privacy_risk_score = float(feature_scores["privacy_risk"].sum())

    for result in results:
        result.approach = approach
        result.data_path = data_path
        result.data_label = data_label
        result.n_features = len(result.join_path_features)
        result.privacy_risk_score = privacy_risk_score
        result.sensitive_features = sensitive_features or []
        result.blocked_features = blocked_features or []
        result.n_sensitive_features = len(result.sensitive_features)
        result.feature_selection_time = feature_selection_time
        result.split_mode = "random"
        result.test_groups = policy
        result.join_name = policy
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local, no-Neo4j EUR AutoFeatPlus benchmark.")
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
    parser.add_argument("--algorithm", default="XGB")
    parser.add_argument(
        "--model-suite",
        choices=["single", *MODEL_SUITES.keys()],
        default="single",
        help="Downstream model suite. Use 'single' to keep --algorithm behavior.",
    )
    parser.add_argument("--target-column", default="lat99")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--policy",
        nargs="+",
        default=["time-private", "target-proxy-private"],
        choices=["none", "all", *EUR_POLICY_PATTERNS.keys()],
        help="Privacy policies to apply to AutoFeatPlus feature selection.",
    )
    parser.add_argument("--autofeat-plus-top-k", type=int, default=20)
    parser.add_argument("--privacy-penalty", type=float, default=0.25)
    parser.add_argument("--missing-penalty", type=float, default=0.10)
    parser.add_argument("--cost-penalty", type=float, default=0.001)
    parser.add_argument("--proxy-correlation-threshold", type=float, default=0.98)
    parser.add_argument("--max-missing-ratio", type=float, default=0.95)
    parser.add_argument(
        "--privacy-mode",
        choices=["hard", "soft"],
        default="hard",
        help="hard blocks risky features; soft keeps them eligible but penalizes their score.",
    )
    parser.add_argument(
        "--preset",
        choices=["single", "eur-privacy-grid"],
        default="single",
        help="Run one policy or a small policy grid for systematic comparison.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/6g_data/EUR/6907619_autofeat_plus_local.csv"),
    )
    parser.add_argument(
        "--feature-scores-output",
        type=Path,
        default=Path("results/6g_data/EUR/6907619_autofeat_plus_scores.csv"),
    )
    args = parser.parse_args()

    base = read_table(args.data_dir, args.base_table)
    joined = join_tables(
        data_dir=args.data_dir,
        base_table=args.base_table,
        join_tables_to_use=args.join_tables,
        join_key=args.join_key,
        join_mode=args.join_mode,
        tolerance_seconds=args.time_tolerance_seconds,
    )

    if args.target_column not in base.columns:
        raise ValueError(f"Target column {args.target_column} not found in {args.base_table}")
    if args.target_column not in joined.columns:
        raise ValueError(f"Target column {args.target_column} not found after joining")

    train_index, test_index, split_description = make_split(base, args.target_column, args.test_size)
    policy_runs = [args.policy]
    if args.preset == "eur-privacy-grid":
        policy_runs = [
            ["none"],
            ["time-private"],
            ["target-proxy-private"],
            ["time-private", "target-proxy-private"],
            ["resource-private"],
            ["workload-private"],
            ["all"],
        ]

    all_results = []
    feature_scores_frames = []
    data_label = f"EUR/{args.data_dir.name}"
    for policy_names in policy_runs:
        patterns = policy_patterns(policy_names)
        policy_name = ",".join(policy_names)
        policy_label = (
            f"{policy_name};privacy_mode={args.privacy_mode};"
            f"{split_description};join_mode={args.join_mode};max_missing={args.max_missing_ratio}"
        )
        selection_start = time.time()
        autofeat_plus_selection = select_autofeat_plus_features(
            dataframe=joined.loc[train_index],
            target_column=args.target_column,
            top_k=args.autofeat_plus_top_k,
            sensitive_patterns=patterns,
            privacy_penalty=args.privacy_penalty,
            missing_penalty=args.missing_penalty,
            cost_penalty=args.cost_penalty,
            block_sensitive=args.privacy_mode == "hard",
            proxy_correlation_threshold=args.proxy_correlation_threshold,
            max_missing_ratio=args.max_missing_ratio,
        )
        feature_selection_time = time.time() - selection_start
        autofeat_plus_dataframe = joined[autofeat_plus_selection.selected_features + [args.target_column]]

        feature_scores = autofeat_plus_selection.feature_scores.copy()
        feature_scores.insert(0, "policy", policy_name)
        feature_scores.insert(1, "privacy_mode", args.privacy_mode)
        feature_scores.insert(2, "target_column", args.target_column)
        feature_scores_frames.append(feature_scores)

        for algorithm in resolve_algorithms(args.algorithm, args.model_suite):
            all_results.extend(
                evaluate_frame(
                    dataframe=base,
                    target_column=args.target_column,
                    algorithm=algorithm,
                    approach=Result.BASE,
                    data_path=args.base_table,
                    data_label=data_label,
                    train_index=train_index,
                    test_index=test_index,
                    policy=policy_label,
                    sensitive_patterns=patterns,
                    proxy_correlation_threshold=args.proxy_correlation_threshold,
                )
            )
            all_results.extend(
                evaluate_frame(
                    dataframe=joined,
                    target_column=args.target_column,
                    algorithm=algorithm,
                    approach=Result.JOIN_ALL_BFS,
                    data_path=",".join(args.join_tables),
                    data_label=data_label,
                    train_index=train_index,
                    test_index=test_index,
                    policy=policy_label,
                    sensitive_patterns=patterns,
                    proxy_correlation_threshold=args.proxy_correlation_threshold,
                )
            )
            all_results.extend(
                evaluate_frame(
                    dataframe=autofeat_plus_dataframe,
                    target_column=args.target_column,
                    algorithm=algorithm,
                    approach=Result.AUTOFEAT_PLUS_LOCAL,
                    data_path=",".join(autofeat_plus_selection.selected_features),
                    data_label=data_label,
                    train_index=train_index,
                    test_index=test_index,
                    policy=policy_label,
                    privacy_risk_score=autofeat_plus_selection.privacy_risk_score,
                    sensitive_features=autofeat_plus_selection.sensitive_features,
                    blocked_features=autofeat_plus_selection.blocked_features,
                    feature_selection_time=feature_selection_time,
                )
            )

    result_frame = pd.DataFrame([vars(result) for result in all_results])
    feature_scores_frame = pd.concat(feature_scores_frames, ignore_index=True) if feature_scores_frames else pd.DataFrame()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.feature_scores_output.parent.mkdir(parents=True, exist_ok=True)
    feature_scores_frame.to_csv(args.feature_scores_output, index=False)
    result_frame.to_csv(args.output, index=False)
    print(f"Saved {len(result_frame)} results to {args.output}")
    print(f"Saved AutoFeatPlus feature scores to {args.feature_scores_output}")
    print(f"Policies: {[','.join(policy) for policy in policy_runs]}")


if __name__ == "__main__":
    main()
