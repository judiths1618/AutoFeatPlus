from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import pandas as pd

from feature_discovery.experiments.autofeat_plus import select_autofeat_plus_features
from feature_discovery.config import AUTO_GLUON_FOLDER
from feature_discovery.experiments.local_benchmark_utils import join_antenna_tables, make_kul_split
from feature_discovery.experiments.result_object import Result


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


def drop_metadata(dataframe: pd.DataFrame, target_column: str) -> pd.DataFrame:
    metadata_columns = [
        column
        for column in ["sample_key", "user_id", "sample_id", "target_y", "target_z", "antenna_id"]
        if column in dataframe.columns and column != target_column
    ]
    return dataframe.drop(columns=metadata_columns)


def evaluate_frame(
    dataframe: pd.DataFrame,
    target_column: str,
    algorithm: str,
    approach: str,
    data_path: str,
    data_label: str,
    privacy_risk_score: float = 0.0,
    sensitive_features: list[str] | None = None,
    blocked_features: list[str] | None = None,
    train_index: pd.Index | None = None,
    test_index: pd.Index | None = None,
    split_mode: str = "random",
    test_groups: str = "",
) -> list[Result]:
    if len([column for column in dataframe.columns if column != target_column]) == 0:
        return [
            Result(
                algorithm=algorithm,
                approach=approach,
                data_path=data_path,
                data_label=data_label,
                accuracy=0.0,
                join_path_features=[],
                n_features=0,
                privacy_risk_score=privacy_risk_score,
                sensitive_features=sensitive_features or [],
                blocked_features=blocked_features or [],
                n_sensitive_features=len(sensitive_features or []),
                split_mode=split_mode,
                test_groups=test_groups,
            )
        ]

    if train_index is None or test_index is None:
        raise ValueError("benchmark_kul_local.py requires an explicit fixed split.")

    results = evaluate_with_fixed_split(
        dataframe=dataframe,
        target_column=target_column,
        algorithm=algorithm,
        train_index=train_index,
        test_index=test_index,
    )

    for result in results:
        result.approach = approach
        result.data_path = data_path
        result.data_label = data_label
        result.n_features = len(result.join_path_features)
        result.privacy_risk_score = privacy_risk_score
        result.sensitive_features = sensitive_features or []
        result.blocked_features = blocked_features or []
        result.n_sensitive_features = len(result.sensitive_features)
        result.split_mode = split_mode
        result.test_groups = test_groups
    return results


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local, no-Neo4j KUL benchmark.")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/KUL/autofeat_nomadic_ula_static"))
    parser.add_argument("--algorithm", default="XGB")
    parser.add_argument(
        "--model-suite",
        choices=["single", *MODEL_SUITES.keys()],
        default="single",
        help="Downstream model suite. Use 'single' to keep --algorithm behavior.",
    )
    parser.add_argument("--target-column", default="target_x")
    parser.add_argument("--output", type=Path, default=Path("results/6g_data/kul_local_benchmark.csv"))
    parser.add_argument(
        "--feature-scores-output",
        type=Path,
        default=Path("results/6g_data/kul_autofeat_plus_feature_scores.csv"),
    )
    parser.add_argument("--antennas", type=int, nargs="+", default=[0, 16, 32, 48])
    parser.add_argument("--autofeat-plus-top-k", type=int, default=50)
    parser.add_argument("--privacy-penalty", type=float, default=0.25)
    parser.add_argument("--missing-penalty", type=float, default=0.10)
    parser.add_argument("--cost-penalty", type=float, default=0.001)
    parser.add_argument(
        "--split-mode",
        choices=["random", "user-holdout", "position-holdout"],
        default="random",
        help="Evaluation split. Holdout modes are stricter than the default random split.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--holdout-user", type=int, default=None)
    parser.add_argument(
        "--holdout-position",
        default=None,
        help="Position holdout tuple formatted as target_x,target_y,target_z. Defaults to the last sorted position.",
    )
    parser.add_argument(
        "--keep-metadata",
        action="store_true",
        help="Keep user/sample/other target coordinate metadata that may make the task trivial.",
    )
    args = parser.parse_args()

    base = pd.read_csv(args.data_dir / "samples.csv")
    joined = join_antenna_tables(args.data_dir, base, args.antennas)
    train_index, test_index, test_groups = make_kul_split(
        metadata=base,
        split_mode=args.split_mode,
        test_size=args.test_size,
        holdout_user=args.holdout_user,
        holdout_position=args.holdout_position,
    )

    if not args.keep_metadata:
        base = drop_metadata(base, args.target_column)
        joined = drop_metadata(joined, args.target_column)

    autofeat_plus_selection = select_autofeat_plus_features(
        dataframe=joined.loc[train_index],
        target_column=args.target_column,
        top_k=args.autofeat_plus_top_k,
        privacy_penalty=args.privacy_penalty,
        missing_penalty=args.missing_penalty,
        cost_penalty=args.cost_penalty,
        block_sensitive=True,
    )
    autofeat_plus_dataframe = joined[autofeat_plus_selection.selected_features + [args.target_column]]

    all_results = []
    data_label = "kul_nomadic_ula_static_local"
    for algorithm in resolve_algorithms(args.algorithm, args.model_suite):
        all_results.extend(
            evaluate_frame(
                dataframe=base,
                target_column=args.target_column,
                algorithm=algorithm,
                approach=Result.BASE,
                data_path="samples.csv",
                data_label=data_label,
                train_index=train_index,
                test_index=test_index,
                split_mode=args.split_mode,
                test_groups=test_groups,
            )
        )
        all_results.extend(
            evaluate_frame(
                dataframe=joined,
                target_column=args.target_column,
                algorithm=algorithm,
                approach=Result.JOIN_ALL_BFS,
                data_path=",".join([f"antenna_{antenna}_features.csv" for antenna in args.antennas]),
                data_label=data_label,
                train_index=train_index,
                test_index=test_index,
                split_mode=args.split_mode,
                test_groups=test_groups,
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
                privacy_risk_score=autofeat_plus_selection.privacy_risk_score,
                sensitive_features=autofeat_plus_selection.sensitive_features,
                blocked_features=autofeat_plus_selection.blocked_features,
                train_index=train_index,
                test_index=test_index,
                split_mode=args.split_mode,
                test_groups=test_groups,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.feature_scores_output.parent.mkdir(parents=True, exist_ok=True)
    autofeat_plus_selection.feature_scores.to_csv(args.feature_scores_output, index=False)
    pd.DataFrame([vars(result) for result in all_results]).to_csv(args.output, index=False)
    print(f"Saved {len(all_results)} results to {args.output}")
    print(f"Saved AutoFeatPlus feature scores to {args.feature_scores_output}")


if __name__ == "__main__":
    main()
