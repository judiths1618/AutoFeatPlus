import logging
import os
import time
from typing import Optional, List

import pandas as pd
import typer as typer
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from sklearn.model_selection import train_test_split

from feature_discovery.config import AUTO_GLUON_FOLDER, SEED
from feature_discovery.experiments.dataset_object import REGRESSION
from feature_discovery.experiments.result_object import Result

AUTO_GLUON_TIME_LIMIT = int(os.getenv("AUTOFEAT_AG_TIME_LIMIT", "120"))

hyper_parameters = [
    {"RF": {}},
    {"GBM": {}},
    {"XT": {}},
    {"XGB": {}},
    {'KNN': {}},
    {'LR': {'penalty': 'L1'}},
]


# AutoGluon renames hyperparameter keys to model display names in its leaderboard
# (e.g. "XGB" → "XGBoost"). Use the same canonical name for the trivial / no-feature
# fallback results so a scenario's rows don't split across an "XGB" and an
# "XGBoost" group in the summary.
_AG_MODEL_NAME = {
    "XGB": "XGBoost",
    "GBM": "LightGBM",
    "RF": "RandomForest",
    "XT": "ExtraTrees",
}


def _canonical_model_name(key: str) -> str:
    return _AG_MODEL_NAME.get(key, key)


# With a single requested model AutoGluon would only add its auto
# WeightedEnsemble_L2, which for one base learner just duplicates that model.
# Instead we disable the auto-ensemble and train a structurally different
# companion model (RandomForest by default) so the evaluation reports two
# genuinely distinct algorithms.
COMPANION_MODEL = "RF"


def _with_companion(algorithms_to_run: dict) -> dict:
    """Add a distinct companion model so the run yields two different algorithms."""
    hp = dict(algorithms_to_run)
    companion = COMPANION_MODEL if COMPANION_MODEL not in hp else "XGB"
    hp.setdefault(companion, {})
    return hp


def get_hyperparameters(algorithm: Optional[str] = None) -> List[dict]:
    if algorithm is None:
        return hyper_parameters

    if algorithm == 'LR':
        return [{'LR': {'penalty': 'L1'}}]

    model = {algorithm: {}}
    if model in hyper_parameters:
        return [model]
    else:
        raise typer.BadParameter(
            "Unsupported algorithm. Choose one from the list: [RF, GBM, XT, XGB, KNN, LR]."
        )


def _resolve_time_column(columns, time_column: Optional[str]) -> Optional[str]:
    """Resolve a (possibly bare) temporal key to an actual dataframe column.

    Joined dataframes prefix columns with the source table (e.g.
    ``scenario2c/rabbitmq-reduced.csv.time``), so an exact match is tried first
    and then a ``.<key>`` suffix match. Returns None when no column matches.
    """
    if not time_column:
        return None
    cols = list(columns)
    if time_column in cols:
        return time_column
    matches = [c for c in cols if c.endswith(f".{time_column}")]
    return matches[0] if matches else None


def _split_train_test(
    dataframe: pd.DataFrame,
    target_column: str,
    time_column: Optional[str],
    test_size: float = 0.2,
) -> tuple:
    """Split data into train/test sets.

    Uses a chronological split when *time_column* resolves to a column in the
    dataframe (avoids temporal leakage and keeps every approach — BASE, JOIN_ALL,
    AutoFeat — on the same test window).  Falls back to a random split otherwise.

    Returns (X_train_full, X_test_full, join_path_features, split_mode).
    Both returned frames include the target column.
    """
    resolved_time_column = _resolve_time_column(dataframe.columns, time_column)
    if resolved_time_column:
        df_sorted = dataframe.sort_values(resolved_time_column).reset_index(drop=True)
        split_idx = int(len(df_sorted) * (1 - test_size))
        X_train = df_sorted.iloc[:split_idx].copy()
        X_test = df_sorted.iloc[split_idx:].copy()
        join_path_features = [c for c in X_train.columns if c != target_column]
        split_mode = "temporal"
    else:
        feat_train, feat_test, y_train, y_test = train_test_split(
            dataframe.drop(columns=[target_column]),
            dataframe[[target_column]],
            test_size=test_size,
            random_state=SEED,
        )
        join_path_features = list(feat_train.columns)
        feat_train[target_column] = y_train
        feat_test[target_column] = y_test
        X_train, X_test = feat_train, feat_test
        split_mode = "random"

    return X_train, X_test, join_path_features, split_mode


def run_auto_gluon(
    dataframe: pd.DataFrame,
    target_column: str,
    problem_type: str,
    algorithms_to_run: dict,
    time_column: Optional[str] = None,
) -> tuple:
    from autogluon.tabular import TabularPredictor

    start = time.time()
    logging.debug(f"Train algorithms: {list(algorithms_to_run.keys())} with AutoGluon ...")

    X_train, X_test, join_path_features, split_mode = _split_train_test(
        dataframe, target_column, time_column
    )

    # Guard: if no usable features remain (e.g. only the join key which gets
    # dropped as a high-cardinality identifier), return a trivial R²=0 result
    # rather than letting AutoGluon crash.
    feature_cols = [c for c in X_train.columns if c != target_column]
    if not feature_cols:
        print("No usable features — returning trivial baseline (R²=0 / accuracy=0).")
        algo_name = _canonical_model_name(list(algorithms_to_run.keys())[0])
        end = time.time()
        return end - start, [Result(
            algorithm=algo_name,
            accuracy=0.0,
            join_path_features=[],
            split_mode=split_mode,
        )]

    predictor = TabularPredictor(
        label=target_column,
        problem_type=problem_type,
        verbosity=0,
        path=AUTO_GLUON_FOLDER / "models",
    ).fit(
        train_data=X_train,
        hyperparameters=_with_companion(algorithms_to_run),
        time_limit=AUTO_GLUON_TIME_LIMIT,
        fit_weighted_ensemble=False,
        raise_on_no_models_fitted=False,
    )

    score_type = 'r2' if problem_type == REGRESSION else 'accuracy'

    model_names = predictor.leaderboard(silent=True)['model'].tolist()
    print(f"Models trained: {model_names}")

    # If AutoGluon trained nothing (e.g. all features were dropped internally),
    # still return a trivial result rather than an empty list.
    if not model_names:
        print("AutoGluon trained no models — returning trivial baseline (R²=0 / accuracy=0).")
        algo_name = _canonical_model_name(list(algorithms_to_run.keys())[0])
        end = time.time()
        return end - start, [Result(
            algorithm=algo_name,
            accuracy=0.0,
            join_path_features=join_path_features,
            split_mode=split_mode,
        )]

    results = []
    for model in model_names:
        result = predictor.evaluate(data=X_test, model=model)
        accuracy = result[score_type]
        ft_imp = predictor.feature_importance(
            data=X_test, model=model, feature_stage="original"
        )
        entry = Result(
            algorithm=model,
            accuracy=accuracy,
            feature_importance=dict(zip(list(ft_imp.index), ft_imp["importance"])),
            join_path_features=join_path_features,
            split_mode=split_mode,
        )
        results.append(entry)

    end = time.time()
    return end - start, results


def evaluate_all_algorithms(
    dataframe: pd.DataFrame,
    target_column: str,
    algorithm: str,
    problem_type: str = 'binary',
    time_column: Optional[str] = None,
) -> tuple:
    hyperparams = get_hyperparameters(algorithm)
    all_results = []
    df = AutoMLPipelineFeatureGenerator(
        enable_text_special_features=False, enable_text_ngram_features=False
    ).fit_transform(X=dataframe)

    # After the generator, check that at least one feature column survived.
    remaining_features = [c for c in df.columns if c != target_column]
    if not remaining_features:
        print(f"No features after AutoMLPipelineFeatureGenerator — skipping training (R²=0).")
        algo_name = _canonical_model_name(list(hyperparams[0].keys())[0] if hyperparams else algorithm)
        return [Result(algorithm=algo_name, accuracy=0.0)], df

    logging.debug(f"Training AutoGluon ... ")
    for model in hyperparams:
        runtime, results = run_auto_gluon(
            dataframe=df,
            target_column=target_column,
            algorithms_to_run=model,
            problem_type=problem_type,
            time_column=time_column,
        )
        for res in results:
            res.train_time = runtime
            res.total_time += res.train_time
        all_results.extend(results)

    return all_results, df
