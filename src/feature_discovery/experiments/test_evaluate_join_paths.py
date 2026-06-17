import pandas as pd
import pytest

pytest.importorskip("autogluon")

from feature_discovery.experiments.evaluate_join_paths import evaluate_paths, join_from_path
from feature_discovery.experiments.result_object import Result


def test_join_from_path_temporal_backward_preserves_rows_and_uses_all_right_candidates(monkeypatch):
    frames = {
        "base.csv": pd.DataFrame(
            {
                "base.csv.time": [3, 1, None, 2],
                "base.csv.target": [30, 10, 99, 20],
            }
        ),
        "lake.csv": pd.DataFrame(
            {
                "lake.csv.time": [2, 4],
                "lake.csv.x": [200, 400],
            }
        ),
    }

    def fake_get_df_with_prefix(node, target_column=None):
        return frames[str(node)].copy(), str(node)

    monkeypatch.setattr(
        "feature_discovery.experiments.evaluate_join_paths.get_df_with_prefix",
        fake_get_df_with_prefix,
    )

    joined = join_from_path(
        [["base.csv", "time", "time", "lake.csv"]],
        target="base.csv.target",
        base_node="base.csv",
        temporal_key="time",
        temporal_tolerance=2,
        temporal_direction="backward",
    )

    assert joined["base.csv.time"].tolist()[:2] == [3, 1]
    assert joined["base.csv.target"].tolist() == [30, 10, 99, 20]
    assert pd.isna(joined.loc[1, "lake.csv.x"])
    assert joined.loc[0, "lake.csv.x"] == 200


def test_evaluate_paths_fallback_preserves_temporal_split(monkeypatch):
    captured = {}

    class DummyBfs:
        base_table_id = "base.csv"
        base_table_label = "scenario_temporal"
        target_column = "base.csv.target"
        temporal_key = "time"
        temporal_tolerance = 0
        temporal_direction = "nearest"
        ranking = {"base.csv": 1.0}
        partial_join_selected_features = {
            "base.csv": ["base.csv.time", "base.csv.x"],
        }

    def fake_get_df_with_prefix(node, target_column=None):
        return pd.DataFrame(
            {
                "base.csv.time": [1, 2, 3],
                "base.csv.x": [10, 20, 30],
                "base.csv.target": [100, 200, 300],
            }
        ), str(node)

    def fake_evaluate_all_algorithms(**kwargs):
        captured["time_column"] = kwargs.get("time_column")
        return [Result(algorithm="XGBoost", split_mode="temporal")], kwargs["dataframe"]

    monkeypatch.setattr(
        "feature_discovery.experiments.evaluate_join_paths.get_df_with_prefix",
        fake_get_df_with_prefix,
    )
    monkeypatch.setattr(
        "feature_discovery.experiments.evaluate_join_paths.evaluate_all_algorithms",
        fake_evaluate_all_algorithms,
    )

    results, _ = evaluate_paths(
        DummyBfs(),
        problem_type="regression",
        algorithm="XGB",
        store_augmented_data=False,
    )

    assert captured["time_column"] == "time"
    assert results[0].split_mode == "temporal"
