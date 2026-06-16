import pandas as pd
import pytest

pytest.importorskip("autogluon")

from feature_discovery.experiments.evaluate_join_paths import join_from_path


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
