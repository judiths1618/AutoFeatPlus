from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("neo4j")

from feature_discovery.autofeat_pipeline.join_data import temporal_join_and_save


def test_temporal_join_preserves_base_order_and_supports_backward(tmp_path: Path):
    left = pd.DataFrame(
        {
            "base.time": [3, 1, None, 2],
            "target": [30, 10, 99, 20],
        }
    )
    right = pd.DataFrame(
        {
            "lake.time": [2, 4],
            "lake.x": [200, 400],
        }
    )

    joined = temporal_join_and_save(
        left,
        right,
        "base.time",
        "lake.time",
        tmp_path / "join.csv",
        tolerance_s=2,
        save_to_disk=False,
        direction="backward",
    )

    assert joined["base.time"].tolist()[:2] == [3, 1]
    assert joined["target"].tolist() == [30, 10, 99, 20]
    assert joined.loc[0, "lake.x"] == 200
    assert pd.isna(joined.loc[1, "lake.x"])
    assert pd.isna(joined.loc[2, "lake.x"])
    assert joined.loc[3, "lake.x"] == 200


def test_temporal_join_nearest_can_use_future_rows_when_requested(tmp_path: Path):
    left = pd.DataFrame({"base.time": [1], "target": [10]})
    right = pd.DataFrame({"lake.time": [2], "lake.x": [200]})

    joined = temporal_join_and_save(
        left,
        right,
        "base.time",
        "lake.time",
        tmp_path / "join.csv",
        tolerance_s=2,
        save_to_disk=False,
        direction="nearest",
    )

    assert joined.loc[0, "lake.x"] == 200
