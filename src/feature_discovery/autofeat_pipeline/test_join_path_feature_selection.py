import pandas as pd
import pytest

pytest.importorskip("ITMO_FS")

from feature_discovery.autofeat_pipeline.join_path_feature_selection import RelevanceRedundancy


def test_relevance_ties_are_ordered_deterministically():
    dataframe = pd.DataFrame(
        {
            "b": [1, 2, 3, 4],
            "a": [1, 2, 3, 4],
            "target": [1, 2, 4, 3],
        }
    )
    scorer = RelevanceRedundancy(target_column="target")

    first = scorer.measure_relevance(
        dataframe=dataframe[["b", "a"]],
        new_features=["a", "b"],
        target_column=dataframe["target"],
    )
    second = scorer.measure_relevance(
        dataframe=dataframe[["b", "a"]],
        new_features=["b", "a"],
        target_column=dataframe["target"],
    )

    assert first == second == [("a", pytest.approx(0.8)), ("b", pytest.approx(0.8))]
