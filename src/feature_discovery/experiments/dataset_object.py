from pathlib import Path
from typing import List, Optional

import pandas as pd


CLASSIFICATION = "binary"
MULTICLASS = "multiclass"
REGRESSION = "regression"

_VALID_DATASET_TYPES = {CLASSIFICATION, MULTICLASS, REGRESSION}


class Dataset:
    def __init__(self, base_table_path: Path, base_table_name: str, base_table_label: str, target_column: str,
                 dataset_type: str, base_table_features: Optional[List] = None,
                 temporal_key: Optional[str] = None, temporal_tolerance: int = 60,
                 temporal_direction: str = "nearest"):
        self.base_table_path = base_table_path
        self.target_column = target_column
        self.base_table_name = base_table_name
        self.base_table_id = base_table_path / base_table_name
        self.base_table_label = base_table_label
        self.base_table_features = base_table_features
        self.base_table_df = None
        # Temporal join settings (None = use exact join)
        self.temporal_key: Optional[str] = temporal_key
        self.temporal_tolerance: int = temporal_tolerance
        if temporal_direction not in ("nearest", "backward", "forward"):
            raise ValueError(
                f"temporal_direction must be nearest|backward|forward, got {temporal_direction!r}"
            )
        self.temporal_direction: str = temporal_direction

        if dataset_type == REGRESSION:
            self.dataset_type = REGRESSION
        elif dataset_type == MULTICLASS:
            self.dataset_type = MULTICLASS
        else:
            # Legacy fall-through: anything non-regression/non-multiclass is binary.
            self.dataset_type = CLASSIFICATION

    def set_features(self):
        if self.base_table_df is not None:
            self.base_table_features = list(self.base_table_df.drop(columns=[self.target_column]).columns)
        else:
            self.base_table_features = list(
                pd.read_csv(self.base_table_id, header=0, engine="python", encoding="utf8", quotechar='"',
                            escapechar='\\', nrows=1).drop(columns=[self.target_column]).columns)

    def set_base_table_df(self):
        self.base_table_df = pd.read_csv(self.base_table_id, header=0, engine="python", encoding="utf8", quotechar='"',
                                         escapechar='\\')
