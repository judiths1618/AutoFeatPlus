"""Configuration object for a single augmentation run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

# Either a path (str/Path) or an already-loaded DataFrame.
TableSource = Union[str, Path, pd.DataFrame]


@dataclass
class AugmentationConfig:
    """Everything the runner needs to know.

    Designed to be safe to serialise (everything is JSON-friendly except for
    in-memory DataFrames). When passing DataFrames, they are staged to a
    tempdir before invoking the pipeline so the existing CSV-reading code
    paths work unchanged.
    """

    # Required
    base: TableSource                  # base table (path or DataFrame)
    target: str                         # target column name

    # Lake tables. Either a list of paths/DataFrames, or a dict mapping
    # "logical name" → source. Logical names show up in result columns.
    lake: Union[List[TableSource], Dict[str, TableSource]] = field(default_factory=list)

    # Inferred at runtime if None.
    dataset_type: Optional[str] = None              # regression | binary | multiclass | None
    temporal_key: Optional[str] = None
    temporal_tolerance: Union[int, str] = 60        # "5min", "200ms", 60, ...
    temporal_direction: str = "nearest"             # nearest | backward | forward

    # Pipeline knobs.
    algorithms: List[str] = field(default_factory=lambda: ["XGB"])
    top_k: int = 15
    value_ratio: float = 0.65
    schema_threshold: float = 0.6
    value_threshold: float = 0.2
    model: str = "sentence-transformers/all-mpnet-base-v2"

    # Behaviour flags.
    use_transformer_discovery: bool = True
    skip_diagnose: bool = False
    label: Optional[str] = None                     # used for output filenames
    output_dir: Optional[Path] = None               # default: results/6g_data/
    keep_workdir: bool = False                      # leave staged files for inspection

    # Connections (optional, explicit PK/FK edges). Either a path to a
    # connections.csv or a list of dicts.
    connections: Optional[Union[Path, List[Dict[str, str]]]] = None

    def __post_init__(self) -> None:
        if self.temporal_direction not in ("nearest", "backward", "forward"):
            raise ValueError(
                f"temporal_direction must be nearest|backward|forward, "
                f"got {self.temporal_direction!r}"
            )
        if self.dataset_type and self.dataset_type not in ("regression", "binary", "multiclass"):
            raise ValueError(
                f"dataset_type must be regression|binary|multiclass, "
                f"got {self.dataset_type!r}"
            )
