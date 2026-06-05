"""Result object returned by `augment_features` / `AutoFeatRunner.run()`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from feature_discovery.config import rel


@dataclass
class AugmentationResult:
    """All the artefacts produced by a single augmentation run.

    Attributes
    ----------
    augmented_df : pandas.DataFrame
        The base table joined with the features AutoFeat selected. This is the
        DataFrame you'd hand to downstream training code or write to a feature
        store. Empty if no augmentation was useful (AutoFeat refused all joined
        columns); in that case it equals the base table.
    selected_features : list[str]
        Column names AutoFeat picked. A subset of ``augmented_df.columns``
        (target excluded).
    selected_join_path : str
        The join path (table | key | key | table | …) AutoFeat picked. Empty
        when AutoFeat refused augmentation.
    summary : pandas.DataFrame
        Per-approach comparison: BASE / Join_All_BFS / Join_All_BFS_Filter /
        AutoFeat, one row per (approach, algorithm) with accuracy + features.
    feature_importance : pandas.DataFrame
        Long-format feature importance: (approach, algorithm, feature,
        source_table, importance).
    summary_path : Path | None
        Path on disk where ``summary`` was written (None if output_dir was None).
    augmented_path : Path | None
        Path on disk where ``augmented_df`` was written (None if no augmentation).
    workdir : Path | None
        Staging directory the runner used. ``None`` if cleaned up.
    diagnose : dict
        Pre-flight introspection report (inferred problem type, temporal key,
        candidate join keys, warnings).
    """
    augmented_df: pd.DataFrame
    selected_features: List[str] = field(default_factory=list)
    selected_join_path: str = ""
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary_path: Optional[Path] = None
    augmented_path: Optional[Path] = None
    workdir: Optional[Path] = None
    diagnose: Dict[str, Any] = field(default_factory=dict)

    # Convenience accessors for ETL DAGs that prefer scalars / dicts.

    @property
    def base_accuracy(self) -> Optional[float]:
        """Best BASE-approach accuracy across algorithms (None if missing)."""
        sub = self.summary[self.summary.approach == "BASE"] if not self.summary.empty else self.summary
        return float(sub.accuracy.max()) if not sub.empty else None

    @property
    def autofeat_accuracy(self) -> Optional[float]:
        sub = self.summary[self.summary.approach == "AutoFeat"] if not self.summary.empty else self.summary
        return float(sub.accuracy.max()) if not sub.empty else None

    @property
    def lift(self) -> Optional[float]:
        if self.base_accuracy is None or self.autofeat_accuracy is None:
            return None
        return self.autofeat_accuracy - self.base_accuracy

    def to_dict(self) -> Dict[str, Any]:
        """Lightweight dict suitable for XCom / job-log serialisation. Drops
        DataFrames; keeps paths, metrics, and selected features."""
        return {
            "selected_features": self.selected_features,
            "selected_join_path": self.selected_join_path,
            "summary": self.summary.to_dict(orient="records") if not self.summary.empty else [],
            "summary_path": rel(self.summary_path) if self.summary_path else None,
            "augmented_path": rel(self.augmented_path) if self.augmented_path else None,
            "base_accuracy": self.base_accuracy,
            "autofeat_accuracy": self.autofeat_accuracy,
            "lift": self.lift,
            "diagnose": self.diagnose,
        }
