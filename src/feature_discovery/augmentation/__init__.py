"""
feature_discovery.augmentation
==============================
Programmatic, pluggable API for the AutoFeat augmentation pipeline.

This is the integration surface for ELT/ETL tools. The pipeline is exposed as
both a function (`augment_features`) and a class (`AutoFeatRunner`); both
accept either DataFrames or file paths, and return an `AugmentationResult`
that hands you the augmented DataFrame, the selected features, and the
comparison metrics in one shot.

Example
-------

    >>> import pandas as pd
    >>> from feature_discovery.augmentation import augment_features
    >>> base = pd.read_csv("data/base.csv")
    >>> lake = {"transactions": pd.read_csv("data/tx.csv")}
    >>> result = augment_features(base, lake, target="churned",
    ...                           temporal_key="event_time",
    ...                           temporal_tolerance="5min")
    >>> result.augmented_df.shape
    (50000, 47)
    >>> result.summary
       approach     accuracy  n_features
    0  BASE         0.7234    8
    1  Join_All_BFS 0.9123    47
    2  AutoFeat     0.9085    22

This module wraps the same code paths driven by ``auto_pipeline.py`` but
removes the CLI / DATA_FOLDER env-var coupling so it can run inside an
Airflow task, a Dagster asset, a Prefect flow, or a vanilla cron job.
"""

from feature_discovery.augmentation.config import AugmentationConfig
from feature_discovery.augmentation.result import AugmentationResult
from feature_discovery.augmentation.runner import AutoFeatRunner, augment_features

__all__ = [
    "AugmentationConfig",
    "AugmentationResult",
    "AutoFeatRunner",
    "augment_features",
]
