"""Utilities for handling the optional Polars dependency.

The original AutoFeat pipeline uses Polars for faster joins when it is
available.  Running the experiments should still be possible when Polars
is missing, so this module exposes a shared flag that callers can use to
switch to the pandas-based fallbacks without importing Polars eagerly.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

POLARS_AVAILABLE = False
pl = None

try:  # pragma: no cover - optional dependency branch
    import polars as _pl  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - executed when Polars is absent
    POLARS_AVAILABLE = False
    pl = None
else:  # pragma: no cover - executed when Polars is installed
    POLARS_AVAILABLE = True
    pl = _pl

if TYPE_CHECKING:  # pragma: no cover - type checkers only
    import polars

    pl = polars  # noqa: F811 - rebind for type checkers
