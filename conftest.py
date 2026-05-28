"""Pytest root configuration.

Adds ``src/`` to ``sys.path`` so ``pytest`` discovers and imports the
``feature_discovery`` package without requiring ``PYTHONPATH=src`` from
the caller. Pytest auto-loads this file from the rootdir before
collecting any tests, so both local runs and CI pick it up.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
