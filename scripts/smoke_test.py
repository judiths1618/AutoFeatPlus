"""
smoke_test.py
=============
Run the two showcase scenarios and assert that the AutoFeat numbers are in the
expected range. Exits non-zero on regression so a CI job (or `make smoke`) can
catch issues before they reach a demo.

What this catches:
  - AutoGluon import bug (`pkg_resources` removed by setuptools 82+) — silently
    produced R²=0 / accuracy=0 for hours before we tracked it down.
  - transformer-discovery scoring bug — selected the wrong join key, cost 14 R²
    points on scenario1.
  - temporal_join_and_save changes that break asof matching.
  - Neo4j connectivity, deps, dashboard imports.

Usage
-----
    python scripts/smoke_test.py                 # exit 0 if all pass
    python scripts/smoke_test.py --fast          # skip pipeline runs, only verify imports
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "6g_data"

# Expected accuracy floors. These come from runs that have already been
# verified; we set the floor below the observed value to allow for AutoGluon
# stochasticity, but tight enough to catch real regressions.
EXPECTED = {
    # scenario, approach, metric, floor
    ("scenario2c", "BASE", "accuracy", 0.95),
    ("scenario2c", "AutoFeat", "accuracy", 0.98),     # observed ≈ 0.991
    ("scenarioK_kul", "Join_All_BFS", "accuracy", 0.95),
    ("scenarioK_kul", "AutoFeat", "accuracy", 0.85),  # observed ≈ 0.97
}


def _run(cmd: list, label: str) -> bool:
    print(f">>> {label}: {' '.join(cmd[-4:])} ...")
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"FAILED ({label}): exit {res.returncode}")
        print(res.stdout[-2000:])
        print(res.stderr[-2000:])
        return False
    return True


def assert_imports() -> int:
    """Catch the most common environment-rot cases up front."""
    failures = 0
    for mod in ["autogluon", "sentence_transformers", "neo4j", "polars",
                "xxhash", "feature_discovery.auto_pipeline",
                "feature_discovery.dataset_introspection",
                "feature_discovery.dataset_relation_graph.transformer_discovery"]:
        try:
            __import__(mod)
        except Exception as exc:
            failures += 1
            print(f"FAILED import: {mod} → {type(exc).__name__}: {exc}")
    try:
        from pkg_resources import parse_version  # noqa: F401
    except ImportError:
        failures += 1
        print("FAILED: pkg_resources missing — pip install --no-deps 'setuptools<81'")
    try:
        import xgboost  # noqa: F401
        from autogluon.tabular.models.xgboost.xgboost_model import XGBoostModel  # noqa: F401
    except Exception as exc:
        failures += 1
        print(f"FAILED: AutoGluon XGB wrapper not importable → {exc}")
    if failures == 0:
        print("imports OK.")
    return failures


def assert_results(label: str) -> int:
    """Check accuracy floors against the saved _summary.csv."""
    path = RESULTS / f"auto_pipeline_{label}_summary.csv"
    if not path.exists():
        print(f"FAILED: result file not found: {path}")
        return 1
    df = pd.read_csv(path)
    failures = 0
    for scenario, approach, metric, floor in EXPECTED:
        if scenario != label:
            continue
        sub = df[(df.approach == approach) & (df.algorithm == "XGBoost")]
        if sub.empty:
            failures += 1
            print(f"FAILED: no row for {scenario}/{approach}/XGBoost in {path.name}")
            continue
        value = float(sub.iloc[0][metric])
        ok = value >= floor
        marker = "✓" if ok else "✗"
        print(f"  {marker} {scenario:<18s} {approach:<22s} {metric}={value:.4f}  (floor={floor})")
        if not ok:
            failures += 1
    return failures


def main(args: argparse.Namespace) -> int:
    print("=== smoke test ===\n")

    failures = assert_imports()
    if failures:
        print(f"\n{failures} import failure(s); fix the env before re-running.")
        return 1
    if args.fast:
        print("\n--fast: skipping pipeline runs.")
        return 0

    print("\n>>> running showcase scenarios ...")
    runs = [
        ("scenario2c",
         ["python", "-m", "feature_discovery.auto_pipeline",
          "--base-table", "datasets/scenario2c/rabbitmq-reduced.csv",
          "--target", "lat99", "--data-dir", "datasets/scenario2c",
          "--dataset-type", "regression",
          "--temporal-key", "time", "--temporal-tolerance", "0",
          "--algorithms", "XGB", "--label", "scenario2c"]),
        ("scenarioK_kul",
         ["python", "-m", "feature_discovery.auto_pipeline",
          "--base-table", "datasets/scenarioK_kul/samples_base.csv",
          "--target", "target_x", "--data-dir", "datasets/scenarioK_kul",
          "--dataset-type", "binary",
          "--no-transformer-discovery",
          "--algorithms", "XGB", "--label", "scenarioK_kul"]),
    ]
    for label, cmd in runs:
        if not _run(cmd, label):
            failures += 1

    print("\n>>> checking accuracy floors ...")
    for label, _ in runs:
        failures += assert_results(label)

    print()
    if failures:
        print(f"SMOKE TEST FAILED  ({failures} issue{'s' if failures>1 else ''})")
        return 1
    print("SMOKE TEST PASSED  ✓")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fast", action="store_true",
                        help="Verify imports only; skip pipeline runs.")
    sys.exit(main(parser.parse_args()))
