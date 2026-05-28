#!/usr/bin/env python3
"""
prepare_scenarios.py
====================
Generates all data artefacts needed for the three 6G benchmark scenarios
and self-checks each scenario for data leakage and join quality.

Design Principles
-----------------
  1. Base table = what you know at prediction time (config + limited runtime).
  2. Correlated latency percentiles (lat50/75/95/min) are NEVER in the base —
     they are measured simultaneously with lat99 and would trivially predict it.
  3. The lake must not expose the target column (lat99) to prevent leakage.
  4. Each scenario lives in its own subdirectory to avoid cross-contamination
     during Neo4j ingestion.

Scenario Designs
----------------
  Scenario 1  — Cross-service workload augmentation
    Base : rabbitmq-s1.csv     = {ram_limit, cpu_limit, ram_usage, cpu_usage, n, lat99}
    Lake : golang-by-n.csv, python-by-n.csv  (median stats per workload level n)
    Join : exact on n  (workload intensity: concurrent requests)
    Goal : does cross-service latency at the same workload level improve rabbitmq lat99 prediction?
    Note : time dropped from base — it's not a predictor, and temporal join gives only ~43% coverage
           due to large measurement gaps in golang/python data.

  Scenario 2C — Feature-recovery augmentation
    Base : rabbitmq-reduced.csv = {time, ram_limit, cpu_limit, n, lat99}
    Lake : rabbitmq-features.csv = rabbitmq minus lat99
           (exact join on time → 100% match, recovers runtime measurements)
    Goal : can we recover dropped runtime measurements from the same table?

  Scenario 3  — Cross-segment generalisation (no change)
    Base : amf seg01
    Lake : amf seg02–seg80  (join on workload key `n`)

Outputs
-------
  EUR/6907619/scenario1/
    rabbitmq-s1.csv, golang-*.csv, python-*.csv, connections.csv
  EUR/6907619/scenario2c/
    rabbitmq-reduced.csv, rabbitmq-features.csv, connections.csv
  EUR/6907619/split_output/connections.csv   (unchanged)
  data/6g_testbed_dataset/datasets.csv       (updated)
"""

from __future__ import annotations

import glob
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent.resolve()
DATA_FOLDER = ROOT / "data" / "6g_testbed_dataset"
EUR_DIR     = DATA_FOLDER / "EUR" / "6907619"
S1_DIR      = EUR_DIR / "scenario1"
S2C_DIR     = EUR_DIR / "scenario2c"
SPLIT_DIR   = EUR_DIR / "split_output"
DATASETS_CSV = DATA_FOLDER / "datasets.csv"

RABBITMQ_SRC = EUR_DIR / "rabbitmq-performance.csv"
GOLANG_SRC   = EUR_DIR / "golang-web-server-performance.csv"
PYTHON_SRC   = EUR_DIR / "python-web-server-performance.csv"

# Columns in rabbitmq that are simultaneous with lat99 → leakage if in base
LEAKAGE_COLS = {"min", "lat50", "lat75", "lat95"}


# ─── Scenario 1 ───────────────────────────────────────────────────────────────
def create_scenario1():
    """
    Build EUR/6907619/scenario1/ with:
      - rabbitmq-s1.csv    : rabbitmq minus correlated latency percentiles and time
      - golang-by-n.csv    : golang aggregated (median) per workload level n
      - python-by-n.csv    : python aggregated (median) per workload level n
      - connections.csv    : golang-by-n→rabbitmq-s1 (n), python-by-n→rabbitmq-s1 (n)

    Why n-based join instead of time-based:
      golang/python have 100+ day measurement gaps → temporal match rate ~43% even at 300s
      tolerance. Joining on n (workload intensity) gives ~48% coverage with clean semantics:
      "how do other services perform under the same number of concurrent requests?"
    """
    S1_DIR.mkdir(parents=True, exist_ok=True)

    # 1a. Rabbitmq base — drop correlated percentiles AND time (not a predictor here)
    #     Also filter zero/near-zero lat99 rows (measurement artifacts: 2.6% of data)
    rmq = pd.read_csv(RABBITMQ_SRC)
    before = len(rmq)
    rmq = rmq[rmq["lat99"] > 1000].reset_index(drop=True)
    print(f"[S1] filtered zero/near-zero lat99 rows: {before - len(rmq)} dropped ({(before-len(rmq))/before:.1%})")
    drop = [c for c in LEAKAGE_COLS | {"time"} if c in rmq.columns]
    rmq_s1 = rmq.drop(columns=drop)
    dst = S1_DIR / "rabbitmq-s1.csv"
    rmq_s1.to_csv(dst, index=False)
    print(f"[S1] rabbitmq-s1.csv  columns: {list(rmq_s1.columns)}")
    print(f"     dropped          : {drop}")
    print(f"     rows             : {len(rmq_s1)}")

    # 1b. Aggregate lake tables by n (median of all numeric metrics)
    agg_cols_drop = {"time", "c"}   # drop time and connection-count (not useful aggregate)
    for src, out_name in [(GOLANG_SRC, "golang-by-n.csv"), (PYTHON_SRC, "python-by-n.csv")]:
        lake = pd.read_csv(src)
        num_cols = [c for c in lake.select_dtypes(include=[np.number]).columns
                    if c not in agg_cols_drop and c != "n"]
        agg = lake.groupby("n")[num_cols].median().reset_index()
        agg.to_csv(S1_DIR / out_name, index=False)
        n_match = rmq_s1["n"].isin(agg["n"]).mean()
        print(f"[S1] {out_name}  rows={len(agg)}  cols={list(agg.columns)}")
        print(f"     rabbitmq n-match rate: {n_match:.1%}")

    # 1c. connections.csv — base table is FK (BFS starts from base, looks up lake)
    connections = [
        ("rabbitmq-s1.csv", "n", "golang-by-n.csv", "n"),
        ("rabbitmq-s1.csv", "n", "python-by-n.csv", "n"),
    ]
    df = pd.DataFrame(connections, columns=["fk_table", "fk_column", "pk_table", "pk_column"])
    df.to_csv(S1_DIR / "connections.csv", index=False)
    print(f"[S1] connections.csv  ({len(df)} edges)\n")
    return rmq_s1


# ─── Scenario 2C ──────────────────────────────────────────────────────────────
def create_scenario2c():
    """
    Build EUR/6907619/scenario2c/ with:
      - rabbitmq-reduced.csv  : config + time + target only (base)
      - rabbitmq-features.csv : rabbitmq minus lat99 (lake — no target leakage)
      - connections.csv       : rabbitmq-features→rabbitmq-reduced (exact time)
    """
    S2C_DIR.mkdir(parents=True, exist_ok=True)

    rmq = pd.read_csv(RABBITMQ_SRC)
    before = len(rmq)
    rmq = rmq[rmq["lat99"] > 1000].reset_index(drop=True)
    print(f"[S2C] filtered zero/near-zero lat99 rows: {before - len(rmq)} dropped ({(before-len(rmq))/before:.1%})")

    # 2a. Base: keep only config + time + target
    base_cols = ["time", "ram_limit", "cpu_limit", "n", "lat99"]
    missing = [c for c in base_cols if c not in rmq.columns]
    if missing:
        raise ValueError(f"Base columns not found in rabbitmq: {missing}")
    reduced = rmq[base_cols]
    reduced.to_csv(S2C_DIR / "rabbitmq-reduced.csv", index=False)
    print(f"[S2C] rabbitmq-reduced.csv  columns: {list(reduced.columns)}")

    # 2b. Lake: rabbitmq minus target (prevents lat99 leakage)
    features = rmq.drop(columns=["lat99"])
    features.to_csv(S2C_DIR / "rabbitmq-features.csv", index=False)
    print(f"[S2C] rabbitmq-features.csv columns: {list(features.columns)}")
    print(f"      lat99 present in lake? {'lat99' in features.columns}  ← must be False")

    # 2c. connections.csv — base table is FK (BFS starts from rabbitmq-reduced, looks up features)
    connections = [
        ("rabbitmq-reduced.csv", "time", "rabbitmq-features.csv", "time"),
    ]
    df = pd.DataFrame(connections, columns=["fk_table", "fk_column", "pk_table", "pk_column"])
    df.to_csv(S2C_DIR / "connections.csv", index=False)
    print(f"[S2C] connections.csv  ({len(df)} edges)\n")
    return reduced, features


# ─── Scenario 3 ───────────────────────────────────────────────────────────────
def create_scenario3():
    """
    Link every amf segment to seg01 via workload key `n`. Unchanged from before.
    """
    seg_files = sorted(glob.glob(str(SPLIT_DIR / "amf-performance_seg*.csv")))
    if not seg_files:
        print("[S3] No segment files found — skipping.")
        return None

    seg_names = [Path(f).name for f in seg_files]
    base_seg  = next((n for n in seg_names if "_seg01_" in n), seg_names[0])
    others    = [n for n in seg_names if n != base_seg]

    # base_seg is FK — BFS starts from seg01, looks up other segments
    connections = [(base_seg, "n", o, "n") for o in others]
    df = pd.DataFrame(connections, columns=["fk_table", "fk_column", "pk_table", "pk_column"])
    df.to_csv(SPLIT_DIR / "connections.csv", index=False)
    print(f"[S3] connections.csv  ({len(df)} edges, base={base_seg})\n")
    return base_seg


# ─── Update datasets.csv ──────────────────────────────────────────────────────
def update_datasets_csv(base_seg: str):
    rows = [
        {
            "base_table_path":  "EUR/6907619/scenario1",
            "base_table_name":  "rabbitmq-s1.csv",
            "base_table_label": "scenario1_rabbitmq",
            "target_column":    "lat99",
            "dataset_type":     "regression",
            "temporal_key":     "",
            "temporal_tolerance": 0,
        },
        {
            "base_table_path":  "EUR/6907619/scenario2c",
            "base_table_name":  "rabbitmq-reduced.csv",
            "base_table_label": "scenario2c_rabbitmq_reduced",
            "target_column":    "lat99",
            "dataset_type":     "regression",
            "temporal_key":     "time",
            "temporal_tolerance": 0,
        },
        {
            "base_table_path":  "EUR/6907619/split_output",
            "base_table_name":  base_seg,
            "base_table_label": "scenario3_amf_seg01",
            "target_column":    "lat99",
            "dataset_type":     "regression",
            "temporal_key":     "",
            "temporal_tolerance": 60,
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(DATASETS_CSV, index=False)
    print(f"[datasets.csv] updated → {DATASETS_CSV}")
    print(df.to_string(index=False))
    print()


# ─── Self-check ───────────────────────────────────────────────────────────────
def self_check(rmq_s1: pd.DataFrame, reduced: pd.DataFrame, features: pd.DataFrame):
    print("\n" + "="*60)
    print("SELF-CHECK")
    print("="*60)

    # ── Scenario 1 checks ────────────────────────────────────────
    print("\n[S1] Scenario 1 — Cross-service workload augmentation (join on n)")

    # 1. No leakage columns in base
    leakage_present = (LEAKAGE_COLS | {"time"}) & set(rmq_s1.columns)
    print(f"  Leakage/time cols in base : {leakage_present or 'none ✓'}")

    # 2. Target in base
    print(f"  Target (lat99) in base    : {'lat99' in rmq_s1.columns} {'✓' if 'lat99' in rmq_s1.columns else '✗ MISSING'}")

    # 3. Base feature correlation with target
    X = rmq_s1.drop(columns=["lat99"], errors="ignore").select_dtypes(include=[np.number])
    y = rmq_s1["lat99"]
    corrs = X.corrwith(y).abs().sort_values(ascending=False)
    print(f"  Base feature |corr| with lat99:")
    for col, val in corrs.items():
        flag = "  ← HIGH, check for leakage" if val > 0.99 else ""
        print(f"    {col:<25} {val:.3f}{flag}")

    # 4. n-based match rate
    golang_agg = pd.read_csv(S1_DIR / "golang-by-n.csv")
    python_agg = pd.read_csv(S1_DIR / "python-by-n.csv")
    go_match  = rmq_s1["n"].isin(golang_agg["n"]).mean()
    py_match  = rmq_s1["n"].isin(python_agg["n"]).mean()
    print(f"  n-based match rate rabbitmq↔golang : {go_match:.1%} {'✓' if go_match > 0.4 else '✗ LOW'}")
    print(f"  n-based match rate rabbitmq↔python : {py_match:.1%} {'✓' if py_match > 0.4 else '✗ LOW'}")
    print(f"  rabbitmq n values not in lake      : {sorted(set(rmq_s1['n'].unique()) - set(golang_agg['n'].unique()))}")

    # ── Scenario 2C checks ───────────────────────────────────────
    print("\n[S2C] Scenario 2C — Feature-recovery augmentation")

    # 1. No lat99 in lake
    print(f"  lat99 in lake (rabbitmq-features) : {'lat99' in features.columns} {'✗ LEAKAGE' if 'lat99' in features.columns else '✓ clean'}")

    # 2. Target in base
    print(f"  Target (lat99) in base            : {'lat99' in reduced.columns} {'✓' if 'lat99' in reduced.columns else '✗ MISSING'}")

    # 3. Exact time match rate (should be 100% — same rows)
    rmq_full = pd.read_csv(RABBITMQ_SRC)
    common_times = set(reduced["time"]) & set(rmq_full["time"])
    rate = len(common_times) / len(reduced)
    print(f"  Exact time match rate             : {rate:.1%} {'✓' if rate > 0.99 else '✗ UNEXPECTED'}")

    # 4. Recoverable columns (in lake, not in base)
    recoverable = set(features.columns) - set(reduced.columns)
    print(f"  Recoverable cols from lake        : {sorted(recoverable)}")

    # 5. Base-only predictability (IQR of lat99 relative to config variance)
    print(f"  Base cols                         : {list(reduced.columns)}")
    lat99_std = reduced["lat99"].std()
    print(f"  lat99 std (target spread)         : {lat99_std:,.0f}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  S1  base cols  : {[c for c in rmq_s1.columns if c != 'lat99']}")
    print(f"  S2C base cols  : {[c for c in reduced.columns if c != 'lat99']}")
    print(f"  S2C lake cols  : {list(features.columns)}")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Preparing 6G benchmark scenarios ...\n")

    rmq_s1            = create_scenario1()
    reduced, features = create_scenario2c()
    base_seg          = create_scenario3()
    base_seg          = base_seg or "amf-performance_seg01_20211110_140618.csv"

    update_datasets_csv(base_seg)
    self_check(rmq_s1, reduced, features)
