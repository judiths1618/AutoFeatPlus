"""
prepare_augmentation_scenarios.py
=================================
One-stop builder for all benchmark scenarios consumed by the
`feature_discovery.auto_pipeline` CLI.

Each scenario writes to its own subdirectory under datasets/ with:
  - the base table (CSV)
  - lake tables (CSVs)
  - metadata.txt (column descriptions for transformer discovery)
  - connections.csv (explicit PK/FK edges where applicable)

Usage
-----
    python scripts/prepare_augmentation_scenarios.py --all
    python scripts/prepare_augmentation_scenarios.py --scenario 2c
    python scripts/prepare_augmentation_scenarios.py --scenario 1 2c k

Scenarios
---------
  1     — Cross-service workload augmentation (rabbitmq + golang/python aggregates, join on n)
  2c    — Feature-recovery (rabbitmq-reduced + rabbitmq-features, join on time)
  a_x   — Cross-app temporal (rabbitmq-reduced + golang/python/amf full, target lat95)
  a_99  — Same as a_x but target lat99
  b     — Within-app via segments (amf seg01 + other amf segments, join on n)
  k     — KUL CSI subset (samples_base + 4 antenna feature tables)
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EUR = ROOT / "datasets" / "EUR" / "6907619"
EUR_META = ROOT / "datasets" / "EUR" / "metadata.txt"
KUL_RAW = ROOT / "datasets" / "KUL" / "nomadic_dataset_ULA_static" / "csi_as_features"
KUL_AUTOFEAT = ROOT / "datasets" / "KUL" / "autofeat_nomadic_ula_static"
SEG_DIR = ROOT / "datasets" / "EUR" / "6907619" / "split_output"

OUT = ROOT / "datasets"

# ─── Common ──────────────────────────────────────────────────────────────────
_LEAKAGE_COLS = {"min", "lat50", "lat75", "lat95"}


def _filter_zero_lat99(df: pd.DataFrame) -> pd.DataFrame:
    """Drop near-zero lat99 measurement artifacts (~6% of rabbitmq rows)."""
    before = len(df)
    out = df[df["lat99"] > 1000].reset_index(drop=True)
    print(f"  filtered near-zero lat99: {before - len(out)} dropped ({(before-len(out))/before:.1%})")
    return out


def _write_metadata(dst: Path, extra: str) -> None:
    """Copy the canonical EUR metadata.txt and append a scenario-specific note."""
    base = EUR_META.read_text() if EUR_META.exists() else ""
    dst.write_text(base + "\n\n" + extra.strip() + "\n")


# ─── Scenario 1 — cross-service via n ────────────────────────────────────────
def build_scenario1() -> None:
    print("[scenario1] cross-service workload augmentation (join on n)")
    dst = OUT / "scenario1"
    dst.mkdir(parents=True, exist_ok=True)

    rmq = _filter_zero_lat99(pd.read_csv(EUR / "rabbitmq-performance.csv"))
    drop = [c for c in _LEAKAGE_COLS | {"time"} if c in rmq.columns]
    rmq_s1 = rmq.drop(columns=drop)
    rmq_s1.to_csv(dst / "rabbitmq-s1.csv", index=False)
    print(f"  rabbitmq-s1: {rmq_s1.shape} cols={list(rmq_s1.columns)}")

    for src_name, out_name in [
        ("golang-web-server-performance.csv", "golang-by-n.csv"),
        ("python-web-server-performance.csv", "python-by-n.csv"),
    ]:
        lake = pd.read_csv(EUR / src_name)
        num = [c for c in lake.select_dtypes(include=[np.number]).columns
               if c not in {"time", "c", "n"}]
        agg = lake.groupby("n")[num].median().reset_index()
        agg.to_csv(dst / out_name, index=False)
        print(f"  {out_name}: {agg.shape}")

    conn = pd.DataFrame([
        {"fk_table": "rabbitmq-s1.csv", "fk_column": "n", "pk_table": "golang-by-n.csv", "pk_column": "n"},
        {"fk_table": "rabbitmq-s1.csv", "fk_column": "n", "pk_table": "python-by-n.csv", "pk_column": "n"},
    ])
    conn.to_csv(dst / "connections.csv", index=False)
    _write_metadata(dst / "metadata.txt", """
Scenario 1 — cross-service workload augmentation. rabbitmq-s1 keeps
configuration + resource usage + lat99 (sibling latencies dropped). golang-by-n
and python-by-n are per-workload medians enabling value-based joins on n.
""")


# ─── Scenario 2C — feature recovery via time ─────────────────────────────────
def build_scenario2c() -> None:
    print("[scenario2c] feature-recovery (self-join via time)")
    dst = OUT / "scenario2c"
    dst.mkdir(parents=True, exist_ok=True)

    rmq = _filter_zero_lat99(pd.read_csv(EUR / "rabbitmq-performance.csv"))
    base_cols = ["time", "ram_limit", "cpu_limit", "n", "lat99"]
    reduced = rmq[base_cols]
    reduced.to_csv(dst / "rabbitmq-reduced.csv", index=False)
    print(f"  base: {reduced.shape} cols={base_cols}")

    features = rmq.drop(columns=["lat99"])
    features.to_csv(dst / "rabbitmq-features.csv", index=False)
    print(f"  lake: {features.shape} (lat99 stripped → no leakage)")

    conn = pd.DataFrame([{
        "fk_table": "rabbitmq-reduced.csv", "fk_column": "time",
        "pk_table": "rabbitmq-features.csv", "pk_column": "time",
    }])
    conn.to_csv(dst / "connections.csv", index=False)
    _write_metadata(dst / "metadata.txt", """
Scenario 2C — feature recovery. Base has configs + time + lat99 only. Lake is
the same rabbitmq table minus the target. Exact time join recovers all the
dropped runtime/latency columns.
""")


# ─── Scenarios A_lat95 / A_lat99 — cross-app temporal ────────────────────────
def build_scenarioA(target: str) -> None:
    assert target in ("lat95", "lat99")
    label = f"scenarioA_{target}"
    print(f"[{label}] cross-app temporal (target={target})")
    dst = OUT / label
    dst.mkdir(parents=True, exist_ok=True)

    rmq = _filter_zero_lat99(pd.read_csv(EUR / "rabbitmq-performance.csv"))
    keep = ["time", "ram_limit", "cpu_limit", "ram_usage", "cpu_usage", "n", target]
    rmq[keep].to_csv(dst / "rabbitmq-reduced.csv", index=False)
    print(f"  base rabbitmq-reduced: cols={keep} rows={len(rmq)}")

    for f in ["golang-web-server-performance.csv",
              "python-web-server-performance.csv",
              "amf-performance.csv"]:
        shutil.copy(EUR / f, dst / f)

    _write_metadata(dst / "metadata.txt", f"""
Scenario A ({target}) — cross-application temporal augmentation. rabbitmq base
strips the sibling latency percentiles so the model has to learn from
configs+resources alone. Lake adds golang/python/amf at matched timestamps
(asof join, Δ=60s).
""")


# ─── Scenario B — within-app amf segments ────────────────────────────────────
def build_scenarioB(n_segments: int = 8) -> None:
    print(f"[scenarioB] within-app via amf segments (n_segments={n_segments})")
    dst = OUT / "scenarioB_seg01"
    dst.mkdir(parents=True, exist_ok=True)

    seg_files = sorted(SEG_DIR.glob("amf-performance_seg0*.csv"))[:n_segments]
    if not seg_files:
        print(f"  ⚠ no amf segment files at {SEG_DIR} — skipping")
        return

    seg01 = next(f for f in seg_files if "seg01" in f.name)
    s = pd.read_csv(seg01)
    keep = ["time", "ram_limit", "cpu_limit", "ram_usage", "cpu_usage", "n", "lat99"]
    s[keep].to_csv(dst / "amf-seg01-reduced.csv", index=False)
    print(f"  base amf-seg01-reduced: {s[keep].shape}")

    others = [f for f in seg_files if f != seg01]
    for f in others:
        shutil.copy(f, dst / f.name)
    print(f"  lake: {len(others)} amf segments")

    _write_metadata(dst / "metadata.txt", """
Scenario B — within-application augmentation across amf time-segments. Base is
seg01 with sibling latencies stripped; lake contains other amf segments with
full schema. Join key is n (workload identifier shared across segments).
""")


# ─── Scenario K — KUL CSI subset ─────────────────────────────────────────────
def build_scenarioK(use_csi_layout: bool = False, antennas: List[int] = None) -> None:
    """Either copy the canonical 4-antenna autofeat layout, or rebuild from raw
    csi_as_features per-antenna at scale (use_csi_layout=True, slower).
    """
    print(f"[scenarioK] KUL CSI {'csi_as_features' if use_csi_layout else 'autofeat layout'}")
    dst = OUT / ("scenarioK_csi" if use_csi_layout else "scenarioK_kul")
    dst.mkdir(parents=True, exist_ok=True)

    if not use_csi_layout:
        if not KUL_AUTOFEAT.exists():
            print(f"  ⚠ no autofeat_nomadic_ula_static at {KUL_AUTOFEAT} — skipping")
            return
        for f in ["samples_base.csv", "antenna_0_features.csv",
                  "antenna_16_features.csv", "antenna_32_features.csv",
                  "antenna_48_features.csv", "connections.csv"]:
            src = KUL_AUTOFEAT / f
            if src.exists():
                shutil.copy(src, dst / f)
        # Use the canonical KUL metadata.txt
        kul_meta = ROOT / "datasets" / "KUL" / "metadata.txt"
        if kul_meta.exists():
            shutil.copy(kul_meta, dst / "metadata.txt")
        print(f"  copied 4-antenna canonical layout (160 samples)")
        return

    antennas = antennas or list(range(0, 64, 4))  # every 4th by default
    pat = re.compile(r"user_(\d+)_sample_(\d+)_antenna_(\d+)\.csv")
    pos_pat = re.compile(r"-?\d+")

    files = [f for f in sorted(KUL_RAW.glob("*.csv"))
             if pat.match(f.name) and int(pat.match(f.name).group(3)) in antennas]
    print(f"  reading {len(files)} files from {KUL_RAW.name} ...")

    antenna_rows = defaultdict(dict)
    for i, f in enumerate(files):
        m = pat.match(f.name)
        u, s_id, ant = (int(g) for g in m.groups())
        df = pd.read_csv(f, usecols=["Unnamed: 0", "real", "imaginary", "key_user_position"])
        sample_key = f"user_{u}_sample_{s_id}"
        row = {"sample_key": sample_key, "user_id": u, "sample_id": s_id}
        for _, r in df.iterrows():
            idx = int(r["Unnamed: 0"])
            row[f"subcarrier_{idx}_real"] = r["real"]
            row[f"subcarrier_{idx}_imaginary"] = r["imaginary"]
        nums = [int(n) for n in pos_pat.findall(df["key_user_position"].iloc[0])]
        row["target_x"], row["target_y"], row["target_z"] = nums
        antenna_rows[ant][sample_key] = row
        if (i + 1) % 3000 == 0:
            print(f"    {i+1}/{len(files)}")

    for ant, rows in antenna_rows.items():
        df = pd.DataFrame(list(rows.values())).drop(
            columns=["target_x", "target_y", "target_z"])
        df.to_csv(dst / f"antenna_{ant:02d}_features.csv", index=False)

    ref = pd.DataFrame(list(antenna_rows[antennas[0]].values()))
    ref[["sample_key", "target_x"]].to_csv(dst / "samples_base.csv", index=False)

    conn = pd.DataFrame([
        {"fk_table": "samples_base.csv", "fk_column": "sample_key",
         "pk_table": f"antenna_{a:02d}_features.csv", "pk_column": "sample_key"}
        for a in antennas
    ])
    conn.to_csv(dst / "connections.csv", index=False)

    kul_meta = ROOT / "datasets" / "KUL" / "metadata.txt"
    if kul_meta.exists():
        shutil.copy(kul_meta, dst / "metadata.txt")
    print(f"  wrote {len(antennas)} antenna tables × {len(ref)} samples")


# ─── Scenario N — target=n on full rabbitmq ──────────────────────────────────
def build_scenarioN() -> None:
    print("[scenarioN] target=n (full rabbitmq + cross-app lake)")
    # Uses the EUR data dir directly; no new files needed. Just confirm.
    print(f"  uses {EUR} as data-dir; no new files (pipeline reads in-place).")


# ─── CLI ─────────────────────────────────────────────────────────────────────
BUILDERS = {
    "1": build_scenario1,
    "2c": build_scenario2c,
    "a_lat95": lambda: build_scenarioA("lat95"),
    "a_lat99": lambda: build_scenarioA("lat99"),
    "b": build_scenarioB,
    "k": build_scenarioK,
    "k_csi": lambda: build_scenarioK(use_csi_layout=True),
    "n": build_scenarioN,
}


def main(targets: Iterable[str]) -> None:
    for t in targets:
        builder = BUILDERS.get(t.lower())
        if builder is None:
            print(f"⚠ unknown scenario '{t}' (valid: {', '.join(BUILDERS)})")
            continue
        builder()
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", "-s", nargs="+", default=None,
                        help=f"One or more of: {', '.join(BUILDERS)}")
    parser.add_argument("--all", action="store_true",
                        help="Build every scenario.")
    args = parser.parse_args()

    if args.all:
        main(BUILDERS.keys())
    elif args.scenario:
        main(args.scenario)
    else:
        parser.print_help()
