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
  k     — KUL CSI (samples_base + per-antenna csi_as_features tables)
  r     — Cross-app resource contention (positive, no target-name leakage)
  u     — Heterogeneous unrelated lake (negative, schema discovery should refuse)
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
SEG_DIR = ROOT / "datasets" / "EUR" / "6907619" / "split_output"

# Synthesised scenarios live under scenarios/ (separate from datasets/EUR & datasets/KUL,
# which hold the raw 6G/MaMIMO data this script reads from).
OUT = ROOT / "scenarios"

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


# Latency-percentile columns we treat as proxies of the target — keeping them in
# a lake makes "feature recovery" look stronger than it really is, because they
# are near-perfect predictors of any other percentile. Stripped from every
# scenario lake whose target is a latency value.
_LAT_PROXY_RE = re.compile(r"^(lat\d+|min|mean)$")


def _strip_target_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """Remove latency-percentile/aggregate columns from a lake dataframe."""
    drop = [c for c in df.columns if _LAT_PROXY_RE.fullmatch(c)]
    return df.drop(columns=drop) if drop else df


def _strip_string_datetime_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Drop string-format datetime columns (e.g. amf's ``dt``, ``datetime``).

    AutoGluon's feature generator decomposes such columns into ``<col>.day`` /
    ``<col>.dayofweek``; downstream ``feature_importance(feature_stage="original")``
    then raises a KeyError because those generated names aren't real columns.
    Since ``time`` (integer epoch) carries the same information, dropping the
    string variants is lossless and avoids the AutoGluon footgun.
    """
    return df.drop(columns=[c for c in ("dt", "datetime") if c in df.columns])


# ─── Scenario 2C — feature recovery via time ─────────────────────────────────
def build_scenario2c() -> None:
    print("[scenario2c] feature-recovery (self-join via time, proxy-free lake)")
    dst = OUT / "scenario2c"
    dst.mkdir(parents=True, exist_ok=True)

    rmq = _filter_zero_lat99(pd.read_csv(EUR / "rabbitmq-performance.csv"))
    base_cols = ["time", "ram_limit", "cpu_limit", "n", "lat99"]
    reduced = rmq[base_cols]
    reduced.to_csv(dst / "rabbitmq-reduced.csv", index=False)
    print(f"  base: {reduced.shape} cols={base_cols}")

    # Strip target column AND target-proxy percentiles so the recovery test
    # measures recovery of ram_usage/cpu_usage rather than lat50→lat99 prediction.
    features = _strip_target_proxies(rmq.drop(columns=["lat99"]))
    features.to_csv(dst / "rabbitmq-features.csv", index=False)
    print(f"  lake: {features.shape} cols={list(features.columns)}")

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

    # Lake tables keep configs + resources + n + time only — every lat* column
    # is stripped so a same-name cross-app lat95/lat99 cannot trivially leak,
    # and the string-format dt/datetime columns are dropped so AutoGluon's
    # feature generator doesn't synthesise dt.day / dt.dayofweek features that
    # later trip up feature_importance(feature_stage="original").
    for f in ["golang-web-server-performance.csv",
              "python-web-server-performance.csv",
              "amf-performance.csv"]:
        cleaned = _strip_string_datetime_cols(_strip_target_proxies(pd.read_csv(EUR / f)))
        cleaned.to_csv(dst / f, index=False)
    print(f"  lake: 3 cross-app tables with lat*/dt/datetime columns stripped")

    _write_metadata(dst / "metadata.txt", f"""
Scenario A ({target}) — cross-application temporal augmentation. rabbitmq base
strips the sibling latency percentiles so the model has to learn from
configs+resources alone. Lake adds golang/python/amf at matched timestamps
(asof join, Δ=60s) with every lat*/min/mean column removed to prevent
target-name leakage.
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
        # Strip lat*/min/mean (no target-percentile shortcut) AND the string
        # dt/datetime columns (avoids AutoGluon's dt.day decomposition that
        # crashes feature_importance later).
        cleaned = _strip_string_datetime_cols(_strip_target_proxies(pd.read_csv(f)))
        cleaned.to_csv(dst / f.name, index=False)
    print(f"  lake: {len(others)} amf segments (lat*/dt/datetime stripped)")

    _write_metadata(dst / "metadata.txt", """
Scenario B — within-application augmentation across amf time-segments. Base is
seg01 with sibling latencies stripped; lake contains other amf segments with
their lat*/min/mean columns also removed. Join key is n (workload identifier
shared across segments) — augmentation must come from resource patterns at the
same workload, not from leaking a sibling segment's percentile.
""")


# ─── Scenario K — KUL CSI (csi_as_features layout) ──────────────────────────
def build_scenarioK(antennas: List[int] = None) -> None:
    """Build scenarioK_csi from the raw KUL csi_as_features per-antenna CSVs.

    Information-poor base (``samples_base.csv`` = sample_key + binary target_x)
    plus one feature table per antenna (subcarrier_<i>_real/imaginary). Default
    selects every 4th antenna for a 16-table lake (960 samples each).
    """
    print("[scenarioK] KUL CSI (csi_as_features layout)")
    dst = OUT / "scenarioK_csi"
    dst.mkdir(parents=True, exist_ok=True)

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
            # Drop position labels AND identity columns. user_id/sample_id
            # would otherwise let AutoFeat shortcut to a per-user lookup;
            # forcing reliance on subcarrier features makes the showcase honest.
            columns=["target_x", "target_y", "target_z", "user_id", "sample_id"])
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


# ─── Scenario R — cross-app resource contention (honest positive) ────────────
def build_scenarioR() -> None:
    """Cross-app resource-contention augmentation, no target-name overlap.

    Base = rabbitmq configs + workload key only (own runtime stripped). Lake =
    {golang, python, amf} stripped down to (time, ram_usage, cpu_usage). Tests
    whether AutoFeat can detect "shared-host contention" — when co-located
    services peak in resource use, rabbitmq slows down — without smuggling
    any lat*/n column from the lake into the feature space.
    """
    print("[scenarioR] cross-app resource contention (honest positive)")
    dst = OUT / "scenarioR_resource"
    dst.mkdir(parents=True, exist_ok=True)

    rmq = _filter_zero_lat99(pd.read_csv(EUR / "rabbitmq-performance.csv"))
    base_cols = ["time", "ram_limit", "cpu_limit", "n", "lat99"]
    rmq[base_cols].to_csv(dst / "rabbitmq-reduced.csv", index=False)
    print(f"  base rabbitmq-reduced: cols={base_cols} rows={len(rmq)}")

    # Lake: only time + resource columns from each peer service. No latency
    # column survives; n is also removed so a workload-level join shortcut
    # cannot pretend to be a host-contention signal.
    resource_cols = ["time", "ram_usage", "cpu_usage"]
    for f in ["golang-web-server-performance.csv",
              "python-web-server-performance.csv",
              "amf-performance.csv"]:
        df = pd.read_csv(EUR / f)
        keep = [c for c in resource_cols if c in df.columns]
        df[keep].to_csv(dst / f.replace(".csv", "-resources.csv"), index=False)
    print(f"  lake: 3 cross-app *-resources.csv tables with {resource_cols}")

    # Explicit FK→PK on time so the discovery phase does not have to invent it.
    conn = pd.DataFrame([
        {"fk_table": "rabbitmq-reduced.csv", "fk_column": "time",
         "pk_table": pk, "pk_column": "time"}
        for pk in ("golang-web-server-performance-resources.csv",
                   "python-web-server-performance-resources.csv",
                   "amf-performance-resources.csv")
    ])
    conn.to_csv(dst / "connections.csv", index=False)
    _write_metadata(dst / "metadata.txt", """
Scenario R — cross-application resource-contention augmentation. Base predicts
rabbitmq lat99 from its configs and workload only. The lake contains exactly
(time, ram_usage, cpu_usage) for each peer service; if AutoFeat lifts BASE here
it is because shared-host load actually correlates with rabbitmq latency, not
because some target-named column slipped through.
""")


# ─── Scenario U — heterogeneous unrelated lake (honest negative) ─────────────
def build_scenarioU() -> None:
    """Heterogeneous unrelated lake — schema discovery should refuse.

    Base = rabbitmq-reduced (configs + lat99). Lake = the KUL CSI tables
    (samples_base + a small set of antenna feature tables). There is no
    semantic shared column or join key. A correct discovery layer should
    find no FK→PK relation and AutoFeat should match BASE exactly.
    """
    print("[scenarioU] heterogeneous unrelated lake (honest negative)")
    dst = OUT / "scenarioU_unrelated"
    dst.mkdir(parents=True, exist_ok=True)

    rmq = _filter_zero_lat99(pd.read_csv(EUR / "rabbitmq-performance.csv"))
    base_cols = ["time", "ram_limit", "cpu_limit", "ram_usage", "cpu_usage", "n", "lat99"]
    rmq[base_cols].to_csv(dst / "rabbitmq-reduced.csv", index=False)
    print(f"  base rabbitmq-reduced: cols={base_cols} rows={len(rmq)}")

    # Re-use whatever scenarioK_csi already produced (if present) — copy a small
    # subset of antenna tables so we have CSI columns in the lake without having
    # to re-read 15k raw files. This is cheap and matches the "unrelated lake"
    # premise: same lake content as the K scenario, but joined to rabbitmq.
    csi_src = OUT / "scenarioK_csi"
    if not csi_src.is_dir():
        print(f"  ⚠ {csi_src} not built yet — run --scenario k first; skipping U")
        return
    copied = 0
    for fname in ("samples_base.csv",
                  "antenna_00_features.csv",
                  "antenna_16_features.csv",
                  "antenna_32_features.csv",
                  "antenna_48_features.csv"):
        src = csi_src / fname
        if src.exists():
            shutil.copy(src, dst / fname)
            copied += 1
    print(f"  lake: {copied} KUL-CSI tables copied (no shared key with rabbitmq)")

    # Intentionally NO connections.csv — there is no valid FK→PK to declare.
    _write_metadata(dst / "metadata.txt", """
Scenario U — unrelated heterogeneous lake. Base is rabbitmq telemetry; lake is
the KUL MaMIMO CSI tables (samples_base + antenna feature tables). No semantic
column is shared between domains, no connections.csv is provided. Schema/
transformer discovery should find no useful FK→PK link, and AutoFeat should
refuse to augment — i.e. match BASE exactly.
""")


# ─── CLI ─────────────────────────────────────────────────────────────────────
BUILDERS = {
    "1": build_scenario1,
    "2c": build_scenario2c,
    "a_lat95": lambda: build_scenarioA("lat95"),
    "a_lat99": lambda: build_scenarioA("lat99"),
    "b": build_scenarioB,
    "k": build_scenarioK,
    "n": build_scenarioN,
    "r": build_scenarioR,
    "u": build_scenarioU,
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
