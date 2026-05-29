"""Split EUR telemetry CSVs into segments at large time gaps.

EUR 6907619 data (amf, rabbitmq, golang, python, …) was collected across
multiple experiment windows separated by long downtimes. Each contiguous
window is its own sub-experiment; splitting at gaps recovers them as
individual CSVs that `prepare_augmentation_scenarios.py` (and downstream
scenarios) can consume.

Usage
-----
    # default: scan datasets/EUR/6907619 for every CSV with a `time` column
    python scripts/split_eur_by_time_gaps.py

    # only specific files
    python scripts/split_eur_by_time_gaps.py --files amf-performance.csv rabbitmq-performance.csv

    # different gap threshold (default 3600 = 1 h)
    python scripts/split_eur_by_time_gaps.py --gap-seconds 1800

    # custom source directory
    python scripts/split_eur_by_time_gaps.py --source-dir datasets/EUR/other_run

Output
------
    <source-dir>/split_output/<stem>_seg<NN>_<YYYYMMDD>_<HHMMSS>.csv

Files without a ``time`` column, or that already live under ``split_output/``,
are skipped silently.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Iterable, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC_DIR = ROOT / "datasets" / "EUR" / "6907619"
TIME_COLUMN = "time"


def split_one(src: Path, out_dir: Path, gap_seconds: int) -> int:
    """Segment a single CSV by gaps in its ``time`` column.

    Returns the number of segment files written. Returns 0 (and prints a hint)
    when the file lacks a ``time`` column, since not every CSV in an EUR run is
    a time series (e.g. precomputed feature dumps).
    """
    df = pd.read_csv(src)
    if TIME_COLUMN not in df.columns:
        print(f"  skip {src.name}: no '{TIME_COLUMN}' column")
        return 0

    df = df.sort_values(TIME_COLUMN).reset_index(drop=True)
    seg_id = (df[TIME_COLUMN].diff() > gap_seconds).cumsum() + 1
    df["__seg"] = seg_id

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for sid, sub in df.groupby("__seg"):
        sub = sub.drop(columns="__seg").reset_index(drop=True)
        start_utc = dt.datetime.utcfromtimestamp(int(sub[TIME_COLUMN].iloc[0]))
        name = f"{src.stem}_seg{int(sid):02d}_{start_utc:%Y%m%d_%H%M%S}.csv"
        sub.to_csv(out_dir / name, index=False)
        written += 1
        print(f"    seg{int(sid):02d}: rows={len(sub):5d}  start={start_utc:%Y-%m-%d %H:%M:%S}  → {name}")
    return written


def candidates(src_dir: Path, files: Iterable[str] | None) -> List[Path]:
    """Resolve the list of CSVs to split.

    With ``--files`` given, those exact names; otherwise every ``*.csv`` under
    ``src_dir`` (top level only — anything already nested under ``split_output/``
    is ignored to keep re-runs idempotent).
    """
    if files:
        out: List[Path] = []
        for name in files:
            p = (src_dir / name) if not Path(name).is_absolute() else Path(name)
            if not p.exists():
                print(f"  ⚠ {p} not found, skipping")
                continue
            out.append(p)
        return out
    return sorted(p for p in src_dir.glob("*.csv") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SRC_DIR,
                        help="Folder of EUR telemetry CSVs to split. "
                             "Default: datasets/EUR/6907619")
    parser.add_argument("--files", nargs="+", default=None,
                        help="Restrict to these filenames (relative to --source-dir). "
                             "Default: every *.csv with a 'time' column.")
    parser.add_argument("--gap-seconds", type=int, default=3600,
                        help="Time gap (seconds) that starts a new segment. Default 3600.")
    args = parser.parse_args()

    src_dir = args.source_dir.resolve()
    if not src_dir.is_dir():
        raise SystemExit(f"source dir not found: {src_dir}")
    out_dir = src_dir / "split_output"

    sources = candidates(src_dir, args.files)
    if not sources:
        raise SystemExit(f"no candidate CSVs under {src_dir}")

    total_files = 0
    total_segments = 0
    for src in sources:
        print(f"\n[{src.name}] splitting on gap > {args.gap_seconds}s")
        n = split_one(src, out_dir, args.gap_seconds)
        if n:
            total_files += 1
            total_segments += n
    print(f"\nWrote {total_segments} segment file(s) across {total_files} source CSV(s) "
          f"→ {out_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
