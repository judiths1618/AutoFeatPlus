#!/usr/bin/env python3
"""
split_by_gaps.py
Split a time-series CSV into segments based on detected large time gaps.

Usage:
    python split_by_gaps.py --input ./EUR/amf-performance.csv
    python split_by_gaps.py --input ./EUR/ --gap-multiplier 10
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# Time column detection
# ─────────────────────────────────────────────

def detect_time_column(df: pd.DataFrame) -> str | None:
    """Auto-detect the time column by name hints and dtype."""
    hints = ["time", "timestamp", "datetime", "date", "ts", "t"]

    # 1. Name-based match
    for col in df.columns:
        if any(h in col.lower() for h in hints):
            return col

    # 2. First numeric column that looks monotonically increasing
    for col in df.select_dtypes(include="number").columns:
        if df[col].is_monotonic_increasing:
            return col

    return df.columns[0]


def parse_time_series(series: pd.Series) -> pd.Series:
    """Parse a time column to datetime, handling Unix seconds/ms automatically."""
    if pd.api.types.is_numeric_dtype(series):
        unit = "ms" if series.max() > 1e12 else "s"
        return pd.to_datetime(series, unit=unit)
    return pd.to_datetime(series, infer_datetime_format=True)


# ─────────────────────────────────────────────
# Gap detection
# ─────────────────────────────────────────────

def detect_split_points(
    time_parsed: pd.Series,
    gap_multiplier: float = 10.0,
    min_gap_seconds: float | None = None,
) -> list[int]:
    """
    Return row indices where a large gap starts (i.e., split before this row).

    A gap is 'large' if:
      - it exceeds median_interval * gap_multiplier, AND
      - (optionally) it exceeds min_gap_seconds
    """
    diffs = time_parsed.diff().dt.total_seconds().fillna(0)
    median_interval = diffs[diffs > 0].median()

    threshold = median_interval * gap_multiplier
    if min_gap_seconds is not None:
        threshold = max(threshold, min_gap_seconds)

    split_indices = diffs[diffs > threshold].index.tolist()

    print(f"\n  Median interval : {median_interval:.2f}s")
    print(f"  Gap threshold   : {threshold:.2f}s  (×{gap_multiplier} multiplier)")
    print(f"  Large gaps found: {len(split_indices)}")
    for idx in split_indices:
        gap_s = diffs[idx]
        t     = time_parsed[idx]
        print(f"    row {idx:6d} — gap {gap_s:.1f}s ({gap_s/3600:.2f}h) before {t}")

    return split_indices


# ─────────────────────────────────────────────
# Splitting
# ─────────────────────────────────────────────

def split_dataframe(
    df: pd.DataFrame,
    split_indices: list[int],
) -> list[pd.DataFrame]:
    """Split df at the given row indices, returning a list of sub-DataFrames."""
    boundaries = [0] + split_indices + [len(df)]
    segments   = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        seg = df.iloc[start:end].copy().reset_index(drop=True)
        if len(seg) > 0:
            segments.append(seg)
    return segments


# ─────────────────────────────────────────────
# Per-file processing
# ─────────────────────────────────────────────

def process_file(
    csv_path: Path,
    output_dir: Path,
    gap_multiplier: float,
    min_gap_seconds: float | None,
    min_segment_rows: int,
) -> list[Path]:
    """Split a single CSV file and save segments."""
    print(f"\n{'='*60}")
    print(f"Processing: {csv_path.name}")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    print(f"  Shape: {df.shape}")

    # ── Detect time column ───────────────────
    time_col = detect_time_column(df)
    print(f"  Time column: '{time_col}'")

    time_parsed = parse_time_series(df[time_col]).reset_index(drop=True)
    print(f"  Time range : {time_parsed.min()}  →  {time_parsed.max()}")
    duration_h = (time_parsed.max() - time_parsed.min()).total_seconds() / 3600
    print(f"  Duration   : {duration_h:.2f}h")

    # ── Detect gaps ──────────────────────────
    split_indices = detect_split_points(time_parsed, gap_multiplier, min_gap_seconds)

    if not split_indices:
        print(f"\n  No large gaps detected — saving as single file.")
        out_path = output_dir / csv_path.name
        df.to_csv(out_path, index=False)
        return [out_path]

    # ── Split ────────────────────────────────
    segments  = split_dataframe(df, split_indices)
    stem      = csv_path.stem
    saved     = []

    print(f"\n  Segments ({len(segments)} total):")
    print(f"  {'#':<4} {'Rows':>8} {'Start':>26} {'End':>26} {'Duration':>12} {'Saved'}")
    print(f"  {'-'*90}")

    for i, seg in enumerate(segments):
        t_seg    = parse_time_series(seg[time_col])
        t_start  = t_seg.min()
        t_end    = t_seg.max()
        dur_h    = (t_end - t_start).total_seconds() / 3600
        date_tag = t_start.strftime("%Y%m%d_%H%M%S")
        out_name = f"{stem}_seg{i+1:02d}_{date_tag}.csv"
        out_path = output_dir / out_name

        if len(seg) < min_segment_rows:
            status = f"SKIP (<{min_segment_rows} rows)"
        else:
            seg.to_csv(out_path, index=False)
            saved.append(out_path)
            status = "✓"

        print(f"  {i+1:<4} {len(seg):>8} {str(t_start):>26} {str(t_end):>26} "
              f"{dur_h:>10.2f}h  {status}")

    print(f"\n  Saved {len(saved)}/{len(segments)} segments to: {output_dir}")
    return saved


# ─────────────────────────────────────────────
# Batch processing
# ─────────────────────────────────────────────

def process_directory(
    input_path: Path,
    output_root: Path,
    gap_multiplier: float,
    min_gap_seconds: float | None,
    min_segment_rows: int,
) -> None:
    """Process all CSV files in a directory."""
    csv_files = sorted(input_path.glob("**/*.csv"))

    if not csv_files:
        print(f"No CSV files found in: {input_path}")
        return

    print(f"Found {len(csv_files)} CSV file(s) in {input_path}")

    all_saved = []
    for csv_path in csv_files:
        # Mirror directory structure under output_root
        rel      = csv_path.parent.relative_to(input_path)
        out_dir  = output_root / rel
        out_dir.mkdir(parents=True, exist_ok=True)

        saved = process_file(
            csv_path, out_dir,
            gap_multiplier, min_gap_seconds, min_segment_rows,
        )
        all_saved.extend(saved)

    print(f"\n{'='*60}")
    print(f"DONE — {len(all_saved)} segment files saved to: {output_root}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────

def save_summary(saved_files: list[Path], output_root: Path) -> None:
    """Save a summary CSV listing all segments with their metadata."""
    rows = []
    for f in saved_files:
        df       = pd.read_csv(f)
        time_col = detect_time_column(df)
        t        = parse_time_series(df[time_col])
        rows.append({
            "file":       f.name,
            "rows":       len(df),
            "start":      t.min(),
            "end":        t.max(),
            "duration_h": round((t.max() - t.min()).total_seconds() / 3600, 3),
            "path":       str(f),
        })

    summary_df = pd.DataFrame(rows)
    summary_path = output_root / "split_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[SUMMARY] {summary_path}")
    print(summary_df.to_string(index=False))


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split time-series CSV files by detected large time gaps."
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to a single CSV file or a directory of CSV files.",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output directory (default: <input_dir>/split_output/).",
    )
    parser.add_argument(
        "--gap-multiplier", "-g", type=float, default=10.0,
        help="Gap threshold = median_interval × this value (default: 10).",
    )
    parser.add_argument(
        "--min-gap-seconds", type=float, default=None,
        help="Hard minimum gap in seconds to trigger a split (optional).",
    )
    parser.add_argument(
        "--min-rows", type=int, default=10,
        help="Skip segments with fewer than this many rows (default: 10).",
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_root = Path(args.output) if args.output else (
        input_path.parent / "split_output" if input_path.is_file()
        else input_path / "split_output"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"\nWaveStitch+ CSV Splitter")
    print(f"  Input          : {input_path}")
    print(f"  Output         : {output_root}")
    print(f"  Gap multiplier : ×{args.gap_multiplier}")
    print(f"  Min gap        : {args.min_gap_seconds}s" if args.min_gap_seconds else
          f"  Min gap        : auto")
    print(f"  Min rows/seg   : {args.min_rows}")

    all_saved = []

    if input_path.is_file():
        saved = process_file(
            input_path, output_root,
            args.gap_multiplier, args.min_gap_seconds, args.min_rows,
        )
        all_saved.extend(saved)

    elif input_path.is_dir():
        csv_files = sorted(input_path.glob("**/*.csv"))
        print(f"\nFound {len(csv_files)} CSV file(s)")

        for csv_path in csv_files:
            rel     = csv_path.parent.relative_to(input_path)
            out_dir = output_root / rel
            out_dir.mkdir(parents=True, exist_ok=True)
            saved   = process_file(
                csv_path, out_dir,
                args.gap_multiplier, args.min_gap_seconds, args.min_rows,
            )
            all_saved.extend(saved)
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")

    if all_saved:
        save_summary(all_saved, output_root)

    print(f"\n{'='*60}")
    print(f"DONE — {len(all_saved)} segment(s) saved to: {output_root}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()