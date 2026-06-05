"""
diagnose.py
===========
Pre-flight inspection of a base table before running ``auto_pipeline``.
Reports inferred problem type, temporal key, candidate join keys, and any
warnings / blocking errors. No Neo4j, no model training — just CSV introspection.

Usage
-----
    python scripts/diagnose.py --base-table path/to/data.csv --target lat99
    python scripts/diagnose.py --base-table data.csv --target y --temporal-key event_time
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from feature_discovery.config import rel
from feature_discovery.dataset_introspection import describe_table, diagnose


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-table", "-b", required=True, type=Path)
    parser.add_argument("--target", "-t", required=True)
    parser.add_argument("--dataset-type", choices=["regression", "binary", "multiclass"], default=None)
    parser.add_argument("--temporal-key", default=None)
    parser.add_argument("--full-schema", action="store_true",
                        help="Print every column's dtype + cardinality, not just summary.")
    args = parser.parse_args()

    if not args.base_table.is_file():
        sys.exit(f"Base table not found: {args.base_table}")

    diag = diagnose(args.base_table, args.target,
                    dataset_type=args.dataset_type, temporal_key=args.temporal_key)

    print(f"Base table : {rel(args.base_table)}")
    print(f"  rows     : {diag.spec.n_rows}")
    print(f"  columns  : {len(diag.spec.columns)}")
    print(f"  target   : '{args.target}' "
          f"(found: {diag.spec.col(args.target) is not None})")
    print()
    print(f"Inferred problem type : {diag.inferred_problem_type}")
    print(f"Inferred temporal key : {diag.inferred_temporal_key} "
          f"(unit: {diag.timestamp_unit})")
    print(f"Candidate join keys   : {diag.candidate_join_keys}")
    print()

    if diag.warnings:
        print("Warnings:")
        for w in diag.warnings:
            print(f"  ⚠  {w}")
        print()
    if diag.errors:
        print("Errors:")
        for e in diag.errors:
            print(f"  ✗  {e}")
        print()

    if args.full_schema:
        print("Column schema:")
        for c in diag.spec.columns:
            flags = []
            if c.is_datetime: flags.append("datetime")
            if c.is_numeric:  flags.append("numeric")
            if c.is_constant: flags.append("CONSTANT")
            print(f"  {c.name:30s} {c.dtype:20s} "
                  f"unique={c.n_unique:>6d}/{c.n_rows:<6d} "
                  f"({c.cardinality_ratio:6.3f})  null={c.n_null:<5d}  "
                  f"{','.join(flags)}")

    sys.exit(0 if diag.ok() else 1)


if __name__ == "__main__":
    main()
