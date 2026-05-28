# Using AutoFeat on your own data

This pipeline is built for the 6G/KUL benchmarks but isn't tied to them.
Most refactoring this round was to make it work on *arbitrary* tabular and
time-series datasets. This guide explains what your data needs to look like
and how to drive the pipeline.

---

## What AutoFeat needs from your data

AutoFeat's job is to find *useful joined features* from related tables. For
that to work:

| Requirement | Why | How to check |
|---|---|---|
| A base table CSV with a target column | Source of training rows + label | `python scripts/diagnose.py --base-table X --target Y` |
| At least one lake table CSV reachable by a join key | Nothing to augment otherwise | inspect candidate join keys in diagnose output |
| Shared values in the join key column across base ↔ lake | Otherwise the join matches 0 rows | the transformer discovery's Jaccard check, or pre-flight |
| Target has > 1 unique value | Nothing to predict | diagnose blocks with a clear error |
| Either a `metadata.txt` describing columns, or descriptive column names | So sentence-transformer discovery can match columns by semantics | a sparse / cryptic schema works less well, but column names alone usually suffice |

**Things that will *not* work out of the box**:
- Targets that are arrays/lists/JSON (must be a scalar per row)
- Hierarchical/nested CSVs
- Tables with no join key relationship to the base
- Very wide tables (200+ columns) where you also want transformer discovery — use `--no-transformer-discovery` with an explicit `connections.csv` (see KUL scenarios)

---

## Quick start on your own data

```bash
# 1. Inspect the base table first (no Neo4j, no training)
python scripts/diagnose.py --base-table mydata/base.csv --target outcome

# 2. If diagnose is happy, run the pipeline
python -m feature_discovery.auto_pipeline \
    --base-table mydata/base.csv \
    --target outcome \
    --data-dir mydata \
    --dataset-type auto                 # infers regression/binary/multiclass
```

The pipeline will:
- Run pre-flight introspection automatically (skip with `--skip-diagnose`).
- Infer `--dataset-type` from the target's unique-value count.
- Auto-detect a temporal key column if you didn't specify one.
- Parse string timestamps to datetime64 automatically.

---

## Time-series augmentation on arbitrary data

The temporal join (`temporal_join_and_save` / `pd.merge_asof`) handles a few
formats out of the box:

### Timestamp formats supported

| In your CSV | Auto-handled |
|---|---|
| `2024-01-15T10:30:00` (ISO string) | ✓ parsed to datetime64 |
| `1705314600` (Unix seconds) | ✓ detected as seconds |
| `1705314600000` (Unix ms) | ✓ detected as milliseconds |
| `1705314600000000000` (Unix ns) | ✓ detected as nanoseconds |
| `datetime64[ns]` (already parsed) | ✓ used directly |
| Mixed time zones | ⚠ not normalised; convert to UTC first |
| Date-only (`2024-01-15`) | ✓ parsed as midnight UTC |

### Tolerance accepts human-readable strings

```bash
--temporal-tolerance 60         # 60 seconds (legacy)
--temporal-tolerance "5min"
--temporal-tolerance "1h"
--temporal-tolerance "200ms"    # millisecond timestamps
--temporal-tolerance "1d"
```

The tolerance unit is converted automatically based on the detected timestamp
unit, so `--temporal-tolerance "1min"` works whether your column stores
seconds, milliseconds, microseconds, or datetime64.

### Direction matters for forecasting

```bash
--temporal-direction nearest    # default; matches the closest row either side
--temporal-direction backward   # only past rows; required to avoid leakage in forecasting
--temporal-direction forward    # only future rows; rare; e.g. "next event"
```

Default is `nearest` because that's what most of the benchmarks want
(snapshot-style analysis). **For forecasting, switch to `backward` or you'll
silently leak the future into training.**

### Example: stock-tick + macro-events join (backward, ms tolerance)

```bash
python -m feature_discovery.auto_pipeline \
    --base-table data/tick_prices.csv \
    --target return_5min \
    --data-dir data \
    --temporal-key timestamp_ms \
    --temporal-tolerance "200ms" \
    --temporal-direction backward \
    --dataset-type regression \
    --label tick_with_macros
```

---

## Multiclass classification

```bash
python -m feature_discovery.auto_pipeline \
    --base-table data/customers.csv --target segment \
    --data-dir data --dataset-type multiclass
# or just --dataset-type auto and let diagnose figure it out
```

`segment` with > 2 levels routes to AutoGluon `problem_type="multiclass"`.
Accuracy is reported instead of R².

---

## What the diagnose script reports

Example on the canonical `scenario2c`:

```
$ python scripts/diagnose.py --base-table datasets/scenario2c/rabbitmq-reduced.csv --target lat99

Base table : datasets/scenario2c/rabbitmq-reduced.csv
  rows     : 21152
  columns  : 5
  target   : 'lat99' (found: True)

Inferred problem type : regression
Inferred temporal key : time (unit: s)
Candidate join keys   : ['cpu_limit', 'n', 'ram_limit', 'time']
```

When something's wrong it tells you:

```
Errors:
  ✗  Target column 'xyz' not found in base.csv. Available: ['time', 'a', 'b', ...]
```

or

```
Warnings:
  ⚠  Target 'y' has 73.2% null values; rows will be dropped before training.
  ⚠  187 feature columns. Consider --no-transformer-discovery + connections.csv ...
```

---

## When does AutoFeat genuinely help?

From the EUR/KUL benchmarks (see [SUMMARY.md](../results/6g_data/SUMMARY.md)):

| Pattern | Will AutoFeat help? |
|---|---|
| Base table info-poor, lake has the missing signal (scenario2c, KUL CSI) | **Yes, big gain** |
| Lake tables share schema with base but are unrelated by physics (scenario1, scenarioA_*) | No — AutoFeat will refuse to include joined cols. Often correct. |
| Lake tables are time-aligned copies of the base (scenarioB amf segments) | Marginal lift — mRMR sees joined cols as redundant |
| Forecasting where the future leaks via `direction="nearest"` | **Misleadingly high** R² — switch to `direction="backward"` |

The pipeline reports BASE alongside AutoFeat so you can see the lift. If
AutoFeat ≈ BASE, augmentation didn't help for your dataset — that's a real
finding, not a bug.

---

## Files added for arbitrary-data support

- [`src/feature_discovery/dataset_introspection.py`](../src/feature_discovery/dataset_introspection.py) — schema/dtype/timestamp-unit detection
- [`scripts/diagnose.py`](../scripts/diagnose.py) — pre-flight CLI
- `temporal_join_and_save` in [`join_data.py`](../src/feature_discovery/autofeat_pipeline/join_data.py) — string-timestamp parsing, ms/us/ns tolerance, `direction` parameter
- `--dataset-type auto`, `--temporal-direction`, parsed `--temporal-tolerance` in [`auto_pipeline.py`](../src/feature_discovery/auto_pipeline.py)
- `multiclass` support in [`Dataset`](../src/feature_discovery/experiments/dataset_object.py)
