# PoC pitch — automated feature augmentation for tabular ML

## The problem

When training a tabular ML model, the difference between **R² = 0.7** and
**R² = 0.97** is usually not the algorithm — it's whether the right *features*
are in front of the model. Most enterprises have those features somewhere
(a sibling table, a per-user dimension table, a daily aggregate) but joining
them in is manual: a data scientist guesses at which lake tables matter, which
column joins to which, and which of the resulting columns to keep.

This pipeline automates that step.

```
your base table  +  unstructured CSV lake  →  joined+selected feature set
   (with target)     (with metadata.txt)        ready for any ML model
```

## What it does, in one paragraph

A sentence-transformer reads each column's name and description from
`metadata.txt`, embeds them, and matches columns across tables by cosine
similarity + value overlap. The matches go into a Neo4j graph. A BFS walks the
graph from the base table, joins reachable tables, and an mRMR-style selector
(`AutoFeat`) keeps the joined features that are relevant *and* non-redundant
with the base. AutoGluon trains and reports R² (regression) or accuracy
(classification) for four head-to-head approaches:

| Approach | What it does |
|---|---|
| **BASE** | Train on the base table alone — lower bound |
| **Join_All_BFS** | Join everything reachable — upper bound on data |
| **Join_All_BFS_Filter** | Same as above + a relevance filter |
| **AutoFeat** | BFS + mRMR selection — the algorithm under test |

## Headline numbers

Two of the seven benchmark scenarios are the clear showcase:

| Scenario | Setup | BASE | AutoFeat | Lift |
|---|---|---|---|---|
| **scenario2c** — synthetic feature recovery | Base = config columns only; target = lat99; lake = same rabbitmq table with the runtime columns | 0.97 R² | **0.99 R²** | **+0.022** |
| **scenarioK_kul** — MaMIMO indoor localisation | Base = sample-key + binary target *only* (no features); lake = 4 antenna CSI tables × 200 subcarrier features | 0.00 acc | **0.97 acc** | **+0.97** (50× compression: picked 15 of 800) |

The remaining 5 scenarios are deliberately negative — cross-application joins
(rabbitmq ↔ golang/python/amf) where no useful augmentation exists, and
AutoFeat correctly refuses to introduce noise. Across all 7, AutoFeat tracks
within 0.002 of BASE when there's nothing to gain, and prevents a 14-R²-point
loss on scenario1 that the naïve `Join_All_BFS` baseline takes.

## Live demo flow (3 commands)

```bash
# 1. Start the graph backend (single docker container)
make neo4j

# 2. Run the two showcase scenarios + generate a summary report
make demo

# 3. Open the interactive dashboard
make dashboard           # → http://localhost:8501
```

The dashboard has four tabs: **Results** (per-scenario drilldown), **Compare**
(cross-scenario pivot enriched with `scenarios.yaml` context), **Run** (upload
your own CSV + lake to test), **Graph** (Cypher console + live Neo4j stats).

After the demo, `make smoke` runs the same two scenarios and **asserts** the
R²/accuracy stays in the expected range — this catches silent regressions
(e.g. the AutoGluon `pkg_resources` bug that gave us R²=0 for hours during
development before we tracked it down).

## Why this is interesting beyond the 6G benchmark

The pipeline is data-agnostic. Three things make it usable on arbitrary data:

1. **Sentence-transformer discovery** — works on any data lake with descriptive
   column names. Doesn't need a hand-curated schema-mapping spec.
2. **`dataset_introspection.py`** — sniffs the target column type
   (regression / binary / multiclass), detects timestamp columns and their
   unit (s / ms / us / ns / datetime64), suggests join keys. Drives the
   pre-flight check that fails fast with a useful message if the data isn't
   suitable.
3. **Configurable temporal semantics** — `temporal_join_and_save` accepts
   tolerance as `"5min"` / `"200ms"` / `"1h"`, picks `direction=nearest`
   (snapshot analysis) vs `backward` (forecasting; no look-ahead leakage).

See [USING_OWN_DATA.md](USING_OWN_DATA.md) for the "bring your own CSV" path.

## What this is, and isn't (PoC honesty)

| In scope for the PoC | Not yet in scope |
|---|---|
| Demonstrate automated discovery + selection on a benchmark | Production multi-tenant SaaS |
| Repro on a laptop in <5 min | Distributed compute over billion-row tables |
| One-command demo + dashboard | Airflow operator, OpenLineage events, DataHub sync |
| Inspect the join graph in Neo4j | High-availability graph backend |
| Detect when augmentation *won't* help and refuse | Real-time / streaming augmentation |

See [USING_OWN_DATA.md](USING_OWN_DATA.md) for what your data needs to satisfy
and [ARCHITECTURE.md](ARCHITECTURE.md) for the component breakdown. The
[maturity assessment](#maturity) at the bottom of the project README is honest
about where this sits on the prototype → production curve.

## After the demo — concrete next steps

If a stakeholder asks "what would it take to put this in our DataOps stack?":

1. **2 weeks of focused work**: structured logging, test suite + CI, OpenLineage
   emission, parquet/warehouse input, idempotent runs, Airflow operator.
2. **One quarter**: replace Neo4j with optional Postgres/DuckDB backend, add
   Dask/Polars backend for the BFS step, container image + Helm chart.

The full breakdown is in the maturity section of the project README.
