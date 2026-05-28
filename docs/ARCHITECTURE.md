# Architecture

Component-by-component walkthrough of the augmentation pipeline.

## Data flow

```
                            ┌────────────────────┐
                            │   --data-dir/      │
                            │   metadata.txt     │  ← natural-language column
                            │   base.csv         │    descriptions per column
                            │   lake_1.csv       │
                            │   …                │
                            │   connections.csv  │  ← optional explicit PK/FK
                            └─────────┬──────────┘
                                      │
                            ┌─────────▼──────────┐
                            │ auto_pipeline.py   │  ← CLI entry
                            │ - parses args      │
                            │ - sets DATA_FOLDER │
                            │ - delegates ↓      │
                            └─────────┬──────────┘
                                      │
       ┌──────────────────────────────┼──────────────────────────────────┐
       │                              │                                  │
       ▼                              ▼                                  ▼
ingest_unprocessed_data    profile_transformer_dataset      BASE/JOIN_ALL/AutoFeat
(reads connections.csv     (sentence-transformer            (experiments/ablation.py
 → Neo4j edges)             over column descriptions         + baselines.py)
                            → cosine + value-Jaccard
                            → Neo4j edges)
```

## Layer 1 — Discovery

**Input**: `metadata.txt` parsed into `{filename: {column: description}}`.
**Encoder**: `sentence-transformers/all-mpnet-base-v2` (CPU, ~420 MB cache).
**Per pair of columns** `(col_a, col_b)`:

1. `schema_sim = cosine(embed("col_a: desc_a"), embed("col_b: desc_b"))`
2. Skip if `schema_sim < 0.6`.
3. `value_sim = Jaccard(unique(col_a), unique(col_b))` over a 10k-row sample.
4. **Score**:
   ```python
   if value_sim > 0:
       score = 0.5 * schema_sim + 0.5 * value_sim
   else:
       score = 0.5 * schema_sim   # schema-only matches are penalised
   ```
5. Keep the edge if `value_sim ≥ 0.2` or `schema_sim ≥ 0.92`.

The penalty in step 4 is the key fix vs the naïve formula. Without it, name-only
matches with identical descriptions but disjoint values (e.g. `cpu_usage` across
unrelated runs) outranked legitimate join keys (`n` in cross-service scenarios).

Each accepted edge is written to Neo4j twice (one per direction) by
`merge_nodes_relation_tables(...)`.

## Layer 2 — Graph storage (Neo4j)

Schema:
- Nodes have `label` (filename) and `id` (relative path).
- Relationships of type `RELATED` carry: `from_column`, `to_column`,
  `from_label`, `to_label`, `weight`.

The BFS in `autofeat.streaming_feature_selection` queries adjacency via
`get_adjacent_nodes(...)` and ranks join keys by `weight`.

Database choice notes (in `config.py`):
- `bolt://localhost:7687` for single-instance Neo4j (use `neo4j://` only for
  routing-capable clusters).
- `NEO4J_DATABASE` defaults to `neo4j` (Neo4j 5 forbids digit-led DB names).
- Auth disabled to match the `NEO4J_AUTH=none` env in docker-compose.

## Layer 3 — Augmentation (autofeat)

`AutoFeat.streaming_feature_selection` does a BFS over `RELATED` edges starting
from the base table. For each (base_node, neighbour) pair:

1. **Pick join keys** (`get_relation_properties_node_name`) — keeps all keys
   tied at the top weight.
2. For each candidate key:
   - **`step_join`** — either `temporal_join_and_save` (asof) if the column
     matches the user-supplied `temporal_key`, else `join_and_save` (left
     merge). Right side is sampled to 1 row per key *except* for temporal
     joins (which need every candidate row to find the nearest neighbour).
   - **`step_data_quality`** — prune joins where the right join column's
     non-null ratio falls below `value_ratio` (default 0.65).
   - **`streaming_relevance_redundancy`** — for the newly joined columns:
     - `measure_relevance` — Spearman correlation with target (or Pearson).
     - `measure_redundancy` — mRMR-style: `relevance − redundancy/N`, optional
       conditional-MI term (`--jmi`).
     - Combined score, top-`k` features per path.

Each successful join adds an entry to `partial_join_selected_features[path]`
keyed by the canonical path name. The BFS terminates when no unvisited
neighbours remain.

## Layer 4 — Evaluation

For each of the top-k paths (`evaluate_paths`):

1. Load the augmented dataframe.
2. AutoGluon `TabularPredictor.fit(hyperparameters={"XGB": {}})`.
3. Compute R² (regression) or accuracy (binary), plus per-feature permutation
   importance.

Three baselines run alongside:
- `non_augmented` — base table only.
- `join_all_bfs` — single merged dataframe across every BFS-reachable table, no
  selection.
- `join_all_bfs` + filter — relevance-filter the merged dataframe.

Results all serialise into a single CSV at
`results/6g_data/auto_pipeline_<label>.csv`.

## Layer 5 — Dashboard

`dashboards/augmentation_dashboard.py` (Streamlit) — three tabs:

| Tab | Reads from | What it shows |
|---|---|---|
| Results | `results/6g_data/auto_pipeline_*.csv` | Cross-scenario bar chart + pivot table + feature-importance drilldown |
| Run | uploaded files → tmpdir → subprocess | Streams `auto_pipeline` stdout, picks up the result CSV |
| Graph | `bolt://localhost:7687` | Live node/relationship counts, link to Neo4j Browser, canned Cypher |

Cache is invalidated (`st.cache_data.clear()`) after each successful run so the
Results tab refreshes automatically.

## Failure modes and how the pipeline reacts

| Symptom | Cause | Handler |
|---|---|---|
| `Unable to retrieve routing information` | `neo4j://` scheme against single instance | `bolt://` in config |
| `MemoryPoolOutOfMemoryError` | Many edges queued by transformer in one transaction | `docker restart feature-discovery-neo4j`; batched delete in graph clear |
| `Models trained: []` | `setuptools 82+` removed `pkg_resources` → autogluon's XGB import silently fails | pin `setuptools<81` |
| `_ARRAY_API not found` | `tensorflow 2.16` compiled against numpy 1.x | `pip uninstall tensorflow keras tensorboard` (we don't use TF) |
| `import xgboost failed` (misleading) | Same `pkg_resources` issue | Same fix |
| `no valid join paths found` | All paths' AutoGluon fits returned empty | Symptom of model-training failure, not of BFS — check logs above |

## Extending the pipeline

To support a new data domain:

1. Add CSVs to a folder, write a `metadata.txt` with `<col>: <description>`
   lines under each `files: a.csv, b.csv` section.
2. (Optional) add a `connections.csv` if you already know the PK/FK edges.
3. `python -m feature_discovery.auto_pipeline --base-table … --target … --data-dir …`.

To benchmark a new algorithm:
- Add a function to `experiments/baselines.py` returning a list of `Result`.
- Wire it into `auto_pipeline.main()` after the existing BASE/JOIN_ALL/AutoFeat
  block. Each `Result` is one row in the output CSV.
