# AutoFeat 2025 — 6G Feature Discovery and Time-Series Benchmarks

This repository contains reproducible experiments for two related, but distinct,
questions on 6G telemetry data:

1. **AutoFeat feature discovery**: can a base table be improved by discovering
   joinable tables and selecting useful external tabular features?
2. **Time-series augmentation**: can train-only time-series transformations
   improve forecasting on real, unchanged test windows?

These are different method families. AutoFeat/AutoFeatPlus widens a tabular
feature matrix through joins. Time-series augmentation creates additional
training windows or rows. The optional downstream bridge experiments ask whether
time-series-style augmentation helps a later tabular application, but those rows
should not be treated as the primary score for either method family.

The original AutoFeat paper README is preserved at
[docs/legacy/README_AutoFeat_paper.md](docs/legacy/README_AutoFeat_paper.md).

## Improvements over the original AutoFeat

This repository keeps the original BFS-discovery / mRMR feature-selection core
([autofeat_pipeline/](src/feature_discovery/autofeat_pipeline/)) intact but fixes
several methodology and reliability issues that affected the published results.
Each row below points at the code that changed.

### Methodology

| Improvement | What was wrong before | Where it's fixed |
|---|---|---|
| **End-to-end reproducibility** — a single `--seed` controls every RNG (train/test split, AutoGluon model seeds, group sampling, feature-selection tie-breaks). `PYTHONHASHSEED` is pinned by a one-shot re-exec so `set`/`dict` ordering is also stable. | The original pipeline used a mix of hard-coded `random_state=10` / `42` and AutoGluon's default seeding — and `set`/`dict` ordering left feature-selection tie-breaks **non-deterministic** even with all RNGs seeded. Two consecutive runs gave different `AutoFeat` numbers. | [`auto_pipeline._ensure_reproducible`](src/feature_discovery/auto_pipeline.py), `config.SEED`, every `random_state=` routed through it |
| **Like-for-like evaluation split** — when `--temporal-key` is set, BASE, JOIN_ALL_BFS, JOIN_ALL_BFS_Filter and AutoFeat all use the same chronological 80/20 split. | Original code only passed `time_column` into AutoFeat's evaluator, so BASE used a *random* split and AutoFeat used a *temporal* split — the headline `Δ AutoFeat − BASE` compared two different test sets. | [`baselines._resolve_time_column`](src/feature_discovery/experiments/baselines.py), `time_column=dataset.temporal_key` threaded through |
| **Non-redundant model comparison** — when `--algorithms XGB` is requested the runner trains **XGBoost + RandomForest** (companion model) instead of XGBoost + AutoGluon's degenerate `WeightedEnsemble_L2` (which for a single base model just duplicates it). | Every result row appeared twice with identical numbers under different algorithm names. | [`evaluation_algorithms._with_companion`](src/feature_discovery/experiments/evaluation_algorithms.py), `fit_weighted_ensemble=False` |
| **Honest scenario lakes** — every benchmark scenario lake is stripped of target-percentile siblings (`lat50/75/95/min`, `mean`) and identity columns (`user_id`, `sample_id`) before evaluation. | The original showcases harvested target proxies — e.g. scenario 2C's "feature recovery" lifted R² by using `lat95` to predict `lat99`; KUL's "compression test" used `user_id` to predict `target_x`. Both inflated the showcase numbers. | [`prepare_augmentation_scenarios._strip_target_proxies`](scripts/prepare_augmentation_scenarios.py), `build_scenarioK` drops identity columns |
| **Three-class scenario taxonomy** — the 9 scenarios are now explicitly tagged 🟢 positive / 🔴 negative / 🟡 ambiguous in the manifest, with two new entries (`scenarioR_resource`, `scenarioU_unrelated`) designed specifically to test honest cross-app augmentation and heterogeneous-lake refusal. | The original "expected_behaviour" labels (refuse/showcase) were inconsistent — some "refuses" actually had partial signal, some "showcases" lifted via target proxies. | [scenarios/scenarios.yaml](scenarios/scenarios.yaml) |

### Reliability / correctness fixes

| Bug | Symptom | Fix |
|---|---|---|
| `merge_asof` rejected a *float* `tolerance` on an *int64* `time` column | Crash on every temporal scenario with `--temporal-tolerance 0` | Coerce tolerance to int when key is integer ([`join_data.py`](src/feature_discovery/autofeat_pipeline/join_data.py)) |
| `--temporal-tolerance` arrived from argparse as a `str`, then compared with `>= 0` | `TypeError: '>=' not supported between str and int` in `evaluate_join_paths` | Robust parse via [`parse_tolerance`](src/feature_discovery/experiments/evaluate_join_paths.py) |
| KUL `connections.csv` had reversed `pk_table,fk_table` order — BFS traverses FK→PK, so the base node had no outgoing edges | Whole KUL scenario silently produced 0 features | Manifest/prep regenerate the canonical FK-first order |
| `transformers` library probed its TensorFlow backend on import and crashed with `ModuleNotFoundError: tf_keras` | Every transformer-discovery scenario failed on fresh envs | `os.environ.setdefault("USE_TF", "0")` at the top of [`transformer_discovery.py`](src/feature_discovery/dataset_relation_graph/transformer_discovery.py) |
| AutoGluon `predictor.feature_importance(feature_stage="original")` raised `KeyError: ['<col>.day', '<col>.dayofweek']` whenever a string-datetime column survived into the lake | Lost the entire JOIN_ALL row even though `evaluate()` succeeded | try/except wrapper preserves the accuracy row ([`evaluation_algorithms.py`](src/feature_discovery/experiments/evaluation_algorithms.py)); prep script also drops `dt`/`datetime` columns |
| Spearman filter ranked features in the *post*-generator frame, then indexed the *pre*-generator dataframe → `KeyError` on AutoGluon-synthesised feature names | `JOIN_ALL_BFS_Filter` row missing whenever any datetime column entered the lake | Filter selected features against `dataframe.columns` first ([`baselines.py`](src/feature_discovery/experiments/baselines.py)) |
| Ingest never explicitly `create_node()`-ed; it relied on `connections.csv`-driven `MERGE` to *implicitly* create nodes | Heterogeneous-lake refusal scenarios with no `connections.csv` blew up with `'NoneType' has no attribute 'get'` | Always `create_node()` per enumerated CSV ([`ingest_data.py`](src/feature_discovery/dataset_relation_graph/ingest_data.py)) |

### Tooling & UX

- **Privacy-safe logging** — `config.rel()` renders every printed path relative to the project root, so neither absolute paths nor your username leak into console output, run logs or committed result files.
- **Inline graph view in the Streamlit dashboard** — `tab_graph` now renders an interactive Graphviz of the live Neo4j graph (or any picked scenario's `connections.csv`) with degree-weighted node sizes, weight-thresholded edge filters, and a top-table panel — instead of just listing Cypher snippets.
- **CI runs every test, not just one file** — [`.github/workflows/ci.yml`](.github/workflows/ci.yml) discovers `test_*.py` under `src/` with `pytest`. [`conftest.py`](conftest.py) at the repo root puts `src/` on `sys.path` so `pytest` works without `PYTHONPATH`.
- **Bandwidth-friendly checkout** — `datasets/` is git-ignored; this version's checkout went from 15 GB to 45 MB on origin (the raw 6G datasets stay external).
- **Time-gap segmenter** for any EUR telemetry CSV with a `time` column → [`scripts/split_eur_by_time_gaps.py`](scripts/split_eur_by_time_gaps.py).
- **One-shot runner for the whole benchmark** — [`scripts/run_all_scenarios.sh`](scripts/run_all_scenarios.sh).

## Repository Map

```text
src/feature_discovery/
├── auto_pipeline.py                     # end-to-end AutoFeat CLI
├── autofeat_pipeline/                   # BFS joins + mRMR feature selection
├── dataset_relation_graph/              # metadata/transformer relationship discovery
├── experiments/                         # baselines, AutoFeatPlus, evaluation helpers
└── pipelines/base_table_pipeline.py     # base-table relationship planning helpers

scripts/
├── prepare_augmentation_scenarios.py    # builds the 9 benchmark scenarios
├── split_eur_by_time_gaps.py            # segment EUR telemetry CSVs at large time gaps
├── run_all_scenarios.sh                 # one-shot 9-scenario benchmark runner
├── benchmark_eur_darts.py               # Darts time-series augmentation benchmark
├── benchmark_eur_ts_augmented_downstream.py
│                                         # exploratory downstream utility check
├── benchmark_eur_autofeat_plus_local.py # local AutoFeatPlus benchmark
├── summarize_results.py                 # AutoFeat scenario summary
├── summarize_autofeat_plus_results.py   # AutoFeatPlus summary
├── plot_base_autofeat_curves.py         # BASE vs AutoFeat plots
└── run_base_table_pipeline.py           # relationship/use-case reports

dashboards/
└── augmentation_dashboard.py            # AutoFeat dashboard (Results, Compare,
                                          # Run-on-your-data, inline graph view)

conftest.py                              # puts src/ on sys.path for pytest
.github/workflows/ci.yml                 # discovers + runs every test_*.py

datasets/
├── EUR/6907619/                         # raw 6G EUR telemetry tables
└── KUL/                                 # raw MaMIMO CSI data layouts

scenarios/
├── scenarios.yaml                       # benchmark manifest
└── scenario*/                           # generated AutoFeat benchmark scenarios

results/6g_data/
├── augmentation/                        # AutoFeat scenario summaries
├── darts/                               # Darts forecasting outputs
├── downstream/                          # bridge/downstream utility outputs
├── figures/base_vs_autofeat/            # BASE vs AutoFeat plots
└── base_table_pipeline/                 # relationship reports
```

## Environment Setup

Use Python 3.10 for the full AutoGluon/Darts stack.

```bash
conda create -n autofeat-6g python=3.10 -y
conda activate autofeat-6g

pip install -r requirements.txt
pip install --no-deps "setuptools<81"
export PYTHONPATH=src   # or prefix commands with PYTHONPATH=src
```

`setuptools<81` is needed because AutoGluon 1.3 imports `pkg_resources`.

Optional dashboard and time-series packages may need to be installed in your
environment if they are not already present:

```bash
pip install streamlit
pip install "u8darts[torch]"
```

The main time-series runner is `scripts/benchmark_eur_darts.py`.

## Start Neo4j For AutoFeat

AutoFeat feature discovery stores candidate table relationships in Neo4j.
Darts/time-series experiments do not require Neo4j.

```bash
docker-compose up -d neo4j
# Browser: http://localhost:7474
# Bolt: bolt://localhost:7687, auth disabled
```

You can also use the convenience target:

```bash
make neo4j
make setup
```

## Data Layout

The checked-in EUR layout used by the scripts is:

```text
datasets/EUR/
├── metadata.txt
└── 6907619/
    ├── rabbitmq-performance.csv
    ├── amf-performance.csv
    ├── golang-web-server-performance.csv
    ├── python-web-server-performance.csv
    ├── connections.csv
    └── split_output/
```

Scenario builders write reproducible benchmark folders under `scenarios/`, for
example `scenarios/scenario2c/` and `scenarios/scenarioB_seg01/`. Raw inputs
(`datasets/EUR/`, `datasets/KUL/`) stay separate so a scenario rebuild can never
corrupt the source data.

## Quick Reproduction Path

The fastest end-to-end AutoFeat reproduction is:

```bash
conda activate autofeat-6g
docker-compose up -d neo4j

python scripts/prepare_augmentation_scenarios.py --scenario 2c k

python -m feature_discovery.auto_pipeline \
  --base-table scenarios/scenario2c/rabbitmq-reduced.csv \
  --target lat99 \
  --data-dir scenarios/scenario2c \
  --dataset-type regression \
  --temporal-key time \
  --temporal-tolerance 0 \
  --algorithms XGB \
  --label scenario2c

python -m feature_discovery.auto_pipeline \
  --base-table scenarios/scenarioK_csi/samples_base.csv \
  --target target_x \
  --data-dir scenarios/scenarioK_csi \
  --dataset-type binary \
  --no-transformer-discovery \
  --algorithms XGB \
  --label scenarioK_csi

python scripts/summarize_results.py
```

The same flow is available as:

```bash
make demo
```

Expected outputs:

```text
results/6g_data/auto_pipeline_scenario2c.csv
results/6g_data/auto_pipeline_scenarioK_csi.csv
results/6g_data/SUMMARY.md
results/6g_data/summary.csv
```

### Make Targets

The `Makefile` wraps the common entry points (each runs inside the
`autofeat-6g` conda env via `conda run`):

| Target | What it does |
|---|---|
| `make setup` | Verify the conda env, core imports, and Neo4j reachability |
| `make neo4j` | Start the Neo4j container and wait for the bolt port |
| `make demo` | Run the two showcase scenarios (2c + KUL) and regenerate the summary |
| `make smoke` | Run `scripts/smoke_test.py`; fails non-zero on an R²/accuracy regression |
| `make summary` | Regenerate `results/6g_data/SUMMARY.md` from saved runs |
| `make dashboard` | Launch the Streamlit dashboard on port 8501 |
| `make reset-graph` | Wipe the Neo4j graph (use between unrelated runs) |
| `make clean` | Remove `auto_pipeline_*` result files (keeps historical runs) |

## Track A — AutoFeat Feature Discovery

This track answers: **does discovering and selecting joined tabular features
improve a downstream tabular model?**

### Build Benchmark Scenarios

```bash
python scripts/prepare_augmentation_scenarios.py --all
```

Available scenario keys (positive / negative / ambiguous):

| Key | Scenario | Class | Purpose |
|---|---|---|---|
| `2c` | `scenario2c` | 🟢 positive | Feature recovery via exact `time` (lake stripped of `lat*`/`min` — proxy-free) |
| `r`  | `scenarioR_resource` | 🟢 positive | Cross-app resource-contention augmentation; lake has only `(time, ram_usage, cpu_usage)` per peer service |
| `k`  | `scenarioK_csi` | 🟢 positive | Wide MaMIMO CSI lake (16 antenna tables, `user_id`/`sample_id` dropped) — discovery + compression |
| `1` | `scenario1` | 🔴 negative | Per-`n` aggregated cross-app lake; joined columns are `f(n)` only, no new info |
| `n` | `scenarioN_target_n` | 🔴 negative | Inverse target `n`; cross-app `n` matches rabbitmq's on ~3.5 % → refusal |
| `u` | `scenarioU_unrelated` | 🔴 negative | Heterogeneous unrelated lake (rabbitmq base + KUL CSI lake); no shared key |
| `a_lat95` | `scenarioA_lat95` | 🟡 ambiguous | Cross-app temporal (asof); lakes stripped of `lat*`/`min`/`mean` |
| `a_lat99` | `scenarioA_lat99` | 🟡 ambiguous | Same as `a_lat95` with target `lat99` |
| `b` | `scenarioB_seg01` | 🟡 ambiguous | Within-AMF segments via `n`; lakes stripped of `lat*`/`min`/`mean` |

### Run One Scenario

```bash
python -m feature_discovery.auto_pipeline \
  --base-table scenarios/scenario2c/rabbitmq-reduced.csv \
  --target lat99 \
  --data-dir scenarios/scenario2c \
  --dataset-type regression \
  --temporal-key time \
  --temporal-tolerance 0 \
  --algorithms XGB \
  --top-k 15 \
  --value-ratio 0.65 \
  --label scenario2c
```

Rows written by `auto_pipeline` include:

- `BASE`: base-table features only.
- `Join_All_BFS`: every BFS-reachable joined feature.
- `Join_All_BFS_Filter`: joined features after a relevance filter.
- `AutoFeat`: BFS plus relevance/redundancy feature selection.

Useful options:

```text
--no-transformer-discovery   use only explicit connections.csv edges
--no-ingest                  reuse the current Neo4j graph
--temporal-direction backward
                              avoid look-ahead for forecasting-style joins
--algorithms XGB,RF,KNN       evaluate multiple AutoGluon model families
--seed 42                     global random seed (default 42)
```

### Reproducibility

`auto_pipeline` is deterministic: a given `--seed` (default `42`) produces
bit-for-bit identical metrics across runs. The seed drives the train/test split,
per-model RNGs, and group sampling, and the pipeline pins `PYTHONHASHSEED` (by
re-exec'ing itself once) so `set`/`dict` ordering — which feature-selection
tie-breaks depend on — is also fixed. Override globally with `--seed N` or the
`AUTOFEAT_SEED` environment variable. Note that changing the seed changes the
split and therefore the absolute metrics; compare approaches only within one seed.

### Summarize And Plot AutoFeat Results

```bash
python scripts/summarize_results.py

python scripts/plot_base_autofeat_curves.py \
  --input results/6g_data/EUR/6907619_autofeat_plus_local.csv \
  --output-dir results/6g_data/figures/base_vs_autofeat
```

Important outputs:

```text
results/6g_data/SUMMARY.md
results/6g_data/summary.csv
results/6g_data/figures/base_vs_autofeat/base_vs_autofeat_curves_data.csv
results/6g_data/figures/base_vs_autofeat/XGBoost_r2_curve.png
```

## Track B — Time-Series Augmentation

This track answers: **does train-only time-series augmentation improve
forecasting on unchanged real test windows?**

It should be interpreted separately from AutoFeat feature discovery.

### Run A Darts Benchmark

```bash
python scripts/benchmark_eur_darts.py \
  --dataset rabbitmq \
  --task forecasting \
  --target-column lat99 \
  --models naive_drift linear_regression random_forest \
  --augmentation-method none scaling magnitude_mask \
  --window-size 32 \
  --stride 8 \
  --horizon 1 \
  --seed 42 \
  --output results/6g_data/darts/evaluation_summary.csv
```

Supported datasets:

```text
rabbitmq  amf  golang_web  python_web
```

Supported augmentation methods:

```text
none  jitter  scaling  time_mask  magnitude_mask
```

The runner writes:

```text
results/6g_data/darts/evaluation_summary.csv
results/6g_data/darts/predictions/*.npz
results/6g_data/darts/metadata/*.json
results/6g_data/darts/profiles/*_profile.csv
```

To forecast every selected non-time numeric feature:

```bash
python scripts/benchmark_eur_darts.py \
  --dataset rabbitmq \
  --task forecasting \
  --target-columns all_features \
  --models naive_drift linear_regression \
  --augmentation-method none scaling \
  --window-size 32 \
  --stride 8 \
  --horizon 1 \
  --output results/6g_data/darts/evaluation_summary.csv
```

## Track C — Bridge / Downstream Utility

This exploratory track asks: **if time-series-style augmentation is applied only
to training rows, does a downstream tabular application improve?**

These rows are bridge evidence, not the primary evaluation for AutoFeat or for
time-series augmentation.

```bash
python scripts/benchmark_eur_ts_augmented_downstream.py \
  --dataset rabbitmq \
  --variants BASE AutoFeatPlus_Local \
  --models ridge rf \
  --augmentation-method none scaling magnitude_mask \
  --target-column lat99 \
  --split-mode time \
  --output results/6g_data/downstream/ts_augmented_downstream.csv
```

Outputs:

```text
results/6g_data/downstream/ts_augmented_downstream.csv
results/6g_data/downstream/metadata/*.json
```

The implementation keeps the real test period unchanged and augments only the
training slice.

## Relationship Discovery And Base-Table Reports

To inspect likely join keys and use cases for a base table:

```bash
python scripts/run_base_table_pipeline.py \
  --data-dir datasets/EUR/6907619 \
  --metadata datasets/EUR/metadata.txt \
  --base-table amf-performance.csv \
  --target-column lat99 \
  --output-dir results/6g_data/base_table_pipeline/6907619/amf-performance
```

Example outputs:

```text
relationship_report.md
benchmark_plan.txt
candidate_relationships.csv
recommended_connections.csv
use_cases.md
```

## Dashboard

A single Streamlit app covers four views:

```bash
streamlit run dashboards/augmentation_dashboard.py
# or:  make dashboard
```

| Tab | What it shows |
|---|---|
| **Results** | Per-scenario `BASE / Join_All_BFS / Filter / AutoFeat` rows with picked algorithm; per-row feature-importance chart |
| **Compare** | Cross-scenario pivot (`scenario × approach × algorithm`), enriched with `scenarios/scenarios.yaml` context |
| **Run on your data** | Upload a base table + lake CSVs (+ optional `connections.csv` and `metadata.txt`) and trigger the pipeline |
| **Graph** | Inline Graphviz view of the live Neo4j graph **or** any picked scenario's `connections*.csv` from disk; degree-weighted node sizes, weight-thresholded edge filters, top-table panel |

If `streamlit` is missing, install it in the active environment:

```bash
pip install streamlit
```

## Reproducing Saved Figures And Tables

From existing result CSVs:

```bash
python scripts/summarize_results.py

python scripts/summarize_autofeat_plus_results.py \
  --output results/6g_data/autofeat_plus_comparison_summary.csv

python scripts/plot_autofeat_plus_results.py \
  --summary results/6g_data/autofeat_plus_comparison_summary.csv \
  --output-dir results/6g_data/figures

python scripts/plot_base_autofeat_curves.py
```

## Current Findings To Expect

AutoFeat feature discovery — every approach is evaluated on the **same**
chronological 80/20 test window when `--temporal-key` is set, so the deltas
below are like-for-like. The 9 scenarios are explicitly tagged 🟢 positive /
🔴 negative / 🟡 ambiguous in [scenarios/scenarios.yaml](scenarios/scenarios.yaml).

XGBoost (representative snapshot — re-run with `make demo` or `bash scripts/run_all_scenarios.sh`):

| Class | Scenario | BASE | AutoFeat | Δ |
|---|---|---|---|---|
| 🟢 P | `scenario2c` (feature recovery, proxy-free lake) | 0.5395 | 0.9417 | **+0.402** |
| 🟢 P | `scenarioR_resource` (cross-app contention, no target overlap) | 0.5395 | 0.9724 | **+0.433** |
| 🟢 P | `scenarioK_csi` (wide-lake compression, identity-free) | 0.0000 | 1.0000 | **+1.000** |
| 🔴 N | `scenario1` (per-`n` aggregated, structurally degenerate) | 0.9662 | 0.9662 | 0.000 |
| 🔴 N | `scenarioN_target_n` (inverse target) | 0.9786 | 0.9794 | +0.001 |
| 🔴 N | `scenarioU_unrelated` (heterogeneous lake — KUL → rabbitmq) | 0.9508 | 0.9765 | +0.026† |
| 🟡 A | `scenarioA_lat95` (cross-app temporal, lake `lat*` stripped) | 0.9551 | 0.9796 | +0.025 |
| 🟡 A | `scenarioA_lat99` (cross-app temporal, lake `lat*` stripped) | 0.9508 | 0.9765 | +0.026 |
| 🟡 A | `scenarioB_amf_seg01` (within-app segment pooling) | 0.9412 | 0.9162 | −0.025 |

† scenarioU's +0.026 is **intra-base feature selection**, not lake augmentation —
`Join_All_BFS` is identical to BASE (no lake features added). The cleanest
single piece of evidence that AutoFeat's selection logic also helps on the
base table alone.

What this tells us:
- **AutoFeat lifts where lake information genuinely exists** (2C: missing
  runtime; R: cross-app resource contention; K_csi: wide CSI lake).
- **AutoFeat refuses cleanly** when there's nothing to add (scenario1's per-`n`
  collapse to `f(n)`; scenarioN's cross-app `n` that agrees on only 3.5 % of
  matched rows; scenarioU's heterogeneous KUL lake).
- **Ambiguous lifts are small** (~±0.03) — within the noise of feature
  selection on the joined frame.

Time-series augmentation:

- RabbitMQ forecasting is the most augmentation-sensitive in the current Darts
  results.
- AMF is already strong without augmentation and current classic transforms can
  slightly hurt.
- Synthetic realism metrics near discriminative accuracy `0.5` mean generated
  windows are not trivially separable, but realism alone does not guarantee
  downstream utility.

Bridge utility:

- The downstream smoke result currently shows that augmentation can help Darts
  forecasting while still hurting a downstream tabular Ridge task. This is why
  the dashboard and README keep the method families separate.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: feature_discovery` | `src/` not on Python path | `pip install -r requirements.txt` then prefix commands with `PYTHONPATH=src` (or `export PYTHONPATH=src`). `pytest` works without it thanks to root `conftest.py` |
| AutoGluon trains no models / `pkg_resources` error | `setuptools` too new | `pip install --no-deps "setuptools<81"` |
| `ModuleNotFoundError: tf_keras` during transformer discovery | `transformers` library tried to load its TF backend | Already pinned via `USE_TF=0` in `transformer_discovery.py`; if you call sentence-transformers from elsewhere, set the env var first |
| `Unable to retrieve routing information` | Neo4j routing URI against local single instance | Use `bolt://localhost:7687`; start with `docker-compose up -d neo4j` |
| `Neo4j is already running (pid:7)` on container start | Stale `neo4j.pid` in the mounted volume after an ungraceful exit | `docker-compose down`, delete `neo4j.pid` under the data volume, then `docker-compose up -d neo4j` |
| `streamlit: command not found` | Dashboard dependency missing | `pip install streamlit` in the active env |
| Darts import missing | Time-series dependency missing | `pip install "u8darts[torch]"` |
| Slow transformer discovery | Large lake or wide tables | Use `--no-transformer-discovery` when `connections.csv` is available |
| `AutoFeat` row reproducible but `Join_All_BFS_Filter` row missing | `dt` (string) column in the lake; AutoGluon's feature generator synthesised `dt.day` / `dt.dayofweek` that downstream indexing then couldn't find | Already guarded in `evaluation_algorithms.py`; for your own scenarios drop string-datetime columns up front (`time` integer is enough) |

## Further Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): AutoFeat pipeline architecture.
- [docs/6g_dataset_setup.md](docs/6g_dataset_setup.md): raw 6G data setup.
- [docs/USING_OWN_DATA.md](docs/USING_OWN_DATA.md): adapting the pipeline to your
  own tables.
- [docs/POC_PITCH.md](docs/POC_PITCH.md): proof-of-concept framing and motivation.
- [results/6g_data/SUMMARY.md](results/6g_data/SUMMARY.md):
  saved AutoFeat scenario summary, regenerated by `scripts/summarize_results.py`.

## License

See [LICENSE](LICENSE).
