# Join-Path-based Data Augmentation — 6G-DALI Feature Discovery and Augmentation Scenarios

This repository is a **methodology-hardened, 5G/6G-tailored fork of the
original AutoFeat join-path discovery system**
([*"AutoFeat: Transitive Feature Discovery over Join Paths"*](docs/assets/papers/ICDE_FeatureDiscovery.pdf),
ICDE). The original BFS-discovery + mRMR-selection core in
[`src/feature_discovery/autofeat_pipeline/`](src/feature_discovery/autofeat_pipeline/)
is preserved verbatim; everything around it has been re-targeted at real
5G/6G telemetry workloads, made reproducible, and stress-tested against the
methodological footguns that affected the published results.

It answers two distinct questions on 5G/6G telemetry data:

1. **AutoFeat feature discovery on telemetry tables** — can a base table from a
   5G/6G core-network function or cloud-side microservice be improved by
   discovering joinable peer tables and selecting useful external tabular
   features?
2. **Time-series augmentation on the same telemetry** — can train-only
   time-series transformations improve forecasting on real, unchanged test
   windows?

These are different method families. AutoFeat / AutoFeatPlus widens a tabular
feature matrix through joins; time-series augmentation creates additional
training windows or rows. The optional downstream bridge experiments ask
whether time-series-style augmentation helps a later tabular application, but
those rows should not be treated as the primary score for either method family.

### What "tailored to 5G/6G" actually means here

| Aspect | The fork's choice (vs. the original AutoFeat) |
|---|---|
| **Datasets** | Real 5G/6G testbed and wireless measurements — EUR 6907619 microservice latency (rabbitmq / golang-web / python-web / AMF), KUL MaMIMO indoor-localisation CSI (4–16 antennas × 200 subcarriers). The original benchmark used Kaggle-style tabular datasets. |
| **Temporal-first evaluation** | Every approach (`BASE / JOIN_ALL / Filter / AutoFeat / AutoFeatPlus`) is evaluated on the *same* chronological 80/20 window when `--temporal-key` is set, so Δ comparisons are like-for-like under genuine time-series leakage rules — required for any 5G/6G stream. The original used random splits. |
| **`merge_asof` is the default join** | Exact-time matches across cloud microservices are rare under realistic 5G/6G observation cadence; `pd.merge_asof` (with `tolerance` and `direction` properly typed and unit-aware) is the default for any temporal join. The original supported only exact joins. |
| **6G-specific scenario suite** | The 9 benchmark scenarios are explicitly tagged 🟢 positive / 🔴 negative / 🟡 ambiguous and built from the 5G/6G datasets — including `scenarioR_resource` (cross-service resource-contention positive) and `scenarioU_unrelated` (heterogeneous-lake refusal) that the original AutoFeat benchmarks didn't cover. |
| **Operator-friendly privacy policies** | `EUR_POLICY_PRESETS` for AutoFeatPlus targets 5G/6G operator concerns: `time-private`, `resource-private`, `workload-private`, `target-proxy-private`. The original AutoFeatPlus had inline-only patterns scattered across the benchmark script. |

The original AutoFeat paper's README is preserved verbatim at
[docs/legacy/README_AutoFeat_paper.md](docs/legacy/README_AutoFeat_paper.md)
for one-click comparison.

## Improvements over the original AutoFeat

This repository keeps the original BFS-discovery / mRMR feature-selection core
([autofeat_pipeline/](src/feature_discovery/autofeat_pipeline/)) intact but fixes several methodology and reliability issues that affected the published results.
Each row below points at the code that changed.

### Methodology

| Improvement | What was wrong before | Where it's fixed |
|---|---|---|
| **End-to-end reproducibility** — a single `--seed` controls every RNG (train/test split, AutoGluon model seeds, group sampling, feature-selection tie-breaks). `PYTHONHASHSEED` is pinned by a one-shot re-exec so `set`/`dict` ordering is also stable. | The original pipeline used a mix of hard-coded `random_state=10` / `42` and AutoGluon's default seeding — and `set`/`dict` ordering left feature-selection tie-breaks **non-deterministic** even with all RNGs seeded. Two consecutive runs gave different `AutoFeat` numbers. | [`auto_pipeline._ensure_reproducible`](src/feature_discovery/auto_pipeline.py), `config.SEED`, every `random_state=` routed through it |
| **Like-for-like evaluation split** — when `--temporal-key` is set, BASE, JOIN_ALL_BFS, JOIN_ALL_BFS_Filter and AutoFeat all use the same chronological 80/20 split. | Original code only passed `time_column` into AutoFeat's evaluator, so BASE used a *random* split and AutoFeat used a *temporal* split — the headline `Δ AutoFeat − BASE` compared two different test sets. | [`baselines._resolve_time_column`](src/feature_discovery/experiments/baselines.py), `time_column=dataset.temporal_key` threaded through |
| **Non-redundant model comparison** — when `--algorithms XGB` is requested the runner trains **XGBoost + RandomForest** (companion model) instead of XGBoost + AutoGluon's degenerate `WeightedEnsemble_L2` (which for a single base model just duplicates it). | Every result row appeared twice with identical numbers under different algorithm names. | [`evaluation_algorithms._with_companion`](src/feature_discovery/experiments/evaluation_algorithms.py), `fit_weighted_ensemble=False` |
| **Honest scenario lakes** — every benchmark scenario lake is stripped of target-percentile siblings (`lat50/75/95/min`, `mean`) and identity columns (`user_id`, `sample_id`) before evaluation. | The original showcases harvested target proxies — e.g. scenario 2C's "feature recovery" lifted R² by using `lat95` to predict `lat99`; KUL's "compression test" used `user_id` to predict `target_x`. Both inflated the showcase numbers. | [`prepare_augmentation_scenarios._strip_target_proxies`](scripts/prepare_augmentation_scenarios.py), `build_scenarioK` drops identity columns |
| **Three-class scenario taxonomy** — the 9 scenarios are now explicitly tagged 🟢 positive / 🔴 negative / 🟡 ambiguous in the manifest, with two new entries (`scenarioR_resource`, `scenarioU_unrelated`) designed specifically to test honest cross-app augmentation and heterogeneous-lake refusal. | The original "expected_behaviour" labels (refuse/showcase) were inconsistent — some "refuses" actually had partial signal, some "showcases" lifted via target proxies. | [scenarios/scenarios.yaml](scenarios/scenarios.yaml) |
| **AutoFeatPlus as a fifth approach in the headline pipeline** — `auto_pipeline` now emits a policy-aware `AutoFeatPlus` row alongside `BASE / JOIN_ALL / Filter / AutoFeat`. Same temporal split, same seed, deterministic Spearman+stable-sort selection, configurable via `--autofeat-plus-policy`. The dashboard's Method picker visualises which tables each method actually consumed. | The original AutoFeatPlus lived in a separate `benchmark_eur_autofeat_plus_local.py` script with its own random split, no shared seed, and policy presets duplicated as inline constants — making "AutoFeat vs AutoFeatPlus" impossible to read off the same SUMMARY row. | [`baselines.join_all_bfs`](src/feature_discovery/experiments/baselines.py) (third pass), [`autofeat_plus.EUR_POLICY_PRESETS`](src/feature_discovery/experiments/autofeat_plus.py) (single source of truth) |
| **Correct `n_features` for every approach** — every `Result` written to the SUMMARY now carries `len(join_path_features)`. The dashboard's `Compare → Feature counts` panel surfaces the cross-method compression story (e.g. K_csi: `0 → 3 200 → 1 600 → 73 → 15`); the `Results → Feature importance drilldown` distinguishes "model trained on 0 features" from "AutoGluon importance crash" via the count. | The original `Result` dataclass defaulted `n_features = 0`; only the AutoFeat path bothered to set it. BASE / JOIN_ALL / Filter / AutoFeatPlus rows all shipped a literal `0`, so the SUMMARY's `n_features` column lied for every row except AutoFeat. | [`run_auto_gluon`](src/feature_discovery/experiments/evaluation_algorithms.py) sets `n_features=len(join_path_features)` on every Result it builds |

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

scripts/                                 # selected entry points — `ls scripts/` for the full ~20
├── prepare_augmentation_scenarios.py    # builds the 9 benchmark scenarios into scenarios/
├── split_eur_by_time_gaps.py            # segments EUR telemetry CSVs at large time gaps
├── run_all_scenarios.sh                 # one-shot 9-scenario benchmark runner
├── smoke_test.py                        # `make smoke` — accuracy-floor regression check
├── benchmark_eur_darts.py               # Track B — Darts forecasting × augmentation
├── benchmark_eur_ts_augmented_downstream.py
│                                         # Track C — bridge / downstream utility
├── benchmark_eur_autofeat_plus_local.py # standalone AutoFeatPlus benchmark (pre-pipeline)
├── benchmark_kul_local.py               # KUL-only local benchmark (no Neo4j)
├── summarize_results.py                 # AutoFeat scenario summary
├── summarize_autofeat_plus_results.py   # AutoFeatPlus summary
├── plot_base_autofeat_curves.py         # BASE vs AutoFeat plots
├── plot_autofeat_plus_results.py        # AutoFeatPlus comparison plots
├── run_base_table_pipeline.py           # relationship/use-case reports
└── …                                    # diagnose, infer_dataset_relationships, etc.

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
├── auto_pipeline_<label>.csv            # raw per-scenario rows (one per approach × algorithm)
├── auto_pipeline_<label>_summary.csv    # the same rows, slim columns
├── auto_pipeline_<label>_features.csv   # long-format feature importance
├── augmented_datasets/                  # AutoFeat-persisted joined frames
├── darts/evaluation_summary.csv         # Track B Darts forecasting × augmentation
├── downstream/ts_augmented_downstream.csv  # Track C bridge utility
├── run_logs/                            # per-scenario stdout/stderr
├── SUMMARY.md  +  summary.csv           # cross-scenario rollup (`summarize_results.py`)
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
| `make demo` | Run the two showcase scenarios (`scenario2c` feature-recovery + `scenarioK_csi` 16-antenna MaMIMO compression) and regenerate the summary |
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

Rows written by `auto_pipeline` (every approach evaluated on the same
chronological 80/20 split when `--temporal-key` is set):

- `BASE`: base-table features only — no augmentation.
- `Join_All_BFS`: every BFS-reachable joined feature.
- `Join_All_BFS_Filter`: joined features after a Spearman top-half filter.
- `AutoFeat`: BFS plus mRMR relevance/redundancy feature selection.
- `AutoFeatPlus`: BFS plus policy-aware utility selection (Spearman utility
  minus privacy/missing-ratio/cost penalties). Default policy is
  `target-proxy-private` — strips sibling-percentile leakers (`lat*`, `min`,
  `mean`) without blocking the temporal key. Configure via
  `--autofeat-plus-policy {time-private,resource-private,workload-private,target-proxy-private,all,none}`
  and `--autofeat-plus-top-k <int>`.

Useful options:

```text
--no-transformer-discovery   use only explicit connections.csv edges
--no-ingest                  reuse the current Neo4j graph
--temporal-direction backward
                              avoid look-ahead for forecasting-style joins
--algorithms XGB,RF,KNN       evaluate multiple AutoGluon model families
--seed 42                     global random seed (default 42)
--autofeat-plus-policy        AutoFeatPlus policy preset(s) layered on top of
  target-proxy-private        DEFAULT_SENSITIVE_PATTERNS. Choose any of
                              {time-private, resource-private,
                              workload-private, target-proxy-private,
                              all, none}; default is target-proxy-private.
--autofeat-plus-top-k 15      top-k for the AutoFeatPlus pass
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
| **Results** | Per-scenario `BASE / Join_All_BFS / Filter / AutoFeat / AutoFeatPlus` rows (accuracy + correct `n_features` for every approach, after the source-side fix), plus a **Feature-importance drilldown**: deduplicated Approach selectbox; **Top-N slider** (1 → min(rows, 200)) lets you sweep down K_csi's 3 200-feature Join_All_BFS tail; a **source-table tally** expander shows how many of the top-N came from each lake table; explicit message distinguishes "trained on 0 input features" from "AutoGluon importance crash" |
| **Compare** | Cross-scenario accuracy pivot (`scenario × approach × algorithm`), enriched with `scenarios/scenarios.yaml` context. `Δ AutoFeat−BASE` and `Δ AutoFeatPlus−BASE` both shown. **Feature counts panel** under the accuracy pivot — same shape, with `n_features` per (scenario × approach), backfilled from the per-scenario features CSV for older summaries that have `n_features = 0` |
| **Run on your data** | Upload a base table + lake CSVs (+ optional `connections.csv` and `metadata.txt`) and trigger the pipeline |
| **Graph** | Three pickers: **Source** (`Live Neo4j` or any scenario), **Method** (`All discovered edges` or one of `BASE` / `Join_All_BFS` / `Filter` / `AutoFeat` / `AutoFeatPlus`), **Algorithm** (`XGBoost` / `RandomForest`). Method mode draws **only the tables that method actually used**, sized by feature count, with a side panel listing every selected feature ranked by \|importance\|. Useful for spotting artefacts — on `scenarioR_resource`, `Join_All_BFS` lights up 4 tables (base + 3 peer-service `*-resources.csv`, 8 features), while `AutoFeat` lights up only 1 (base alone, same 4 columns as BASE) — yet AutoFeat's R² is +0.43 above BASE, evidence that the lift is from the trivial-self-join detour in `evaluate_paths`, not from any joined feature |

Time-series augmentation results (Track B + Track C) are written to
`results/6g_data/darts/evaluation_summary.csv` and
`results/6g_data/downstream/ts_augmented_downstream.csv`. They are **not**
surfaced in the dashboard — inspect them directly with `pandas` or your editor.

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

| Class | Scenario | BASE | AutoFeat | AutoFeatPlus | Δ AF | Δ AF+ |
|---|---|---:|---:|---:|---:|---:|
| 🟢 P | `scenario2c` (feature recovery, proxy-free lake) | 0.5395 | 0.9417 | **0.9518** | +0.402 | **+0.412** |
| 🟢 P | `scenarioR_resource` (cross-app contention, no target overlap) | 0.5395 | **0.9724** | 0.5373 | **+0.433** | −0.002 |
| 🟢 P | `scenarioK_csi` (wide-lake compression, identity-free) | 0.0000 | **1.0000** | 0.9948 | **+1.000** | **+0.995** |
| 🔴 N | `scenario1` (per-`n` aggregated, structurally degenerate) | 0.9662 | 0.9662 | 0.9454 | 0.000 | −0.021 |
| 🔴 N | `scenarioN_target_n` (inverse target) | 0.9786 | 0.9794 | 0.9158 | +0.001 | −0.063 |
| 🔴 N | `scenarioU_unrelated` (heterogeneous lake — KUL → rabbitmq) | 0.9508 | 0.9765 | 0.9508 | +0.026† | 0.000 |
| 🟡 A | `scenarioA_lat95` (cross-app temporal, lake `lat*` stripped) | 0.9551 | 0.9796 | 0.9048 | +0.025 | −0.050 |
| 🟡 A | `scenarioA_lat99` (cross-app temporal, lake `lat*` stripped) | 0.9508 | 0.9765 | 0.8845 | +0.026 | −0.066 |
| 🟡 A | `scenarioB_amf_seg01` (within-app segment pooling) | 0.9412 | 0.9162 | 0.7742 | −0.025 | −0.167 |

† scenarioU's +0.026 (AF) and 0.000 (AF+) is **intra-base feature selection**,
not lake augmentation — `Join_All_BFS` is identical to BASE.

What this tells us:
- **AutoFeat lifts where lake information genuinely exists** (2C: missing
  runtime; K_csi: wide CSI lake; B: AMF segment pooling).
- **AutoFeat refuses cleanly** when there's nothing to add (scenario1's per-`n`
  collapse to `f(n)`; scenarioN's cross-app `n` that agrees on only 3.5 % of
  matched rows; scenarioU's heterogeneous KUL lake).
- **Ambiguous lifts are small** (~±0.03) — within the noise of feature
  selection on the joined frame.

### How many features each method actually uses (`n_features`)

XGBoost input-column count (= `len(join_path_features)`, what AutoGluon
trains on; **not** the AutoGluon-importance-dict size, which can fan out under
generator decomposition):

| Class | Scenario | BASE | JOIN_ALL_BFS | Filter | AutoFeat | AutoFeatPlus |
|---|---|---:|---:|---:|---:|---:|
| 🟢 P | scenario2c | 4 | 9 | 5 | **8** | 9 |
| 🟢 P | scenarioR_resource | 4 | 8 | 4 | **4** | 4 |
| 🟢 P | scenarioK_csi | 0 | **3 200** | 1 600 | **73** | **15** |
| 🔴 N | scenario1 | 5 | 28 | 14 | 5 | 10 |
| 🔴 N | scenarioN_target_n | 10 | 446 | 223 | 14 | 15 |
| 🔴 N | scenarioU_unrelated | 6 | 6 | 4 | 6 | 6 |
| 🟡 A | scenarioA_lat95 / lat99 | 6 | 22 | 11 | **6** | 16 |
| 🟡 A | scenarioB_amf_seg01 | 6 | 38 | 20 | 28 | 16 |

Read from this:
- **K_csi shows the cleanest compression funnel** — 3 200 → 1 600 → 73 → 15.
  AutoFeatPlus's `--autofeat-plus-top-k=15` cap is doing the work; one of the
  15 carries the accuracy and the rest are zero-importance ballast.
- **AutoFeat picks 0 lake features in 1 / A_lat95 / A_lat99 / R / U** —
  `n_features(AutoFeat) == n_features(BASE)` and the source-table breakdown
  (Compare-tab "Feature counts" panel in the dashboard) shows AutoFeat used
  only the base table. On the refusal scenarios (1, N, U) that's the correct
  behaviour. On scenarioR_resource it's **surprising** — `Join_All_BFS`
  reaches 3 peer-service tables (8 features), but AutoFeat's mRMR ranks the
  base columns higher and keeps them only. The +0.433 R² lift comes from
  AutoFeat's evaluation path (the trivial-self-join detour through
  `evaluate_paths`), not from new lake features. This is a known artifact
  worth tracking — see the open-issues note in the AutoFeat audit notebook.

### AutoFeat vs AutoFeatPlus — when does each one win?

| Observation | Mechanism |
|---|---|
| **AutoFeatPlus matches or beats AutoFeat when the lake's strongest features are top-utility** (scenario2c +0.412 vs +0.402; K_csi +0.995 vs +1.000) | `score_plus` ranks `ram_usage` / `cpu_usage` / subcarrier features highly on their own merit; no interaction effect is needed |
| **AutoFeat's +0.43 R² lift on `scenarioR_resource` is *not* from selecting lake features** — the `n_features` pivot shows AutoFeat used the same 4 base columns as BASE (1 source table). The lift comes from `evaluate_paths` running a base-only "join" detour that shifts the temporal split's row ordering. AutoFeatPlus doesn't benefit from that artefact and reports the cleaner −0.002 number. | `ablation.autofeat → evaluate_paths` always wraps the chosen path through `pd.merge_asof`/`merge` even when the path is just the base node; the row order in the resulting frame differs from `non_augmented`'s raw read, and that order change propagates through the chronological split |
| **AutoFeatPlus is more conservative on refusal scenarios** (1, N, U deltas all ≤ 0) | The top-k cap + privacy/missing/cost penalties trim the candidate set; fewer features means a tighter fit on already-strong BASEs and a slightly worse number |
| **The honest privacy trade-off lives in the negative deltas on ambiguous cases** (A_lat95 −0.050; A_lat99 −0.066; B −0.167) | What you sacrifice in raw accuracy you buy back in policy guarantees AutoFeat doesn't provide — `pattern_risk` hard-blocks PII/identifier names, `proxy_risk`/`identifier_risk` penalise (but don't ban) data-derived risky columns |

In the **Graph** tab, switch the Method picker between `Join_All_BFS` and
`AutoFeat` for `scenarioR_resource` and you'll see this visually:
`Join_All_BFS` lights up 4 tables (base + 3 peer-service `*-resources.csv`,
8 features); **`AutoFeat` lights up only 1 (base alone, same 4 features as
BASE)**. The +0.433 R² lift is therefore *not* from selecting lake features —
it's from AutoFeat's `evaluate_paths` re-running the join even with a base-
only path, which materially shifts the temporal split's row ordering.
`AutoFeatPlus` shows the same single-table picture but without the lift,
making the artefact easy to spot.

<!-- Time-series augmentation:

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
  the dashboard and README keep the method families separate. -->

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
