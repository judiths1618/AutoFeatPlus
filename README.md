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

## Repository Map

```text
src/feature_discovery/
├── auto_pipeline.py                     # end-to-end AutoFeat CLI
├── autofeat_pipeline/                   # BFS joins + mRMR feature selection
├── dataset_relation_graph/              # metadata/transformer relationship discovery
├── experiments/                         # baselines, AutoFeatPlus, evaluation helpers
└── pipelines/base_table_pipeline.py     # base-table relationship planning helpers

scripts/
├── prepare_augmentation_scenarios.py    # builds reproducible AutoFeat scenarios
├── benchmark_eur_darts.py               # Darts time-series augmentation benchmark
├── benchmark_eur_ts_augmented_downstream.py
│                                         # exploratory downstream utility check
├── benchmark_eur_autofeat_plus_local.py # local AutoFeatPlus benchmark
├── summarize_results.py                 # AutoFeat scenario summary
├── summarize_autofeat_plus_results.py   # AutoFeatPlus summary
├── plot_base_autofeat_curves.py         # BASE vs AutoFeat plots
└── run_base_table_pipeline.py           # relationship/use-case reports

dashboards/
├── augmentation_dashboard.py            # AutoFeat feature discovery dashboard
└── eur_time_series_dashboard.py         # separated TS / AutoFeat / bridge views

datasets/
├── EUR/6907619/                         # 6G EUR telemetry tables
├── scenario*/                           # generated AutoFeat benchmark scenarios
└── KUL/                                 # MaMIMO CSI data layouts

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

pip install -e .
pip install --no-deps "setuptools<81"
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

Scenario builders write reproducible benchmark folders under `datasets/`, for
example `datasets/scenario2c/` and `datasets/scenarioB_seg01/`.

## Quick Reproduction Path

The fastest end-to-end AutoFeat reproduction is:

```bash
conda activate autofeat-6g
docker-compose up -d neo4j

python scripts/prepare_augmentation_scenarios.py --scenario 2c k

python -m feature_discovery.auto_pipeline \
  --base-table datasets/scenario2c/rabbitmq-reduced.csv \
  --target lat99 \
  --data-dir datasets/scenario2c \
  --dataset-type regression \
  --temporal-key time \
  --temporal-tolerance 0 \
  --algorithms XGB \
  --label scenario2c

python -m feature_discovery.auto_pipeline \
  --base-table datasets/scenarioK_kul/samples_base.csv \
  --target target_x \
  --data-dir datasets/scenarioK_kul \
  --dataset-type binary \
  --no-transformer-discovery \
  --algorithms XGB \
  --label scenarioK_kul

python scripts/summarize_results.py
```

The same flow is available as:

```bash
make demo
```

Expected outputs:

```text
results/6g_data/auto_pipeline_scenario2c.csv
results/6g_data/auto_pipeline_scenarioK_kul.csv
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

Available scenario keys:

| Key | Scenario | Purpose |
|---|---|---|
| `1` | `scenario1` | Cross-service workload augmentation via `n`; should often refuse |
| `2c` | `scenario2c` | Feature recovery via exact `time`; should discover useful features |
| `a_lat95` | `scenarioA_lat95` | Cross-application temporal joins for `lat95`; control/refusal case |
| `a_lat99` | `scenarioA_lat99` | Cross-application temporal joins for `lat99`; control/refusal case |
| `b` | `scenarioB_seg01` | Within-AMF segments via `n`; partial/weak-signal case |
| `n` | `scenarioN_target_n` | Inverse target `n`; control/refusal case |
| `k` | `scenarioK_kul` | MaMIMO CSI feature compression; should discover |
| `k_csi` | `scenarioK_csi` | Larger CSI layout built from raw CSI-as-features files |

### Run One Scenario

```bash
python -m feature_discovery.auto_pipeline \
  --base-table datasets/scenario2c/rabbitmq-reduced.csv \
  --target lat99 \
  --data-dir datasets/scenario2c \
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

## Dashboards

AutoFeat feature discovery dashboard:

```bash
streamlit run dashboards/augmentation_dashboard.py
```

EUR evaluation dashboard with separated tabs for time-series augmentation,
AutoFeat feature discovery, and bridge utility:

```bash
streamlit run dashboards/eur_time_series_dashboard.py
```

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

AutoFeat feature discovery (every approach evaluated on the same chronological
test window when `--temporal-key` is set, so the deltas are like-for-like):

- `scenario2c` is the clean feature-recovery case; recovering the RabbitMQ
  runtime features through `time` lifts R² from a weak BASE (~0.54 on the future
  window) to ~0.99 — the strongest showcase.
- `scenarioK_kul` / `scenarioK_csi` are the high-dimensional CSI cases; AutoFeat
  compresses a wide antenna lake into ~15 features at ~1.0 accuracy, while the
  information-poor BASE (key + target only) cannot train.
- `scenarioA_lat95` / `scenarioA_lat99` give AutoFeat a modest lift (~+0.025)
  once BASE and AutoFeat share the temporal split — cross-app timing telemetry
  helps a little (these were previously read as pure refusal cases).
- `scenario1` (cross-service workload via `n`) and `scenarioN_target_n` (inverse
  target) remain refusal/control cases: extra joins add ~0. `scenarioB` is a
  weak-signal case where mRMR can under-select and slightly hurt.

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
| `ModuleNotFoundError: feature_discovery` | Package not installed | Run `pip install -e .` or prefix commands with `PYTHONPATH=src` |
| AutoGluon trains no models / `pkg_resources` error | `setuptools` too new | `pip install --no-deps "setuptools<81"` |
| `Unable to retrieve routing information` | Neo4j routing URI against local single instance | Use `bolt://localhost:7687`; start with `docker-compose up -d neo4j` |
| `streamlit: command not found` | Dashboard dependency missing | `pip install streamlit` in the active env |
| Darts import missing | Time-series dependency missing | `pip install "u8darts[torch]"` |
| Slow transformer discovery | Large lake or wide tables | Use `--no-transformer-discovery` when `connections.csv` is available |

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
