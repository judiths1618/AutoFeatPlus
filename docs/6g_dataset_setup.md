# Using the 6G testbed datasets with AutoFeat

The 6G testbed release contains four independent CSV tables (`golang-web-server-performance.csv`, `python-web-server-performance.csv`, `rabbitmq-performance.csv`, and `amf-performance.csv`). The datasets ship without declared primary/foreign key relations, so AutoFeat will not automatically generate join-path features unless we either hand craft a `connections.csv` file or let the built-in discovery pipeline infer candidate relationships with Valentine. This guide shows how to register the data lake metadata and drive the discovery workflow when you only have the raw CSV files.

## 1. Organise the files on disk

1. Create a directory under `data/6g_testbed_dataset` that follows the Zenodo download structure. For example:
   ```text
   data/6g_testbed_dataset/
   └── EUR/
       └── 6907619/
           ├── amf-performance.csv
           ├── golang-web-server-performance.csv
           ├── python-web-server-performance.csv
           └── rabbitmq-performance.csv
   ```
2. (Optional) If you know the exact key relationships, add a `connections.csv` inside the same directory. Each row should describe a pair of tables and the matching columns (`pk_table`, `pk_column`, `fk_table`, `fk_column`). When the file is absent the CLI can fall back to automated discovery.

## 2. Describe the base table in `datasets.csv`

Create `data/6g_testbed_dataset/datasets.csv` with the usual schema:

```csv
base_table_path,base_table_name,base_table_label,target_column,dataset_type
EUR/6907619/rabbitmq-performance.csv,rabbitmq-performance.csv,EUR/6907619,label,binary
```

Key points:

- `base_table_path` is the path to the base CSV relative to `data/<DATASET_TYPE>`.
- `base_table_name` is only the file name and must match the actual CSV file.
- `base_table_label` is the identifier you will pass to the CLI (`--dataset-label`). Re-using the folder (`EUR/6907619`) keeps discovery scoped to the right subtree.
- `target_column` is the supervised learning label in the base table. Replace `label` with the real target column from your dataset (for example, `lat99` if you create a regression target).
- `dataset_type` controls which AutoML heads run (`binary`, `multiclass`, or `regression`). Pick the appropriate value for your task.

If you want to model multiple tables from the same download separately, add one row per base table.

## 3. Point AutoFeat to the 6G data directory

`feature_discovery.config` now honours an environment variable called `DATASET_TYPE`. Export it before launching any CLI command so the runtime looks under `data/6g_testbed_dataset` instead of the default `data/benchmark`:

```bash
export DATASET_TYPE=6g_testbed_dataset
```

When you connect to a long-running shell (for example inside the Docker container) you only need to export the variable once per session. The same variable is also used as the default Neo4j database name unless you override `NEO4J_DATABASE`.

## 4. Ingest nodes and discover relationships

With the environment set, ingest the tables and trigger relationship discovery. There are two typical flows:

### A. Work with a single dataset folder

```bash
feature-discovery-cli ingest-kfk-data --dataset-label EUR/6907619 --discover-connections-dataset
```

- The command loads every CSV file underneath `data/6g_testbed_dataset/EUR/6907619/`.
- When `--discover-connections-dataset` is present, Valentine searches for attribute matches among the tables in that folder (default similarity threshold `0.55`).
- If the folder only contains one table, the CLI still loads it, but no joins are produced because there are no other tables to connect to.

### B. Treat the 6G tables as part of a larger data lake

```bash
feature-discovery-cli ingest-data --discover-connections-data-lake --data-discovery-threshold 0.55
```

- `ingest-data` first creates graph nodes for every CSV in the configured `DATA_FOLDER`.
- Setting both flags enables Valentine across all tables (adjust the threshold if you want stricter or more permissive matches).

After either ingestion flow finishes, you can run the standard AutoFeat commands (`run-all`, `run-arda`, etc.) and pass `--dataset-label EUR/6907619` to restrict experiments to your 6G base table.

## 5. Understanding the limitations

- Without real key relationships AutoFeat cannot create meaningful join paths. Discovery helps, but you should inspect the suggested matches in Neo4j Studio or via Cypher queries before trusting them in production experiments.
- A single-table dataset will only benefit from AutoFeat's unary transformations. To exploit join-path features you need at least two tables with relatable columns.
- The Valentine-based workflow is heuristic. If you already know the correct join columns, encoding them in `connections.csv` gives you deterministic results and faster ingestion.

Following these steps lets you experiment with the 6G testbed release while keeping metadata and ingestion consistent with the benchmark datasets.
