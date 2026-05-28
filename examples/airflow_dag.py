"""
Airflow DAG example — AutoFeat as one task in an ELT pipeline.

Plumbing:
  extract       — read source tables from a warehouse (Snowflake, BigQuery, ...)
  augment       — run AutoFeat to discover + select joined features
  load_features — push the augmented DataFrame to a feature store / table
  train         — train a downstream model (off-DAG handoff)

Save under `dags/` of your Airflow deployment. The Airflow worker must have:
  - the project's dependencies installed (`pip install -r requirements.txt`) with `src/` on `PYTHONPATH`
  - Neo4j reachable at the location set in `feature_discovery.config.NEO4J_HOST`
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator


# ─── Tasks ──────────────────────────────────────────────────────────────────
def extract(ti, **_):
    """Pull source data. Replace these reads with whatever your warehouse hooks
    look like; for example: SnowflakeHook(...).get_pandas_df(sql)."""
    base = pd.read_csv("/data/raw/customers.csv")
    transactions = pd.read_csv("/data/raw/transactions.csv")
    sessions = pd.read_csv("/data/raw/sessions.csv")

    # Hand DataFrames to the next task via XCom-friendly parquet paths.
    out = Path("/data/tmp/extract"); out.mkdir(parents=True, exist_ok=True)
    base.to_parquet(out / "customers.parquet")
    transactions.to_parquet(out / "transactions.parquet")
    sessions.to_parquet(out / "sessions.parquet")

    ti.xcom_push(key="base_path", value=str(out / "customers.parquet"))
    ti.xcom_push(key="lake_paths", value=[
        str(out / "transactions.parquet"),
        str(out / "sessions.parquet"),
    ])


def augment(ti, **_):
    """The AutoFeat step. Drop-in replacement for any feature-engineering task."""
    from feature_discovery.augmentation import augment_features

    base = pd.read_parquet(ti.xcom_pull(task_ids="extract", key="base_path"))
    lake = {
        Path(p).stem: pd.read_parquet(p)
        for p in ti.xcom_pull(task_ids="extract", key="lake_paths")
    }

    result = augment_features(
        base=base,
        lake=lake,
        target="churned",
        dataset_type="binary",
        temporal_key="event_time",
        temporal_tolerance="5min",
        temporal_direction="backward",   # forecasting — no look-ahead leakage
        algorithms=["XGB"],
        label=f"churn_run_{ti.execution_date:%Y%m%d_%H}",
    )

    # Persist the augmented DataFrame for the load step.
    out_path = Path(f"/data/tmp/augmented/{ti.dag_run.run_id}.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.augmented_df.to_parquet(out_path)

    # Push metrics + path to XCom for downstream tasks and observability.
    ti.xcom_push(key="augmented_path", value=str(out_path))
    ti.xcom_push(key="metrics", value=result.to_dict())

    # Soft assertion: skip the load step if AutoFeat didn't actually help.
    if (result.lift or 0) < 0.005:
        raise ValueError(
            f"Augmentation lift below threshold "
            f"(base={result.base_accuracy:.3f}, autofeat={result.autofeat_accuracy:.3f}). "
            "Not loading degraded features."
        )


def load_features(ti, **_):
    """Push to a feature store / warehouse table."""
    path = ti.xcom_pull(task_ids="augment", key="augmented_path")
    df = pd.read_parquet(path)
    # Example: FeastFeatureStore().push("churn_features", df)
    # Example: SnowflakeHook(...).insert_rows("ANALYTICS.CHURN_FEATURES", df.values.tolist())
    print(f"Loaded {len(df)} rows × {df.shape[1]} columns from {path}")


# ─── DAG ────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="autofeat_churn_features",
    start_date=datetime(2026, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    default_args={
        "owner": "platform",
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
    },
    tags=["features", "autofeat"],
) as dag:
    t_extract = PythonOperator(task_id="extract",       python_callable=extract)
    t_augment = PythonOperator(task_id="augment",       python_callable=augment)
    t_load    = PythonOperator(task_id="load_features", python_callable=load_features)

    t_extract >> t_augment >> t_load
