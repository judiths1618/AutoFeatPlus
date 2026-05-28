"""
augmentation_dashboard.py
=========================
Streamlit dashboard for the AutoFeat feature-discovery pipeline.

Four tabs:
  1. Results    — per-scenario comparison + feature-importance drilldown
                  (reads `auto_pipeline_<label>_summary.csv` and `_features.csv`).
  2. Compare    — cross-scenario pivot, enriched with `datasets/scenarios.yaml`
                  context (purpose, expected behaviour).
  3. Run        — upload base + lake CSVs, configure target/temporal-key, run the
                  pipeline end-to-end with live log streaming.
  4. Graph      — Neo4j connection state, edge/node counts, canned Cypher
                  queries, and a direct link to the Neo4j Browser UI.

Launch:
    conda activate autofeat-6g
    streamlit run dashboards/augmentation_dashboard.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import altair as alt
import pandas as pd
import streamlit as st

# ─── Paths / config ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "6g_data"
SCENARIOS_YAML = ROOT / "datasets" / "scenarios.yaml"
DEFAULT_NEO4J_HOST = os.environ.get("NEO4J_HOST", "bolt://localhost:7687")
NEO4J_BROWSER_URL = "http://localhost:7474"

APPROACHES = ["BASE", "Join_All_BFS", "Join_All_BFS_Filter", "AutoFeat"]
APPROACH_COLOURS = {
    "BASE": "#888888",
    "Join_All_BFS": "#5ba9ff",
    "Join_All_BFS_Filter": "#9c8bff",
    "AutoFeat": "#ff7f50",
}

st.set_page_config(page_title="AutoFeat Feature Discovery", layout="wide")


# ─── Helpers ─────────────────────────────────────────────────────────────────
# AutoGluon reports the "XGB" hyperparameter key as the model name "XGBoost";
# the trivial no-feature fallback used the bare key. Fold them together so the
# dashboard never shows a standalone "XGB" option. (Maps other AG keys too.)
ALGORITHM_ALIASES = {"XGB": "XGBoost", "GBM": "LightGBM", "RF": "RandomForest", "XT": "ExtraTrees"}


@st.cache_data(show_spinner=False)
def load_summaries() -> pd.DataFrame:
    """Concat every `auto_pipeline_<label>_summary.csv`."""
    rows = []
    for f in sorted(RESULTS_DIR.glob("auto_pipeline_*_summary.csv")):
        try:
            rows.append(pd.read_csv(f))
        except Exception as exc:
            st.warning(f"Could not read {f.name}: {exc}")
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    if "algorithm" in df.columns:
        df["algorithm"] = df["algorithm"].replace(ALGORITHM_ALIASES)
        df = df.drop_duplicates(subset=["scenario", "approach", "algorithm"], keep="first")
    return df


@st.cache_data(show_spinner=False)
def load_features(label: str) -> pd.DataFrame:
    """Long-format feature importance for one scenario."""
    f = RESULTS_DIR / f"auto_pipeline_{label}_features.csv"
    if f.exists():
        return pd.read_csv(f)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_scenario_manifest() -> Dict[str, dict]:
    """Parse datasets/scenarios.yaml (best-effort)."""
    if not SCENARIOS_YAML.exists():
        return {}
    try:
        import yaml
        with SCENARIOS_YAML.open() as f:
            data = yaml.safe_load(f)
        return {s["label"]: s for s in data.get("scenarios", [])}
    except Exception:
        return {}


def neo4j_stats() -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """Return (node_count, rel_count, error_message)."""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(DEFAULT_NEO4J_HOST, auth=None)
        with driver.session() as s:
            nodes = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            rels = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        driver.close()
        return nodes, rels, None
    except Exception as exc:
        return None, None, str(exc)


def run_pipeline(args: List[str], log_box) -> int:
    """Run the auto_pipeline as a subprocess, streaming output to log_box."""
    env = os.environ.copy()
    cmd = [sys.executable, "-m", "feature_discovery.auto_pipeline"] + args
    log_box.text("$ " + " ".join(cmd) + "\n")
    proc = subprocess.Popen(
        cmd, cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    lines: List[str] = []
    for line in iter(proc.stdout.readline, ""):
        lines.append(line.rstrip())
        # Cap to last 500 lines to avoid streamlit overload
        log_box.code("\n".join(lines[-500:]), language="text")
    proc.wait()
    return proc.returncode


# ─── Sidebar nav ─────────────────────────────────────────────────────────────
tab_results, tab_compare, tab_run, tab_graph = st.tabs(
    ["Results", "Compare", "Run on your data", "Graph browser"]
)


# ─── Tab 1: Results ──────────────────────────────────────────────────────────
with tab_results:
    st.header("Pipeline results")
    summary = load_summaries()
    manifest = load_scenario_manifest()

    if summary.empty:
        st.info("No `auto_pipeline_*_summary.csv` files found in `results/6g_data/` yet. "
                "Run a scenario in the *Run on your data* tab to populate.")
    else:
        scenarios_available = sorted(summary.scenario.unique())
        algorithms_available = sorted(summary.algorithm.unique())

        c1, c2 = st.columns(2)
        scenario = c1.selectbox("Scenario", scenarios_available, index=0)
        algorithm = c2.selectbox("Algorithm", algorithms_available,
                                 index=algorithms_available.index("XGBoost")
                                 if "XGBoost" in algorithms_available else 0)
        scen_sum = summary[(summary.scenario == scenario) & (summary.algorithm == algorithm)]

        # Scenario context strip
        ctx = manifest.get(scenario, {})
        if ctx:
            st.caption(
                f"**{ctx.get('name','—')}** · target=`{ctx.get('target_column','—')}` · "
                f"expected: **{ctx.get('expected_behaviour','—')}** · "
                f"join: `{','.join(ctx.get('join_keys', [])) or '—'}`"
            )

        # Per-approach metric table
        st.subheader("Approaches")
        metric_cols = [c for c in ["approach", "accuracy", "n_features",
                                   "train_time", "total_time"] if c in scen_sum.columns]
        st.dataframe(scen_sum[metric_cols].reset_index(drop=True),
                     use_container_width=True)

        st.divider()
        st.subheader("Feature importance drilldown")
        approach = st.selectbox("Approach", scen_sum.approach.tolist())
        feats = load_features(scenario)
        feats = feats[(feats.approach == approach) & (feats.algorithm == algorithm)]
        if feats.empty:
            st.write("(no feature importance recorded for this row)")
        else:
            top = (
                feats.assign(abs_importance=feats.importance.abs())
                .sort_values("abs_importance", ascending=False)
                .head(20)
            )
            chart_imp = (
                alt.Chart(top)
                .mark_bar()
                .encode(
                    y=alt.Y("feature:N", title=None, sort="-x"),
                    x=alt.X("importance:Q", title="Feature importance"),
                    color=alt.Color("source_table:N", title="Source table"),
                    tooltip=["feature", alt.Tooltip("importance:Q", format=".2f"),
                             "source_table"],
                )
                .properties(height=24 * len(top))
            )
            st.altair_chart(chart_imp, use_container_width=True)


# ─── Tab 2: Compare ──────────────────────────────────────────────────────────
with tab_compare:
    st.header("Cross-scenario comparison")
    summary = load_summaries()
    manifest = load_scenario_manifest()

    if summary.empty:
        st.info("Run a scenario first to populate this view.")
    else:
        algorithms_available = sorted(summary.algorithm.unique())
        algorithm = st.selectbox("Algorithm",
                                 algorithms_available,
                                 index=algorithms_available.index("XGBoost")
                                 if "XGBoost" in algorithms_available else 0,
                                 key="cmp_alg")
        sub = summary[summary.algorithm == algorithm]

        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.subheader("Accuracy by scenario × approach")
            chart = (
                alt.Chart(sub)
                .mark_bar()
                .encode(
                    y=alt.Y("scenario:N", title=None, sort=None),
                    x=alt.X("accuracy:Q", title="Accuracy / R²",
                            scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("approach:N", scale=alt.Scale(
                        domain=list(APPROACH_COLOURS.keys()),
                        range=list(APPROACH_COLOURS.values()),
                    )),
                    yOffset="approach:N",
                    tooltip=["scenario", "approach",
                             alt.Tooltip("accuracy:Q", format=".4f"),
                             "n_features"],
                )
                .properties(height=max(220, 60 * sub.scenario.nunique()))
            )
            st.altair_chart(chart, use_container_width=True)

        with col_b:
            st.subheader("Pivot with Δ vs BASE")
            wide = (
                sub.pivot_table(index="scenario", columns="approach",
                                values="accuracy", aggfunc="first")
                .reindex(columns=[a for a in APPROACHES if a in sub.approach.unique()])
                .round(4)
            )
            if "AutoFeat" in wide.columns and "BASE" in wide.columns:
                wide["Δ AutoFeat−BASE"] = (wide["AutoFeat"] - wide["BASE"]).round(4)
            st.dataframe(wide, use_container_width=True)

        if manifest:
            st.subheader("Scenario context")
            ctx_rows = []
            for label in sorted(sub.scenario.unique()):
                s = manifest.get(label, {})
                ctx_rows.append({
                    "scenario": label,
                    "purpose": s.get("name", "—"),
                    "expected": s.get("expected_behaviour", "—"),
                    "target": s.get("target_column", "—"),
                    "join_keys": ", ".join(s.get("join_keys", [])) or "—",
                })
            st.dataframe(pd.DataFrame(ctx_rows), use_container_width=True)


# ─── Tab 2: Run on your data ─────────────────────────────────────────────────
with tab_run:
    st.header("Run AutoFeat feature discovery on your own data")
    st.caption(
        "Upload one base table CSV (must include the target column) plus one or more "
        "lake tables. The pipeline will discover joins via PK/FK edges in any "
        "uploaded `connections.csv`, optionally extend them with a sentence-transformer, "
        "then run BASE / Join_All_BFS / AutoFeat and compare tabular feature sets."
    )

    with st.form("custom_run"):
        base_file = st.file_uploader("Base table (CSV, includes target column)", type=["csv"], accept_multiple_files=False)
        lake_files = st.file_uploader("Lake tables (CSV, one or more)", type=["csv"], accept_multiple_files=True)
        meta_file = st.file_uploader("metadata.txt (optional, column descriptions for transformer discovery)", type=["txt"], accept_multiple_files=False)
        conn_file = st.file_uploader("connections.csv (optional, explicit PK/FK edges)", type=["csv"], accept_multiple_files=False)

        c1, c2 = st.columns(2)
        with c1:
            target_col = st.text_input("Target column", placeholder="e.g. lat99 / target_x")
            dataset_type = st.selectbox("Dataset type", ["regression", "binary"])
            label = st.text_input("Run label (output file name suffix)", value="custom_run")
        with c2:
            temporal_key = st.text_input("Temporal key (optional, e.g. 'time')", value="")
            temporal_tol = st.number_input("Temporal tolerance (seconds)", min_value=0, max_value=3600, value=60, step=10)
            skip_transformer = st.checkbox(
                "Skip transformer discovery (use only uploaded connections.csv)",
                value=False,
                help="Faster when you've already provided explicit PK/FK edges.",
            )

        submitted = st.form_submit_button("Run pipeline", type="primary")

    if submitted:
        if not base_file:
            st.error("Please upload a base table CSV.")
        elif not lake_files:
            st.error("Please upload at least one lake table CSV.")
        elif not target_col.strip():
            st.error("Please specify the target column.")
        else:
            # Stage all uploads in a tmpdir under the project so DATA_FOLDER resolves cleanly
            tmpdir = Path(tempfile.mkdtemp(prefix="aug_dash_", dir=ROOT / "datasets"))
            (tmpdir / base_file.name).write_bytes(base_file.getbuffer())
            for f in lake_files:
                (tmpdir / f.name).write_bytes(f.getbuffer())
            if meta_file is not None:
                (tmpdir / "metadata.txt").write_bytes(meta_file.getbuffer())
            if conn_file is not None:
                (tmpdir / "connections.csv").write_bytes(conn_file.getbuffer())

            # Sanity check: target column exists in base
            try:
                head = pd.read_csv(tmpdir / base_file.name, nrows=1)
                if target_col not in head.columns:
                    st.error(f"Target column `{target_col}` not found in base table. "
                             f"Columns: {list(head.columns)}")
                    shutil.rmtree(tmpdir, ignore_errors=True)
                    st.stop()
            except Exception as exc:
                st.error(f"Could not read base CSV: {exc}")
                shutil.rmtree(tmpdir, ignore_errors=True)
                st.stop()

            args = [
                "--base-table", str(tmpdir / base_file.name),
                "--target", target_col,
                "--data-dir", str(tmpdir),
                "--dataset-type", dataset_type,
                "--label", label,
            ]
            if temporal_key.strip():
                args += ["--temporal-key", temporal_key.strip(), "--temporal-tolerance", str(temporal_tol)]
            if skip_transformer:
                args += ["--no-transformer-discovery"]

            st.info(f"Staging files in `{tmpdir.relative_to(ROOT)}`. Running pipeline …")
            log_box = st.empty()
            t0 = time.time()
            rc = run_pipeline(args, log_box)
            dt = time.time() - t0

            if rc == 0:
                st.success(f"Pipeline finished in {dt:.1f}s (exit 0).")
                out_path = RESULTS_DIR / f"auto_pipeline_{label}.csv"
                if out_path.exists():
                    res_df = pd.read_csv(out_path)
                    st.subheader("Results")
                    st.dataframe(
                        res_df.drop_duplicates(subset=["approach"])[
                            ["approach", "accuracy", "n_features", "train_time"]
                        ],
                        use_container_width=True,
                    )
                    st.cache_data.clear()
                    st.caption("Reload the *Results* tab to see this run in the comparison view.")
                else:
                    st.warning(f"Pipeline reported success but no output CSV found at {out_path}")
            else:
                st.error(f"Pipeline failed with exit code {rc}.")

            # Note: leaving tmpdir in place so users can inspect what was staged.
            # User can clean up `datasets/aug_dash_*` directories manually.


# ─── Tab 3: Graph browser ────────────────────────────────────────────────────
with tab_graph:
    st.header("Neo4j join graph")
    st.markdown(
        f"Neo4j Browser → **[{NEO4J_BROWSER_URL}]({NEO4J_BROWSER_URL})** "
        f"(bolt: `{DEFAULT_NEO4J_HOST}`, no auth)"
    )

    nodes, rels, err = neo4j_stats()
    if err:
        st.error(f"Cannot reach Neo4j: {err}")
        st.caption("Start the database with `docker start feature-discovery-neo4j` "
                   "or `docker-compose up -d neo4j`.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Nodes (tables)", nodes)
        c2.metric("Relationships (join edges)", rels)

    st.divider()
    st.subheader("Canned Cypher queries")
    st.caption("Copy any of these into the Neo4j Browser to inspect the graph.")

    queries = [
        ("All discovered join edges with weight",
         "MATCH (a)-[r]->(b)\n"
         "RETURN a.label AS from_table, r.from_column, b.label AS to_table, r.to_column, r.weight\n"
         "ORDER BY r.weight DESC LIMIT 100;"),
        ("Reachable lake tables from a given base",
         "MATCH path = (base {label: 'rabbitmq-reduced.csv'})-[*1..3]-(lake)\n"
         "RETURN path LIMIT 50;"),
        ("Edges grouped by source table (sanity check)",
         "MATCH (a)-[r]->()\n"
         "RETURN a.label AS table, count(r) AS edge_count\n"
         "ORDER BY edge_count DESC;"),
        ("All tables in the graph",
         "MATCH (n)\n"
         "RETURN n.label AS table, n.id AS path\n"
         "ORDER BY n.label;"),
    ]
    for title, body in queries:
        with st.expander(title):
            st.code(body, language="cypher")
