"""
augmentation_dashboard.py
=========================
Streamlit dashboard for the AutoFeat / AutoFeatPlus feature-discovery pipeline.

Four tabs:
  1. Results    — per-scenario BASE / Join_All_BFS / Filter / AutoFeat /
                  AutoFeatPlus rows with feature-importance drilldown
                  (reads `auto_pipeline_<label>_summary.csv` + `_features.csv`).
  2. Compare    — cross-scenario pivot, enriched with `scenarios/scenarios.yaml`
                  context (purpose, expected behaviour). XGB → XGBoost is
                  aliased so each scenario shows up under one row.
  3. Run        — upload base + lake CSVs, configure target/temporal-key, run
                  the pipeline end-to-end with live log streaming.
  4. Graph      — inline Graphviz view with three pickers:
                    • Source:    Live Neo4j  *or*  any scenario from the manifest
                    • Method:    All discovered edges *or* one of
                                 BASE / Join_All_BFS / Filter / AutoFeat / AutoFeatPlus
                                 (drawn from `auto_pipeline_<label>_features.csv`)
                    • Algorithm: XGBoost / RandomForest (only in method mode)
                  Method mode draws only the tables that method actually
                  consumed features from, with a side panel listing every
                  selected feature ranked by |importance|.

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
SCENARIOS_YAML = ROOT / "scenarios" / "scenarios.yaml"
DEFAULT_NEO4J_HOST = os.environ.get("NEO4J_HOST", "bolt://localhost:7687")
NEO4J_BROWSER_URL = "http://localhost:7474"

APPROACHES = ["BASE", "Join_All_BFS", "Join_All_BFS_Filter", "AutoFeat", "AutoFeatPlus"]
APPROACH_COLOURS = {
    "BASE": "#888888",
    "Join_All_BFS": "#5ba9ff",
    "Join_All_BFS_Filter": "#9c8bff",
    "AutoFeat": "#ff7f50",
    # Olive-ish — distinguishable from the AutoFeat orange without clashing
    # with the cool palette used for the lake/filter rows.
    "AutoFeatPlus": "#5b8c00",
}

st.set_page_config(page_title="AutoFeat Feature Discovery", layout="wide")


# ─── Helpers ─────────────────────────────────────────────────────────────────
# AutoGluon reports the "XGB" hyperparameter key as the model name "XGBoost";
# the trivial no-feature fallback used the bare key. Fold them together so the
# dashboard never shows a standalone "XGB" option. (Maps other AG keys too.)
ALGORITHM_ALIASES = {"XGB": "XGBoost", "GBM": "LightGBM", "RF": "RandomForest", "XT": "ExtraTrees"}


@st.cache_data(show_spinner=False)
def load_summaries() -> pd.DataFrame:
    """Concat every `auto_pipeline_<label>_summary.csv`.

    Old result files left ``n_features`` at 0 for every approach except
    AutoFeat — a tracking bug fixed at the source on 2026-05-31. To keep the
    dashboard honest for previously-saved results, backfill ``n_features`` from
    the per-scenario ``_features.csv`` row counts whenever the summary value is
    missing or 0.
    """
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

    # Backfill n_features from the features CSV when the summary CSV has 0.
    if {"scenario", "approach", "algorithm"}.issubset(df.columns):
        counts = _feature_counts_index()
        if counts is not None and not counts.empty:
            df = df.merge(counts, on=["scenario", "approach", "algorithm"],
                          how="left", suffixes=("", "_from_features"))
            if "n_features" not in df.columns:
                df["n_features"] = 0
            df["n_features"] = df["n_features"].where(
                df["n_features"].fillna(0) > 0, df.get("n_features_from_features"))
            df["n_features"] = pd.to_numeric(df["n_features"], errors="coerce").fillna(0).astype(int)
            df = df.drop(columns=[c for c in df.columns if c.endswith("_from_features")])
    return df


@st.cache_data(show_spinner=False)
def _feature_counts_index() -> pd.DataFrame:
    """Tally one row per (scenario, approach, algorithm) → n_features.

    Reads every ``auto_pipeline_*_features.csv`` once and counts unique feature
    names per group; cached so the merge in ``load_summaries`` stays cheap.
    """
    rows = []
    for f in sorted(RESULTS_DIR.glob("auto_pipeline_*_features.csv")):
        try:
            fdf = pd.read_csv(f, usecols=["scenario", "approach", "algorithm", "feature"])
        except Exception:
            continue
        g = (fdf.groupby(["scenario", "approach", "algorithm"])["feature"]
             .nunique().reset_index(name="n_features_from_features"))
        rows.append(g)
    if not rows:
        return pd.DataFrame(columns=["scenario", "approach", "algorithm", "n_features_from_features"])
    return pd.concat(rows, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_features(label: str) -> pd.DataFrame:
    """Long-format feature importance for one scenario."""
    f = RESULTS_DIR / f"auto_pipeline_{label}_features.csv"
    if f.exists():
        return pd.read_csv(f)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_scenario_manifest() -> Dict[str, dict]:
    """Parse scenarios/scenarios.yaml (best-effort)."""
    if not SCENARIOS_YAML.exists():
        return {}
    try:
        import yaml
        with SCENARIOS_YAML.open() as f:
            data = yaml.safe_load(f)
        return {s["label"]: s for s in data.get("scenarios", [])}
    except Exception:
        return {}


@st.cache_data(show_spinner=False, ttl=10)
def fetch_graph() -> Tuple[List[Dict], List[Dict], Optional[str]]:
    """Return (nodes, edges, error) for the current Neo4j graph.

    Each node is ``{label, id}``; each edge is
    ``{from_table, to_table, from_col, to_col, weight}``. Cached for 10 s so
    the graph tab stays snappy without missing fresh ingests.
    """
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(DEFAULT_NEO4J_HOST, auth=None)
        with driver.session() as s:
            nodes = [
                {"label": r["label"], "id": r["id"]}
                for r in s.run("MATCH (n) RETURN n.label AS label, n.id AS id")
            ]
            edges = [
                {
                    "from_table": r["from_table"],
                    "to_table": r["to_table"],
                    "from_col": r["from_col"],
                    "to_col": r["to_col"],
                    "weight": float(r["weight"]) if r["weight"] is not None else 1.0,
                }
                for r in s.run(
                    "MATCH (a)-[r]->(b) "
                    "RETURN a.label AS from_table, b.label AS to_table, "
                    "       r.from_column AS from_col, r.to_column AS to_col, "
                    "       coalesce(r.weight, 1.0) AS weight"
                )
            ]
        driver.close()
        return nodes, edges, None
    except Exception as exc:
        return [], [], str(exc)


@st.cache_data(show_spinner=False, ttl=30)
def read_scenario_graph(label: str, base_dir: str) -> Tuple[List[Dict], List[Dict], Optional[str]]:
    """Build (nodes, edges) by reading a scenario's connections files from disk.

    Reads ``connections.csv`` (explicit FK→PK edges, weight 1.0) and
    ``connections_transformer.csv`` (transformer-discovered edges, weight from
    the file) under ``base_dir``. Empty when neither file exists. Decouples the
    per-scenario graph view from whichever scenario is currently ingested in
    Neo4j, so any scenario in the manifest is browsable.
    """
    base = ROOT / base_dir if not Path(base_dir).is_absolute() else Path(base_dir)
    if not base.is_dir():
        return [], [], f"base_dir '{base_dir}' not found"
    edges: List[Dict] = []
    nodes_seen: Dict[str, str] = {}
    for fname, source_tag in [("connections.csv", "explicit"),
                              ("connections_transformer.csv", "transformer")]:
        f = base / fname
        if not f.exists():
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        # Accept both column orders (FK-first or PK-first); reuse the same
        # semantic mapping (the actual data direction is FK→PK).
        cols = set(df.columns)
        if not {"fk_table", "fk_column", "pk_table", "pk_column"}.issubset(cols):
            continue
        for _, row in df.iterrows():
            ft = str(row["fk_table"])
            tt = str(row["pk_table"])
            fc = str(row["fk_column"])
            tc = str(row["pk_column"])
            try:
                w = float(row["weight"]) if "weight" in cols and pd.notna(row["weight"]) else 1.0
            except (TypeError, ValueError):
                w = 1.0
            edges.append({"from_table": ft, "to_table": tt, "from_col": fc,
                          "to_col": tc, "weight": w, "source": source_tag})
            nodes_seen[ft] = ft
            nodes_seen[tt] = tt
    nodes = [{"label": n, "id": f"{base_dir}/{n}"} for n in nodes_seen]
    return nodes, edges, None


@st.cache_data(show_spinner=False, ttl=30)
def load_method_selection(label: str) -> pd.DataFrame:
    """Return the per-(approach, algorithm, feature, source_table) selection rows.

    Empty DataFrame if the features CSV doesn't exist (scenario not run yet).
    """
    f = RESULTS_DIR / f"auto_pipeline_{label}_features.csv"
    if not f.exists():
        return pd.DataFrame(columns=["approach", "algorithm", "feature", "source_table", "importance"])
    return pd.read_csv(f)


def method_graph(
    method_df: pd.DataFrame,
    approach: str,
    algorithm: str,
    base_table_name: str,
    discovered_edges: List[Dict],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Build (nodes, edges, feature_rows) for a single (scenario, approach, algorithm).

    Uses the per-feature ``source_table`` rows in the features CSV to decide
    which lake tables the method actually consumed; intersects ``discovered_edges``
    so only joins the method materialised are drawn. Standalone source tables
    that are not in any discovered edge are connected to ``base_table_name``
    by synthetic dashed edges (weight 0) so the viewer can still see them.
    """
    sub = method_df[(method_df.approach == approach) & (method_df.algorithm == algorithm)]
    if sub.empty:
        return [], [], []

    # Source tables the method actually selected features from. "base" is the
    # special sentinel for BASE-only rows; map it to the configured base table.
    raw_sources = sub.source_table.fillna("base").unique().tolist()
    sources = {s if s != "base" else base_table_name for s in raw_sources}
    if base_table_name not in sources:
        sources.add(base_table_name)  # always show the base for orientation

    # Keep only edges whose endpoints are both in the selected sources.
    used_edges = [e for e in discovered_edges
                  if e["from_table"] in sources and e["to_table"] in sources]

    # Per-source feature count (used for node sizing + side panel).
    feat_count = (
        sub.assign(source_table=sub.source_table.fillna("base").replace({"base": base_table_name}))
        .groupby("source_table").size()
        .to_dict()
    )
    nodes = [
        {"label": n, "id": n, "feature_count": int(feat_count.get(n, 0))}
        for n in sources
    ]

    feature_rows = (
        sub.assign(source_table=sub.source_table.fillna("base").replace({"base": base_table_name}))
        .sort_values("importance", key=lambda s: s.abs(), ascending=False)
        [["feature", "source_table", "importance"]]
        .to_dict("records")
    )
    return nodes, used_edges, feature_rows


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
            if "AutoFeatPlus" in wide.columns and "BASE" in wide.columns:
                wide["Δ AutoFeatPlus−BASE"] = (wide["AutoFeatPlus"] - wide["BASE"]).round(4)
            st.dataframe(wide, use_container_width=True)

        # ── Feature counts panel ────────────────────────────────────────────
        # How many features each method *actually used* per scenario. Reveals
        # the compression story (AutoFeatPlus hard-capped at top_k, K_csi's
        # 3200 → 213 → 15 funnel, scenarios where AutoFeat picks 0 lake
        # features and the "lift" is really just the trivial-self-join detour).
        st.subheader("Feature counts (number of features each method selected)")
        if "n_features" in sub.columns:
            counts_pivot = (
                sub.pivot_table(index="scenario", columns="approach",
                                values="n_features", aggfunc="first")
                .reindex(columns=[a for a in APPROACHES if a in sub.approach.unique()])
                .fillna(0).astype(int)
            )
            st.dataframe(counts_pivot, use_container_width=True)
            with st.expander("How to read this", expanded=False):
                st.markdown(
                    "- `BASE` ≈ base-table column count (target excluded).\n"
                    "- `Join_All_BFS` = every column the BFS join reached.\n"
                    "- `Join_All_BFS_Filter` ≈ Spearman top-half cut.\n"
                    "- `AutoFeat` = mRMR-style relevance/redundancy selection.\n"
                    "- `AutoFeatPlus` is hard-capped at `--autofeat-plus-top-k` "
                    "(default 15) and policy-aware — a row that equals BASE "
                    "means AutoFeat picked **zero lake features** even though "
                    "the lake was reachable. Compare against the Δ vs BASE "
                    "pivot above to spot scenarios where the headline lift is "
                    "intra-base feature selection rather than augmentation."
                )

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
    st.subheader("Inline graph view")

    # ---- source picker: scenarios from the manifest + live Neo4j -----------
    LIVE = "Live Neo4j (current ingest)"
    ALL_EDGES = "All discovered edges (connections files)"
    scenarios = load_scenario_manifest()  # {label: scenario_dict}
    scenario_labels = sorted(scenarios.keys())

    pc1, pc2, pc3 = st.columns([1.2, 1.4, 1.1])
    source = pc1.selectbox(
        "Source",
        [LIVE] + scenario_labels,
        index=0,
        key="graph_source",
        help="Live shows what's currently in Neo4j. Pick a scenario to read its "
             "connections files (or any method's actual selection) from disk.",
    )

    method_df = pd.DataFrame()
    method_choices = [ALL_EDGES]
    algorithm_choices: List[str] = []
    if source != LIVE:
        method_df = load_method_selection(source)
        if not method_df.empty:
            # Canonical order — same one the summary CSV uses
            order = ["BASE", "Join_All_BFS", "Join_All_BFS_Filter", "AutoFeat", "AutoFeatPlus"]
            seen = list(method_df.approach.unique())
            method_choices += [m for m in order if m in seen] + [m for m in seen if m not in order]
            algorithm_choices = sorted(method_df.algorithm.unique().tolist())

    method = pc2.selectbox(
        "Method",
        method_choices,
        index=0,
        key="graph_method",
        help="`All discovered edges` shows what the schema/transformer pipeline "
             "uncovered. Pick a method to see only the tables that method "
             "actually consumed features from (driven by the per-scenario "
             "`auto_pipeline_<label>_features.csv`).",
    )
    algorithm = ""
    if method != ALL_EDGES and algorithm_choices:
        default_idx = algorithm_choices.index("XGBoost") if "XGBoost" in algorithm_choices else 0
        algorithm = pc3.selectbox("Algorithm", algorithm_choices, index=default_idx, key="graph_algo")

    # ---- fetch the underlying graph + (optionally) method overlay -----------
    if source == LIVE:
        g_nodes, g_edges, g_err = fetch_graph()
        source_caption = "Source: live Neo4j ingest"
        feature_rows: List[Dict] = []
    else:
        scn = scenarios.get(source, {})
        all_nodes, all_edges, g_err = read_scenario_graph(source, scn.get("base_dir", ""))
        if method == ALL_EDGES or method_df.empty:
            g_nodes, g_edges = all_nodes, all_edges
            feature_rows = []
            extra = "" if method == ALL_EDGES else f" — no features CSV for `{source}` yet"
            source_caption = (
                f"Source: `{scn.get('base_dir', source)}/connections*.csv` "
                f"(expected behaviour: {scn.get('expected_behaviour', '—')}){extra}"
            )
        else:
            g_nodes, g_edges, feature_rows = method_graph(
                method_df=method_df,
                approach=method,
                algorithm=algorithm,
                base_table_name=scn.get("base_table", ""),
                discovered_edges=all_edges,
            )
            source_caption = (
                f"Source: `{source}` × **{method}** × `{algorithm}` "
                f"({len(g_nodes)} tables, {len(feature_rows)} selected features)"
            )

    st.caption(source_caption)

    if g_err:
        st.warning(f"Could not fetch graph data: {g_err}")
    elif not g_edges:
        if source == LIVE:
            st.info("Graph has no edges yet — run a scenario in the Run tab to ingest tables.")
        else:
            # Map scenario label → prepare-script CLI key. Falls back to --all
            # (always valid) when the label isn't one of the canonical ones.
            _CLI_KEY = {
                "scenario1": "1", "scenario2c": "2c",
                "scenarioA_lat95": "a_lat95", "scenarioA_lat99": "a_lat99",
                "scenarioB_amf_seg01": "b", "scenarioN_target_n": "n",
                "scenarioK_csi": "k",
                "scenarioR_resource": "r", "scenarioU_unrelated": "u",
            }
            cli_arg = "--scenario " + _CLI_KEY.get(source, source) if source in _CLI_KEY else "--all"
            st.info(
                f"No connections files found under `{scn.get('base_dir', source)}/`. "
                f"Build this scenario first: "
                f"`python scripts/prepare_augmentation_scenarios.py {cli_arg}`."
            )
    else:
        import collections
        import graphviz

        # ---- filter controls --------------------------------------------------
        weights = [e["weight"] for e in g_edges]
        wmin, wmax = (min(weights), max(weights)) if weights else (0.0, 1.0)
        c1, c2, c3, c4 = st.columns([1, 1, 1.5, 1])
        max_edges = c1.slider("Max edges", 10, max(50, len(g_edges)),
                              min(100, len(g_edges)), 10,
                              help="Render at most this many top-weight edges.")
        # Streamlit's slider requires min < max. When every edge has the same
        # weight (e.g. scenario2c's single time→time edge with weight 1.0) show
        # a static metric instead of a degenerate slider.
        if wmax > wmin:
            min_w = c2.slider("Min weight", float(wmin), float(wmax),
                              float(wmin), step=max((wmax - wmin) / 20, 0.01))
        else:
            c2.metric("Edge weight", f"{wmin:.2f}",
                      help="All edges share the same weight; no filter slider needed.")
            min_w = wmin
        name_q = c3.text_input("Filter by table name", "",
                               help="Show only edges whose endpoints contain this substring.")
        rankdir = c4.selectbox("Layout", ["LR", "TB"], index=0,
                               help="Left→Right or Top→Bottom.")

        # ---- apply filters and rank by weight --------------------------------
        kept = [
            e for e in g_edges
            if e["weight"] >= min_w
            and (not name_q or name_q.lower() in e["from_table"].lower()
                 or name_q.lower() in e["to_table"].lower())
        ]
        kept.sort(key=lambda e: -e["weight"])
        kept = kept[:max_edges]

        if not kept:
            st.info("No edges match the current filters.")
        else:
            # ---- degree → node sizing/colour ---------------------------------
            deg = collections.Counter()
            for e in kept:
                deg[e["from_table"]] += 1
                deg[e["to_table"]] += 1
            top_deg = max(deg.values())

            dot = graphviz.Digraph(
                graph_attr={"rankdir": rankdir, "bgcolor": "transparent",
                            "splines": "spline", "concentrate": "true"},
                node_attr={"shape": "box", "style": "filled,rounded",
                           "fontname": "Menlo", "fontsize": "10"},
                edge_attr={"fontname": "Menlo", "fontsize": "9",
                           "color": "#3b6ea8", "arrowsize": "0.6"},
            )
            seen = set()
            for e in kept:
                for n in (e["from_table"], e["to_table"]):
                    if n in seen:
                        continue
                    # Scale node size + colour by degree.
                    rel = deg[n] / top_deg if top_deg else 0
                    fill = "#fff2c2" if deg[n] == top_deg else ("#cfe2ff" if rel > 0.5 else "#e7f5e7")
                    dot.node(n, label=n, fillcolor=fill,
                             fontsize=str(10 + int(4 * rel)))
                    seen.add(n)
                label = (e["from_col"] if e["from_col"] == e["to_col"]
                         else f'{e["from_col"]}→{e["to_col"]}')
                pen = 1 + 3 * (e["weight"] - wmin) / max(wmax - wmin, 1e-9)
                dot.edge(e["from_table"], e["to_table"],
                         label=label, penwidth=f"{pen:.1f}")

            st.graphviz_chart(dot, use_container_width=True)
            st.caption(
                f"Showing {len(kept)} of {len(g_edges)} edges over "
                f"{len(seen)} of {len(g_nodes)} tables · "
                f"weight ∈ [{wmin:.2f}, {wmax:.2f}] · gold = highest-degree table."
            )

            # ---- companion stats: top tables by degree ----------------------
            with st.expander("Top tables by degree (in shown subgraph)", expanded=False):
                deg_df = pd.DataFrame(
                    sorted(deg.items(), key=lambda kv: -kv[1])[:15],
                    columns=["table", "degree"],
                )
                st.dataframe(deg_df, use_container_width=True, hide_index=True)

            # ---- companion stats: features the method actually picked -------
            if feature_rows:
                with st.expander(
                    f"Features selected by **{method}** × `{algorithm}` "
                    f"({len(feature_rows)} rows, ranked by |importance|)",
                    expanded=False,
                ):
                    st.dataframe(
                        pd.DataFrame(feature_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

    st.divider()
    with st.expander("Canned Cypher queries (paste into Neo4j Browser)", expanded=False):
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
            st.markdown(f"**{title}**")
            st.code(body, language="cypher")
