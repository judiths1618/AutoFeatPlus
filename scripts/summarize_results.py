"""
summarize_results.py
====================
Consolidate every scenario's `auto_pipeline_<label>_summary.csv` into a single
cross-scenario comparison view, enriched with context from `scenarios/scenarios.yaml`.

Writes:
  results/6g_data/summary.csv     — one row per (scenario, approach, algorithm)
  results/6g_data/SUMMARY.md      — markdown pivot table for quick reading

Usage
-----
    python scripts/summarize_results.py
    python scripts/summarize_results.py --algorithm XGB      # filter
    python scripts/summarize_results.py --markdown-only      # skip summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "6g_data"
SCENARIOS_YAML = ROOT / "scenarios" / "scenarios.yaml"


def _rel(path) -> str:
    """Render a path relative to the project root (keeps usernames out of logs)."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return Path(path).name

APPROACHES = ["BASE", "Join_All_BFS", "Join_All_BFS_Filter", "AutoFeat"]

# AutoGluon reports the XGB hyperparameter key as the model name "XGBoost"; the
# trivial no-feature fallback used the bare key. Treat them as one algorithm so a
# scenario's rows don't split across two tables. (Maps other AG keys too.)
ALGORITHM_ALIASES = {"XGB": "XGBoost", "GBM": "LightGBM", "RF": "RandomForest", "XT": "ExtraTrees"}


def load_scenarios() -> Dict[str, dict]:
    """Parse scenarios.yaml into {label: scenario_dict}. PyYAML preferred; fall
    back to a tiny hand-rolled parser if unavailable.
    """
    if not SCENARIOS_YAML.exists():
        return {}
    try:
        import yaml
        with SCENARIOS_YAML.open() as f:
            data = yaml.safe_load(f)
        return {s["label"]: s for s in data.get("scenarios", [])}
    except ImportError:
        return _parse_yaml_fallback(SCENARIOS_YAML)


def _parse_yaml_fallback(path: Path) -> Dict[str, dict]:
    """Minimal YAML extractor — handles only the flat fields we need (label,
    name, expected_behaviour, target_column). Used only when PyYAML isn't installed.
    """
    scenarios: Dict[str, dict] = {}
    current: Optional[dict] = None
    for raw in path.read_text().splitlines():
        line = raw.rstrip()
        if line.startswith("  - label:"):
            if current is not None:
                scenarios[current["label"]] = current
            current = {"label": line.split(":", 1)[1].strip()}
        elif current is not None and line.startswith("    ") and ":" in line:
            key, val = line.strip().split(":", 1)
            val = val.strip().strip('"').strip("'")
            if val and not val.startswith(">"):
                current[key] = val
    if current is not None:
        scenarios[current["label"]] = current
    return scenarios


def load_run_summaries() -> pd.DataFrame:
    """Read every `auto_pipeline_<label>_summary.csv` into one DataFrame.
    Falls back to the raw `auto_pipeline_<label>.csv` if the summary file
    doesn't exist yet (older runs)."""
    rows: List[pd.DataFrame] = []
    for f in sorted(RESULTS_DIR.glob("auto_pipeline_*_summary.csv")):
        df = pd.read_csv(f)
        df["__source_file"] = f.name
        rows.append(df)

    # Fall back to raw CSVs for any scenario without a _summary.csv yet.
    summaries_seen = {f.stem.replace("_summary", "") for f in
                      RESULTS_DIR.glob("auto_pipeline_*_summary.csv")}
    for f in sorted(RESULTS_DIR.glob("auto_pipeline_*.csv")):
        if f.stem in summaries_seen or "_summary" in f.name or "_features" in f.name:
            continue
        raw = pd.read_csv(f)
        keep = ["data_label", "approach", "algorithm", "accuracy",
                "rmse", "mae", "n_features", "train_time", "total_time"]
        sub = (
            raw[[c for c in keep if c in raw.columns]]
            .drop_duplicates(subset=["approach", "algorithm"], keep="first")
            .rename(columns={"data_label": "scenario"})
        )
        sub["__source_file"] = f.name + " (raw fallback)"
        rows.append(sub)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def compute_lift(df: pd.DataFrame) -> pd.DataFrame:
    """Add Δ vs BASE per (scenario, algorithm)."""
    base = (
        df[df.approach == "BASE"][["scenario", "algorithm", "accuracy"]]
        .rename(columns={"accuracy": "base_accuracy"})
    )
    return df.merge(base, on=["scenario", "algorithm"], how="left").assign(
        delta_vs_base=lambda x: (x.accuracy - x.base_accuracy).round(4)
    )


def write_markdown(df: pd.DataFrame, scenarios: Dict[str, dict],
                   out_path: Path) -> None:
    """Render a single-glance markdown report."""
    if df.empty:
        out_path.write_text("# Benchmark summary\n\n_No runs found._\n")
        return

    sections: List[str] = []
    sections.append("# Augmentation benchmark — cross-scenario summary\n")
    sections.append(f"_{len(df)} rows from {df.scenario.nunique()} scenarios, "
                    f"{df.algorithm.nunique()} algorithms._\n")

    # Headline pivot: (scenario × approach) with default algorithm
    algorithms = sorted(df.algorithm.unique())
    for alg in algorithms:
        sub = df[df.algorithm == alg]
        if sub.empty:
            continue
        pivot = (
            sub.pivot_table(index="scenario", columns="approach",
                            values="accuracy", aggfunc="first")
            .reindex(columns=[a for a in APPROACHES if a in sub.approach.unique()])
            .round(4)
        )
        if "AutoFeat" in pivot.columns and "BASE" in pivot.columns:
            pivot["Δ AutoFeat−BASE"] = (pivot["AutoFeat"] - pivot["BASE"]).round(4)
        sections.append(f"\n## Algorithm: `{alg}`\n")
        sections.append(pivot.to_markdown())

    # Scenario context
    sections.append("\n\n## Scenario context\n")
    sections.append("| Scenario | Purpose | Expected | Target |")
    sections.append("|---|---|---|---|")
    for label in sorted(df.scenario.unique()):
        s = scenarios.get(label, {})
        purpose = (s.get("name") or s.get("purpose") or "—").split("\n")[0].strip()
        expected = s.get("expected_behaviour", "—")
        target = s.get("target_column", "—")
        sections.append(f"| `{label}` | {purpose} | {expected} | `{target}` |")

    out_path.write_text("\n".join(sections) + "\n")


def main(args: argparse.Namespace) -> None:
    scenarios = load_scenarios()
    df = load_run_summaries()
    if df.empty:
        print("No auto_pipeline result files found in", RESULTS_DIR)
        return

    df["algorithm"] = df["algorithm"].replace(ALGORITHM_ALIASES)
    # A scenario's BASE may be the trivial fallback (now "XGBoost") while the same
    # scenario also has a real "XGBoost" row from another approach — collapse any
    # exact duplicates that the aliasing introduces.
    df = df.drop_duplicates(subset=["scenario", "approach", "algorithm"], keep="first")

    if args.algorithm:
        wanted = ALGORITHM_ALIASES.get(args.algorithm, args.algorithm)
        df = df[df.algorithm == wanted]

    df = compute_lift(df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not args.markdown_only:
        csv_path = RESULTS_DIR / "summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"summary.csv → {_rel(csv_path)}  ({len(df)} rows)")

    md_path = RESULTS_DIR / "SUMMARY.md"
    write_markdown(df, scenarios, md_path)
    print(f"SUMMARY.md  → {_rel(md_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithm", "-a", default=None,
                        help="Filter to a single algorithm (e.g. XGB).")
    parser.add_argument("--markdown-only", action="store_true",
                        help="Write SUMMARY.md only; skip summary.csv.")
    main(parser.parse_args())
