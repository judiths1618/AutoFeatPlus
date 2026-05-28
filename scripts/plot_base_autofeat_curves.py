from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


APPROACH_LABELS = {
    "BASE": "BASE",
    "AutoFeat": "AutoFeat",
    "AutoFeatPlus": "AutoFeat",
    "AutoFeatPlus_Local": "AutoFeat",
}

APPROACH_COLORS = {
    "BASE": "#4C78A8",
    "AutoFeat": "#54A24B",
}

POLICY_ORDER = [
    "none",
    "time-private",
    "resource-private",
    "workload-private",
    "target-proxy-private",
    "time-private,target-proxy-private",
    "all",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ML performance curves comparing BASE and AutoFeat results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/6g_data/EUR/6907619_autofeat_plus_local.csv"),
        help="Benchmark CSV with approach, algorithm, accuracy/rmse/mae, and policy metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/6g_data/figures/base_vs_autofeat"),
        help="Directory where plots and the normalized comparison CSV are written.",
    )
    parser.add_argument(
        "--algorithm",
        default=None,
        help="Optional model name filter. Defaults to the first algorithm in the input file.",
    )
    parser.add_argument(
        "--include-join-all",
        action="store_true",
        help="Also include Join_All/Join_All_BFS curves when present.",
    )
    return parser.parse_args()


def policy_from_row(row: pd.Series) -> str:
    for column in ["join_name", "test_groups", "privacy_policy"]:
        value = row.get(column)
        if isinstance(value, str) and value and value != "nan":
            return value.split(";", 1)[0]
    return "unknown"


def normalize_approach(value: object, include_join_all: bool) -> str | None:
    text = str(value)
    if include_join_all and text in {"Join_All", "Join_All_BFS", "Join_All_Local"}:
        return "Join All"
    return APPROACH_LABELS.get(text)


def load_results(path: Path, algorithm: str | None, include_join_all: bool) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    frame = pd.read_csv(path)
    required = {"approach", "algorithm"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    metric_aliases = {
        "r2": ["r2", "accuracy"],
        "rmse": ["rmse"],
        "mae": ["mae"],
        "n_features": ["n_features"],
    }
    for target, candidates in metric_aliases.items():
        for candidate in candidates:
            if candidate in frame.columns:
                frame[target] = pd.to_numeric(frame[candidate], errors="coerce")
                break

    if algorithm is None:
        algorithm = str(frame["algorithm"].dropna().iloc[0])
    frame = frame[frame["algorithm"].astype(str) == algorithm].copy()
    if frame.empty:
        raise ValueError(f"No rows found for algorithm: {algorithm}")

    frame["approach_label"] = frame["approach"].map(
        lambda value: normalize_approach(value, include_join_all)
    )
    frame = frame[frame["approach_label"].notna()].copy()
    if frame.empty:
        raise ValueError("No BASE or AutoFeat rows found in input CSV.")

    frame["policy"] = frame.apply(policy_from_row, axis=1)
    ordered = [policy for policy in POLICY_ORDER if policy in set(frame["policy"])]
    extra = sorted(set(frame["policy"]) - set(ordered))
    categories = ordered + extra
    frame["policy"] = pd.Categorical(frame["policy"], categories=categories, ordered=True)

    keep_columns = [
        column
        for column in [
            "policy",
            "approach_label",
            "algorithm",
            "r2",
            "rmse",
            "mae",
            "n_features",
            "total_time",
            "train_time",
            "feature_selection_time",
            "n_sensitive_features",
            "blocked_features",
        ]
        if column in frame.columns
    ]
    normalized = frame[keep_columns].sort_values(["policy", "approach_label"]).reset_index(drop=True)
    return normalized, algorithm


def plot_metric(frame: pd.DataFrame, metric: str, ylabel: str, title: str, output_path: Path) -> None:
    if metric not in frame.columns:
        return

    plot_frame = frame.dropna(subset=[metric]).copy()
    if plot_frame.empty:
        return

    import matplotlib.pyplot as plt

    policies = [str(policy) for policy in plot_frame["policy"].cat.categories]
    x_positions = range(len(policies))

    fig, ax = plt.subplots(figsize=(11, 5.8))
    for approach in plot_frame["approach_label"].drop_duplicates():
        subset = plot_frame[plot_frame["approach_label"] == approach]
        values = []
        for policy in plot_frame["policy"].cat.categories:
            rows = subset[subset["policy"] == policy]
            values.append(float(rows[metric].iloc[0]) if not rows.empty else float("nan"))
        ax.plot(
            list(x_positions),
            values,
            marker="o",
            linewidth=2.4,
            markersize=6,
            label=approach,
            color=APPROACH_COLORS.get(str(approach)),
        )

    ax.set_title(title)
    ax.set_xlabel("Privacy / feature policy")
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(policies, rotation=25, ha="right")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_feature_frontier(frame: pd.DataFrame, output_path: Path) -> None:
    if not {"n_features", "r2"}.issubset(frame.columns):
        return
    plot_frame = frame.dropna(subset=["n_features", "r2"]).copy()
    if plot_frame.empty:
        return

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.8))
    for approach in plot_frame["approach_label"].drop_duplicates():
        subset = plot_frame[plot_frame["approach_label"] == approach]
        ax.scatter(
            subset["n_features"],
            subset["r2"],
            s=72,
            alpha=0.85,
            label=approach,
            color=APPROACH_COLORS.get(str(approach)),
        )
        if len(subset) > 1:
            ordered = subset.sort_values("n_features")
            ax.plot(
                ordered["n_features"],
                ordered["r2"],
                alpha=0.45,
                color=APPROACH_COLORS.get(str(approach)),
            )

    ax.set_title("R2 vs Feature Count")
    ax.set_xlabel("Number of features")
    ax.set_ylabel("R2")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp/matplotlib-codex")))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, algorithm = load_results(args.input, args.algorithm, args.include_join_all)
    normalized_path = args.output_dir / "base_vs_autofeat_curves_data.csv"
    frame.to_csv(normalized_path, index=False)

    safe_algorithm = algorithm.replace("/", "_").replace(" ", "_")
    plot_metric(
        frame,
        "r2",
        "R2",
        f"{algorithm}: R2 Curve",
        args.output_dir / f"{safe_algorithm}_r2_curve.png",
    )
    plot_metric(
        frame,
        "rmse",
        "RMSE",
        f"{algorithm}: RMSE Curve",
        args.output_dir / f"{safe_algorithm}_rmse_curve.png",
    )
    plot_metric(
        frame,
        "mae",
        "MAE",
        f"{algorithm}: MAE Curve",
        args.output_dir / f"{safe_algorithm}_mae_curve.png",
    )
    plot_feature_frontier(frame, args.output_dir / f"{safe_algorithm}_r2_vs_features.png")

    print(f"Saved normalized data to {normalized_path}")
    print(f"Saved BASE vs AutoFeat plots to {args.output_dir}")


if __name__ == "__main__":
    main()
