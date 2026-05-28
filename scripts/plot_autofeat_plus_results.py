from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


APPROACH_ORDER = ["BASE", "Join_All_BFS", "AutoFeatPlus_Local"]
APPROACH_COLORS = {
    "BASE": "#4C78A8",
    "Join_All_BFS": "#F58518",
    "AutoFeatPlus_Local": "#54A24B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot AutoFeatPlus benchmark results.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/6g_data/autofeat_plus_comparison_summary.csv"),
        help="Summary CSV produced by scripts/summarize_autofeat_plus_results.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/6g_data/figures"),
        help="Directory where PNG plots will be written.",
    )
    parser.add_argument(
        "--dataset-family",
        choices=["all", "EUR", "KUL"],
        default="all",
        help="Optionally filter plots to one dataset family.",
    )
    parser.add_argument(
        "--algorithm",
        default="all",
        help="Optionally filter plots to one downstream model label, e.g. XGBoost or ExtraTrees.",
    )
    return parser.parse_args()


def load_summary(path: Path, dataset_family: str, algorithm: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")

    frame = pd.read_csv(path)
    if dataset_family != "all":
        frame = frame[frame["dataset_family"] == dataset_family]
    if algorithm != "all":
        frame = frame[frame["algorithm"] == algorithm]

    frame = frame.copy()
    frame["policy_label"] = frame["privacy_policy"].astype(str)
    if "privacy_mode" in frame.columns:
        frame["policy_label"] = frame["policy_label"] + " / " + frame["privacy_mode"].astype(str)
    if "split_mode" in frame.columns:
        frame["policy_label"] = frame["policy_label"] + " / " + frame["split_mode"].astype(str)

    frame["approach"] = pd.Categorical(frame["approach"], categories=APPROACH_ORDER, ordered=True)
    return frame.sort_values(["dataset_family", "policy_label", "algorithm", "approach"])


def bar_plot(frame: pd.DataFrame, y: str, title: str, output_path: Path, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    if frame.empty or y not in frame.columns:
        return

    plot_frame = frame.dropna(subset=[y]).copy()
    if plot_frame.empty:
        return

    labels = (
        plot_frame["dataset_family"].astype(str)
        + "\n"
        + plot_frame["privacy_policy"].astype(str)
        + "\n"
        + plot_frame["privacy_mode"].astype(str)
        + "\n"
        + plot_frame["algorithm"].astype(str)
    )
    plot_frame["x_label"] = labels

    groups = plot_frame["x_label"].drop_duplicates().tolist()
    approaches = [approach for approach in APPROACH_ORDER if approach in set(plot_frame["approach"].astype(str))]
    x_positions = range(len(groups))
    width = min(0.8 / max(len(approaches), 1), 0.25)

    fig_width = max(10, len(groups) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    for idx, approach in enumerate(approaches):
        subset = plot_frame[plot_frame["approach"].astype(str) == approach]
        values = []
        for group in groups:
            rows = subset[subset["x_label"] == group]
            values.append(float(rows[y].iloc[0]) if not rows.empty else float("nan"))
        offsets = [pos + (idx - (len(approaches) - 1) / 2) * width for pos in x_positions]
        ax.bar(offsets, values, width=width, label=approach, color=APPROACH_COLORS.get(approach))

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_frontier(frame: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    required = {"r2", "n_features", "approach", "dataset_family", "privacy_policy", "privacy_mode", "algorithm"}
    if frame.empty or not required.issubset(frame.columns):
        return

    plot_frame = frame.dropna(subset=["r2", "n_features"]).copy()
    plot_frame = plot_frame[plot_frame["approach"].astype(str).isin(APPROACH_ORDER)]
    if plot_frame.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    for approach in APPROACH_ORDER:
        subset = plot_frame[plot_frame["approach"].astype(str) == approach]
        if subset.empty:
            continue
        ax.scatter(
            subset["n_features"],
            subset["r2"],
            label=approach,
            s=70,
            alpha=0.8,
            color=APPROACH_COLORS.get(approach),
        )

    ax.set_title("Performance vs Feature Count")
    ax.set_xlabel("Number of features")
    ax.set_ylabel("R²")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frame = load_summary(args.summary, args.dataset_family, args.algorithm)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if frame.empty:
        print("No rows to plot after filtering.")
        return

    stem_suffix = f"{args.dataset_family}_{args.algorithm}".replace("/", "_")
    bar_plot(frame, "r2", "R² By Method", args.output_dir / f"r2_by_method_{stem_suffix}.png", "R²")
    bar_plot(
        frame,
        "error_1_minus_r2",
        "Unexplained Error (1 - R²)",
        args.output_dir / f"error_1_minus_r2_{stem_suffix}.png",
        "1 - R²",
    )
    bar_plot(
        frame,
        "n_features",
        "Feature Count By Method",
        args.output_dir / f"feature_count_{stem_suffix}.png",
        "Number of features",
    )
    bar_plot(
        frame,
        "n_sensitive_features",
        "Sensitive/Risky Features By Method",
        args.output_dir / f"sensitive_features_{stem_suffix}.png",
        "Number of sensitive/risky features",
    )
    bar_plot(
        frame,
        "total_time",
        "Runtime By Method",
        args.output_dir / f"runtime_{stem_suffix}.png",
        "Total time (s)",
    )
    if "rmse" in frame.columns and frame["rmse"].notna().any():
        bar_plot(frame, "rmse", "RMSE By Method", args.output_dir / f"rmse_{stem_suffix}.png", "RMSE")
    if "mae" in frame.columns and frame["mae"].notna().any():
        bar_plot(frame, "mae", "MAE By Method", args.output_dir / f"mae_{stem_suffix}.png", "MAE")

    plot_frontier(frame, args.output_dir / f"performance_vs_features_{stem_suffix}.png")
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
