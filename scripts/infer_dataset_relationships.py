from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feature_discovery.dataset_relation_graph.hybrid_discovery import (
    build_relationship_report,
    infer_dataset_relationships,
    recommend_connections,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer candidate relationships between dataset files using metadata + content heuristics."
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing CSV data files.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional metadata txt file describing the tables and columns.",
    )
    parser.add_argument("--sample-rows", type=int, default=5000)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/6g_data/relation_discovery"),
    )
    args = parser.parse_args()

    relationships = infer_dataset_relationships(
        data_dir=args.data_dir,
        metadata_path=args.metadata,
        sample_rows=args.sample_rows,
    )
    recommended = recommend_connections(
        relationships,
        confidence_threshold=args.confidence_threshold,
    )

    output_dir = args.output_dir / args.data_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    relationships_path = output_dir / "candidate_relationships.csv"
    recommended_path = output_dir / "recommended_connections.csv"
    report_path = output_dir / "relationship_report.md"

    relationships.to_csv(relationships_path, index=False)
    recommended.to_csv(recommended_path, index=False)
    report = build_relationship_report(
        data_dir=args.data_dir,
        metadata_path=args.metadata,
        relationships=relationships,
        recommended=recommended,
    )
    report_path.write_text(report, encoding="utf-8")

    print(f"Saved candidate relationships to {relationships_path}")
    print(f"Saved recommended connections to {recommended_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
