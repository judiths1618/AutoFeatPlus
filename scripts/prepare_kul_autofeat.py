from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


CSI_NAME = re.compile(r"user_(?P<user>\d+)_sample_(?P<sample>\d+)_antenna_(?P<antenna>\d+)\.csv$")
POSITION = re.compile(r"-?\d+")


def parse_position(value: str) -> tuple[int, int, int]:
    coords = [int(v) for v in POSITION.findall(value)]
    if len(coords) != 3:
        raise ValueError(f"Expected three position coordinates, got {value!r}")
    return coords[0], coords[1], coords[2]


def build_feature_row(path: Path, user_id: int, sample_id: int, antenna_id: int) -> dict[str, float | int | str]:
    df = pd.read_csv(path)
    row: dict[str, float | int | str] = {
        "sample_key": f"user_{user_id}_sample_{sample_id}",
        "user_id": user_id,
        "sample_id": sample_id,
        "antenna_id": antenna_id,
    }
    for _, feature in df.iterrows():
        subcarrier = int(feature["Unnamed: 0"])
        row[f"subcarrier_{subcarrier}_real"] = feature["real"]
        row[f"subcarrier_{subcarrier}_imaginary"] = feature["imaginary"]
    return row


def prepare_kul_autofeat(
    kul_root: Path,
    output_dir: Path,
    antennas: list[int],
    max_samples_per_user: int | None,
) -> None:
    csi_dir = kul_root / "nomadic_dataset_ULA_static" / "csi_as_features"
    if not csi_dir.is_dir():
        raise FileNotFoundError(f"CSI directory not found: {csi_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_files: dict[int, list[tuple[int, int, Path]]] = {antenna: [] for antenna in antennas}
    sample_positions: dict[str, dict[str, int | str]] = {}

    for path in sorted(csi_dir.glob("*.csv")):
        match = CSI_NAME.match(path.name)
        if match is None:
            continue

        user_id = int(match.group("user"))
        sample_id = int(match.group("sample"))
        antenna_id = int(match.group("antenna"))
        if antenna_id not in selected_files:
            continue

        if max_samples_per_user is not None:
            user_count = sum(1 for _, existing_user, _ in selected_files[antenna_id] if existing_user == user_id)
            if user_count >= max_samples_per_user:
                continue

        selected_files[antenna_id].append((sample_id, user_id, path))

    for antenna_id, paths in selected_files.items():
        rows = []
        for sample_id, user_id, path in sorted(paths):
            df = pd.read_csv(path, usecols=["key_user_position"], nrows=1)
            target_x, target_y, target_z = parse_position(df["key_user_position"].iloc[0])
            sample_key = f"user_{user_id}_sample_{sample_id}"
            sample_positions[sample_key] = {
                "sample_key": sample_key,
                "user_id": user_id,
                "sample_id": sample_id,
                "target_x": target_x,
                "target_y": target_y,
                "target_z": target_z,
            }
            rows.append(build_feature_row(path, user_id, sample_id, antenna_id))

        pd.DataFrame(rows).to_csv(output_dir / f"antenna_{antenna_id}_features.csv", index=False)

    pd.DataFrame(sample_positions.values()).sort_values(["user_id", "sample_id"]).to_csv(
        output_dir / "samples.csv", index=False
    )

    with (output_dir / "connections.csv").open("w", encoding="utf8") as connections:
        connections.write("pk_table,pk_column,fk_table,fk_column\n")
        for antenna_id in antennas:
            connections.write(f"samples.csv,sample_key,antenna_{antenna_id}_features.csv,sample_key\n")

    data_root = kul_root.parent
    with (data_root / "datasets.csv").open("w", encoding="utf8") as datasets:
        datasets.write("base_table_path,base_table_name,base_table_label,target_column,dataset_type\n")
        datasets.write(f"{output_dir.relative_to(data_root)},samples.csv,kul_nomadic_ula_static,target_x,regression\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare KUL CSI files for AutoFeat.")
    parser.add_argument("--kul-root", type=Path, default=Path("datasets/KUL"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/KUL/autofeat_nomadic_ula_static"),
    )
    parser.add_argument("--antennas", type=int, nargs="+", default=[0, 16, 32, 48])
    parser.add_argument("--max-samples-per-user", type=int, default=None)
    args = parser.parse_args()

    prepare_kul_autofeat(
        kul_root=args.kul_root,
        output_dir=args.output_dir,
        antennas=args.antennas,
        max_samples_per_user=args.max_samples_per_user,
    )


if __name__ == "__main__":
    main()
