from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from feature_discovery.experiments.autofeat_plus import select_autofeat_plus_features
from feature_discovery.experiments.local_benchmark_utils import join_antenna_tables, make_kul_split


def load_models_module():
    module_path = Path("downstream ML/models.py")
    spec = importlib.util.spec_from_file_location("downstream_ml_models", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load model definitions from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_position_labels(base: pd.DataFrame) -> tuple[np.ndarray, dict]:
    positions = (
        base[["target_x", "target_y", "target_z"]]
        .astype(int)
        .apply(lambda row: (int(row["target_x"]), int(row["target_y"]), int(row["target_z"])), axis=1)
    )
    unique_positions = sorted(positions.unique())
    position_to_idx = {position: idx for idx, position in enumerate(unique_positions)}
    labels = np.array([position_to_idx[position] for position in positions], dtype=np.int64)
    return labels, position_to_idx


def build_structured_csi_tensor(
    data_dir: Path,
    base: pd.DataFrame,
    antennas: list[int],
) -> np.ndarray:
    n_samples = len(base)
    n_ant = len(antennas)
    n_sub = 100
    structured = np.zeros((n_samples, n_ant, n_sub * 2), dtype=np.float32)
    sample_order = {sample_key: idx for idx, sample_key in enumerate(base["sample_key"].tolist())}

    for ant_idx, antenna in enumerate(antennas):
        features = pd.read_csv(data_dir / f"antenna_{antenna}_features.csv")
        features = features.set_index("sample_key")
        for sub in range(n_sub):
            real_col = f"subcarrier_{sub}_real"
            imag_col = f"subcarrier_{sub}_imaginary"
            real_values = features.loc[base["sample_key"], real_col].to_numpy(dtype=np.float32)
            imag_values = features.loc[base["sample_key"], imag_col].to_numpy(dtype=np.float32)
            structured[:, ant_idx, sub] = real_values
            structured[:, ant_idx, n_sub + sub] = imag_values

    return structured


def selected_feature_mask(selected_features: list[str], antennas: list[int], n_sub: int = 100) -> np.ndarray:
    antenna_to_idx = {antenna: idx for idx, antenna in enumerate(antennas)}
    mask = np.zeros((len(antennas), n_sub * 2), dtype=np.float32)
    for feature in selected_features:
        parts = feature.split("_")
        if len(parts) < 4 or parts[0] != "antenna":
            continue
        antenna = int(parts[1])
        if antenna not in antenna_to_idx:
            continue
        try:
            sub_idx = int(parts[3])
        except ValueError:
            continue
        channel = parts[4] if len(parts) > 4 else ""
        channel_offset = 0 if channel == "real" else n_sub
        if 0 <= sub_idx < n_sub:
            mask[antenna_to_idx[antenna], channel_offset + sub_idx] = 1.0
    return mask


def transform_for_model(model_name: str, structured_x: np.ndarray) -> np.ndarray:
    if model_name == "cnn":
        real = structured_x[:, :, :100].reshape(len(structured_x), -1)
        imag = structured_x[:, :, 100:].reshape(len(structured_x), -1)
        return np.concatenate([real, imag], axis=1)
    return structured_x.reshape(len(structured_x), -1)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    accuracy = float((y_true == y_pred).mean())
    classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    f1_scores = []
    for cls in classes:
        tp = int(((y_true == cls) & (y_pred == cls)).sum())
        fp = int(((y_true != cls) & (y_pred == cls)).sum())
        fn = int(((y_true == cls) & (y_pred != cls)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    return accuracy, macro_f1


def train_and_evaluate(
    torch,
    models_module,
    model_name: str,
    x_train_structured: np.ndarray,
    y_train: np.ndarray,
    x_test_structured: np.ndarray,
    y_test: np.ndarray,
    n_ant: int,
    n_sub: int,
    num_classes: int,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
) -> dict:
    nn = torch.nn
    data_utils = torch.utils.data

    x_train = transform_for_model(model_name, x_train_structured)
    x_test = transform_for_model(model_name, x_test_structured)

    train_dataset = data_utils.TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_dataset = data_utils.TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = models_module.get_model(
        model_name,
        task="positioning",
        input_dim=x_train.shape[1],
        num_classes=num_classes,
        n_ant=n_ant,
        n_sub=n_sub,
        n_channels=2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    predictions = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            predictions.append(logits.argmax(dim=1).cpu().numpy())

    y_pred = np.concatenate(predictions) if predictions else np.array([], dtype=np.int64)
    accuracy, macro_f1 = compute_metrics(y_test, y_pred)
    runtime = time.time() - start

    return {
        "model": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "train_time": runtime,
        "n_params": int(models_module.count_params(model)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark custom downstream CSI models on KUL augmented data.")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/KUL/autofeat_nomadic_ula_static"))
    parser.add_argument("--antennas", type=int, nargs="+", default=[0, 16, 32, 48])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mlp", "cnn", "patchtst", "timesnet", "lstm", "freqmlp"],
        help="Model names from downstream ML/models.py",
    )
    parser.add_argument("--split-mode", choices=["random", "user-holdout", "position-holdout"], default="random")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--holdout-user", type=int, default=None)
    parser.add_argument("--holdout-position", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--autofeat-plus-top-k", type=int, default=50)
    parser.add_argument("--privacy-penalty", type=float, default=0.25)
    parser.add_argument("--missing-penalty", type=float, default=0.10)
    parser.add_argument("--cost-penalty", type=float, default=0.001)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/6g_data/kul_downstream_models.csv"),
    )
    args = parser.parse_args()

    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None:
        raise ImportError("PyTorch is required to run downstream ML models. Please install torch in your env.")
    import torch

    models_module = load_models_module()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base = pd.read_csv(args.data_dir / "samples.csv")
    joined = join_antenna_tables(args.data_dir, base, args.antennas)
    train_index, test_index, test_groups = make_kul_split(
        metadata=base,
        split_mode=args.split_mode,
        test_size=args.test_size,
        holdout_user=args.holdout_user,
        holdout_position=args.holdout_position,
    )

    labels, position_to_idx = build_position_labels(base)
    structured_x = build_structured_csi_tensor(args.data_dir, base, args.antennas)

    drop_columns = [c for c in ["sample_key", "user_id", "sample_id", "target_y", "target_z"] if c in joined.columns]
    joined_privacy_clean = joined.drop(columns=drop_columns)

    selection = select_autofeat_plus_features(
        dataframe=joined_privacy_clean.iloc[train_index],
        target_column="target_x",
        top_k=args.autofeat_plus_top_k,
        privacy_penalty=args.privacy_penalty,
        missing_penalty=args.missing_penalty,
        cost_penalty=args.cost_penalty,
        block_sensitive=True,
    )
    mask = selected_feature_mask(selection.selected_features, args.antennas)
    masked_structured_x = structured_x * mask[None, :, :]

    rows = []
    base_zero_x = np.zeros_like(structured_x, dtype=np.float32)

    for variant_name, x_structured in [
        ("Base_ZeroCSI", base_zero_x),
        ("Join_All_CSI", structured_x),
        ("AutoFeatPlus_Masked", masked_structured_x),
    ]:
        for model_name in args.models:
            result = train_and_evaluate(
                torch=torch,
                models_module=models_module,
                model_name=model_name,
                x_train_structured=x_structured[train_index],
                y_train=labels[train_index],
                x_test_structured=x_structured[test_index],
                y_test=labels[test_index],
                n_ant=len(args.antennas),
                n_sub=100,
                num_classes=len(position_to_idx),
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                device=device,
            )
            result.update(
                {
                    "variant": variant_name,
                    "split_mode": args.split_mode,
                    "test_groups": test_groups,
                    "device": device,
                    "n_classes": len(position_to_idx),
                    "selected_features": json.dumps(selection.selected_features),
                    "n_selected_features": len(selection.selected_features),
                    "blocked_features": json.dumps(selection.blocked_features),
                    "n_blocked_features": len(selection.blocked_features),
                }
            )
            rows.append(result)

    output_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print(f"Saved downstream KUL benchmark results to {args.output}")


if __name__ == "__main__":
    main()
