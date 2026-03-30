#!/usr/bin/env python3
"""Run Y-scrambling / permutation tests on a saved best model to quantify chance correlations.

Usage:
    python Scripts/step03_validate_model_robustness.py \
        -p models_out/classification_20260330_202114//split_seed_29 \
        -m RFC \
        -s 42 \
        --data models_out/classification_20260330_202114/split_seed_29/data/splits/external_test.npz \
        --n-permutations 5 \
        --task classification

By default the script uses the metadata saved alongside the model to apply the same feature mask.
It currently supports sklearn models and the PyTorch MLP (ResidualMLP)."""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import joblib
except ImportError:
    joblib = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from sklearn.base import clone
from sklearn.metrics import roc_auc_score, r2_score

from Scripts.step01_train_qsar_models import ResidualMLP


def _load_npz(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        features = data.get("features")
        labels = data.get("labels")
    if features is None or labels is None:
        raise ValueError("Permutation data must contain 'features' and 'labels'.")
    return {
        "features": np.asarray(features, dtype=float),
        "labels": np.asarray(labels)
    }


def _load_metadata(seed_dir: Path) -> Dict[str, Any]:
    meta_path = seed_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata missing: {meta_path}")
    return json.loads(meta_path.read_text())


def _find_model_file(seed_dir: Path) -> Path:
    for suffix in (".joblib", ".pt"):
        candidate = seed_dir / f"model{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No saved model artifact found in {seed_dir}")


def _apply_feature_mask(X: np.ndarray, mask_path: Optional[Path]) -> np.ndarray:
    if mask_path is None or not mask_path.exists():
        return X
    mask = np.load(mask_path)
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if mask.shape[0] != X.shape[1]:
        print("Feature mask length does not match data columns; skipping mask", file=sys.stderr)
        return X
    return X[:, mask]


def _evaluate_sklearn(model, X: np.ndarray, task: str):
    if task == "classification":
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X)
        else:
            raise AttributeError("Model lacks probability or decision function for classification")
        y_pred = (y_proba >= 0.5).astype(int)
    else:
        y_proba = model.predict(X)
        y_pred = y_proba
    return y_pred, y_proba


def _evaluate_pytorch(model: nn.Module, X: np.ndarray, task: str, device: torch.device):
    model = model.to(device).eval()
    tensor = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        output = model(tensor).squeeze()
    output = output.detach().cpu().numpy()
    if task == "classification":
        proba = 1 / (1 + np.exp(-output))
        pred = (proba >= 0.5).astype(int)
    else:
        proba = output
        pred = proba
    return pred, proba


def _instantiate_mlp(config: Dict[str, Any], input_dim: int) -> ResidualMLP:
    params = config.get("params", {})
    return ResidualMLP(
        input_dim=input_dim,
        hidden_dims=params.get("hidden_dims", [512, 256]),
        output_dim=params.get("output_dim", 1),
        dropout=params.get("dropout", 0.3),
        activation=params.get("activation", "mish"),
        use_residual=params.get("use_residual", True),
        norm_type=params.get("norm_type", "layernorm"),
    )


def _train_pytorch_model(config: Dict[str, Any], X: np.ndarray, y: np.ndarray, task: str, device: torch.device,
                         epochs: int, batch_size: int, lr: float):
    input_dim = X.shape[1]
    model = _instantiate_mlp(config, input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss() if task == "classification" else torch.nn.MSELoss()
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb).squeeze()
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
    return model


def _metric_from_probs(y_true: np.ndarray, y_proba: np.ndarray, task: str) -> float:
    if task == "classification":
        try:
            return float(roc_auc_score(y_true, y_proba))
        except ValueError:
            return float("nan")
    else:
        return float(r2_score(y_true, y_proba))


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    return float(np.nan_to_num(corr))


def _plot_histogram(values: List[float], actual: float, path: Path, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=30, color="tab:blue", alpha=0.8)
    ax.axvline(actual, color="red", linewidth=2, linestyle="--", label="Actual")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".svg"), dpi=300)
    plt.close(fig)


def _plot_scatter(correlations: List[float], metrics: List[float], path: Path, xlabel: str, ylabel: str) -> None:
    corr_arr = np.asarray(correlations)
    metric_arr = np.asarray(metrics)
    mask = np.isfinite(corr_arr) & np.isfinite(metric_arr)
    corr_arr = corr_arr[mask]
    metric_arr = metric_arr[mask]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(corr_arr, metric_arr, color="tab:purple", alpha=0.7)
    if corr_arr.size >= 2:
        coeffs = np.polyfit(corr_arr, metric_arr, 1)
        line = np.poly1d(coeffs)
        xs = np.linspace(corr_arr.min(), corr_arr.max(), 50)
        ax.plot(xs, line(xs), color="k", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Permutation correlation vs. performance")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".svg"), dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--project-root", type=Path, default=Path("models_out"))
    parser.add_argument("-m", "--model-key", required=True)
    parser.add_argument("-s", "--seed", required=True, type=int)
    parser.add_argument("--data", type=Path, help="NPZ with features/labels (default: <project>/data/splits/external_test.npz)")
    parser.add_argument("--task", choices=["classification", "regression"], help="Task override")
    parser.add_argument("--n-permutations", type=int, default=100, help="Number of Y-scrambling runs")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per PyTorch scramble")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=Path, help="Where to save validation plots (default: <project>/validation/<model>/seed_<seed>)")
    parser.add_argument("--seed-value", type=int, default=42, help="Random seed for permutations")
    args = parser.parse_args()

    seed_dir = args.project_root / "models" / "full_dev" / args.model_key / f"seed_{args.seed}"
    if not seed_dir.exists():
        raise FileNotFoundError(f"Model directory missing: {seed_dir}")

    metadata = _load_metadata(seed_dir)
    task = args.task or metadata.get("task", "classification")
    data_path = args.data or args.project_root / "data" / "splits" / "external_test.npz"
    data = _load_npz(data_path)
    X = data["features"]
    y = data["labels"]
    mask_path = args.project_root / "feature_processors" / "feature_mask.npy"
    X = _apply_feature_mask(X, mask_path)

    model_path = _find_model_file(seed_dir)
    model_type = metadata.get("model_type", "sklearn")

    if model_type == "sklearn":
        if joblib is None:
            raise RuntimeError("joblib is required for sklearn models")
        model = joblib.load(model_path)
        y_pred, y_proba = _evaluate_sklearn(model, X, task)
    elif model_type == "pytorch":
        config_path = seed_dir / "model_config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        params = config if "params" in config else {}
        mlp = _instantiate_mlp(params, X.shape[1])
        mlp.load_state_dict(torch.load(model_path, map_location="cpu"))
        y_pred, y_proba = _evaluate_pytorch(mlp, X, task, torch.device("cpu"))
        model = mlp
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not supported for permutation testing")

    actual_metric = _metric_from_probs(y, y_proba, task)
    rng = np.random.default_rng(args.seed_value)

    rand_metrics: List[float] = []
    correlations: List[float] = []

    for idx in range(args.n_permutations):
        y_rand = rng.permutation(y)
        correlations.append(_safe_corr(y, y_rand))

        if model_type == "sklearn":
            perm_model = clone(model)
            perm_model.fit(X, y_rand)
            _, perm_proba = _evaluate_sklearn(perm_model, X, task)
        else:
            perm_model = _train_pytorch_model(
                params,
                X,
                y_rand.astype(float),
                task,
                torch.device("cpu"),
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
            )
            _, perm_proba = _evaluate_pytorch(perm_model, X, task, torch.device("cpu"))

        rand_metrics.append(_metric_from_probs(y_rand, perm_proba, task))

    rand_arr = np.asarray(rand_metrics, dtype=float)
    valid_rand = rand_arr[~np.isnan(rand_arr)]
    mu_rand = float(np.mean(valid_rand)) if valid_rand.size else float("nan")
    sigma_rand = float(np.std(valid_rand, ddof=1)) if valid_rand.size > 1 else float("nan")
    z_score = float("nan")
    if not math.isnan(mu_rand) and not math.isnan(sigma_rand) and sigma_rand > 0:
        z_score = (actual_metric - mu_rand) / sigma_rand

    p_value = (np.sum(valid_rand >= actual_metric) + 1) / (len(valid_rand) + 1) if valid_rand.size else float("nan")
    crp2 = float("nan")
    if task == "regression" and not math.isnan(actual_metric) and not math.isnan(mu_rand) and actual_metric > mu_rand:
        delta = actual_metric - mu_rand
        if delta > 0:
            crp2 = actual_metric * math.sqrt(delta)

    base_output = args.output_dir or args.project_root / "validation" / args.model_key / f"seed_{args.seed}"
    output_dir = base_output
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model_key": args.model_key,
        "seed": args.seed,
        "task": task,
        "actual_metric": actual_metric,
        "mean_random": mu_rand,
        "std_random": sigma_rand,
        "z_score": z_score,
        "p_value": p_value,
        "crp2": crp2 if not math.isnan(crp2) else None,
        "metric_name": "AUC" if task == "classification" else "R2",
        "n_permutations": args.n_permutations,
    }
    (output_dir / "permutation_summary.json").write_text(json.dumps(summary, indent=2))

    _plot_histogram(valid_rand.tolist(), actual_metric, output_dir / "permutation_histogram", "Y-scrambling performance", "Metric")
    _plot_scatter(correlations, rand_metrics, output_dir / "permutation_scatter", "Correlation r^2 between Y and Y_rand", "Performance")

    print(f"Permutation summary saved to: {output_dir / 'permutation_summary.json'}")
    print(f"Histogram: {output_dir / 'permutation_histogram.png'}")
    print(f"Scatter: {output_dir / 'permutation_scatter.png'}")


if __name__ == "__main__":
    main()
