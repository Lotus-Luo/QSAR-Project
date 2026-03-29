#!/usr/bin/env python3
"""Applicability domain tooling: Williams plot, SOM mapping, Tanimoto clustering, and AD-aware prediction exporting.

Usage:
    python Scripts/step04_applicability_domain.py \
        -p models_out/classification_20260329_221753/split_seed_29 \
        --train-data models_out/classification_20260329_221753/split_seed_29/data/train_features.npz \
        --external-data models_out/classification_20260329_221753/split_seed_29/data/splits/external_test.npz \
        --predictions models_out/classification_20260329_221753/split_seed_29/predictions/external_test_predictions.csv \
        -m RFC -s 42

The script requires numpy-friendly NPZ files that include `features` and `labels`, and optional `ids`/`smiles`.  MiniSom must be installed for the SOM method (`pip install minisom`)."""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    from minisom import MiniSom
except ImportError:
    MiniSom = None  # type: ignore


def _load_npz(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        features = data.get("features")
        if features is None:
            raise ValueError("NPZ must contain 'features' array")
        return {
            "features": np.asarray(features, dtype=float),
            "labels": data.get("labels"),
            "ids": np.atleast_1d(data.get("ids")) if data.get("ids") is not None else None,
            "smiles": np.atleast_1d(data.get("smiles")) if data.get("smiles") is not None else None,
        }


def _compute_leverage(X_train: np.ndarray, X_query: np.ndarray) -> Tuple[np.ndarray, float]:
    XtX = X_train.T @ X_train
    inv = np.linalg.pinv(XtX)
    h_query = np.einsum("ij,jk,ik->i", X_query, inv, X_query)
    p = X_train.shape[1]
    n = X_train.shape[0]
    h_star = (3 * p) / n if n > 0 else np.inf
    return h_query, h_star


def _standardized_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    resid = y_true - y_pred
    sigma = np.std(resid, ddof=1)
    if sigma == 0 or np.isnan(sigma):
        sigma = 1.0
    return resid / sigma


def _train_som(X_train: np.ndarray, rows: int, cols: int, iterations: int) -> Tuple[MiniSom, set]:
    if MiniSom is None:
        raise SystemExit("MiniSom is required for SOM AD analysis. Install with `pip install minisom`." )
    som = MiniSom(rows, cols, X_train.shape[1], sigma=1.0, learning_rate=0.5)
    som.random_weights_init(X_train)
    som.train_random(X_train, iterations)
    occupied = {som.winner(x) for x in X_train}
    return som, occupied


def _compute_som_flags(som: MiniSom, occupied: set, X_query: np.ndarray) -> np.ndarray:
    flags = []
    for x in X_query:
        winner = som.winner(x)
        flags.append(winner in occupied)
    return np.array(flags, dtype=bool)


def _binarize(X: np.ndarray, threshold: float) -> np.ndarray:
    return (X >= threshold).astype(float)


def _tanimoto_similarity(matrix_bin: np.ndarray, vector_bin: np.ndarray) -> np.ndarray:
    intersection = matrix_bin @ vector_bin
    union = matrix_bin.sum(axis=1) + vector_bin.sum() - intersection
    union = np.where(union == 0, 1e-8, union)
    return intersection / union


def _compute_tanimoto_flags(X_train: np.ndarray, X_query: np.ndarray, binary_threshold: float,
                           tanimoto_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    train_bin = _binarize(X_train, binary_threshold)
    max_vals = []
    for x in X_query:
        vec_bin = _binarize(x.reshape(1, -1), binary_threshold).flatten()
        similarities = _tanimoto_similarity(train_bin, vec_bin)
        max_vals.append(np.max(similarities))
    max_arr = np.array(max_vals)
    return max_arr, max_arr >= tanimoto_threshold


def _plot_williams(leverage: np.ndarray, residuals: np.ndarray, h_star: float, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(leverage, residuals, c="tab:blue", edgecolor="k", alpha=0.7)
    ax.axvline(h_star, color="tab:orange", linestyle="--", label=f"h*={h_star:.2f}")
    ax.axhline(3, color="red", linestyle=":")
    ax.axhline(-3, color="red", linestyle=":")
    ax.set_xlabel("Leverage (h)")
    ax.set_ylabel("Standardized residual")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".svg"), dpi=300)
    plt.close(fig)


def _plot_pca_tsne(X_train: np.ndarray, X_external: np.ndarray, in_domain: np.ndarray, path: Path) -> None:
    pca = PCA(n_components=2)
    pcs_train = pca.fit_transform(X_train)
    pcs_external = pca.transform(X_external)
    tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42)
    tsne_emb = tsne.fit_transform(np.vstack([X_train, X_external]))
    tsne_train = tsne_emb[:len(X_train)]
    tsne_external = tsne_emb[len(X_train):]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(pcs_train[:, 0], pcs_train[:, 1], color="lightgray", s=15, label="Train")
    axes[0].scatter(pcs_external[in_domain, 0], pcs_external[in_domain, 1], c="tab:green", s=30, label="In-domain")
    axes[0].scatter(pcs_external[~in_domain, 0], pcs_external[~in_domain, 1], c="tab:red", s=30, label="Out-of-domain")
    axes[0].set_title("PCA: Training vs. External")
    axes[0].legend()
    axes[1].scatter(tsne_train[:, 0], tsne_train[:, 1], color="lightgray", s=15, label="Train")
    axes[1].scatter(tsne_external[in_domain, 0], tsne_external[in_domain, 1], c="tab:green", s=30, label="In-domain")
    axes[1].scatter(tsne_external[~in_domain, 0], tsne_external[~in_domain, 1], c="tab:red", s=30, label="Out-of-domain")
    axes[1].set_title("t-SNE: Application Set")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".svg"), dpi=300)
    plt.close(fig)


def _backup_predictions(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy(path, backup)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--project-root", type=Path, default=Path("models_out"))
    parser.add_argument("-m", "--model-key", required=True)
    parser.add_argument("-s", "--seed", required=True, type=int)
    parser.add_argument("--train-data", type=Path, help="NPZ with training features (defaults to <project>/data/train_features.npz)")
    parser.add_argument("--external-data", type=Path, help="NPZ with external features (defaults to <project>/data/splits/external_test.npz)")
    parser.add_argument("--predictions", type=Path, help="CSV predictions file to annotate (defaults to <project>/predictions/external_test_predictions.csv)")
    parser.add_argument("--som-rows", type=int, default=8)
    parser.add_argument("--som-cols", type=int, default=8)
    parser.add_argument("--som-iterations", type=int, default=1000)
    parser.add_argument("--tanimoto-threshold", type=float, default=0.5)
    parser.add_argument("--binary-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, help="Where to save figures/statistics (default: <project>/validation/<model>/seed_<seed>)")
    parser.add_argument("--skip-som", action="store_true", help="Skip the SOM-based analysis (MiniSom not required)")
    args = parser.parse_args()

    train_path = args.train_data or args.project_root / "data" / "train_features.npz"
    external_path = args.external_data or args.project_root / "data" / "splits" / "external_test.npz"
    predictions_path = args.predictions or args.project_root / "predictions" / "external_test_predictions.csv"

    train_data = _load_npz(train_path)
    external_data = _load_npz(external_path)
    X_train = train_data["features"]
    X_external = external_data["features"]

    preds = pd.read_csv(predictions_path)
    mask = preds["model"].astype(str).str.upper() == args.model_key.upper()
    if not mask.any():
        raise ValueError(f"No predictions found for model {args.model_key} in {predictions_path}")
    ad_preds = preds.loc[mask].copy()
    if len(ad_preds) != len(X_external):
        raise ValueError("Number of external samples does not match predictions subset; check data alignment")
    if "predicted_probability" in ad_preds.columns:
        pred_values = ad_preds["predicted_probability"].to_numpy(dtype=float)
    elif "predicted_value" in ad_preds.columns:
        pred_values = ad_preds["predicted_value"].to_numpy(dtype=float)
    elif "predicted_label" in ad_preds.columns:
        pred_values = ad_preds["predicted_label"].to_numpy(dtype=float)
    else:
        raise ValueError("Predictions CSV must contain predicted_probability/predicted_value/predicted_label")
    y_true = ad_preds["true_label"].to_numpy(dtype=float)

    leverage, h_star = _compute_leverage(X_train, X_external)
    std_resid = _standardized_residuals(y_true, pred_values)
    leverage_flag = leverage <= h_star

    som_flag = np.ones(len(X_external), dtype=bool)
    if not args.skip_som:
        som, occupied = _train_som(X_train, args.som_rows, args.som_cols, args.som_iterations)
        som_flag = _compute_som_flags(som, occupied, X_external)
    similarity_scores, similarity_flag = _compute_tanimoto_flags(
        X_train,
        X_external,
        args.binary_threshold,
        args.tanimoto_threshold,
    )

    in_domain = leverage_flag & som_flag & similarity_flag

    ad_preds["Leverage"] = leverage
    ad_preds["StdResidual"] = std_resid
    ad_preds["Williams_Outlier"] = np.abs(std_resid) > 3
    ad_preds["Leverage_In_Domain"] = leverage_flag
    ad_preds["SOM_In_Domain"] = som_flag
    ad_preds["Tanimoto_max"] = similarity_scores
    ad_preds["Similarity_In_Domain"] = similarity_flag
    ad_preds["In_Domain"] = in_domain
    preds.loc[mask, ad_preds.columns] = ad_preds

    _backup_predictions(predictions_path)
    preds.to_csv(predictions_path, index=False)

    output_dir = args.output_dir or args.project_root / "validation" / args.model_key / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model_key": args.model_key,
        "seed": args.seed,
        "h_star": h_star,
        "n_training_samples": len(X_train),
        "n_external_samples": len(X_external),
        "in_domain_fraction": float(np.mean(in_domain)),
        "tanimoto_threshold": args.tanimoto_threshold,
    }
    (output_dir / "ad_summary.json").write_text(json.dumps(summary, indent=2))

    _plot_williams(leverage, std_resid, h_star, "Williams Plot", output_dir / "williams_plot")
    _plot_pca_tsne(X_train, X_external, in_domain, output_dir / "pca_tsne")

    print(f"Updated predictions CSV with In_Domain flag: {predictions_path}")
    print(f"AD summary saved to: {output_dir / 'ad_summary.json'}")
    print(f"Figures: {output_dir / 'williams_plot.png'}, {output_dir / 'pca_tsne.png'}")


if __name__ == "__main__":
    main()
