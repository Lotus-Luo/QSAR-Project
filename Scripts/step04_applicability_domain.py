#!/usr/bin/env python3
"""Applicability domain tooling: Williams plot, SOM mapping, and multimodal AD filtering.

Usage:
    python Scripts/step04_applicability_domain.py \
        -p models_out/classification_20260330_212716/split_seed_30 \
        --train-data models_out/classification_20260330_212716/split_seed_30/data/train_features.npz \
        --external-data models_out/classification_20260330_212716/split_seed_30/data/splits/external_test.npz \
        --predictions models_out/classification_20260330_212716/split_seed_30/predictions/external_test_predictions.csv \
        -m RFC -s 42

The script requires numpy-friendly NPZ files that include `features` and `labels`, and optional `ids`/`smiles`.  MiniSom must be installed for the SOM method (`pip install minisom`)."""

# %%
'''
import numpy as np

# 替换成你实际的 NPZ 路径
path = "../models_out/classification_20260408_131840/split_seed_3/data/train_features.npz"
data = np.load(path, allow_pickle=True)

features = data['features']
print(f"矩阵形状 (Samples, Features): {features.shape}")

# 查看第一行数据的前 20 列和最后 20 列
print("\n前 20 列数值 (通常是指纹):")
print(features[0, :20])

print("\n最后 20 列数值 (通常是物理描述符):")
print(features[0, -20:])

# 统计分析：如果均值接近 0，标准差接近 1，说明已经过标准化
print(f"\n全矩阵均值: {np.mean(features):.4f}")
print(f"全矩阵标准差: {np.std(features):.4f}")

# %%
# 检查每一列的最大值和最小值
col_max = np.max(features, axis=0)
col_min = np.min(features, axis=0)

# 找出数值范围异常大的列（比如范围 > 10 的）
outlier_cols = np.where((col_max - col_min) > 10)[0]
if len(outlier_cols) > 0:
    print(f"注意：发现 {len(outlier_cols)} 列特征数值范围过大，可能未经过归一化！")
    print(f"异常列索引: {outlier_cols}")
else:
    print("特征列数值范围看起来都在正常缩放区间内。")
'''
# %%
from pathlib import Path

BASE_CONFIG = {
    "project_root": Path("models_out/classification_20260408_131840"),
    "model_key": "ETC",
    "seed": 42,
    "train_data": Path("models_out/classification_20260408_131840/split_seed_3/data/train_features.npz"),
    "external_data": Path("models_out/classification_20260408_131840/split_seed_3/data/splits/external_test.npz"),
    "predictions": Path("models_out/classification_20260408_131840/split_seed_3/predictions/external_test_predictions.csv"),
    "som_rows": 12,
    "som_cols": 12,
    "som_iterations": 5000,
    "tanimoto_threshold": 0.7,
    "cosine_threshold": 0.75,
    "descriptor_columns": 20,
    "output_dir": Path("models_out/classification_20260408_131840/split_seed_3/validation/ETC/seed_42"),
    "skip_som": False,
    "use_strict_similarity": True,
    "expected_input_dim": None,
}

STYLE_CONFIG = {
    "font_family": "Cambria",
    "font_size": 8,
    "label_size": 9,
    "title_size": 10,
    "tick_size": 8,
    "dpi": 600,
    "color_palette": {
        "train": "#B0BEC5",
        "in_domain": "#2E7D32",
        "out_domain": "#D32F2F",
        "h_line": "#000000",
        "residual": "#757575",
    },
    "fig_size_single": (3.5, 3),
    "fig_size_double": (10, 5),
}

# %%
import json
import shutil
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({
    "font.family": STYLE_CONFIG["font_family"],
    "font.size": STYLE_CONFIG["font_size"],
    "axes.labelsize": STYLE_CONFIG["label_size"],
    "axes.titlesize": STYLE_CONFIG["title_size"],
    "xtick.labelsize": STYLE_CONFIG["tick_size"],
    "ytick.labelsize": STYLE_CONFIG["tick_size"],
    "figure.dpi": STYLE_CONFIG["dpi"],
    "savefig.dpi": STYLE_CONFIG["dpi"],
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.4,
    "lines.markeredgewidth": 0.3,
})

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


def _compute_leverage(X_train: np.ndarray, X_query: np.ndarray, variance_ratio: float = 0.95) -> Tuple[np.ndarray, float, int]:
    """Compute leverage after projecting data into a PCA subspace explaining variance_ratio of variance."""
    pca = PCA(n_components=variance_ratio, svd_solver="full")
    X_train_pca = pca.fit_transform(X_train)
    X_query_pca = pca.transform(X_query)
    XtX = X_train_pca.T @ X_train_pca
    inv = np.linalg.pinv(XtX)
    h_query = np.einsum("ij,jk,ik->i", X_query_pca, inv, X_query_pca)
    p = X_train_pca.shape[1]
    n = X_train_pca.shape[0]
    h_star = (3 * p) / n if n > 0 else np.inf
    return h_query, h_star, p


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


def _binary_column_indices(X: np.ndarray) -> np.ndarray:
    rounded = np.round(X)
    is_integer = np.isclose(X, rounded, atol=1e-5)
    is_binary = np.logical_or(np.isclose(rounded, 0, atol=1e-5), np.isclose(rounded, 1, atol=1e-5))
    mask = np.logical_and(np.all(is_integer, axis=0), np.all(is_binary, axis=0))
    return np.where(mask)[0]


def _tanimoto_similarity(matrix_bin: np.ndarray, vector_bin: np.ndarray) -> np.ndarray:
    intersection = matrix_bin @ vector_bin
    union = matrix_bin.sum(axis=1) + vector_bin.sum() - intersection
    union = np.where(union == 0, 1e-8, union)
    return intersection / union


def _compute_tanimoto_flags(
    X_train: np.ndarray,
    X_query: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    binary_cols = _binary_column_indices(X_train)
    if binary_cols.size == 0:
        return np.zeros(len(X_query)), np.zeros(len(X_query), dtype=bool), 0
    train_bin = np.clip(np.round(X_train[:, binary_cols]), 0, 1).astype(float)
    query_bin = np.clip(np.round(X_query[:, binary_cols]), 0, 1).astype(float)
    max_vals = []
    for x in query_bin:
        similarities = _tanimoto_similarity(train_bin, x)
        max_vals.append(np.max(similarities))
    max_arr = np.array(max_vals)
    return max_arr, max_arr >= threshold, binary_cols.size


def _validate_feature_dimensions(expected_dim: Optional[int], *arrays: np.ndarray) -> int:
    """Ensure every array has the same number of features and matches the expected input size."""
    dims = {arr.shape[1] for arr in arrays}
    if len(dims) != 1:
        raise ValueError("Feature files do not share the same number of feature columns.")
    actual_dim = dims.pop()
    if expected_dim is not None and actual_dim != expected_dim:
        raise ValueError(f"Feature dimension {actual_dim} conflicts with expected {expected_dim}.")
    return actual_dim


def _style_axis(ax: plt.Axes) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.4)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.0)
    ax.tick_params(labelsize=STYLE_CONFIG["tick_size"])


def _plot_williams(leverage: np.ndarray, residuals: np.ndarray, h_star: float, title: str, path: Path) -> None:
    palette = STYLE_CONFIG["color_palette"]
    fig, ax = plt.subplots(figsize=STYLE_CONFIG["fig_size_single"])
    ax.scatter(
        leverage,
        residuals,
        c=palette["train"],
        edgecolor="white",
        linewidth=0.3,
        alpha=0.7,
        s=24,
        label="Samples",
    )
    ax.axvline(
        h_star,
        color=palette["h_line"],
        linestyle="--",
        linewidth=0.8,
        label=f"h*={h_star:.2f}",
    )
    ax.axhline(
        3,
        color=palette["residual"],
        linestyle=":",
        linewidth=0.8,
        alpha=0.5,
    )
    ax.axhline(
        -3,
        color=palette["residual"],
        linestyle=":",
        linewidth=0.8,
        alpha=0.5,
    )
    ax.set_xlabel("Leverage (h)", size=STYLE_CONFIG["label_size"])
    ax.set_ylabel("Standardized residual", size=STYLE_CONFIG["label_size"])
    ax.set_title(title)
    _style_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            fontsize=7,
            frameon=False,
            ncol=len(handles),
        )
    span_min = np.floor(np.min(leverage) / 0.25) * 0.25
    span_max = np.ceil(np.max(leverage) / 0.25) * 0.25
    ticks = np.arange(span_min, span_max + 1e-6, 0.25)
    if len(ticks) > 0:
        ax.set_xticks(ticks)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=STYLE_CONFIG["dpi"])
    fig.savefig(path.with_suffix(".svg"), dpi=STYLE_CONFIG["dpi"])
    plt.close(fig)


def _plot_pca_tsne(X_train: np.ndarray, X_external: np.ndarray, in_domain: np.ndarray, path: Path) -> None:
    pca = PCA(n_components=2)
    pcs_train = pca.fit_transform(X_train)
    pcs_external = pca.transform(X_external)
    tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42)
    tsne_emb = tsne.fit_transform(np.vstack([X_train, X_external]))
    tsne_train = tsne_emb[:len(X_train)]
    tsne_external = tsne_emb[len(X_train):]

    palette = STYLE_CONFIG["color_palette"]
    fig, axes = plt.subplots(1, 2, figsize=STYLE_CONFIG["fig_size_double"])
    for ax, title, emb in zip(
        axes,
        ("PCA: Training vs. External", "t-SNE: Application Set"),
        (pcs_train, tsne_train),
    ):
        ax.scatter(
            emb[:, 0],
            emb[:, 1],
            color=palette["train"],
            s=18,
            edgecolor="white",
            linewidth=0.3,
            alpha=0.55,
            label="Train",
        )
        ax.set_title(title, size=STYLE_CONFIG["title_size"])
        ax.set_xlabel("Component 1", size=STYLE_CONFIG["label_size"])
        ax.set_ylabel("Component 2", size=STYLE_CONFIG["label_size"])
    axes[0].scatter(
        pcs_external[in_domain, 0],
        pcs_external[in_domain, 1],
        c=palette["in_domain"],
        marker="D",
        edgecolor="white",
        linewidth=0.4,
        s=20,
        label="External (in-domain)",
    )
    axes[0].scatter(
        pcs_external[~in_domain, 0],
        pcs_external[~in_domain, 1],
        c=palette["out_domain"],
        marker="D",
        edgecolor="white",
        linewidth=0.4,
        s=20,
        label="External (out-of-domain)",
    )
    axes[1].scatter(
        tsne_external[in_domain, 0],
        tsne_external[in_domain, 1],
        c=palette["in_domain"],
        marker="D",
        edgecolor="white",
        linewidth=0.4,
        s=20,
        label="External (in-domain)",
    )
    axes[1].scatter(
        tsne_external[~in_domain, 0],
        tsne_external[~in_domain, 1],
        c=palette["out_domain"],
        marker="D",
        edgecolor="white",
        linewidth=0.4,
        s=20,
        label="External (out-of-domain)",
    )
    for ax in axes:
        _style_axis(ax)
    axes[0].legend(
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        fontsize=7,
        frameon=False,
    )
    axes[1].legend(
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        fontsize=7,
        frameon=False,
    )
    fig.subplots_adjust(wspace=0.3)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=STYLE_CONFIG["dpi"])
    fig.savefig(path.with_suffix(".svg"), dpi=STYLE_CONFIG["dpi"])
    plt.close(fig)


def _plot_som_density(som: MiniSom, X_train: np.ndarray, X_external: np.ndarray, path: Path) -> None:
    """Render the SOM U-Matrix with training/external sample occupancy markers."""
    distance_map = som.distance_map()
    train_winners = np.array([som.winner(x) for x in X_train])
    external_winners = np.array([som.winner(x) for x in X_external])
    fig, ax = plt.subplots(figsize=STYLE_CONFIG["fig_size_double"])
    im = ax.imshow(distance_map, cmap="magma", origin="lower")
    if len(train_winners) > 0:
        ax.scatter(
            train_winners[:, 1],
            train_winners[:, 0],
            c=STYLE_CONFIG["color_palette"]["train"],
            s=15,
            edgecolor="white",
            alpha=0.9,
            label="Train samples",
        )
    if len(external_winners) > 0:
        ax.scatter(
            external_winners[:, 1],
            external_winners[:, 0],
            c=STYLE_CONFIG["color_palette"]["out_domain"],
            marker="x",
            s=15,
            linewidths=2,
            label="External samples",
        )
    ax.set_title("SOM U-Matrix Density", size=STYLE_CONFIG["title_size"])
    ax.set_xlabel("Columns", size=STYLE_CONFIG["label_size"])
    ax.set_ylabel("Rows", size=STYLE_CONFIG["label_size"])
    _style_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            fontsize=7,
            frameon=False,
            ncol=len(handles),
        )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Inter-neuron distance")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=STYLE_CONFIG["dpi"])
    fig.savefig(path.with_suffix(".svg"), dpi=STYLE_CONFIG["dpi"])
    plt.close(fig)


def _plot_som_umatrix(som: MiniSom, X_train: np.ndarray, path: Path) -> None:
    """Render the SOM U-Matrix with training sample positions overlaid."""
    distance_map = som.distance_map()
    winners = np.array([som.winner(x) for x in X_train])
    fig, ax = plt.subplots(figsize=STYLE_CONFIG["fig_size_double"])
    im = ax.imshow(distance_map, cmap="magma", origin="lower")
    ax.scatter(
        winners[:, 1],
        winners[:, 0],
        c=STYLE_CONFIG["color_palette"]["train"],
        s=15,
        edgecolor="white",
        alpha=0.75,
        label="Train samples",
    )
    ax.set_title("SOM U-Matrix + Training Samples")
    ax.set_xlabel("Columns", size=STYLE_CONFIG["label_size"])
    ax.set_ylabel("Rows", size=STYLE_CONFIG["label_size"])
    _style_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            fontsize=7,
            frameon=False,
            ncol=len(handles),
        )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Inter-neuron distance")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=STYLE_CONFIG["dpi"])
    fig.savefig(path.with_suffix(".svg"), dpi=STYLE_CONFIG["dpi"])
    plt.close(fig)


def _backup_predictions(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy(path, backup)


def main():
    train_path = BASE_CONFIG["train_data"] or BASE_CONFIG["project_root"] / "data" / "train_features.npz"
    external_path = BASE_CONFIG["external_data"] or BASE_CONFIG["project_root"] / "data" / "splits" / "external_test.npz"
    predictions_path = BASE_CONFIG["predictions"] or BASE_CONFIG["project_root"] / "predictions" / "external_test_predictions.csv"

    train_data = _load_npz(train_path)
    external_data = _load_npz(external_path)
    X_train_raw = train_data["features"]
    X_external_raw = external_data["features"]
    feature_dim = _validate_feature_dimensions(BASE_CONFIG.get("expected_input_dim"), X_train_raw, X_external_raw)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_external_scaled = scaler.transform(X_external_raw)
    descriptor_cols = BASE_CONFIG["descriptor_columns"]
    if descriptor_cols <= 0 or descriptor_cols >= feature_dim:
        raise ValueError("descriptor_columns must be positive and smaller than total feature count")
    X_fp_train = X_train_raw[:, :-descriptor_cols]
    X_fp_external = X_external_raw[:, :-descriptor_cols]

    preds = pd.read_csv(predictions_path)
    mask = preds["model"].astype(str).str.upper() == BASE_CONFIG["model_key"].upper()
    if not mask.any():
        raise ValueError(f"No predictions found for model {BASE_CONFIG['model_key']} in {predictions_path}")
    ad_preds = preds.loc[mask].copy()
    if len(ad_preds) != len(X_external_raw):
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

    leverage, h_star, n_components = _compute_leverage(X_train_scaled, X_external_scaled)
    std_resid = _standardized_residuals(y_true, pred_values)
    leverage_flag = leverage <= h_star

    som_flag = np.ones(len(X_external_scaled), dtype=bool)
    som: Optional[MiniSom] = None
    if not BASE_CONFIG.get("skip_som", False):
        som, occupied = _train_som(X_train_scaled, BASE_CONFIG["som_rows"], BASE_CONFIG["som_cols"], BASE_CONFIG["som_iterations"])
        som_flag = _compute_som_flags(som, occupied, X_external_scaled)
    tanimoto_scores, tanimoto_flag, binary_cols = _compute_tanimoto_flags(
        X_fp_train,
        X_fp_external,
        BASE_CONFIG["tanimoto_threshold"],
    )
    cosine_matrix = cosine_similarity(X_external_scaled, X_train_scaled)
    cosine_scores = np.max(cosine_matrix, axis=1)
    cosine_flag = cosine_scores >= BASE_CONFIG["cosine_threshold"]
    total_external = len(X_external_scaled)
    def _stat_line(name: str, mask: np.ndarray) -> str:
        count = int(np.sum(mask))
        pct = count / total_external if total_external else 0.0
        return f"{name}: {count}/{total_external} ({pct:.1%})"
    print(_stat_line("Leverage pass rate", leverage_flag))
    print(_stat_line("SOM pass rate", som_flag))
    print(_stat_line("Tanimoto pass rate", tanimoto_flag))
    print(_stat_line("Cosine pass rate", cosine_flag))
    print(f"PCA components used: {n_components}")
    print(f"Binary fingerprint columns: {binary_cols}")

    in_domain = leverage_flag & som_flag
    if BASE_CONFIG.get("use_strict_similarity", True):
        in_domain &= tanimoto_flag & cosine_flag

    ad_preds["Leverage"] = leverage
    ad_preds["StdResidual"] = std_resid
    ad_preds["Williams_Outlier"] = np.abs(std_resid) > 3
    ad_preds["Leverage_In_Domain"] = leverage_flag
    ad_preds["SOM_In_Domain"] = som_flag
    ad_preds["Tanimoto_max"] = tanimoto_scores
    ad_preds["Tanimoto_In_Domain"] = tanimoto_flag
    ad_preds["Cosine_max"] = cosine_scores
    ad_preds["Cosine_In_Domain"] = cosine_flag
    ad_preds["In_Domain"] = in_domain
    preds.loc[mask, ad_preds.columns] = ad_preds

    _backup_predictions(predictions_path)
    preds.to_csv(predictions_path, index=False)

    output_dir = BASE_CONFIG["output_dir"] or BASE_CONFIG["project_root"] / "validation" / BASE_CONFIG["model_key"] / f"seed_{BASE_CONFIG['seed']}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model_key": BASE_CONFIG["model_key"],
        "seed": BASE_CONFIG["seed"],
        "h_star": h_star,
        "n_training_samples": len(X_train_raw),
        "n_external_samples": len(X_external_raw),
        "in_domain_fraction": float(np.mean(in_domain)),
        "feature_dim": feature_dim,
        "descriptor_columns": descriptor_cols,
        "pca_components": n_components,
        "binary_feature_columns": binary_cols,
        "tanimoto_threshold": BASE_CONFIG["tanimoto_threshold"],
        "cosine_threshold": BASE_CONFIG["cosine_threshold"],
        "use_strict_similarity": BASE_CONFIG.get("use_strict_similarity", True),
    }
    (output_dir / "ad_summary.json").write_text(json.dumps(summary, indent=2))

    _plot_williams(leverage, std_resid, h_star, "Williams Plot", output_dir / "williams_plot")
    if som is not None:
        _plot_som_umatrix(som, X_train_scaled, output_dir / "som_u_matrix")
        _plot_som_density(som, X_train_scaled, X_external_scaled, output_dir / "som_density")
    _plot_pca_tsne(X_train_raw, X_external_raw, in_domain, output_dir / "pca_tsne")

    figures = [output_dir / 'williams_plot.png', output_dir / 'pca_tsne.png']
    if som is not None:
        figures.append(output_dir / 'som_u_matrix.png')
        figures.append(output_dir / 'som_density.png')
    print(f"Updated predictions CSV with In_Domain flag: {predictions_path}")
    print(f"AD summary saved to: {output_dir / 'ad_summary.json'}")
    print(f"Figures: {', '.join(str(f) for f in figures)}")



if __name__ == "__main__":
    main()
