#!/usr/bin/env python3
"""Run Y-scrambling / permutation tests on a saved best model to quantify chance correlations.

Usage:
    python Scripts/step03_validate_model_robustness.py \
        -p models_out/classification_20260408_131840/split_seed_3 \
        -m Hybrid_GAT_FP \
        -s 42 \
        --data models_out/classification_20260408_131840/split_seed_3/data/splits/external_test.npz \
        --n-permutations 1 \
        --task classification

By default the script uses the metadata saved alongside the model to apply the same feature mask.
It currently supports sklearn models and the PyTorch MLP (ResidualMLP).
"""

# %%
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from torch_geometric.loader import DataLoader as GeometricDataLoader
except ImportError:
    GeometricDataLoader = None

try:
    import joblib
except ImportError:
    joblib = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from sklearn.base import clone
from sklearn.metrics import roc_auc_score, r2_score

from Scripts.step01_train_qsar_models import (
    CHEMBERTA_AVAILABLE,
    GAT_AVAILABLE,
    GATModel,
    HybridGATChemBERTaFusionModel,
    HybridGATFingerprintModel,
    MoleculeDataset,
    ResidualMLP,
    ChemBERTaEncoder,
    compute_morgan_fingerprints,
    hybrid_collate_fn,
    predict_pytorch_model,
    train_pytorch_model,
    QSARConfig,
)

# %%
# Change `font_family` here (e.g., "Times New Roman", "Cambria") for publication-ready labels.
PLOT_CONFIG = {
    "default_backend": "Agg",
    "interactive_backend": "Qt5Agg",
    "use_interactive": True,
    "display_plots": True,
    "font_family": "Cambria", # "Times New Roman" , "Cambria"
    "font_size": 11,
    "title_fontsize": 14,
    "label_fontsize": 12,
    "legend_fontsize": 10,
    "title_fontweight": "semibold",
    "label_fontweight": "normal",
    "line_width": 2.0,
    "grid_alpha": 0.15,
    "grid_color": "#d9d9d9",
    "tick_color": "#4c4c4c",
    "hist_color": "#003366",
    "hist_edgecolor": "black",
    "hist_edgewidth": 0.5,
    "hist_bins": 22,
    "hist_alpha": 1.0,
    "scatter_color": "#9467bd",
    "scatter_alpha": 0.72,
    "scatter_line_color": "#2c2c2c",
    "scatter_line_width": 1.25,
    "scatter_line_style": "-",
    "placeholder_color": "#737373",
    "dpi": 300,
}

def _has_display() -> bool:
    return bool(
        os.environ.get("DISPLAY")
        or os.environ.get("WAYLAND_DISPLAY")
        or os.environ.get("MIR_SOCKET")
    )

backend_override = os.environ.get("MATPLOTLIB_BACKEND")
if backend_override:
    _backend = backend_override
elif PLOT_CONFIG["use_interactive"] and _has_display():
    _backend = PLOT_CONFIG["interactive_backend"]
else:
    _backend = PLOT_CONFIG["default_backend"]

try:
    matplotlib.use(_backend)
except Exception as exc:  # pragma: no cover - environment dependent
    logging.getLogger(__name__).warning(
        "Unable to switch matplotlib backend to %s (%s); falling back to Agg.",
        _backend,
        exc,
    )
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import rcParams

if PLOT_CONFIG["use_interactive"] and not matplotlib.get_backend().lower().startswith("agg"):
    plt.ion()

# %%
def _configure_rcparams() -> None:
    rcParams.update({
        "font.family": "serif",
        "font.serif": [PLOT_CONFIG["font_family"]],
        "font.size": PLOT_CONFIG["font_size"],
        "axes.titlesize": PLOT_CONFIG["title_fontsize"],
        "axes.labelsize": PLOT_CONFIG["label_fontsize"],
        "legend.fontsize": PLOT_CONFIG["legend_fontsize"],
        "axes.titleweight": PLOT_CONFIG["title_fontweight"],
        "axes.labelweight": PLOT_CONFIG["label_fontweight"],
        "axes.linewidth": PLOT_CONFIG["line_width"],
        "axes.edgecolor": "#333333",
        "axes.grid": True,
        "grid.alpha": PLOT_CONFIG["grid_alpha"],
        "grid.color": PLOT_CONFIG["grid_color"],
        "grid.linestyle": "-",
        "savefig.dpi": PLOT_CONFIG["dpi"],
        "figure.dpi": PLOT_CONFIG["dpi"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "xtick.direction": "out",
        "ytick.direction": "out",
    })
    rcParams["axes.spines.right"] = False
    rcParams["axes.spines.top"] = False


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(True, alpha=PLOT_CONFIG["grid_alpha"], color=PLOT_CONFIG["grid_color"], linewidth=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(1.5)
        spine.set_visible(True)
    ax.tick_params(colors=PLOT_CONFIG["tick_color"], which="both")


def _maybe_show(fig: plt.Figure) -> None:
    if not PLOT_CONFIG["display_plots"]:
        return
    backend = matplotlib.get_backend().lower()
    if backend.startswith("agg"):
        return
    try:
        fig.canvas.draw_idle()
        plt.show(block=False)
        plt.pause(0.001)
    except Exception:
        pass


def _save_plot(fig: plt.Figure, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for suffix in (".png", ".svg"):
        fig.savefig(base_path.with_suffix(suffix), dpi=PLOT_CONFIG["dpi"])
    _maybe_show(fig)


_configure_rcparams()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_CONFIG = {
    "project_root": Path("models_out/classification_20260408_131840"),
    "split_seed": 3, #
    "model_key": "ETC", #choices: ["GAT","ChemBERTa","LR","MLP","Hybrid_GAT_FP","Hybrid_GAT_BERT"],
    "seed": 42,
    "data_path": Path("models_out/classification_20260408_131840/split_seed_3/data/splits/external_test.npz"),
    "task": "classification",
    "n_permutations": 5,
    "epochs": 20,
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "early_stopping_patience": 10,
    "output_root": Path("models_out/classification_20260408_131840/split_seed_3/validation"),
    "seed_value": 42,
}


def _setup_logger(name: str = "step03_validate_model_robustness") -> logging.Logger:
    logger = logging.getLogger(name)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def _load_npz(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        features = data.get("features")
        labels = data.get("labels")
        smiles = data.get("smiles")
        ids = data.get("ids")
        feature_names = data.get("feature_names")
    if features is None or labels is None:
        raise ValueError("Permutation data must contain 'features' and 'labels'.")
    arr: Dict[str, Any] = {
        "features": np.asarray(features, dtype=float),
        "labels": np.asarray(labels),
    }
    arr["smiles"] = [str(s) for s in np.asarray(smiles, dtype=object)] if smiles is not None else None
    arr["ids"] = [str(i) for i in np.asarray(ids, dtype=object)] if ids is not None else None
    arr["feature_names"] = [str(fn) for fn in np.asarray(feature_names, dtype=object)] if feature_names is not None else None
    return arr


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


def _permutation_attempt_limit(n_permutations: int) -> int:
    """Determine a sane number of permutation trials to collect the requested metrics."""
    return max(10, n_permutations * 4)


def _plot_histogram(values: List[float], actual: float, path: Path, title: str, xlabel: str,
                    xlim: Optional[Tuple[float, float]] = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    _style_axis(ax)
    bins = np.linspace(0.0, 1.0, 50)
    counts, bins, patches = ax.hist(
        values,
        bins=bins,
        color=PLOT_CONFIG["hist_color"],
        alpha=PLOT_CONFIG["hist_alpha"],
        edgecolor=PLOT_CONFIG["hist_edgecolor"],
        linewidth=0.3,
        rwidth=0.8,
        zorder=3,
    )
    ax.axvline(
        actual,
        color=PLOT_CONFIG["scatter_line_color"],
        linewidth=2,
        linestyle=PLOT_CONFIG["scatter_line_style"],
        label="Actual",
        zorder=4,
    )
    for patch in patches:
        patch.set_joinstyle("round")
        patch.set_zorder(3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend()
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_xlim(0.0, 1.0)
    if counts.size > 0:
        ax.set_ylim(0, max(1.0, np.nanmax(counts)) * 1.1)
    stats = np.asarray(values, dtype=float)
    median = float(np.nanmedian(stats)) if stats.size else float("nan")
    mean = float(np.nanmean(stats)) if stats.size else float("nan")
    ax.text(
        0.04,
        0.95,
        f"perm mean {mean:.3f}\nperm median {median:.3f}",
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=PLOT_CONFIG["label_fontsize"] - 1,
        fontweight="bold",
        color=PLOT_CONFIG["tick_color"],
        bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.4", alpha=0.0),
        zorder=5,
    )
    _save_plot(fig, path)
    plt.close(fig)


def _plot_scatter(correlations: List[float], metrics: List[float], path: Path, xlabel: str, ylabel: str,
                  xlim: Optional[Tuple[float, float]] = None,
                  ylim: Optional[Tuple[float, float]] = None) -> None:
    corr_arr = np.asarray(correlations)
    metric_arr = np.asarray(metrics)
    mask = np.isfinite(corr_arr) & np.isfinite(metric_arr)
    corr_arr = corr_arr[mask]
    metric_arr = metric_arr[mask]
    fig, ax = plt.subplots(figsize=(6, 6))
    _style_axis(ax)
    ax.scatter(
        corr_arr,
        metric_arr,
        color=PLOT_CONFIG["scatter_color"],
        alpha=PLOT_CONFIG["scatter_alpha"],
        edgecolors="none",
        linewidths=0,
        s=40,
        zorder=3,
    )
    if corr_arr.size >= 2:
        coeffs = np.polyfit(corr_arr, metric_arr, 1)
        line = np.poly1d(coeffs)
        xs = np.linspace(corr_arr.min(), corr_arr.max(), 50)
        ax.plot(
            xs,
            line(xs),
            color=PLOT_CONFIG["scatter_line_color"],
            linewidth=PLOT_CONFIG["scatter_line_width"],
            linestyle=PLOT_CONFIG["scatter_line_style"],
            zorder=4,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Permutation correlation vs. performance")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if corr_arr.size:
        ax.text(
            0.98,
            0.08,
            f"r̄={np.nanmean(corr_arr):.3f}",
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            color=PLOT_CONFIG["tick_color"],
            fontsize=PLOT_CONFIG["label_fontsize"] - 1,
        )
    _save_plot(fig, path)
    plt.close(fig)


def _build_hybrid_dataset(smiles: List[str], labels: np.ndarray, fp_features: Optional[np.ndarray],
                          tokenizer=None, max_length: int = 128, logger: logging.Logger = None) -> Tuple[MoleculeDataset, np.ndarray]:
    if smiles is None:
        raise ValueError("SMILES data is required for hybrid models.")
    if not smiles:
        raise ValueError("SMILES list is empty; cannot build hybrid dataset.")
    if fp_features is not None:
        fp_arr = np.asarray(fp_features, dtype=np.float32)
    else:
        fp_arr = compute_morgan_fingerprints(smiles, logger=logger)
    dataset = MoleculeDataset(
        smiles,
        labels,
        logger=logger,
        extra_fp_features=fp_arr,
        tokenizer=tokenizer,
        max_length=max_length,
        return_extra=True
    )
    if len(dataset) == 0:
        raise ValueError("No valid graphs were generated from the provided SMILES.")
    return dataset, fp_arr


def _extract_graph_dims(dataset: MoleculeDataset) -> Tuple[int, int]:
    sample = dataset.graphs[0]
    num_node_features = int(sample.x.shape[1])
    num_edge_features = int(sample.edge_attr.shape[1]) if getattr(sample, "edge_attr", None) is not None else 0
    return num_node_features, num_edge_features


def _build_gat_dataset(smiles: List[str], labels: np.ndarray, logger: logging.Logger = None) -> MoleculeDataset:
    if smiles is None:
        raise ValueError("SMILES data is required for GAT models.")
    dataset = MoleculeDataset(smiles, labels, logger=logger)
    if len(dataset) == 0:
        raise ValueError("No valid graphs were generated from the provided SMILES.")
    return dataset


def _instantiate_gat_model(model_cfg: Dict[str, Any], dataset: MoleculeDataset) -> GATModel:
    params = dict(model_cfg.get("params", {}))
    node_dim, edge_dim = _extract_graph_dims(dataset)
    params["num_node_features"] = node_dim
    params["num_edge_features"] = edge_dim
    return GATModel(**params)


def _load_shared_bert_encoder(model_path: Path, params: Dict[str, Any], logger: logging.Logger) -> ChemBERTaEncoder:
    if not CHEMBERTA_AVAILABLE:
        raise RuntimeError("ChemBERTa dependencies are not available in this environment.")
    bert_dir = model_path.with_suffix("")
    if not bert_dir.exists():
        logger.warning("Expected ChemBERTa folder missing: %s", bert_dir)
    dropout = params.get("dropout", 0.3)
    freeze_layers = params.get("freeze_transformer_layers", 0)
    encoder = ChemBERTaEncoder(
        model_name=str(bert_dir),
        dropout=dropout,
        freeze_transformer_layers=freeze_layers,
    )
    return encoder


def _instantiate_hybrid_model(model_type: str,
                              model_cfg: Dict[str, Any],
                              dataset: MoleculeDataset,
                              fingerprint_dim: int,
                              model_path: Path,
                              shared_bert_encoder: Optional[ChemBERTaEncoder],
                              logger: logging.Logger) -> nn.Module:
    if not GAT_AVAILABLE:
        raise RuntimeError("PyTorch Geometric is not available; cannot rebuild hybrid GAT model.")
    params = dict(model_cfg.get("params", {}))
    gat_params = dict(params.get("gat_params", {}))
    node_dim, edge_dim = _extract_graph_dims(dataset)
    gat_params["num_node_features"] = node_dim
    gat_params["num_edge_features"] = edge_dim
    params["gat_params"] = gat_params
    params["fingerprint_dim"] = fingerprint_dim
    if model_type == "hybrid_gat_fp":
        return HybridGATFingerprintModel(**params)
    elif model_type == "hybrid_gat_bert":
        if shared_bert_encoder is None:
            shared_bert_encoder = _load_shared_bert_encoder(model_path, params, logger)
        params["bert_encoder"] = shared_bert_encoder
        params.pop("bert_model_name", None)
        params.pop("fingerprint_dim", None)
        return HybridGATChemBERTaFusionModel(**params)
    else:
        raise NotImplementedError(f"Unsupported hybrid model type: {model_type}")


def _build_training_config(task: str, base_config: Dict[str, Any], model_cfg: Dict[str, Any]) -> QSARConfig:
    config = QSARConfig()
    config.task = task
    config.batch_size = base_config["batch_size"]
    config.max_epochs = base_config["epochs"]
    config.learning_rate = base_config["lr"]
    config.deep_learning_weight_decay = base_config.get("weight_decay", 0.0)
    config.early_stopping_patience = base_config.get("early_stopping_patience", 10)
    gat_params = model_cfg.get("params", {}).get("gat_params", {})
    config.gat_hyperparams = {
        "learning_rate": config.learning_rate,
        "weight_decay": config.deep_learning_weight_decay,
        "hidden_dim": gat_params.get("hidden_dim", 64),
        "num_heads": gat_params.get("num_heads", 4),
        "num_layers": gat_params.get("num_layers", 3),
        "dropout": gat_params.get("dropout", 0.3),
    }
    chem_params = model_cfg.get("params", {})
    config.chemberta_hyperparams = {
        "model_name": chem_params.get("bert_model_name", "DeepChem/ChemBERTa-77M-MLM"),
        "dropout": chem_params.get("dropout", 0.3),
        "freeze_transformer_layers": chem_params.get("freeze_transformer_layers", 0),
        "learning_rate": config.learning_rate,
        "weight_decay": config.deep_learning_weight_decay,
    }
    return config


def _slice_array(array: np.ndarray, indices: Optional[Sequence[int]]) -> np.ndarray:
    if indices is None:
        return array
    return array[np.asarray(indices, dtype=int)]


def _align_labels(labels: np.ndarray, valid_indices: Optional[List[int]]) -> np.ndarray:
    return _slice_array(labels, valid_indices)


# %%
def main():
    logger = _setup_logger()
    split_dir = BASE_CONFIG["project_root"] / f"split_seed_{BASE_CONFIG['split_seed']}"
    seed_dir = split_dir / "models" / "full_dev" / BASE_CONFIG["model_key"] / f"seed_{BASE_CONFIG['seed']}"
    if not seed_dir.exists():
        raise FileNotFoundError(f"Model directory missing: {seed_dir}")
    metadata = _load_metadata(seed_dir)
    model_key = metadata.get("model_key", BASE_CONFIG["model_key"])
    task = BASE_CONFIG["task"] or metadata.get("task", "classification")
    data_path = BASE_CONFIG["data_path"] or (BASE_CONFIG["project_root"] / "data" / "splits" / "external_test.npz")
    data = _load_npz(data_path)
    X = data["features"]
    y = data["labels"]
    smiles = data.get("smiles")
    mask_path = BASE_CONFIG["project_root"] / "feature_processors" / "feature_mask.npy"
    X = _apply_feature_mask(X, mask_path)
    model_path = _find_model_file(seed_dir)
    model_type = metadata.get("model_type", "sklearn")
    model_cfg_path = seed_dir / "model_config.json"
    model_cfg = json.loads(model_cfg_path.read_text()) if model_cfg_path.exists() else {}
    shared_encoder = None
    tokenizer = None
    tokenizer_max_length = 128
    if model_type == "hybrid_gat_bert":
        shared_encoder = _load_shared_bert_encoder(model_path, model_cfg, logger)
        tokenizer = shared_encoder.tokenizer
        tokenizer_max_length = getattr(shared_encoder, "max_length", 128)
    y_pred: Optional[np.ndarray] = None
    y_proba: Optional[np.ndarray] = None
    valid_indices: Optional[List[int]] = None

    if model_type == "sklearn":
        if joblib is None:
            raise RuntimeError("joblib is required for sklearn models")
        model = joblib.load(model_path)
        y_pred, y_proba = _evaluate_sklearn(model, X, task)
    elif model_type == "pytorch":
        params = model_cfg if "params" in model_cfg else {}
        mlp = _instantiate_mlp(params, X.shape[1])
        mlp.load_state_dict(torch.load(model_path, map_location="cpu"))
        y_pred, y_proba = _evaluate_pytorch(mlp, X, task, DEVICE)
        model = mlp
    elif model_type == "pytorch_geometric":
        if GeometricDataLoader is None:
            raise RuntimeError("PyTorch Geometric DataLoader is not available.")
        if smiles is None:
            raise ValueError("SMILES data missing for GAT model evaluation")
        gat_dataset = _build_gat_dataset(smiles, y, logger=logger)
        gat_model = _instantiate_gat_model(model_cfg, gat_dataset)
        state_dict = torch.load(model_path, map_location="cpu")
        gat_model.load_state_dict(state_dict)
        loader = GeometricDataLoader(gat_dataset, batch_size=BASE_CONFIG["batch_size"], shuffle=False)
        y_pred, y_proba = predict_pytorch_model(gat_model, loader, task, threshold=0.5)
        valid_indices = gat_dataset.valid_indices
        model = gat_model
    elif model_type in ["hybrid_gat_fp", "hybrid_gat_bert"]:
        smiles = data.get("smiles")
        if smiles is None:
            raise ValueError("SMILES data missing for hybrid model evaluation")
        hybrid_dataset, fp_features = _build_hybrid_dataset(
            smiles,
            y,
            X,
            tokenizer=tokenizer,
            max_length=tokenizer_max_length,
            logger=logger,
        )
        fp_dim = fp_features.shape[1]
        model = _instantiate_hybrid_model(model_type, model_cfg, hybrid_dataset, fp_dim, model_path, shared_encoder, logger)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        dataloader = DataLoader(hybrid_dataset, batch_size=BASE_CONFIG["batch_size"], shuffle=False, collate_fn=hybrid_collate_fn)
        y_pred, y_proba = predict_pytorch_model(model, dataloader, task, threshold=0.5)
        valid_indices = hybrid_dataset.valid_indices
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not supported for permutation testing")

    actual_labels = _align_labels(y, valid_indices)
    if y_pred is None or y_proba is None:
        raise RuntimeError("Failed to obtain predictions for the selected model.")
    actual_metric = _metric_from_probs(actual_labels, y_proba, task)
    rng = np.random.default_rng(BASE_CONFIG["seed_value"])
    n_permutations = max(0, int(BASE_CONFIG["n_permutations"]))
    rand_metrics: List[float] = []
    correlations: List[float] = []

    if model_type == "sklearn":
        trained_model = model  # type: ignore
        target = n_permutations
        if target <= 0:
            logger.warning("n_permutations=%d; skipping sklearn y-scrambling loop.", target)
        else:
            attempts = 0
            max_attempts = _permutation_attempt_limit(target)
            while len(rand_metrics) < target and attempts < max_attempts:
                attempts += 1
                y_rand = rng.permutation(y)
                perm_model = clone(trained_model)
                perm_model.fit(X, y_rand)
                _, perm_proba = _evaluate_sklearn(perm_model, X, task)
                perm_proba_aligned = _slice_array(perm_proba, valid_indices)
                perm_metric = _metric_from_probs(actual_labels, perm_proba_aligned, task)
                if math.isnan(perm_metric):
                    logger.warning(
                        "Sklearn permutation %d/%d produced NaN AUC (likely single-class shuffle); retrying.",
                        attempts,
                        target,
                    )
                    continue
                aligned_true = actual_labels
                aligned_rand = _slice_array(y_rand, valid_indices)
                correlations.append(_safe_corr(aligned_true, aligned_rand))
                rand_metrics.append(perm_metric)
            if len(rand_metrics) < target:
                logger.warning(
                    "Only %d/%d sklearn permutations produced valid metrics after %d attempts.",
                    len(rand_metrics),
                    target,
                    attempts,
                )
    elif model_type == "pytorch":
        params = model_cfg if "params" in model_cfg else {}
        device = DEVICE
        target = n_permutations
        if target <= 0:
            logger.warning("n_permutations=%d; skipping PyTorch permutation loop.", target)
        else:
            attempts = 0
            max_attempts = _permutation_attempt_limit(target)
            while len(rand_metrics) < target and attempts < max_attempts:
                attempts += 1
                y_rand = rng.permutation(y)
                perm_model = _train_pytorch_model(
                    params,
                    X,
                    y_rand.astype(float),
                    task,
                    device,
                    epochs=BASE_CONFIG["epochs"],
                    batch_size=BASE_CONFIG["batch_size"],
                    lr=BASE_CONFIG["lr"],
                )
                _, perm_proba = _evaluate_pytorch(perm_model, X, task, device)
                perm_proba_aligned = _slice_array(perm_proba, valid_indices)
                perm_metric = _metric_from_probs(actual_labels, perm_proba_aligned, task)
                if math.isnan(perm_metric):
                    logger.warning(
                        "PyTorch permutation %d/%d produced NaN metric (single-class); retrying.",
                        attempts,
                        target,
                    )
                    continue
                aligned_true = actual_labels
                aligned_rand = _slice_array(y_rand, valid_indices)
                correlations.append(_safe_corr(aligned_true, aligned_rand))
                rand_metrics.append(perm_metric)
            if len(rand_metrics) < target:
                logger.warning(
                    "Only %d/%d PyTorch permutations produced valid metrics after %d attempts.",
                    len(rand_metrics),
                    target,
                    attempts,
                )
    elif model_type == "pytorch_geometric":
        if GeometricDataLoader is None:
            raise RuntimeError("PyTorch Geometric DataLoader is required for GAT permutation tests.")
        if smiles is None:
            raise ValueError("SMILES data missing for GAT permutation testing")
        training_config = _build_training_config(task, BASE_CONFIG, model_cfg)
        optimizer_hyperparams = {
            "learning_rate": training_config.gat_hyperparams.get("learning_rate", training_config.learning_rate),
            "weight_decay": training_config.gat_hyperparams.get("weight_decay", training_config.deep_learning_weight_decay),
        }
        batch_size = BASE_CONFIG["batch_size"]
        target = n_permutations
        if target <= 0:
            logger.warning("n_permutations=%d; skipping PyTorch Geometric y-scrambling loop.", target)
        else:
            attempts = 0
            max_attempts = _permutation_attempt_limit(target)
            while len(rand_metrics) < target and attempts < max_attempts:
                attempts += 1
                y_rand = rng.permutation(y)
                perm_dataset = _build_gat_dataset(smiles, y_rand, logger=logger)
                train_loader = GeometricDataLoader(perm_dataset, batch_size=batch_size, shuffle=True)
                perm_model = _instantiate_gat_model(model_cfg, perm_dataset)
                perm_model = train_pytorch_model(
                    perm_model,
                    train_loader,
                    None,
                    training_config,
                    task,
                    logger,
                    model_type="pytorch_geometric",
                    optimizer_hyperparams=optimizer_hyperparams,
                )
                eval_loader = GeometricDataLoader(perm_dataset, batch_size=batch_size, shuffle=False)
                _, perm_proba = predict_pytorch_model(perm_model, eval_loader, task, threshold=0.5)
                perm_valid_indices = perm_dataset.valid_indices
                aligned_labels = _slice_array(y_rand, perm_valid_indices)
                perm_proba_aligned = _slice_array(perm_proba, perm_valid_indices)
                perm_metric = _metric_from_probs(_slice_array(y, perm_valid_indices), perm_proba_aligned, task)
                if math.isnan(perm_metric):
                    logger.warning(
                        "PyTorch Geometric permutation %d/%d produced NaN metric; retrying with a new shuffle.",
                        attempts,
                        target,
                    )
                    continue
                aligned_true = _slice_array(y, perm_valid_indices)
                correlations.append(_safe_corr(aligned_true, aligned_labels))
                rand_metrics.append(perm_metric)
            if len(rand_metrics) < target:
                logger.warning(
                    "Only %d/%d PyTorch Geometric permutations produced valid metrics after %d attempts.",
                    len(rand_metrics),
                    target,
                    attempts,
                )
    elif model_type in ["hybrid_gat_fp", "hybrid_gat_bert"]:
        hybrid_labels = y
        training_config = _build_training_config(task, BASE_CONFIG, model_cfg)
        optimizer_hyperparams = {
            "learning_rate": training_config.learning_rate,
            "weight_decay": training_config.deep_learning_weight_decay,
        }
        fp_features = X
        batch_size = BASE_CONFIG["batch_size"]
        target = n_permutations
        if target <= 0:
            logger.warning("n_permutations=%d; skipping hybrid y-scrambling loop.", target)
        else:
            attempts = 0
            max_attempts = _permutation_attempt_limit(target)
            while len(rand_metrics) < target and attempts < max_attempts:
                attempts += 1
                y_rand = rng.permutation(hybrid_labels)
                perm_dataset, _ = _build_hybrid_dataset(
                    smiles,
                    y_rand,
                    fp_features,
                    tokenizer=tokenizer,
                    max_length=tokenizer_max_length,
                    logger=logger,
                )
                train_loader = DataLoader(perm_dataset, batch_size=batch_size, shuffle=True, collate_fn=hybrid_collate_fn)
                eval_loader = DataLoader(perm_dataset, batch_size=batch_size, shuffle=False, collate_fn=hybrid_collate_fn)
                perm_model = _instantiate_hybrid_model(
                    model_type,
                    model_cfg,
                    perm_dataset,
                    fp_features.shape[1],
                    model_path,
                    None,
                    logger,
                )
                perm_model = train_pytorch_model(
                    perm_model,
                    train_loader,
                    None,
                    training_config,
                    task,
                    logger,
                    model_type="pytorch_geometric",
                    optimizer_hyperparams=optimizer_hyperparams,
                )
                _, perm_proba = predict_pytorch_model(perm_model, eval_loader, task, threshold=0.5)
                perm_valid_indices = perm_dataset.valid_indices
                aligned_labels = _slice_array(y_rand, perm_valid_indices)
                perm_proba_aligned = _slice_array(perm_proba, perm_valid_indices)
                perm_metric = _metric_from_probs(_slice_array(hybrid_labels, perm_valid_indices), perm_proba_aligned, task)
                if math.isnan(perm_metric):
                    logger.warning(
                        "Hybrid permutation %d/%d produced NaN metric; retrying with a new shuffle.",
                        attempts,
                        target,
                    )
                    continue
                aligned_true = _slice_array(hybrid_labels, perm_valid_indices)
                correlations.append(_safe_corr(aligned_true, aligned_labels))
                rand_metrics.append(perm_metric)
            if len(rand_metrics) < target:
                logger.warning(
                    "Only %d/%d hybrid permutations produced valid metrics after %d attempts.",
                    len(rand_metrics),
                    target,
                    attempts,
                )
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not supported for permutation testing")

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

    base_output = BASE_CONFIG["output_root"] / model_key / f"seed_{BASE_CONFIG['seed']}"
    output_dir = base_output
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model_key": model_key,
        "seed": BASE_CONFIG["seed"],
        "task": task,
        "actual_metric": actual_metric,
        "mean_random": mu_rand,
        "std_random": sigma_rand,
        "z_score": z_score,
        "p_value": p_value,
        "crp2": crp2 if not math.isnan(crp2) else None,
        "metric_name": "AUC" if task == "classification" else "R2",
        "n_permutations": n_permutations,
    }
    (output_dir / "permutation_summary.json").write_text(json.dumps(summary, indent=2))

    hist_xlim = (0.0, 1.0) if task == "classification" else None
    scatter_ylim = (0.0, 1.0) if task == "classification" else None
    _plot_histogram(valid_rand.tolist(), actual_metric, output_dir / "permutation_histogram",
                    "Y-scrambling performance", "Metric", xlim=hist_xlim)
    _plot_scatter(correlations, rand_metrics, output_dir / "permutation_scatter",
                  "Correlation r^2 between Y and Y_rand", "Performance",
                  xlim=(-1.0, 1.0), ylim=scatter_ylim)

    logger.info(f"Permutation summary saved to: {output_dir / 'permutation_summary.json'}")
    logger.info(f"Histogram: {output_dir / 'permutation_histogram.png'}")
    logger.info(f"Scatter: {output_dir / 'permutation_scatter.png'}")


if __name__ == "__main__":
    main()
