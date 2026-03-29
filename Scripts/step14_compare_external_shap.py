#!/usr/bin/env python3
"""Run SHAP explanations for full-development models on the External Test Set.
Usage:
    python Scripts/step14_compare_external_shap.py \
    -p models_out/classification_20260329_165120/split_seed_29 \
    -m RFC \
    -s 42 \
    --max-display 25 \
    --heatmap-samples 50

    note: -m can be sklearn model keys (XGBC, SVC, LR)
    optional:
    --task overrides the task if metadata is missing.
    --external-data points to a custom .npz split.
    --output-dir overrides the default models_out/shap/....
    --sample-size, --max-display, --heatmap-samples control SHAP sampling/plot sizes
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

try:
    import shap
except ImportError:  # pragma: no cover - user must install shap to use this script
    shap = None


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": False,
})


def _save_fig(fig: plt.Figure, path_noext: Path) -> None:
    png = path_noext.with_suffix(".png")
    svg = path_noext.with_suffix(".svg")
    fig.tight_layout()
    fig.savefig(png, bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(svg, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)


def _load_external_data(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"External data file not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        features = data.get("features")
        labels = data.get("labels")
        ids = data.get("ids")
        smiles = data.get("smiles")
        fnames = data.get("feature_names")

    if features is None or features.size == 0:
        raise ValueError("External feature matrix is empty; SHAP cannot run without numeric features.")

    return {
        "features": np.asarray(features, dtype=float),
        "labels": np.asarray(labels) if labels is not None else None,
        "ids": [str(x) for x in np.atleast_1d(ids).tolist()] if ids is not None and len(ids) else None,
        "smiles": [str(x) for x in np.atleast_1d(smiles).tolist()] if smiles is not None and len(smiles) else None,
        "feature_names": [str(x) for x in np.atleast_1d(fnames).tolist()] if fnames is not None and len(fnames) else None,
    }


def _find_model_file(seed_dir: Path) -> Path:
    for suffix in (".joblib", ".pt"):
        candidate = seed_dir / f"model{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No saved model file found in {seed_dir}")


def _load_model(model_path: Path):
    if model_path.suffix == ".joblib":
        return joblib.load(model_path)
    elif model_path.suffix == ".pt":
        raise NotImplementedError("PyTorch models require custom SHAP wiring; currently only sklearn models are supported.")
    else:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")


def _build_kmeans_background(X: pd.DataFrame, k: int):
    target_k = min(k, max(1, len(X)))
    while target_k > 0:
        try:
            background = shap.kmeans(X, k=target_k)
            if background.weights is not None and background.weights.shape[0] != background.data.shape[0]:
                background.weights = np.ones(background.data.shape[0])
            return background
        except ValueError:
            target_k -= 1
    return X


def _create_explainer(model, X_sample: pd.DataFrame, task: str):
    if shap is None:
        raise ImportError("Shap is not installed. Install with `pip install shap` to run this script.")

    def _safe_tree_explainer():
        try:
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(X_sample)
            return values, explainer
        except Exception as exc:  # pragma: no cover - compatibility issue
            raise RuntimeError(f"TreeExplainer failed ({exc}); falling back")

    if hasattr(model, 'feature_importances_'):
        try:
            return _safe_tree_explainer()
        except RuntimeError:
            pass

    if hasattr(model, 'coef_'):
        explainer = shap.LinearExplainer(model, X_sample, feature_perturbation='interventional')
        values = explainer.shap_values(X_sample)
        return values, explainer

    if hasattr(model, 'feature_importances_'):
        background = _build_kmeans_background(X_sample, k=min(50, max(1, len(X_sample))))
        try:
            explainer = shap.Explainer(model, background)
            values = explainer(X_sample)
            return values, explainer
        except TypeError:
            pass

    def predict_fn(data):
        if task == 'classification' and hasattr(model, 'predict_proba'):
            return model.predict_proba(data)[:, 1]
        return model.predict(data)

    background = _build_kmeans_background(X_sample, k=min(50, max(1, len(X_sample))))
    explainer = shap.KernelExplainer(predict_fn, background)
    values = explainer.shap_values(X_sample)
    return values, explainer


def _normalize_shap_values(shap_values, task: str):
    if isinstance(shap_values, list) and len(shap_values) > 1 and task == 'classification':
        return shap_values[1]
    if hasattr(shap_values, 'values'):
        return shap_values.values
    return shap_values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--project-root", default="models_out", help="Output directory produced by the main QSAR pipeline")
    parser.add_argument("-m", "--model-key", required=True, help="Model key (e.g., XGBC, RFC, MLP)")
    parser.add_argument("-s", "--seed", required=True, type=int, help="Seed identifier used when training the full Development model")
    parser.add_argument("-t", "--task", choices=['classification', 'regression'], help="Task type (defaults to the one stored in metadata)")
    parser.add_argument("--external-data", type=Path, help="Path to the saved external test split (.npz). Defaults to <project_root>/data/splits/external_test.npz")
    parser.add_argument("-o", "--output-dir", type=Path, help="Where to save SHAP outputs (defaults to <project_root>/shap/<model>/<seed>)")
    parser.add_argument("--sample-size", type=int, default=500, help="Maximum number of samples to include in SHAP computations")
    parser.add_argument("--max-display", type=int, default=20, help="Max features to show on summary/heatmap plots")
    parser.add_argument("--heatmap-samples", type=int, default=64, help="Number of samples to visualize in the heatmap")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    seed_dir = project_root / "models" / "full_dev" / args.model_key / f"seed_{args.seed}"
    if not seed_dir.exists():
        raise FileNotFoundError(f"Seed directory not found: {seed_dir}")

    external_data_path = args.external_data or project_root / "data" / "splits" / "external_test.npz"
    external_data_path = Path(external_data_path)
    if not external_data_path.exists():
        raise FileNotFoundError(f"External test data not found: {external_data_path}")

    output_dir = args.output_dir or project_root / "shap" / args.model_key / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = seed_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as fh:
            metadata = json.load(fh)

    model_path = _find_model_file(seed_dir)
    model_type = metadata.get('model_type', 'sklearn')

    model = _load_model(model_path)
    scaler_path = seed_dir / "scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    external_data = _load_external_data(external_data_path)
    X_ext = external_data['features']
    if scaler is not None:
        X_for_model = scaler.transform(X_ext)
    else:
        X_for_model = X_ext

    feature_names = metadata.get('feature_names') or external_data.get('feature_names')
    if feature_names is None or len(feature_names) != X_for_model.shape[1]:
        feature_names = [f"f_{i}" for i in range(X_for_model.shape[1])]

    task = args.task or metadata.get('task', 'classification')
    X_df = pd.DataFrame(X_for_model, columns=feature_names)

    sample_size = min(args.sample_size, len(X_df))
    if sample_size < len(X_df):
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(X_df), sample_size, replace=False)
        X_sample = X_df.iloc[sample_idx]
    else:
        X_sample = X_df

    shap_values, explainer = _create_explainer(model, X_sample, task)
    shap_array = _normalize_shap_values(shap_values, task)

    mean_abs_shap = np.abs(shap_array).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)
    importance_path = output_dir / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    base_value = explainer.expected_value
    if isinstance(base_value, (list, tuple, np.ndarray)):
        base_value = base_value[1] if len(base_value) > 1 else base_value[0]

    height = max(8, len(feature_names) * 0.28)
    plt.figure(figsize=(12, height))
    shap.summary_plot(shap_array, X_sample, feature_names=feature_names, max_display=args.max_display, show=False)
    plt.title(f"SHAP Summary ({args.model_key} seed={args.seed})", fontsize=14, fontweight='bold')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.tick_params(axis='x', direction='out')
    summary_fig = plt.gcf()
    summary_fig.subplots_adjust(left=0.32, right=0.98, top=0.92, bottom=0.12)
    _save_fig(summary_fig, output_dir / "shap_summary")

    heatmap_samples = min(args.heatmap_samples, len(X_sample))
    explanation = shap.Explanation(
        values=shap_array[:heatmap_samples],
        base_values=base_value,
        data=X_sample.iloc[:heatmap_samples],
        feature_names=feature_names
    )
    shap.plots.heatmap(explanation, max_display=args.max_display, show=False)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(5)) # 每 5 个实例显示一个刻度
    ax.tick_params(axis='x', direction='out', labelsize=12, pad=5) # 
    ax.set_xlabel("Instances", fontsize=14, fontfamily='Times New Roman', labelpad=10)
    heatmap_fig = plt.gcf()
    heatmap_fig.subplots_adjust(left=0.4, right=0.95, top=0.90, bottom=0.15)
    _save_fig(heatmap_fig, output_dir / "shap_heatmap")

    shap_values_path = output_dir / "shap_values.npz"
    np.savez_compressed(shap_values_path, shap_values=shap_array, feature_names=feature_names)

    print(f"SHAP outputs saved to: {output_dir}")
    print(f"  - Feature importance: {importance_path}")
    print(f"  - Summary plot: {output_dir / 'shap_summary.png'}")
    print(f"  - Heatmap: {output_dir / 'shap_heatmap.png'}")
    print(f"  - Raw SHAP values: {shap_values_path}")


if __name__ == "__main__":
    main()
