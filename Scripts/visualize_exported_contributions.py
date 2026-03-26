#!/usr/bin/env python3
"""Visualize exported PyTorch attributions/joint contributions.
Usage:
python Scripts/visualize_exported_contributions.py \
  -i models_out/classification_20260326_164025/split_seed_3/exports/MLP/seed_42/pytorch_shap_export.npz \
  -o models_out/classification_20260326_164025/split_seed_3/shap/MLP/  \
  -m MLP \
  -s 42 \
  --max-display 25  \
  --heatmap-samples 50

--o ./model_out/classificationXXX/split_seed_x/shape/<model>/seed_y (defualt)
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

try:
    import shap
except ImportError:
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


def _ensure_feature_names(feature_names: Optional[np.ndarray], n_features: int):
    if feature_names is None:
        return [f"f_{i}" for i in range(n_features)]
    return [str(f) for f in feature_names]

def _normalize_shap_matrix(shap_array: np.ndarray) -> np.ndarray:
    arr = np.asarray(shap_array)
    return np.squeeze(arr)


def _save_fig(fig: plt.Figure, path_noext: Path) -> None:
    png = path_noext.with_suffix(".png")
    svg = path_noext.with_suffix(".svg")
    fig.tight_layout()
    fig.savefig(png, bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(svg, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)


def _prepare_base_values(base_values, n_outputs: int):
    if base_values is None:
        return np.zeros((n_outputs,))
    arr = np.asarray(base_values)
    if arr.ndim == 0:
        arr = arr.reshape((1,))
    return arr


def main():
    if shap is None:
        raise SystemExit("Shap is required for visualization. Install with `pip install shap`")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", type=Path, required=True, help="Exported .npz with shap_values")
    parser.add_argument("-m", "--model-key", required=True, help="Model key used during training (e.g., MLP)")
    parser.add_argument("-s", "--seed", required=True, help="Seed identifier stored under seed_<seed>")
    parser.add_argument("-o", "--output-dir", type=Path, help="Directory for plots (default: <run_root>/shape/<model>/seed)")
    parser.add_argument("--shape-root", type=Path, help="Optional base run directory for shape outputs")
    parser.add_argument("--max-display", type=int, default=25, help="Max features to display")
    parser.add_argument("--heatmap-samples", type=int, default=64, help="Samples for the heatmap")
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    shap_values = _normalize_shap_matrix(data["shap_values"])
    features = data.get("features")
    feature_names = _ensure_feature_names(data.get("feature_names"), shap_values.shape[1])

    if features is None or features.size == 0:
        raise ValueError("Exported feature matrix is empty. Nothing to plot.")

    input_path = Path(args.input)
    if args.output_dir:
        shape_root = Path(args.output_dir)
    else:
        if args.shape_root:
            base_root = Path(args.shape_root)
        else:
            split_root = next((p for p in input_path.parents if p.name.startswith("split_seed")), None)
            base_root = split_root.parent if split_root is not None else input_path.parents[2]
        shape_root = base_root / "shape" / args.model_key / f"seed_{args.seed}"
    shape_root.mkdir(parents=True, exist_ok=True)
    output_dir = shape_root

    X_df = pd.DataFrame(features, columns=feature_names)
    sample_size = min(args.max_display * 2, len(X_df))
    if sample_size < len(X_df):
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(X_df), sample_size, replace=False)
        X_sample = X_df.iloc[sample_idx]
        shap_sample = shap_values[sample_idx]
    else:
        X_sample = X_df
        shap_sample = shap_values

    # Summary plot
    height = max(8, len(feature_names) * 0.28)
    plt.figure(figsize=(12, height))
    shap.summary_plot(
        shap_sample,
        X_sample,
        feature_names=feature_names,
        max_display=args.max_display,
        show=False,
        cmap="RdBu_r",
    )
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.tick_params(axis='x', direction='out')
    summary_fig = plt.gcf()
    summary_fig.subplots_adjust(left=0.32, right=0.98, top=0.92, bottom=0.12)
    _save_fig(summary_fig, output_dir / "exported_summary")

    # Heatmap
    heatmap_samples = min(args.heatmap_samples, len(X_sample))
    base_vals = _prepare_base_values(data.get("base_values"), shap_sample.shape[-1])
    explanation = shap.Explanation(
        values=shap_sample[:heatmap_samples],
        base_values=base_vals,
        data=X_sample.iloc[:heatmap_samples],
        feature_names=feature_names,
    )
    n_instances = explanation.values.shape[0]
    n_features = explanation.values.shape[1]
    instance_order = np.arange(n_instances)
    feature_display = min(n_features, args.max_display)
    feature_order = np.arange(feature_display)
    shap.plots.heatmap(
        explanation,
        max_display=args.max_display,
        show=False,
        cmap="RdBu_r",
        instance_order=instance_order,
        feature_order=feature_order
    )
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.tick_params(axis='x', direction='out')
    heatmap_fig = plt.gcf()
    heatmap_fig.subplots_adjust(left=0.35, right=0.98, top=0.92, bottom=0.12)
    _save_fig(heatmap_fig, output_dir / "exported_heatmap")

    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
