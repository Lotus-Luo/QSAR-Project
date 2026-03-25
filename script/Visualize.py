"""
Aggregate split-seed metrics and predictions for publication-ready figures.

Usage:
    python aggregate_split_predictions.py \
  --base-dir models_out/classification_20260325_141059 \
  --include-external \
  --include-cv \
  --boxplot-stage both


Features:
  * Nature/Science–style ROC/PR panels (white background, Times New Roman, 1.5pt lines, alpha=0.1 shading).
  * Multi-metric boxplots (MCC, F1, ACC, AUC, PR-AUC, EF%) with swarm overlays showing every split_seed datapoint.
  * Config dict + CLI toggles to enable/disable CV/External plots and customize metrics.
  * SVG outputs saved at 300+ DPI for perfect vector quality.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from sklearn.metrics import auc, precision_recall_curve, roc_curve

DEFAULT_CONFIG: Dict[str, object] = {
    "include_external": True,
    "include_cv": False,
    "boxplot_stage": "external",  # choices: external | cv | both
    "boxplot_metrics": ["MCC", "F1", "ACC", "AUC", "PR_AUC", "EF5%"],
    "dpi": 600,
    "palette": "colorblind",
    "font": "Times New Roman"
}


def _configure_plotting(font: str):
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = font
    rcParams["text.usetex"] = False
    rcParams["axes.edgecolor"] = "black"
    rcParams["axes.linewidth"] = 1.2
    rcParams["axes.facecolor"] = "white"
    rcParams["grid.color"] = "white"
    rcParams["figure.facecolor"] = "white"
    rcParams["savefig.facecolor"] = "white"


def _collect_prediction_files(base_dir: Path, stage: str) -> List[Path]:
    split_dirs = sorted(p for p in base_dir.glob("split_seed_*") if p.is_dir())
    files = []
    for split_dir in split_dirs:
        seed = split_dir.name.split("_")[-1]
        if stage == "external":
            candidate = split_dir / "predictions" / f"external_test_predictions_seed_{seed}.csv"
            if candidate.exists():
                files.append(candidate)
        elif stage == "cv":
            for match in sorted((split_dir / "results").glob("cv_predictions_fold_*.csv")):
                files.append(match)
    return files


def _interpolate_curve(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, x, y, left=y[0], right=y[-1])


def _prepare_curves(prediction_files: List[Path]) -> Dict[str, Dict[str, List[np.ndarray]]]:
    curves: Dict[str, Dict[str, List[np.ndarray]]] = {}
    fpr_grid = np.linspace(0, 1, 400)
    recall_grid = np.linspace(0, 1, 400)

    def _record_curve(model_name: str, y_true_arr: np.ndarray, scores_arr: np.ndarray):
        if len(scores_arr) == 0 or np.all(np.isnan(scores_arr)):
            return
        fpr, tpr, _ = roc_curve(y_true_arr, scores_arr)
        precision, recall, _ = precision_recall_curve(y_true_arr, scores_arr)
        entry = curves.setdefault(model_name, {'roc': [], 'pr': []})
        entry['roc'].append(_interpolate_curve(fpr, tpr, fpr_grid))
        entry.setdefault('roc_grid', fpr_grid)
        entry['pr'].append(_interpolate_curve(recall[::-1], precision[::-1], recall_grid))
        entry.setdefault('pr_grid', recall_grid)

    for path in prediction_files:
        data = pd.read_csv(path)
        if 'true_label' not in data.columns:
            continue
        if any(col.endswith("_score") for col in data.columns):
            y_true = data['true_label'].astype(float).to_numpy()
            model_cols = [col for col in data.columns if col.endswith("_score")]
            for model_col in model_cols:
                scores = data[model_col].astype(float).to_numpy()
                _record_curve(model_col.replace("_score", ""), y_true, scores)
        elif {'model', 'predicted_probability'}.issubset(data.columns):
            for model_name, group in data.groupby("model"):
                y_true = group['true_label'].astype(float).to_numpy()
                scores = group['predicted_probability'].astype(float).to_numpy()
                _record_curve(model_name, y_true, scores)
        else:
            continue
    return curves


def _plot_roc_pr(curves: Dict[str, Dict[str, List[np.ndarray]]],
                 output_path: Path,
                 stage: str,
                 palette_name: str,
                 dpi: int,
                 font: str):
    if not curves:
        return
    _configure_plotting(font)
    palette = sns.color_palette(palette_name)
    colors = palette * 3
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for idx, (model, data) in enumerate(sorted(curves.items())):
        if not data['roc']:
            continue
        color = colors[idx % len(colors)]
        fpr_grid = data['roc_grid']
        mean_tpr = np.mean(data['roc'], axis=0)
        std_tpr = np.std(data['roc'], axis=0)
        auc_vals = [auc(fpr_grid, arr) for arr in data['roc']]
        auc_mean = np.mean(auc_vals)
        auc_std = np.std(auc_vals)
        axes[0].plot(
            fpr_grid,
            mean_tpr,
            label=f"{model} (AUC={auc_mean:.3f}±{auc_std:.3f})",
            linewidth=1.5,
            color=color
        )
        axes[0].fill_between(
            fpr_grid,
            np.clip(mean_tpr - std_tpr, 0, 1),
            np.clip(mean_tpr + std_tpr, 0, 1),
            color=color,
            alpha=0.1,
            linewidth=0
        )

        recall_grid = data['pr_grid']
        mean_prec = np.mean(data['pr'], axis=0)
        std_prec = np.std(data['pr'], axis=0)
        pr_auc_vals = [auc(recall_grid, arr) for arr in data['pr']]
        pr_auc_mean = np.mean(pr_auc_vals)
        pr_auc_std = np.std(pr_auc_vals)
        axes[1].plot(
            recall_grid,
            mean_prec,
            label=f"{model} (AUC={pr_auc_mean:.3f}±{pr_auc_std:.3f})",
            linewidth=1.5,
            color=color
        )
        axes[1].fill_between(
            recall_grid,
            np.clip(mean_prec - std_prec, 0, 1),
            np.clip(mean_prec + std_prec, 0, 1),
            color=color,
            alpha=0.1,
            linewidth=0
        )

    axes[0].set_title(f"{stage.capitalize()} ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(loc="lower right", frameon=False)
    axes[0].grid(False)

    axes[1].set_title(f"{stage.capitalize()} PR Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(loc="lower left", frameon=False)
    axes[1].grid(False)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)
    sns.despine(fig=fig, left=False, bottom=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", dpi=dpi)
    plt.close(fig)


def _metric_column(stage_df: pd.DataFrame, metric: str, stage: str) -> Optional[str]:
    if stage == "external":
        return metric if metric in stage_df.columns else None
    candidates = [
        f"{metric}_val_mean",
        f"{metric}_mean",
        f"{metric}_val",
        f"{metric}_train_mean",
    ]
    for col in candidates:
        if col in stage_df.columns:
            return col
    return None


def _prepare_metric_dataframe(row_data_path: Path, stage: str, metrics: List[str]) -> pd.DataFrame:
    if not row_data_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(row_data_path)
    stage_df = df[df['stage'] == stage].copy()
    if stage_df.empty:
        return pd.DataFrame()
    data = []
    for metric in metrics:
        column = _metric_column(stage_df, metric, stage)
        if column is None:
            continue
        for _, row in stage_df.iterrows():
            value = row.get(column)
            if pd.isna(value):
                continue
            data.append({
                "stage": stage,
                "metric": metric,
                "model": row["model"],
                "split_seed": row.get("split_seed"),
                "value": float(value)
            })
    return pd.DataFrame(data)


def _plot_metric_boxplots(metric_df: pd.DataFrame, metrics: List[str], stage: str, output_path: Path,
                          palette_name: str, dpi: int, font: str):
    if metric_df.empty:
        return
    _configure_plotting(font)
    palette = sns.color_palette(palette_name)
    fig, axes = plt.subplots(3, 2, figsize=(18, 12), sharey=False)
    axes = axes.flatten()
    unique_models = sorted(metric_df['model'].unique())
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        sub_df = metric_df[metric_df['metric'] == metric]
        if sub_df.empty:
            ax.set_visible(False)
            continue
        sns.boxplot(
            x="model",
            y="value",
            data=sub_df,
            ax=ax,
            order=unique_models,
            palette=palette,
            showcaps=True,
            boxprops={'alpha': 0.7},
            showfliers=False
        )
        sns.swarmplot(
            x="model",
            y="value",
            data=sub_df,
            ax=ax,
            order=unique_models,
            color="black",
            size=3,
            alpha=0.45
        )
        ax.set_title(f"{stage.capitalize()} {metric}")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(False)
    for idx in range(len(metrics), len(axes)):
        axes[idx].set_visible(False)
    sns.despine(fig=fig, left=True, bottom=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Aggregate split seed metrics with publication styling.")
    parser.add_argument("--base-dir", required=True, help="Run output directory containing split_seed_* folders.")
    parser.add_argument("--output-dir", default=None, help="Directory to save resulting SVGs (defaults to base_dir/figures).")
    parser.add_argument("--include-external", action="store_true", help="Enable External-stage curves/metrics.")
    parser.add_argument("--include-cv", action="store_true", help="Enable CV-stage curves/metrics.")
    parser.add_argument("--boxplot-stage", choices=["external", "cv", "both"], default="external",
                        help="Which stage(s) to render the multi-metric boxplots for.")
    parser.add_argument("--metrics", help="Comma-separated list of metric names for boxplots.")
    parser.add_argument("--dpi", type=int, help="DPI used when saving the SVGs (vector format).")
    parser.add_argument("--palette", choices=["colorblind", "deep"], default="colorblind",
                        help="Seaborn palette for lines and boxes.")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["include_external"] = args.include_external or config["include_external"]
    config["include_cv"] = args.include_cv or config["include_cv"]
    config["boxplot_stage"] = args.boxplot_stage
    if args.metrics:
        config["boxplot_metrics"] = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if args.dpi:
        config["dpi"] = args.dpi
    config["palette"] = args.palette

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "figures"
    row_data_path = base_dir / "results" / "all_split_row_data.csv"

    stages = []
    if config["include_external"]:
        stages.append("external")
    if config["include_cv"]:
        stages.append("cv")
    if not stages:
        raise SystemExit("At least one stage (--include-external or --include-cv) must be enabled.")

    for stage in stages:
        prediction_files = _collect_prediction_files(base_dir, stage)
        curves = _prepare_curves(prediction_files)
        if curves:
            rocpr_path = output_dir / f"{stage}_roc_pr.svg"
            _plot_roc_pr(curves, rocpr_path, stage, config["palette"], config["dpi"], config["font"])
            print(f"Saved {stage} ROC/PR curves to {rocpr_path}")

    boxplot_stages = []
    if config["boxplot_stage"] in ("external", "both"):
        boxplot_stages.append("external")
    if config["boxplot_stage"] in ("cv", "both"):
        boxplot_stages.append("cv")

    for stage in boxplot_stages:
        metric_df = _prepare_metric_dataframe(row_data_path, stage, config["boxplot_metrics"])
        if metric_df.empty:
            print(f"[WARN] No boxplot data found for stage '{stage}' (checked {row_data_path}).")
            continue
        boxplot_path = output_dir / f"{stage}_metric_boxplots.svg"
        _plot_metric_boxplots(metric_df, config["boxplot_metrics"], stage, boxplot_path, config["palette"], config["dpi"], config["font"])
        print(f"Saved {stage} metric boxplots to {boxplot_path}")


if __name__ == "__main__":
    main()
