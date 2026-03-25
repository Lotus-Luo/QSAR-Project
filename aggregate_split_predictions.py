"""
Aggregate predictions across split seeds and plot average ROC / PR curves.

Usage:
    python aggregate_split_predictions.py --base-dir models_out/classification_20260325_115105

Outputs:
    - results/external_roc_pr_curves.svg
"""

import argparse
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def _collect_prediction_files(base_dir: Path) -> List[Path]:
    split_dirs = sorted(p for p in base_dir.glob("split_seed_*") if p.is_dir())
    files = []
    for split_dir in split_dirs:
        seed = split_dir.name.split("_")[-1]
        candidate = split_dir / "predictions" / f"external_test_predictions_seed_{seed}.csv"
        if candidate.exists():
            files.append(candidate)
    return files


def _interpolate_curve(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, x, y, left=y[0], right=y[-1])


def _prepare_curves(prediction_files: List[Path]) -> Dict[str, Dict[str, List[np.ndarray]]]:
    curves = {}
    fpr_grid = np.linspace(0, 1, 200)
    recall_grid = np.linspace(0, 1, 200)

    for path in prediction_files:
        data = pd.read_csv(path)
        if {'true_label', 'molecule_id'}.issubset(set(data.columns)):
            y_true = data['true_label'].astype(float).to_numpy()
        else:
            continue
        model_cols = [col for col in data.columns if col.endswith("_score")]
        for model_col in model_cols:
            scores = data[model_col].astype(float).to_numpy()
            if np.all(np.isnan(scores)):
                continue

            fpr, tpr, _ = roc_curve(y_true, scores)
            precision, recall, _ = precision_recall_curve(y_true, scores)

            entry = curves.setdefault(model_col.replace("_score", ""), {'roc': [], 'pr': []})
            entry['roc'].append(_interpolate_curve(fpr, tpr, fpr_grid))
            entry.setdefault('roc_grid', fpr_grid)
            entry['pr'].append(_interpolate_curve(recall[::-1], precision[::-1], recall_grid))
            entry.setdefault('pr_grid', recall_grid)

    return curves


def plot_curves(curves: Dict[str, Dict[str, List[np.ndarray]]], output_path: Path):
    plt.style.use("seaborn-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for model, data in curves.items():
        if not data['roc']:
            continue
        fpr_grid = data['roc_grid']
        mean_tpr = np.mean(data['roc'], axis=0)
        std_tpr = np.std(data['roc'], axis=0)
        auc_val = auc(fpr_grid, mean_tpr)
        axes[0].plot(fpr_grid, mean_tpr, label=f"{model} (AUC={auc_val:.3f})")
        axes[0].fill_between(fpr_grid, np.clip(mean_tpr - std_tpr, 0, 1), np.clip(mean_tpr + std_tpr, 0, 1), alpha=0.1)

        recall_grid = data['pr_grid']
        mean_prec = np.mean(data['pr'], axis=0)
        std_prec = np.std(data['pr'], axis=0)
        pr_auc_val = auc(recall_grid, mean_prec)
        axes[1].plot(recall_grid, mean_prec, label=f"{model} (AUC={pr_auc_val:.3f})")
        axes[1].fill_between(recall_grid, np.clip(mean_prec - std_prec, 0, 1), np.clip(mean_prec + std_prec, 0, 1), alpha=0.1)

    axes[0].set_title("Average ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")

    axes[1].set_title("Average PR Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")


def main():
    parser = argparse.ArgumentParser(description="Aggregate split seed prediction curves and plot ROC/PR.")
    parser.add_argument("--base-dir", required=True, help="Run output directory containing split_seed_* folders.")
    parser.add_argument("--output", default="results/external_roc_pr_curves.svg", help="Relative path for the aggregated SVG.")
    args = parser.parse_args()

    base_path = Path(args.base_dir)
    prediction_files = _collect_prediction_files(base_path)
    if not prediction_files:
        raise SystemExit("No prediction CSV files found. Run the main pipeline with split_seeds first.")

    curves = _prepare_curves(prediction_files)
    output_path = base_path / args.output
    plot_curves(curves, output_path)

    print(f"Aggregated ROC/PR curves saved to: {output_path}")


if __name__ == "__main__":
    main()
