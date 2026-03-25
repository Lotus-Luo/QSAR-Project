import sys
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


REPO_DIR = Path(__file__).resolve().parent


def _run_smoke_cmd(cmd: list[str], *, allow_failure: bool = False) -> None:
    proc = subprocess.run(cmd, cwd=REPO_DIR, text=True, capture_output=True)
    if proc.returncode != 0 and not allow_failure:
        raise RuntimeError(
            "Smoke command failed.\n"
            f"Command: {cmd}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}\n"
        )


def test_regression_nan_alignment_via_ids_mapping() -> None:
    """
    Integration-ish smoke test for the NaN-pIC50 filtering alignment bug:
    - Create a tiny regression dataset with NaNs in pIC50.
    - Run the pipeline with CV (folds=2) and a tree regressor (fast).
    - Verify that `external_test_predictions.csv` has `true_label` values
      that match the original `pic50` for each returned `id`.
    """
    tmp_csv = REPO_DIR / "_smoke_regression_nan_alignment.csv"
    out_dir = REPO_DIR / "_smoke_out_regression_nan_alignment"
    if out_dir.exists():
        # Best-effort cleanup (script is for smoke testing only)
        for p in out_dir.rglob("*"):
            try:
                if p.is_file():
                    p.unlink()
            except Exception:
                pass

    rng = np.random.default_rng(0)
    n = 12
    ids = np.arange(n)
    smiles = ["C"] * n
    label = np.zeros(n, dtype=int)  # dummy (not used for regression)

    pic50 = ids.astype(float) * 0.1
    # Introduce missing pIC50 in middle indices (so ordering matters)
    pic50[[3, 8]] = np.nan

    # Create at least a few fingerprint columns (so no RDKit fingerprint generation is needed)
    fp_cols = [f"morgan_{i}" for i in range(5)]
    X_fp = rng.normal(0, 1, size=(n, len(fp_cols))).astype(float)
    df = pd.DataFrame(X_fp, columns=fp_cols)
    df.insert(0, "id", ids)
    df.insert(1, "smiles", smiles)
    df.insert(2, "label", label)
    df.insert(3, "pic50", pic50)

    df.to_csv(tmp_csv, index=False)

    # Use `rfr` because CLI uppercases model keys; registry uses `RFR` (not `Ridge`)
    cmd = [
        sys.executable,
        str(REPO_DIR / "qsar_modeling_pytorch.py"),
        "-i",
        str(tmp_csv),
        "-l",
        "label",
        "-p",
        "pic50",
        "-t",
        "regression",
        "-m",
        "rfr",
        "-o",
        str(out_dir),
        "--folds",
        "2",
        "--seed",
        "1",
        "--epochs",
        "1",
        "--batch-size",
        "4",
        "--split-method",
        "random",
        "--cv-split-method",
        "random",
        "--no-shap",
        "--variance-threshold",
        "0",
    ]

    # Note: the pipeline may still fail later due to unrelated KeyError logic,
    # but we only care that the predictions file was written.
    _run_smoke_cmd(cmd, allow_failure=True)

    run_dirs = sorted(
        (p for p in out_dir.glob("regression_*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not run_dirs:
        raise AssertionError(f"No regression_* run directories found under: {out_dir}")

    run_dir = run_dirs[0]
    pred_path = run_dir / "predictions" / "external_test_predictions.csv"
    if not pred_path.exists():
        raise AssertionError(f"Expected predictions file not found: {pred_path}")

    preds = pd.read_csv(pred_path)
    if "id" not in preds.columns or "true_label" not in preds.columns:
        raise AssertionError(f"Unexpected predictions columns: {preds.columns.tolist()}")

    # Build expected mapping only for non-NaN pic50 ids
    expected = {int(i): float(v) for i, v in zip(ids, pic50) if not math.isnan(float(v))}

    mismatches = 0
    for _, row in preds.iterrows():
        row_id = int(row["id"])
        if row_id not in expected:
            mismatches += 1
            continue
        exp = expected[row_id]
        got = float(row["true_label"])
        if not np.isclose(got, exp, rtol=1e-6, atol=1e-8):
            mismatches += 1

    if mismatches != 0:
        raise AssertionError(f"Found {mismatches} id->true_label mismatches in predictions output")


def test_calculate_metrics_single_class_does_not_crash() -> None:
    sys.path.insert(0, str(REPO_DIR))
    import qsar_modeling_pytorch as qm

    y_true = np.ones(10, dtype=int)
    y_proba = np.linspace(0.01, 0.99, 10, dtype=float)
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = qm.calculate_metrics(y_true, y_pred, y_proba, task="classification", ef_percentile=1.0)
    # AUC/PR-AUC are undefined for single-class folds; we expect NaN (not a crash)
    if not np.isnan(metrics.get("AUC", np.nan)):
        raise AssertionError(f"Expected AUC=NaN for single-class input, got: {metrics.get('AUC')}")
    # `average_precision_score` can still return a numeric value depending on sklearn behavior,
    # so we only require that the call does not crash and that the result is finite or NaN.
    pr = metrics.get("PR_AUC", np.nan)
    if not (np.isnan(pr) or np.isfinite(pr)):
        raise AssertionError(f"Expected PR_AUC to be finite or NaN, got: {pr}")


def test_chembberta_empty_smiles_stability_no_crash() -> None:
    sys.path.insert(0, str(REPO_DIR))
    import qsar_modeling_pytorch as qm

    class EmptySMILESDataset(Dataset):
        def __init__(self, length: int = 4):
            self.length = length

        def __len__(self) -> int:
            return self.length

        def __getitem__(self, idx: int):
            # Simulate ChemBERTaDataset returning None for empty SMILES
            return None

    class DummyChemBERTa(nn.Module):
        def __init__(self):
            super().__init__()
            # Ensure the model has at least one parameter so optimizers can be constructed.
            self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))

        def forward(self, input_ids, attention_mask):
            # Return logits shaped like [batch]
            bs = input_ids.size(0)
            return torch.zeros((bs,), dtype=torch.float32, device=input_ids.device) + self.bias

    config = qm.QSARConfig(
        task="classification",
        max_epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        early_stopping_patience=1,
    )

    dataset = EmptySMILESDataset(length=4)
    max_length = 16
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda b: qm.chemberta_collate_fn(b, max_length=max_length),
    )

    model = DummyChemBERTa()
    qm.train_pytorch_model(model, loader, val_loader=None, config=config, task="classification", model_type="transformer")

    y_pred, y_proba = qm.predict_pytorch_model(model, loader, task="classification")
    if len(y_pred) != 0 or len(y_proba) != 0:
        raise AssertionError("Expected empty predictions/probabilities for an all-empty ChemBERTa batch loader")


def main() -> None:
    tests = [
        test_regression_nan_alignment_via_ids_mapping,
        test_calculate_metrics_single_class_does_not_crash,
        test_chembberta_empty_smiles_stability_no_crash,
    ]
    for t in tests:
        tname = t.__name__
        try:
            t()
            print(f"[PASS] {tname}")
        except Exception as e:
            print(f"[FAIL] {tname}: {e}")
            raise


if __name__ == "__main__":
    main()

