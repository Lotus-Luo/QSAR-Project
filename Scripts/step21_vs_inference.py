#!/usr/bin/env python3
"""Generate predictions from a finished QSAR run.

Reads a `split_seed_*` directory, loads the saved classifiers/MLP/GAT/ChemBERTa
artifacts under `models/full_dev/`, reuses their fingerprint filtering + scaler,
and writes a CSV with one row per molecule per model. Fingerprint
columns are added automatically when missing (RDKit required) and the script
honors the original task (`classification` vs `regression`).

Usage example:

python Scripts/step21_vs_inference.py \
     --run-dir models_out/classification_20260326_213228/split_seed_3 \
    --input Data/prediction_test_data_fingerprints.csv \
     --models "GAT,ChemBERTa,RFC,SVC" \
    --batch-size 64
        
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Allow importing helpers from step01_train_qsar_models even though Scripts/
# is not a package.
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    TRANSFORMERS_AVAILABLE = False

try:
    from torch_geometric.loader import DataLoader as GeometricDataLoader
except ImportError:  # pragma: no cover
    GeometricDataLoader = None

from step01_train_qsar_models import (
    GATModel,
    GAT_NUM_EDGE_FEATURES,
    GAT_NUM_NODE_FEATURES,
    MoleculeDataset,
    ResidualMLP,
    ChemBERTaDataset,
    chemberta_collate_fn,
    GAT_AVAILABLE,
    generate_fingerprints_from_csv,
    predict_pytorch_model,
    read_table,
)

HIGHER_IS_BETTER = {"AUC", "PR_AUC", "ACC", "F1", "MCC", "R2"}
LOWER_IS_BETTER = {"RMSE", "MAE"}


class HuggingFaceChemBERTa(nn.Module):
    """Wrap a saved HuggingFace ChemBERTa checkpoint for the training API."""

    def __init__(self, model_dir: Path):
        super().__init__()
        if AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise ImportError("Transformers is required to run ChemBERTa inference")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), local_files_only=True
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits.squeeze(-1)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict using saved QSAR models")
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Path to a split_seed_* directory (models_out/.../split_seed_3)")
    parser.add_argument("--input", "-i", required=True, type=Path,
                        help="CSV/Parquet file with id, smiles, and optional fingerprint columns")
    parser.add_argument("--output", "-o", type=Path,
                        help="Prediction CSV path (default: virtual_screening/<run>_<timestamp>.csv)")
    parser.add_argument("--models", help="Comma-separated subset of model names to predict with")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for DataLoader used by MLP/GAT/ChemBERTa")
    parser.add_argument("--id-column", default=None, help="Override ID column name")
    parser.add_argument("--smiles-column", default=None, help="Override SMILES column name")
    parser.add_argument("--force-fingerprints", action="store_true",
                        help="Regenerate fingerprint columns even if they already exist")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity")
    return parser.parse_args()


def setup_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )
    return logging.getLogger("qsar_predict")


def load_json(path: Path) -> Dict:
    with path.open("r") as reader:
        return json.load(reader)


def load_config(run_dir: Path) -> Dict:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in run dir: {config_path}")
    return load_json(config_path)


def load_results(run_dir: Path) -> Optional[pd.DataFrame]:
    results_path = run_dir / "results" / "external_test_results.csv"
    if not results_path.exists():
        return None
    return pd.read_csv(results_path)


def determine_best_seeds(results: pd.DataFrame, metric: str, task: str) -> Dict[str, int]:
    direction = "higher"
    if metric in LOWER_IS_BETTER or task == "regression":
        direction = "lower"
    best = {}
    for _, row in results.iterrows():
        model = row["model"]
        seed = int(row["seed"])
        value = row.get(metric)
        if pd.isna(value):
            continue
        candidate = float(value)
        if model not in best:
            best[model] = {"seed": seed, "value": candidate}
            continue
        current = best[model]["value"]
        if (direction == "higher" and candidate > current) or (direction == "lower" and candidate < current):
            best[model] = {"seed": seed, "value": candidate}
    return {model: info["seed"] for model, info in best.items()}


def filter_models(available: List[str], requested: Optional[List[str]], logger: logging.Logger) -> List[str]:
    if not requested:
        return available
    mapping = {name.upper(): name for name in available}
    cleaned = [m.strip().upper() for m in requested if m.strip()]
    missing = [m for m in cleaned if m not in mapping]
    if missing:
        logger.warning(f"Requested models not in run: {', '.join(missing)}")
    result = [mapping[m] for m in cleaned if m in mapping]
    if not result:
        raise SystemExit("None of the requested models were available in the run")
    return result


def ensure_fingerprints(
    df: pd.DataFrame,
    required: List[str],
    config: Dict,
    logger: logging.Logger,
    force: bool,
) -> pd.DataFrame:
    missing = [col for col in required if col not in df.columns]
    if not missing and not force:
        return df
    if not config.get("auto_generate_fingerprints", True) and not force:
        raise SystemExit("Missing fingerprint columns and automatic generation is disabled")
    logger.info("Generating fingerprint columns for prediction inputs")
    fingerprints = config.get("fingerprint_types", ["morgan"])
    df = generate_fingerprints_from_csv(df, config.get("smiles_column", "smiles"), fingerprint_types=fingerprints, logger=logger)
    missing_after = [col for col in required if col not in df.columns]
    if missing_after:
        raise SystemExit(f"Still missing fingerprint columns after regeneration: {missing_after[:5]}...")
    return df


def prepare_matrix(df: pd.DataFrame, columns: List[str], cache: Dict[str, np.ndarray]) -> np.ndarray:
    key = "||".join(columns)
    if key in cache:
        return cache[key]
    matrix = df[columns].astype(float).to_numpy()
    cache[key] = matrix
    return matrix


def align_predictions(length: int, valid_indices: List[int], values: np.ndarray) -> np.ndarray:
    aligned = np.full(length, np.nan)
    if valid_indices and values is not None and len(values):
        aligned[np.array(valid_indices, dtype=int)] = values
    return aligned


def load_scaler(seed_dir: Path) -> Optional[object]:
    scaler_path = seed_dir / "scaler.joblib"
    if scaler_path.exists():
        return joblib.load(scaler_path)
    return None


def predict_sklearn(
    seed_dir: Path,
    model_name: str,
    feature_matrix: np.ndarray,
    scaler: Optional[object],
    ids: List,
    smiles: List[str],
    seed: int,
    task: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    logger.info(f"  → Loading sklearn model for {model_name}")
    model_path = seed_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    model = joblib.load(model_path)
    X = feature_matrix
    if scaler is not None:
        X = scaler.transform(X)
    if task == "classification":
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            raw = model.decision_function(X)
            scores = 1 / (1 + np.exp(-raw))
        else:
            scores = model.predict(X).astype(float)
        labels = (scores >= 0.5).astype(int)
    else:
        scores = model.predict(X).astype(float)
        labels = np.full(len(scores), np.nan)
    return pd.DataFrame({
        "id": ids,
        "smiles": smiles,
        "model": model_name,
        "seed": seed,
        "predicted_label": labels,
        "predicted_score": scores,
    })


def predict_mlp(
    seed_dir: Path,
    model_name: str,
    feature_matrix: np.ndarray,
    scaler: Optional[object],
    ids: List,
    smiles: List[str],
    seed: int,
    task: str,
    logger: logging.Logger,
    batch_size: int,
    model_config: Dict,
    metadata: Dict,
) -> pd.DataFrame:
    logger.info(f"  → Loading MLP checkpoint for {model_name}")
    model_path = seed_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    params = model_config.get("params", {})
    input_dim = int(metadata.get("input_dim", feature_matrix.shape[1]))
    mlp = ResidualMLP(input_dim=input_dim, **{k: v for k, v in params.items() if k != "input_dim"})
    state = torch.load(model_path, map_location="cpu")
    mlp.load_state_dict(state)
    X = feature_matrix
    if scaler is not None:
        X = scaler.transform(X)
    dataset = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32), torch.zeros(len(X))),
        batch_size=batch_size,
        shuffle=False,
    )
    logits, scores = predict_pytorch_model(mlp, dataset, task)
    return pd.DataFrame({
        "id": ids,
        "smiles": smiles,
        "model": model_name,
        "seed": seed,
        "predicted_label": logits,
        "predicted_score": scores,
    })


def predict_gat(
    seed_dir: Path,
    model_name: str,
    smiles_list: List[str],
    ids: List,
    seed: int,
    task: str,
    logger: logging.Logger,
    batch_size: int,
    model_config: Dict,
) -> pd.DataFrame:
    if not GAT_AVAILABLE or GeometricDataLoader is None:
        raise RuntimeError("GAT dependencies are missing (torch_geometric/RDKit)")
    logger.info(f"  → Building GAT graphs for {model_name}")
    dataset = MoleculeDataset(smiles_list, [0] * len(smiles_list), logger=logger)
    loader = GeometricDataLoader(dataset, batch_size=batch_size, shuffle=False)
    params = model_config.get("params", {})
    model = GATModel(
        num_node_features=GAT_NUM_NODE_FEATURES,
        num_edge_features=GAT_NUM_EDGE_FEATURES,
        hidden_dim=params.get("hidden_dim", 64),
        num_heads=params.get("num_heads", 4),
        num_layers=params.get("num_layers", 3),
        dropout=params.get("dropout", 0.3),
    )
    model_path = seed_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    logits, scores = predict_pytorch_model(model, loader, task)
    aligned_labels = align_predictions(len(smiles_list), dataset.valid_indices, logits)
    aligned_scores = align_predictions(len(smiles_list), dataset.valid_indices, scores)
    return pd.DataFrame({
        "id": ids,
        "smiles": smiles_list,
        "model": model_name,
        "seed": seed,
        "predicted_label": aligned_labels,
        "predicted_score": aligned_scores,
    })


def predict_chemberta(
    seed_dir: Path,
    model_name: str,
    smiles_list: List[str],
    ids: List,
    seed: int,
    task: str,
    logger: logging.Logger,
    batch_size: int,
) -> pd.DataFrame:
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("Transformers is required for ChemBERTa predictions")
    model_folder = seed_dir / "model"
    if not model_folder.exists():
        raise FileNotFoundError(model_folder)
    logger.info(f"  → Loading ChemBERTa from {model_folder}")
    hf = HuggingFaceChemBERTa(model_folder)
    dataset = ChemBERTaDataset(smiles_list, [0] * len(smiles_list), tokenizer=hf.tokenizer, logger=logger)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch, ml=dataset.max_length: chemberta_collate_fn(batch, max_length=ml),
    )
    logits, scores = predict_pytorch_model(hf, loader, task)
    aligned_labels = align_predictions(len(smiles_list), dataset.valid_indices, logits)
    aligned_scores = align_predictions(len(smiles_list), dataset.valid_indices, scores)
    return pd.DataFrame({
        "id": ids,
        "smiles": smiles_list,
        "model": model_name,
        "seed": seed,
        "predicted_label": aligned_labels,
        "predicted_score": aligned_scores,
    })


def main():
    args = parse_args()
    logger = setup_logger(args.log_level)
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")
    config = load_config(run_dir)
    task = config.get("task", "classification")
    id_col = args.id_column or config.get("id_column", "id")
    smiles_col = args.smiles_column or config.get("smiles_column", "smiles")
    config["smiles_column"] = smiles_col
    results_df = load_results(run_dir)
    if results_df is None:
        raise SystemExit("external_test_results.csv is missing; run must have completed first")
    metric = config.get("external_test_metric", "MCC")
    best_seeds = determine_best_seeds(results_df, metric, task)
    full_dev_dir = run_dir / "models" / "full_dev"
    if not full_dev_dir.exists():
        raise SystemExit(f"Expected checkpoints in {full_dev_dir}")
    available_models = sorted([d.name for d in full_dev_dir.iterdir() if d.is_dir()])
    selected_models = filter_models(available_models, args.models.split(",") if args.models else None, logger)
    df = read_table(args.input)
    if id_col not in df.columns or smiles_col not in df.columns:
        raise SystemExit(f"Input file must contain '{id_col}' and '{smiles_col}' columns")
    ids = df[id_col].tolist()
    smiles_list = df[smiles_col].astype(str).fillna("").tolist()
    feature_cache: Dict[str, np.ndarray] = {}
    fingerprint_requirements: List[str] = []

    # Determine fingerprint features required by the selected sklearn/MLP models
    for model_name in selected_models:
        seed = best_seeds.get(model_name)
        if seed is None:
            logger.warning(f"No metric entry found for {model_name}; skipping")
            continue
        metadata_path = full_dev_dir / model_name / f"seed_{seed}" / "metadata.json"
        if not metadata_path.exists():
            logger.warning(f"Missing metadata for {model_name} (seed {seed})")
            continue
        metadata = load_json(metadata_path)
        if metadata.get("model_type") in {"sklearn", "pytorch"}:
            fingerprint_requirements.extend(metadata.get("feature_names", []))
    if fingerprint_requirements:
        df = ensure_fingerprints(df, sorted(set(fingerprint_requirements)), config, logger, args.force_fingerprints)

    result_df = df[[id_col, smiles_col]].copy()
    used_models: List[str] = []
    for model_name in selected_models:
        seed = best_seeds.get(model_name)
        if seed is None:
            continue
        seed_dir = full_dev_dir / model_name / f"seed_{seed}"
        if not seed_dir.exists():
            logger.warning(f"Checkpoint missing for {model_name} seed {seed}")
            continue
        metadata = load_json(seed_dir / "metadata.json")
        model_config = load_json(seed_dir / "model_config.json")
        model_type = metadata.get("model_type")
        logger.info(f"\nPredicting with {model_name} (seed {seed}, type {model_type})")
        try:
            if model_type == "sklearn":
                features = metadata.get("feature_names", [])
                matrix = prepare_matrix(df, features, feature_cache)
                scaler = load_scaler(seed_dir)
                frame = predict_sklearn(seed_dir, model_name, matrix, scaler, ids, smiles_list, seed, task, logger)
            elif model_type == "pytorch":
                features = metadata.get("feature_names", [])
                matrix = prepare_matrix(df, features, feature_cache)
                scaler = load_scaler(seed_dir)
                frame = predict_mlp(
                    seed_dir, model_name, matrix, scaler, ids, smiles_list, seed, task,
                    logger, args.batch_size, model_config, metadata
                )
            elif model_type == "pytorch_geometric":
                frame = predict_gat(seed_dir, model_name, smiles_list, ids, seed, task, logger, args.batch_size, model_config)
            elif model_type == "transformer":
                frame = predict_chemberta(seed_dir, model_name, smiles_list, ids, seed, task, logger, args.batch_size)
            else:
                logger.warning(f"Skipping unsupported model type: {model_type}")
                continue
        except Exception as exc:
            logger.exception(f"Prediction failed for {model_name}: {exc}")
            continue
        if len(frame) != len(df):
            logger.warning(
                f"{model_name}: prediction frame length {len(frame)} does not match input rows {len(df)}"
            )
        result_df[f"{model_name}_predicted"] = frame["predicted_label"]
        result_df[f"{model_name}_predicted_score"] = frame["predicted_score"]
        used_models.append(model_name)

    if not used_models:
        raise SystemExit("No predictions could be generated")

    predicted_cols = [col for col in result_df.columns if col.endswith("_predicted")]
    if predicted_cols:
        result_df["Consensus_Sum"] = result_df[predicted_cols].fillna(0).sum(axis=1).astype(int)

    default_root = Path("virtual_screening")
    parent_name = run_dir.parent.name or run_dir.name
    input_stem = Path(args.input).stem  # 获取输入文件的文件名（不带后缀）
    default_path = default_root / f"VS_{input_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_path = args.output or default_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    logger.info(f"Saved predictions for {len(used_models)} models to {output_path}")


if __name__ == "__main__":
    main()
