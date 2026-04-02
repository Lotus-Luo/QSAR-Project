#!/usr/bin/env python3
"""Compute ChemBERTa token-level SHAP/IG contributions.

python Scripts/step13_shap_interpreter_chemberta.py \
  -p models_out/classification_20260330_151751/split_seed_30 \
  -m ChemBERTa \
  -s 42 \
  -e models_out/classification_20260330_151751/split_seed_30/exports/ChemBERTa/seed_42/pytorch_shap_export.npz \
  -o models_out/classification_20260330_151751/split_seed_30/shap/ChemBERTa/seed_42 \
  --max-samples 64 \
  --n-steps 32

"""

# %%
from pathlib import Path

BASE_CONFIG = {
    "project_root": Path("models_out/classification_20260330_151751"),
    "model_key": "ChemBERTa",
    "seed": "42",
    "export": Path("models_out/classification_20260330_151751/split_seed_30/exports/ChemBERTa/seed_42/pytorch_shap_export.npz"),
    "output_dir": Path("models_out/classification_20260330_151751/split_seed_30/shap/ChemBERTa/seed_42"),
    "max_samples": 64,
    "n_steps": 32,
    "device": None,
}

# %%
import sys
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch

from captum.attr import LayerIntegratedGradients

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from Scripts.step01_train_qsar_models import CHEMBERTA_AVAILABLE


def _ensure_transformers():
    if AutoModelForSequenceClassification is None or AutoTokenizer is None:
        raise SystemExit("Transformers library is required for ChemBERTa SHAP (pip install transformers).")


def _load_metadata(project_root: Path, model_key: str, seed: str) -> Dict[str, Any]:
    meta_path = project_root / "models" / "full_dev" / model_key / f"seed_{seed}" / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata missing: {meta_path}")
    import json
    with open(meta_path, "r") as fh:
        return json.load(fh)


def _determine_output_dir(cfg: Dict[str, Any], export_path: Path) -> Path:
    if cfg.get("output_dir"):
        return cfg["output_dir"]
    split_root = next((p for p in export_path.parents if p.name.startswith("split_seed")), None)
    if split_root is None:
        base_root = export_path.parents[2]
    else:
        base_root = split_root.parent
    return base_root / "shape" / cfg["model_key"] / f"seed_{cfg['seed']}"


def _forward_fn(model, input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    return logits.squeeze(-1)


def _attribute_tokens(
    model,
    token_ids_list: List[np.ndarray],
    attention_masks: List[np.ndarray],
    device: torch.device,
    n_steps: int,
    max_samples: Optional[int],
) -> List[np.ndarray]:
    layer = model.get_input_embeddings()
    lig = LayerIntegratedGradients(lambda ids, mask: _forward_fn(model, ids, mask), layer)
    selected = list(range(len(token_ids_list)))
    if max_samples is not None:
        selected = selected[:max_samples]
    contributions = []
    for idx in selected:
        ids = torch.tensor(token_ids_list[idx], dtype=torch.long, device=device).unsqueeze(0)
        mask = torch.tensor(attention_masks[idx], dtype=torch.long, device=device).unsqueeze(0)
        baseline_ids = torch.zeros_like(ids)
        attr = lig.attribute(
            inputs=ids,
            baselines=baseline_ids,
            additional_forward_args=(mask,),
            n_steps=n_steps,
        )
        token_attr = attr.sum(dim=-1).squeeze(0)
        token_attr = token_attr * mask.squeeze(0).float()
        contributions.append(token_attr.detach().cpu().numpy())
    return contributions, selected


def _flatten_token_data(
    data_array,
    indices: List[int],
) -> List[np.ndarray]:
    return [np.asarray(data_array[idx]) for idx in indices]


def main():
    if not CHEMBERTA_AVAILABLE:
        raise SystemExit("ChemBERTa support is disabled in this environment.")
    _ensure_transformers()

    meta = _load_metadata(BASE_CONFIG["project_root"], BASE_CONFIG["model_key"], BASE_CONFIG["seed"])
    if meta.get("model_type") != "transformer":
        raise ValueError("Metadata does not describe a ChemBERTa transformer model.")

    export_path = BASE_CONFIG["export"]
    if not export_path.exists():
        raise FileNotFoundError(f"Export file missing: {export_path}")

    output_dir = _determine_output_dir(BASE_CONFIG, export_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(export_path, allow_pickle=True)
    token_ids_all = list(data["token_ids"])
    attention_masks_all = list(data["attention_mask"])
    token_strings_all = list(data["token_strings"])
    token_offsets_all = list(data["token_offsets"])
    valid_indices_raw = data.get("valid_indices")
    valid_indices = (
        [int(idx) for idx in np.atleast_1d(valid_indices_raw)]
        if valid_indices_raw is not None
        else list(range(len(token_ids_all)))
    )
    smiles_raw = data.get("smiles")
    smiles_all = list(np.atleast_1d(smiles_raw)) if smiles_raw is not None else []
    ids_raw = data.get("ids")
    ids_all = list(np.atleast_1d(ids_raw)) if ids_raw is not None else []

    model_dir = BASE_CONFIG["project_root"] / "models" / "full_dev" / BASE_CONFIG["model_key"] / f"seed_{BASE_CONFIG['seed']}" / "model"
    if not model_dir.exists():
        raise FileNotFoundError(f"Saved ChemBERTa model missing: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device_str = BASE_CONFIG["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    model = model.to(device).eval()

    contributions, processed_indices = _attribute_tokens(
        model,
        token_ids_all,
        attention_masks_all,
        device,
        BASE_CONFIG["n_steps"],
        BASE_CONFIG["max_samples"],
    )

    selected_token_ids = _flatten_token_data(token_ids_all, processed_indices)
    selected_attention_masks = _flatten_token_data(attention_masks_all, processed_indices)
    selected_token_strings = _flatten_token_data(token_strings_all, processed_indices)
    selected_token_offsets = _flatten_token_data(token_offsets_all, processed_indices)
    selected_valid_indices = [valid_indices[i] for i in processed_indices]

    csv_rows = []
    for idx, sample_idx in enumerate(processed_indices):
        dataset_idx = selected_valid_indices[idx]
        token_values = selected_token_strings[idx]
        offsets = selected_token_offsets[idx]
        mask = selected_attention_masks[idx]
        attr_values = contributions[idx]
        for token_idx, (token, value, offset, attn) in enumerate(zip(token_values, attr_values, offsets, mask)):
            if int(attn) == 0:
                break
            row = {
                "sample_index": dataset_idx,
                "token_index": token_idx,
                "token": str(token),
                "contribution": float(value),
                "char_start": int(offset[0]),
                "char_end": int(offset[1]),
                "smiles": str(smiles_all[dataset_idx]) if dataset_idx < len(smiles_all) else "",
                "id": str(ids_all[dataset_idx]) if dataset_idx < len(ids_all) else "",
            }
            csv_rows.append(row)

    csv_path = output_dir / "chemberta_token_contributions.csv"
    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    else:
        print("Warning: No tokens were recorded in the CSV output.")

    np.savez_compressed(
        output_dir / "chemberta_token_contributions.npz",
        token_contributions=np.array(contributions, dtype=object),
        token_ids=np.array(selected_token_ids, dtype=object),
        attention_mask=np.array(selected_attention_masks, dtype=object),
        token_offsets=np.array(selected_token_offsets, dtype=object),
        token_strings=np.array(selected_token_strings, dtype=object),
        valid_indices=np.array(selected_valid_indices, dtype=int),
        smiles=smiles_all,
        ids=ids_all,
        tokenizer_name=data.get("tokenizer_name"),
        tokenizer_max_length=data.get("tokenizer_max_length"),
    )

    print(f"Saved ChemBERTa token contributions:\n  - {csv_path}\n  - {output_dir / 'chemberta_token_contributions.npz'}")


if __name__ == "__main__":
    main()
