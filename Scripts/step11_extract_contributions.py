#!/usr/bin/env python3
"""Export SHAP-style contributions for PyTorch models trained by the QSAR pipeline.
Usage:
python Scripts/step11_extract_contributions.py \
  -p models_out/classification_20260330_151751/split_seed_30 \
  -m GAT \
  -s 42 \
  --background-size 50

python Scripts/step11_extract_contributions.py \
  -p models_out/classification_20260330_151751/split_seed_30 \
  -m ChemBERTa \
  -s 42

python Scripts/step11_extract_contributions.py \
  -p models_out/classification_20260330_151751/split_seed_30 \
  -m MLP \
  -s 42 \
  --background-size 50
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

try:
    import shap
except ImportError:
    shap = None

# Allow imports from the main pipeline
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from Scripts.step01_train_qsar_models import (
    ResidualMLP,
    DEVICE,
    GATModel,
    smiles_to_graph,
    GAT_AVAILABLE,
    GAT_NUM_NODE_FEATURES,
    GAT_NUM_EDGE_FEATURES,
)

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover
    AutoTokenizer = None


def load_npz_data(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"External data file not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {
            "features": np.asarray(data.get("features"), dtype=float),
            "labels": np.asarray(data.get("labels")) if data.get("labels") is not None else None,
            "ids": np.atleast_1d(data.get("ids")).tolist() if data.get("ids") is not None else None,
            "smiles": np.atleast_1d(data.get("smiles")).tolist() if data.get("smiles") is not None else None,
            "feature_names": np.atleast_1d(data.get("feature_names")).tolist() if data.get("feature_names") is not None else None,
        }


def _load_metadata(seed_dir: Path) -> Dict[str, Any]:
    meta_path = seed_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    with open(meta_path, 'r') as fh:
        metadata = json.load(fh)
    config_path = seed_dir / "model_config.json"
    if config_path.exists():
        with open(config_path, 'r') as fh:
            metadata['model_config'] = json.load(fh)
    return metadata


def _determine_input_dim(metadata: Dict[str, Any], features: np.ndarray, model_config: Dict[str, Any]) -> int:
    # Priority: metadata saved input_dim > model_config params > feature vector length
    input_dim = metadata.get('input_dim')
    if input_dim is None:
        params = model_config.get('params', {})
        input_dim = params.get('input_dim')
    if input_dim is None or input_dim <= 0:
        input_dim = features.shape[1]
    return int(input_dim)


def _instantiate_mlp(model_config: Dict[str, Any], input_dim: int) -> ResidualMLP:
    params = model_config.get('params', {})
    hidden_dims = params.get('hidden_dims', [512, 256])
    dropout = params.get('dropout', 0.3)
    activation = params.get('activation', 'mish')
    use_residual = params.get('use_residual', True)
    norm_type = params.get('norm_type', 'layernorm')
    output_dim = params.get('output_dim', 1)
    model = ResidualMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        activation=activation,
        dropout=dropout,
        use_residual=use_residual,
        norm_type=norm_type,
    )
    return model
    hidden_dims = params.get('hidden_dims', [512, 256])
    dropout = params.get('dropout', 0.3)
    activation = params.get('activation', 'mish')
    use_residual = params.get('use_residual', True)
    norm_type = params.get('norm_type', 'layernorm')
    output_dim = params.get('output_dim', 1)
    model = ResidualMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        activation=activation,
        dropout=dropout,
        use_residual=use_residual,
        norm_type=norm_type,
    )
    return model


def _to_numpy(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_numpy(v) for v in obj)
    return obj


def _select_shap_array(shap_values: Any, n_features: int) -> np.ndarray:
    if isinstance(shap_values, list):
        for arr in shap_values:
            arr_np = np.asarray(_to_numpy(arr))
            if arr_np.ndim == 2 and arr_np.shape[1] == n_features:
                return arr_np
        return np.asarray(_to_numpy(shap_values[0]))
    if hasattr(shap_values, 'values'):
        return np.asarray(_to_numpy(shap_values.values))
    return np.asarray(_to_numpy(shap_values))


def _encode_chemb_smiles(tokenizer, smiles_list, max_length: int):
    if tokenizer is None:
        raise SystemExit("Transformers are required for ChemBERTa export (pip install transformers).")
    token_ids = []
    attention_masks = []
    token_offsets = []
    token_strings = []
    valid_indices = []
    for idx, smiles in enumerate(smiles_list):
        if not smiles or str(smiles).strip() == "":
            continue
        encoding = tokenizer(
            smiles,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt',
            return_offsets_mapping=True,
        )
        ids = encoding['input_ids'][0].cpu().numpy()
        mask = encoding['attention_mask'][0].cpu().numpy()
        offsets = encoding['offset_mapping'][0].cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(ids.tolist())
        token_ids.append(ids)
        attention_masks.append(mask)
        token_offsets.append(offsets)
        token_strings.append(np.array(tokens, dtype=object))
        valid_indices.append(idx)
    if not token_ids:
        raise ValueError("No valid SMILES strings were tokenized for ChemBERTa export.")
    return token_ids, attention_masks, token_offsets, token_strings, valid_indices


def main():
    if shap is None:
        sys.exit("Shap is required: pip install shap")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--project-root", default="models_out", help="QSAR output directory")
    parser.add_argument("-m", "--model-key", required=True, help="Model key (e.g., MLP)")
    parser.add_argument("-s", "--seed", required=True, type=int, help="Seed identifier")
    parser.add_argument("--external-data", type=Path, help="Path to saved external split (.npz)")
    parser.add_argument("-o", "--output-dir", type=Path, help="Where to store exported contributions")
    parser.add_argument("--background-size", type=int, default=50, help="Number of samples to build background reference")
    parser.add_argument("--chemberta-max-length", type=int, default=128, help="Tokenization max length for ChemBERTa export")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for model inference")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    seed_dir = project_root / "models" / "full_dev" / args.model_key / f"seed_{args.seed}"
    if not seed_dir.exists():
        raise FileNotFoundError(f"Seed directory not found: {seed_dir}")

    external_data_path = args.external_data or project_root / "data" / "splits" / "external_test.npz"
    external_data_path = Path(external_data_path)
    external_data = load_npz_data(external_data_path)

    output_dir = args.output_dir or project_root / "exports" / args.model_key / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_metadata(seed_dir)
    model_type = metadata.get('model_type', 'pytorch')
    export_payload = {}

    X_ext = external_data['features']
    if X_ext.size == 0:
        raise ValueError("External features are empty, nothing to explain.")

    def _object_array(items):
        arr = np.empty(len(items), dtype=object)
        for idx, item in enumerate(items):
            arr[idx] = np.asarray(item)
        return arr

    if model_type == 'pytorch':
        model_config = metadata.get('model_config', {})
        input_dim = _determine_input_dim(metadata, X_ext, model_config)
        model = _instantiate_mlp(model_config, input_dim)
        model_path = seed_dir / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file missing: {model_path}")
        device = torch.device(args.device)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model = model.to(device).eval()

        background = torch.tensor(X_ext[:args.background_size], dtype=torch.float32).to(device)
        dataset = torch.tensor(X_ext, dtype=torch.float32).to(device)

        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(dataset)
        shap_array = _select_shap_array(shap_values, X_ext.shape[1])

        base_values = getattr(explainer, 'expected_value', None)
        if base_values is None:
            base_values = getattr(explainer, 'expected_values', None)
        base_values = _to_numpy(base_values)

        export_payload.update({
            "shap_values": np.asarray(shap_array),
            "base_values": np.asarray(base_values) if base_values is not None else None,
        })
    elif model_type == 'pytorch_geometric':
        if not GAT_AVAILABLE:
            raise ImportError("PyTorch Geometric (GAT) support is not installed in this environment.")
        model = GATModel(
            num_node_features=GAT_NUM_NODE_FEATURES,
            num_edge_features=GAT_NUM_EDGE_FEATURES,
            hidden_dim=64,
            num_heads=4,
            num_layers=3,
            dropout=0.3,
        )
        model_path = seed_dir / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"GAT model file missing: {model_path}")
        device = torch.device(args.device)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model = model.to(device).eval()

        smiles_list = external_data.get('smiles') or []
        if not smiles_list:
            raise ValueError("SMILES strings are required to build GAT graphs.")

        graphs = []
        valid_indices = []
        for idx, smiles in enumerate(smiles_list):
            graph = smiles_to_graph(smiles)
            if graph is not None:
                graphs.append(graph)
                valid_indices.append(idx)

        if not graphs:
            raise ValueError("No valid GAT graphs could be constructed from the external SMILES.")

        node_features = [graph.x.detach().cpu().numpy() for graph in graphs]
        edge_indices = [graph.edge_index.detach().cpu().numpy() for graph in graphs]
        edge_attrs = [graph.edge_attr.detach().cpu().numpy() if graph.edge_attr is not None else np.empty((0, GAT_NUM_EDGE_FEATURES)) for graph in graphs]

        export_payload.update({
            "shap_values": None,
            "node_features": _object_array(node_features),
            "edge_indices": _object_array(edge_indices),
            "edge_attrs": _object_array(edge_attrs),
            "valid_indices": np.array(valid_indices, dtype=int),
        })
    elif model_type == 'transformer':
        if AutoTokenizer is None:
            raise SystemExit("Transformers are required for ChemBERTa export (pip install transformers).")
        tokenizer_dir = seed_dir / "model"
        if not tokenizer_dir.exists():
            raise FileNotFoundError(f"ChemBERTa assets missing: {tokenizer_dir}")
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
        smiles_list = external_data.get('smiles') or []
        if not smiles_list:
            raise ValueError("SMILES strings are required for ChemBERTa export.")
        max_length = args.chemberta_max_length
        token_ids, attention_masks, token_offsets, token_strings, valid_indices = _encode_chemb_smiles(
            tokenizer,
            smiles_list,
            max_length,
        )
        export_payload.update({
            "shap_values": None,
            "token_ids": _object_array(token_ids),
            "attention_mask": _object_array(attention_masks),
            "token_offsets": _object_array(token_offsets),
            "token_strings": _object_array(token_strings),
            "valid_indices": np.array(valid_indices, dtype=int),
            "tokenizer_name": metadata.get('model_config', {}).get('params', {}).get('model_name'),
            "tokenizer_max_length": max_length,
        })
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not supported yet.")

    save_path = output_dir / "pytorch_shap_export.npz"
    np.savez_compressed(
        save_path,
        **export_payload,
        features=np.asarray(X_ext, dtype=float),
        feature_names=np.array(external_data['feature_names'], dtype=object) if external_data['feature_names'] else None,
        ids=np.array(external_data['ids'], dtype=object) if external_data['ids'] else None,
        smiles=np.array(external_data['smiles'], dtype=object) if external_data['smiles'] else None
    )

    print(f"Exported PyTorch contributions to: {save_path}")


if __name__ == "__main__":
    main()
