#!/usr/bin/env python3
"""GAT SHAP runner: compute per-atom attributions and apply global scaling.
Usage:
python Scripts/step12_shap_interpreter_gat.py \
  -p models_out/classification_20260329_165120/split_seed_29 \
  -m GAT \
  -s 42 \
  -e models_out/classification_20260329_165120/split_seed_29/exports/GAT/seed_42/pytorch_shap_export.npz \
  -o models_out/classification_20260329_165120/split_seed_29/shap/GAT/seed_42/
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from captum.attr import IntegratedGradients

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from Scripts.step01_train_qsar_models import GATModel, GAT_NUM_EDGE_FEATURES, GAT_NUM_NODE_FEATURES

DEFAULT_CONFIG_PATH = REPO_ROOT / "Shap_config" / "shap_gat_runner_config.yaml"


def _load_config(path: Optional[Path]) -> Dict[str, Any]:
    config_path = path or DEFAULT_CONFIG_PATH
    with open(config_path, 'r') as fh:
        data = yaml.safe_load(fh) or {}
    return data.get('gat_interpretation', {})


def _load_metadata(project_root: Path, model_key: str, seed: str) -> Dict[str, Any]:
    meta_path = project_root / 'models' / 'full_dev' / model_key / f"seed_{seed}" / 'metadata.json'
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata missing: {meta_path}")
    with open(meta_path, 'r') as fh:
        return json.load(fh)


def _instantiate_gat(metadata: Dict[str, Any]) -> GATModel:
    params = metadata.get('model_config', {}).get('params', {})
    return GATModel(
        num_node_features=GAT_NUM_NODE_FEATURES,
        num_edge_features=GAT_NUM_EDGE_FEATURES,
        hidden_dim=params.get('hidden_dim', 64),
        num_heads=params.get('num_heads', 4),
        num_layers=params.get('num_layers', 3),
        dropout=params.get('dropout', 0.3)
    )


def _gat_forward(model: GATModel, edge_index: torch.Tensor, edge_attr: torch.Tensor):
    def forward(x: torch.Tensor) -> torch.Tensor:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return model(x, edge_index, batch, edge_attr=edge_attr)
    return forward


def _aggregation(attributions: torch.Tensor, method: str) -> np.ndarray:
    if method == 'sum':
        return attributions.sum(dim=-1).detach().cpu().numpy()
    raise NotImplementedError(f"Aggregation '{method}' is not implemented")


def _normalize_values(values: Iterable[np.ndarray], clip_threshold: float) -> List[np.ndarray]:
    flat = np.concatenate([v.flatten() for v in values]) if values else np.array([])
    if flat.size == 0:
        return [np.zeros_like(v) for v in values]
    clip_val = np.percentile(np.abs(flat), clip_threshold * 100)
    clip_val = max(clip_val, 1e-8)
    normalized = []
    for v in values:
        clipped = np.clip(v, -clip_val, clip_val)
        normalized.append(clipped / clip_val)
    return normalized


def _determine_output_dir(args, export_path: Path) -> Path:
    if args.output_dir:
        return args.output_dir
    split_root = next((p for p in export_path.parents if p.name.startswith('split_seed')), None)
    if split_root is None:
        base_root = export_path.parents[2]
    else:
        base_root = split_root.parent
    return base_root / 'shape' / args.model_key / f"seed_{args.seed}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-p', '--project-root', type=Path, default='models_out', help='QSAR output root')
    parser.add_argument('-m', '--model-key', required=True, help='Model key (GAT)')
    parser.add_argument('-s', '--seed', required=True, help='Seed identifier')
    parser.add_argument('-e', '--export', type=Path, required=True, help='Exported .npz (GAT graphs)')
    parser.add_argument('-c', '--config', type=Path, help='shap runner config path')
    parser.add_argument('-o', '--output-dir', type=Path, help='Where to save shape outputs (defaults to shape/<model>/seed)')
    args = parser.parse_args()

    cfg = _load_config(args.config)
    agg_method = cfg.get('aggregation_method', 'sum')
    clip_threshold = cfg.get('clipping_threshold', 0.995)
    output_format = cfg.get('output_format', 'csv_plus_npz')
    device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    export_data = np.load(args.export, allow_pickle=True)
    node_features = [np.asarray(v) for v in export_data['node_features']]
    edge_indices = [np.asarray(v) for v in export_data['edge_indices']]
    edge_attrs = [np.asarray(v) for v in export_data['edge_attrs']]
    valid_indices = export_data.get('valid_indices')

    metadata = _load_metadata(args.project_root, args.model_key, args.seed)
    if metadata.get('model_type') != 'pytorch_geometric':
        raise ValueError('Metadata does not correspond to a GAT model')

    model = _instantiate_gat(metadata)
    model_path = args.project_root / 'models' / 'full_dev' / args.model_key / f"seed_{args.seed}" / 'model.pt'
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint missing: {model_path}")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device).eval()

    scores_list: List[np.ndarray] = []
    info_rows: List[Dict[str, Any]] = []
    scores_list = []
    info_rows: List[Dict[str, Any]] = []

    for graph_idx, (node_feat_np, edge_index_np, edge_attr_np) in enumerate(zip(node_features, edge_indices, edge_attrs)):
        x = torch.tensor(node_feat_np.astype(np.float32), device=device)
        edge_index = torch.tensor(edge_index_np.astype(np.int64), device=device)
        edge_attr = torch.tensor(edge_attr_np.astype(np.float32), device=device)
        forward = _gat_forward(model, edge_index, edge_attr)
        ig = IntegratedGradients(forward)
        baseline = torch.zeros_like(x)
        attr_tensor = ig.attribute(x, baselines=baseline, target=None)
        node_scores = _aggregation(attr_tensor, agg_method)
        scores_list.append(node_scores)
        for node_idx, score in enumerate(node_scores):
            info_rows.append({
                'graph': graph_idx,
                'node': node_idx,
                'score': score,
            })

    normalized_list = _normalize_values(scores_list, clip_threshold)
    df = pd.DataFrame(info_rows)
    if normalized_list:
        offset = 0
        for idx, norm_scores in enumerate(normalized_list):
            n = len(norm_scores)
            df.loc[offset:offset+n-1, 'score_norm'] = norm_scores
            offset += n

    output_dir = _determine_output_dir(args, args.export)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / 'gat_atom_contributions.csv'
    df.to_csv(csv_path, index=False)

    npz_path = output_dir / 'gat_atom_contributions.npz'
    def _to_object_array(items: List[np.ndarray]) -> np.ndarray:
        arr = np.empty(len(items), dtype=object)
        for idx, item in enumerate(items):
            arr[idx] = item
        return arr

    np.savez_compressed(
        npz_path,
        node_scores=np.array(scores_list, dtype=object),
        node_scores_norm=np.array(normalized_list, dtype=object),
        edge_indices=_to_object_array(edge_indices),
        edge_attrs=_to_object_array(edge_attrs),
        valid_indices=valid_indices if valid_indices is not None else None,
    )

    print(f"Saved GAT contribution CSV: {csv_path}")
    print(f"Saved GAT contribution NPZ: {npz_path}")


if __name__ == '__main__':
    main()
