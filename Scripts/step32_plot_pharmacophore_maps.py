#!/usr/bin/env python3
"""Visualize exported PyTorch attributions/joint contributions.
Usage:
python Scripts/step32_plot_pharmacophore_maps.py \
  -i models_out/classification_20260329_165120/split_seed_29/exports/MLP/seed_42/pytorch_shap_export.npz \
  -o models_out/classification_20260329_165120/split_seed_29/shap/MLP/  \
  -m MLP \
  -s 42 \
  --max-display 25  \
  --heatmap-samples 50

Optional GAT mode example:
python Scripts/step32_plot_pharmacophore_maps.py \
  -i models_out/classification_20260326_180529/split_seed_3/exports/GAT/seed_42/pytorch_shap_export.npz \
  -m GAT \
  -s 42 \
  --gat-contributions models_out/classification_20260326_180529/split_seed_3/shap/GAT/seed_42/gat_atom_contributions.csv \
  --gat-max-molecules 16 \
  --gat-image-size 600

python Scripts/step32_plot_pharmacophore_maps.py \
  -i models_out/classification_20260326_213228/split_seed_3/exports/ChemBERTa/seed_42/pytorch_shap_export.npz \
  -m ChemBERTa \
  -s 42 \
  --chemberta-token-contributions models_out/classification_20260326_213228/split_seed_3/shape/ChemBERTa/seed_42/chemberta_token_contributions.npz \
  --heatmap-samples 64 \
  --output-dir models_out/classification_20260326_213228/split_seed_3/shape/ChemBERTa/seed_42/figures


--o ./model_out/classificationXXX/split_seed_x/shape/<model>/seed_y (defualt)
"""

import argparse
from pathlib import Path
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
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


def _normalize_text_array(values):
    if values is None:
        return []
    array = np.atleast_1d(values)
    normalized = []
    for entry in array:
        if entry is None or (isinstance(entry, (float, np.floating)) and np.isnan(entry)):
            normalized.append("")
        elif isinstance(entry, bytes):
            normalized.append(entry.decode("utf-8", "ignore"))
        else:
            normalized.append(str(entry))
    return normalized


def _sanitize_label(value: str, fallback: str) -> str:
    if not value:
        return fallback
    cleaned = []
    for ch in value:
        if ch.isalnum() or ch in "-_.":
            cleaned.append(ch)
        else:
            cleaned.append("_")
    cleaned_value = "".join(cleaned).strip("-_.")
    return cleaned_value[:64] or fallback


def _prepare_molecule_for_visualization(Chem, smiles: str, weights: np.ndarray, graph_idx: int):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Skipping GAT graph {graph_idx}: RDKit failed to parse SMILES.")
        return None, None
    mol = Chem.AddHs(mol)
    weight_array = np.asarray(weights, dtype=float)
    if mol.GetNumAtoms() != weight_array.size:
        print(
            f"Skipping GAT graph {graph_idx}: atom count ({mol.GetNumAtoms()}) "
            f"does not match contribution length ({weight_array.size})."
        )
        return None, None

    heavy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() != "H"]
    mol_no_h = Chem.RemoveHs(mol)
    if len(heavy_indices) != mol_no_h.GetNumAtoms():
        print(
            f"Skipping GAT graph {graph_idx}: RemoveHs filtered {len(heavy_indices)} atoms "
            f"but drawing molecule has {mol_no_h.GetNumAtoms()} atoms."
        )
        return None, None

    if not heavy_indices:
        print(f"Skipping GAT graph {graph_idx}: no heavy atoms after filtering.")
        return None, None

    symbols_before = [mol.GetAtomWithIdx(idx).GetSymbol() for idx in heavy_indices]
    symbols_after = [atom.GetSymbol() for atom in mol_no_h.GetAtoms()]
    if symbols_before != symbols_after:
        print(f"Skipping GAT graph {graph_idx}: atom ordering mismatch after RemoveHs.")
        return None, None

    heavy_weights = weight_array[heavy_indices]
    return mol_no_h, heavy_weights


def _select_column(columns: Iterable[str], candidates):
    lookup = {col.lower(): col for col in columns}
    for candidate in candidates:
        candidate_key = candidate.lower()
        if candidate_key in lookup:
            return lookup[candidate_key]
    return None


def _canonicalize_gat_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("GAT contributions file contains no data.")
    graph_col = _select_column(df.columns, ["graph", "graph_index", "molecule", "mol_index", "sample_index"])
    atom_col = _select_column(df.columns, ["atom_index", "node", "atom", "node_index"])
    weight_col = _select_column(df.columns, ["score_norm", "weight", "contribution_score", "score"])
    if graph_col is None or atom_col is None or weight_col is None:
        raise ValueError("Missing required columns (graph, atom_index/node, score) in GAT contributions file.")
    df = df.rename(columns={
        graph_col: "graph",
        atom_col: "atom_index",
        weight_col: "weight",
    })
    df = df[["graph", "atom_index", "weight"]].copy()
    df["graph"] = df["graph"].astype(int)
    df["atom_index"] = df["atom_index"].astype(int)
    df["weight"] = df["weight"].astype(float)
    df = df.dropna(subset=["weight"])
    return df


def _load_gat_contribution_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"GAT contributions file missing: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".npz":
        rows = []
        with np.load(path, allow_pickle=True) as raw:
            norm_scores = raw.get("node_scores_norm")
            scores = raw.get("node_scores")
            data_source = norm_scores if norm_scores is not None else scores
            if data_source is None:
                raise ValueError("GAT contributions NPZ is missing node score data.")
            for graph_idx, weights in enumerate(np.atleast_1d(data_source)):
                if weights is None:
                    continue
                weights_array = np.asarray(weights)
                for atom_idx, value in enumerate(weights_array):
                    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
                        continue
                    rows.append({
                        "graph": int(graph_idx),
                        "atom_index": int(atom_idx),
                        "weight": float(value),
                    })
        df = pd.DataFrame(rows)
    else:
        raise ValueError("GAT contributions must be provided as a CSV or NPZ file.")
    return _canonicalize_gat_dataframe(df)


def _render_similarity_map_assets(mol, weights, size: int, cmap):
    from rdkit.Chem.Draw import rdMolDraw2D, SimilarityMaps

    width = height = int(size)
    png_drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    SimilarityMaps.GetSimilarityMapFromWeights(
        mol,
        weights.tolist() if hasattr(weights, "tolist") else weights,
        draw2d=png_drawer,
        colorMap=cmap,
        scale=1.0,
        size=(width, height),
    )
    png_drawer.FinishDrawing()
    png_bytes = png_drawer.GetDrawingText()

    svg_drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    SimilarityMaps.GetSimilarityMapFromWeights(
        mol,
        weights.tolist() if hasattr(weights, "tolist") else weights,
        draw2d=svg_drawer,
        colorMap=cmap,
        scale=1.0,
        size=(width, height),
    )
    svg_drawer.FinishDrawing()
    svg_text = svg_drawer.GetDrawingText()
    return png_bytes, svg_text


def _write_similarity_map_files(path_noext: Path, png_bytes, svg_text: str):
    path_noext.with_suffix(".png").write_bytes(png_bytes)
    path_noext.with_suffix(".svg").write_text(svg_text, encoding="utf-8")


def _build_weight_map(df: pd.DataFrame) -> dict[int, np.ndarray]:
    weight_map = {}
    for graph_idx, group in df.groupby("graph", sort=True):
        atom_indices = group["atom_index"].astype(int).values
        if atom_indices.size == 0:
            continue
        max_idx = int(atom_indices.max())
        weights = np.zeros(max_idx + 1, dtype=float)
        for atom_idx, weight in zip(atom_indices, group["weight"].astype(float).values):
            weights[int(atom_idx)] = float(weight)
        weight_map[int(graph_idx)] = weights
    return weight_map


def _select_top_graphs(weight_map: dict[int, np.ndarray], max_molecules: int) -> list[int]:
    scored = [
        (graph_idx, float(np.sum(np.abs(weights))))
        for graph_idx, weights in weight_map.items()
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    limit = min(max_molecules, len(scored))
    return [graph_idx for graph_idx, _ in scored[:limit]]


def _visualize_gat_contributions(
    data,
    contributions_path: Path,
    output_dir: Path,
    max_molecules: int,
    image_size: int,
) -> int:
    if max_molecules <= 0:
        print("--gat-max-molecules must be greater than 0 to render GAT maps.")
        return 0
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise SystemExit("RDKit is required for GAT visualization (pip install rdkit).") from exc

    df = _load_gat_contribution_dataframe(contributions_path)
    weight_map = _build_weight_map(df)
    if not weight_map:
        raise ValueError("No GAT contributions were loaded for visualization.")

    selected_graphs = _select_top_graphs(weight_map, max_molecules)
    smiles_list = _normalize_text_array(data.get("smiles"))
    ids_list = _normalize_text_array(data.get("ids"))
    valid_indices_raw = data.get("valid_indices")
    valid_indices = (
        [int(idx) for idx in np.atleast_1d(valid_indices_raw)]
        if valid_indices_raw is not None
        else None
    )
    cmap = plt.get_cmap("RdBu_r")
    saved = 0

    for graph_idx in selected_graphs:
        dataset_idx = (
            valid_indices[graph_idx] if valid_indices and graph_idx < len(valid_indices) else graph_idx
        )
        smiles = smiles_list[dataset_idx] if dataset_idx < len(smiles_list) else ""
        if not smiles:
            print(f"Skipping GAT graph {graph_idx}: missing SMILES string.")
            continue
        weights = weight_map[graph_idx]
        mol, heavy_weights = _prepare_molecule_for_visualization(Chem, smiles, weights, graph_idx)
        if mol is None or heavy_weights is None:
            continue
        try:
            png_bytes, svg_text = _render_similarity_map_assets(mol, heavy_weights, image_size, cmap)
        except Exception as exc:
            print(f"Failed to draw GAT graph {graph_idx}: {exc}")
            continue
        label_idx = dataset_idx if dataset_idx < len(ids_list) else graph_idx
        label = ids_list[label_idx] if label_idx < len(ids_list) else ""
        label = _sanitize_label(label, f"graph_{graph_idx:04d}")
        _write_similarity_map_files(output_dir / f"gat_similarity_map_{graph_idx:04d}_{label}", png_bytes, svg_text)
        saved += 1

    return saved


def _load_chemberta_token_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"ChemBERTa token contributions file missing: {path}")
    data = np.load(path, allow_pickle=True)
    token_contributions = [np.asarray(entry) for entry in np.atleast_1d(data["token_contributions"])]
    return {
        "token_contributions": token_contributions,
        "token_strings": [np.asarray(entry, dtype=object) for entry in np.atleast_1d(data["token_strings"])],
        "attention_mask": [np.asarray(entry) for entry in np.atleast_1d(data["attention_mask"])],
        "token_offsets": [np.asarray(entry) for entry in np.atleast_1d(data["token_offsets"])],
        "valid_indices": np.atleast_1d(data.get("valid_indices", np.arange(len(token_contributions)))).astype(int),
        "ids": np.atleast_1d(data.get("ids") if data.get("ids") is not None else []),
        "smiles": np.atleast_1d(data.get("smiles") if data.get("smiles") is not None else []),
    }


def _render_chemberta_token_heatmap(data, output_dir: Path, max_samples: int):
    token_contributions = data["token_contributions"]
    attention_masks = data["attention_mask"]
    token_strings = data["token_strings"]
    valid_indices = data["valid_indices"]
    ids = data["ids"]
    smiles = data["smiles"]
    contrib_scores = [float(np.sum(np.abs(attr))) for attr in token_contributions]
    if not contrib_scores:
        print("ChemBERTa token contributions contain no samples.")
        return
    order = np.argsort(contrib_scores)[::-1][:max_samples]
    max_tokens = int(max(np.sum(mask) for mask in attention_masks))
    max_samples = min(max_samples, len(order))
    vmax = max(np.max(np.abs(attr)) for attr in token_contributions) if token_contributions else 1.0
    vmax = float(max(vmax, 1e-6))
    norm = mpl_colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")
    height = max(4, max_samples * 0.6 + 1)
    fig, ax = plt.subplots(figsize=(12, height))
    ax.set_xlim(-2, max_tokens + 2)
    ax.set_ylim(-0.5, max_samples - 0.5)
    ax.set_axis_off()
    for row_idx, sample_pos in enumerate(order):
        y = max_samples - row_idx - 1
        tokens = token_strings[sample_pos]
        mask = attention_masks[sample_pos]
        contributions = token_contributions[sample_pos]
        x = 0
        for token, weight, attn in zip(tokens, contributions, mask):
            if int(attn) == 0:
                break
            ax.text(
                x,
                y,
                str(token),
                fontname="Times New Roman",
                fontsize=12,
                color=cmap(norm(weight)),
                ha="left",
                va="center",
            )
            x += 1
        label_idx = valid_indices[sample_pos] if sample_pos < len(valid_indices) else sample_pos
        label = ids[label_idx] if label_idx < len(ids) else f"idx_{label_idx}"
        ax.text(-1.5, y, str(label), fontname="Times New Roman", fontsize=10, ha="right", va="center")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.02, label="Token contribution")
    chart_path = output_dir / "chemberta_token_heatmap"
    _save_fig(fig, chart_path)
    print(f"ChemBERTa token heatmap saved to: {chart_path}.png")


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
    parser.add_argument("--chemberta-token-contributions", type=Path, help="ChemBERTa token contributions .npz for text heatmap")
    parser.add_argument("--chemberta-max-samples", type=int, default=16, help="Max sequences to render in ChemBERTa heatmap")
    parser.add_argument("--gat-contributions", type=Path, help="CSV or NPZ with per-atom GAT contributions")
    parser.add_argument("--gat-max-molecules", type=int, default=16, help="Max number of GAT molecules to visualize")
    parser.add_argument("--gat-image-size", type=int, default=360, help="Square pixel size for each GAT similarity map")
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
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

    if args.chemberta_token_contributions:
        chem_data = _load_chemberta_token_data(args.chemberta_token_contributions)
        _render_chemberta_token_heatmap(chem_data, output_dir, args.chemberta_max_samples)
        return

    if args.gat_contributions:
        saved = _visualize_gat_contributions(
            data,
            args.gat_contributions,
            output_dir,
            args.gat_max_molecules,
            args.gat_image_size,
        )
        if saved:
            print(f"GAT similarity maps saved to: {output_dir}")
        else:
            print("No GAT similarity maps were generated.")
        return

    shap_values = _normalize_shap_matrix(data["shap_values"])
    features = data.get("features")
    feature_names = _ensure_feature_names(data.get("feature_names"), shap_values.shape[1])

    if features is None or features.size == 0:
        raise ValueError("Exported feature matrix is empty. Nothing to plot.")

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
