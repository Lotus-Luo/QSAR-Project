#!/usr/bin/env python3
"""GTA Global SHAP summary for GAT models.

Loads exported GAT atom contributions, detects recurring substructures, aggregates
their SHAP contributions, and renders a Times New Roman bar plot showing each
pattern’s average effect (positive values increase activity; negative values
decrease it).

python Scripts/step15_summarize_global_features.py \
  --contributions models_out/classification_20260330_151751/split_seed_30/shap/GAT/seed_42/gat_atom_contributions.csv \
  --shap-export models_out/classification_20260330_151751/split_seed_30/exports/GAT/seed_42/pytorch_shap_export.npz \
  --output-dir models_out/classification_20260330_151751/split_seed_30/shap/GAT/seed_42/global_summary

"""

# %%
from pathlib import Path
from typing import Iterable, Optional

BASE_CONFIG = {
    "contributions": Path("models_out/classification_20260330_151751/split_seed_30/shap/GAT/seed_42/gat_atom_contributions.csv"),
    "shap_export": Path("models_out/classification_20260330_151751/split_seed_30/exports/GAT/seed_42/pytorch_shap_export.npz"),
    "output_dir": Path("models_out/classification_20260330_151751/split_seed_30/shap/GAT/seed_42/global_summary"),
}

# %%
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


SUBSTRUCTURE_SMARTS = OrderedDict([
    ("Benzene ring", "c1ccccc1"),
    ("Amide bond", "[NX3][CX3](=O)[#6]"),
    ("Hydroxyl group", "[OX2H]"),
    ("Carbonyl group", "[CX3]=O"),
    ("Nitro motif", "[NX3](=O)=O"),
])


def _ensure_smiles(values: Iterable) -> list[str]:
    if values is None:
        return []
    result = []
    for entry in np.atleast_1d(list(values)):
        if entry is None:
            continue
        if isinstance(entry, bytes):
            result.append(entry.decode("utf-8", "ignore"))
        else:
            result.append(str(entry))
    return result


def _save_fig(fig: plt.Figure, path_noext: Path) -> None:
    png = path_noext.with_suffix(".png")
    svg = path_noext.with_suffix(".svg")
    fig.tight_layout()
    fig.savefig(png, bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(svg, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)


def _load_gat_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"GAT contributions file not found: {path}")
    df = pd.read_csv(path)
    graph_col = _select_column(df.columns, ["graph", "graph_index", "mol_index", "sample_index"])
    atom_col = _select_column(df.columns, ["atom_index", "node", "atom", "node_index"])
    score_col = _select_column(df.columns, ["score_norm", "weight", "contribution_score", "score"])
    if graph_col is None or atom_col is None or score_col is None:
        raise ValueError("Expected graph/atom/score columns in GAT contributions CSV.")
    df = df.rename(columns={
        graph_col: "graph",
        atom_col: "atom_index",
        score_col: "weight",
    })
    df = df[["graph", "atom_index", "weight"]].copy()
    df["graph"] = df["graph"].astype(int)
    df["atom_index"] = df["atom_index"].astype(int)
    df["weight"] = df["weight"].astype(float)
    return df


def _build_weight_map(df: pd.DataFrame) -> dict[int, np.ndarray]:
    weight_map: dict[int, np.ndarray] = {}
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


def _select_column(columns: Iterable[str], choices):
    lookup = {col.lower(): col for col in columns}
    for choice in choices:
        key = choice.lower()
        if key in lookup:
            return lookup[key]
    return None


def _prepare_heavy_molecule(Chem, smiles: str, weights: np.ndarray, graph_idx: int):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Skipping graph {graph_idx}: invalid SMILES.")
        return None, None
    mol = Chem.AddHs(mol)
    weight_array = np.asarray(weights, dtype=float)
    if mol.GetNumAtoms() != weight_array.size:
        print(
            f"Skipping graph {graph_idx}: hydrogen-inclusive atom count "
            f"{mol.GetNumAtoms()} does not match contributions ({weight_array.size})."
        )
        return None, None
    heavy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() != "H"]
    mol_no_h = Chem.RemoveHs(mol)
    if len(heavy_indices) != mol_no_h.GetNumAtoms():
        print(f"Skipping graph {graph_idx}: RemoveHs produced {mol_no_h.GetNumAtoms()} heavy atoms "
              f"but expected {len(heavy_indices)}.")
        return None, None
    heavy_weights = weight_array[heavy_indices]
    return mol_no_h, heavy_weights


def _aggregate_scores(
    weight_map: dict[int, np.ndarray],
    smiles_list: list[str],
    valid_indices: Optional[list[int]],
) -> dict[str, float]:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise SystemExit("RDKit is required for substructure detection (pip install rdkit).") from exc

    patterns = []
    for name, smarts in SUBSTRUCTURE_SMARTS.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            raise ValueError(f"Unable to parse SMARTS for {name}: {smarts}")
        patterns.append((name, patt))

    scores: dict[str, list[float]] = {name: [] for name in SUBSTRUCTURE_SMARTS}
    for graph_idx, weights in weight_map.items():
        dataset_idx = (
            valid_indices[graph_idx] if valid_indices and graph_idx < len(valid_indices) else graph_idx
        )
        if dataset_idx >= len(smiles_list):
            continue
        smiles = smiles_list[dataset_idx]
        if not smiles:
            continue
        mol, heavy_weights = _prepare_heavy_molecule(Chem, smiles, weights, graph_idx)
        if mol is None or heavy_weights is None:
            continue
        for name, patt in patterns:
            matches = mol.GetSubstructMatches(patt, uniquify=True)
            for match in matches:
                match_weights = heavy_weights[list(match)]
                scores[name].append(float(np.sum(match_weights)))

    aggregated = {}
    for name, values in scores.items():
        if not values:
            continue
        aggregated[name] = float(np.mean(values))
    return aggregated


def _draw_bar_chart(output_dir: Path, aggregated: dict[str, float]):
    if not aggregated:
        print("No substructures were matched; nothing to plot.")
        return
    items = sorted(aggregated.items(), key=lambda kv: kv[1], reverse=True)
    labels, values = zip(*items)
    colors = ["#d73027" if val >= 0 else "#4575b4" for val in values]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 5))
    bars = ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Average SHAP contribution", fontsize=14)
    ax.set_title("GTA Global SHAP Substructure Summary", fontsize=16)
    y_range = max(0.2, max(abs(v) for v in values))
    for idx, bar in enumerate(bars):
        value = values[idx]
        va = "bottom" if value >= 0 else "top"
        offset = 0.02 * y_range if value >= 0 else -0.02 * y_range
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:+.2f}",
            ha="center",
            va=va,
            fontname="Times New Roman",
        )
    chart_path = output_dir / "gat_global_substructure_summary"
    _save_fig(fig, chart_path)
    print(f"Substructure summary chart saved to {chart_path.with_suffix('.png')}")


def main():
    output_dir = BASE_CONFIG["output_dir"] or BASE_CONFIG["contributions"].parent / "gat_global_shap_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_gat_dataframe(BASE_CONFIG["contributions"])
    weight_map = _build_weight_map(df)
    data = np.load(BASE_CONFIG["shap_export"], allow_pickle=True)
    smiles_list = _ensure_smiles(data.get("smiles"))
    valid_indices_raw = data.get("valid_indices")
    valid_indices = (
        [int(idx) for idx in np.atleast_1d(valid_indices_raw)]
        if valid_indices_raw is not None
        else None
    )
    aggregated = _aggregate_scores(weight_map, smiles_list, valid_indices)
    if not aggregated:
        print("No substructure contributions were aggregated.")
        return
    for name, value in aggregated.items():
        print(f"{name}: {value:+.2f}")

    _draw_bar_chart(output_dir, aggregated)


if __name__ == "__main__":
    main()
