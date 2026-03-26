# QSAR Modeling with PyTorch Deep Learning

Streamlined QSAR pipeline that trains traditional ML, GAT, and ChemBERTa models, exports SHAP-ready artifacts, and visualizes contributions with a consistent Times New Roman + `RdBu_r` style.

## Quick steps

1. **Install dependencies** (CPU stack shown; add CUDA builds as needed):
   ```bash
   pip install numpy pandas scikit-learn matplotlib tqdm pyyaml shap joblib torch torchvision rdkit torch-geometric transformers xgboost lightgbm
   ```
2. **Prepare your CSV** with `id`, `smiles`, `label`, and optional `pic50`.
3. **Run training** via the config (e.g., `Config/test_config.yaml`):
   ```bash
   python Scripts/qsar_modeling_pytorch.py -c Config/test_config.yaml
   ```
4. **Inspect `models_out/{task}_{timestamp}`** for logs, results, predictions, saved models, and fingerprint/scaler processors.

## Core scripts

- `Scripts/qsar_modeling_pytorch.py`: entry point for single-split or two-stage CV training of LR/RFC/SVC/XGBC/LGBMC/ETC, MLP, GAT, and ChemBERTa models.
- `Scripts/export_pytorch_contributions.py`: exports Stage 1 tensors (nodes/fingerprints or tokens/masks) required by every downstream SHAP runner.
- `Scripts/run_gat_shap_runner.py`: Captum IG on GAT node features; configuration lives in `Shap_config/shap_runner_config.yaml`.
- `Scripts/run_chemberta_shap_runner.py`: LayerIntegratedGradients on ChemBERTa embeddings, saving per-token contributions plus offsets/IDs.
- `Scripts/visualize_exported_contributions.py`: renders SHAP summaries, GAT similarity maps, and ChemBERTa token heatmaps with Times New Roman + `RdBu_r` styling.
- `Scripts/gat_global_shap_summary.py`: accumulates SMARTS-based patterns (benzene, amide, hydroxyl, etc.) by mean SHAP effect and plots a bar chart.
- `Scripts/external_shap_analysis.py`: SHAP for classical models on the external split, producing consistent summary/heatmap figures.

## SHAP workflows

- **Traditional models (LR/RFC/SVC/XGBC/LGBMC/ETC/Ridge/RFR/ETR)**: train, then run `Scripts/external_shap_analysis.py -p models_out/... -m <model> -s <seed>` to compute SHAP on the saved external split.
- **GAT**: export with `Scripts/export_pytorch_contributions.py -m GAT -s <seed>`, run `Scripts/run_gat_shap_runner.py --export .../pytorch_shap_export.npz` (config overrides from `Shap_config/shap_runner_config.yaml`), and visualize via `Scripts/visualize_exported_contributions.py --gat-contributions .../gat_atom_contributions.csv`.
- **ChemBERTa**: export with `Scripts/export_pytorch_contributions.py -m ChemBERTa`, run `Scripts/run_chemberta_shap_runner.py --export .../pytorch_shap_export.npz --max-samples 64 --n-steps 32`, and use `Scripts/visualize_exported_contributions.py --chemberta-contributions .../chemberta_token_contributions.npz` for token-level heatmaps.

## Outputs & metrics

- **Outputs**: `models_out/.../results/` (metrics), `predictions/`, `models/`, `feature_processors/`, `exports/`, and `shape/` (SHAP artifacts + figures).  
- **Metrics**: classification (AUC, PR-AUC, MCC, ACC, F1, EF1%, Hit Rate); regression (R², RMSE, MAE).  
- **Validation**: single split for fast iteration, two-stage (external + K-fold CV) for publication-ready evaluation, optional multi-seed (`--seeds 42,123,2025`).

## Notes

- Default split is scaffold to prevent scaffold leakage; switch via `--split-method` or `--cv-split-method` as needed.  
- SHAP helpers reuse the exported fingerprints/pytorch tensors; visualizers trust the Runner’s global Max-Abs scaling so no further normalization is applied.  
- All figures (ROC/PR, similarity maps, ChemBERTa heatmaps, bar summaries) follow the Times New Roman + `RdBu_r` palette mandated in this flow.

## License & contact

Academic/research use only. Questions? Contact Toby Lo.
