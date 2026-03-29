# QSAR Modeling with PyTorch Deep Learning

Streamlined QSAR pipeline that trains traditional ML, GAT, and ChemBERTa models, exports SHAP-ready artifacts, and visualizes contributions.

## Quick steps

1. **Install dependencies** (CPU stack shown; add CUDA builds as needed):
   ```bash
   pip install numpy pandas scikit-learn matplotlib tqdm pyyaml shap joblib torch torchvision rdkit torch-geometric transformers xgboost lightgbm
   ```
2. **Prepare your CSV** with `id`, `smiles`, `label`, and optional `pic50`.
3. **Run training** via the config (e.g., `Config/test_config.yaml`):
   ```bash
   python Scripts/step01_train_qsar_models.py -c Config/test_config.yaml
   ```
4. **Inspect `models_out/{task}_{timestamp}`** for logs, results, predictions, saved models, and fingerprint/scaler processors.
5. [**Pretrained models download(ChemBERT from Huggingface)**](https://drive.google.com/drive/folders/1sYv695rq7FSW5fiH6dIScCWBZDLmmtni?usp=drive_link)

## Scripts by stage

### Stage 1 – Core Modeling
- `Scripts/step01_train_qsar_models.py`: entry point for single-split or two-stage CV training of LR/RFC/SVC/XGBC/LGBMC/ETC, MLP, GAT, and ChemBERTa models.
- `Scripts/step02_run_smoke_test.py`: lightweight regression/classification sanity checks covering NaNs, metrics, and ChemBERTa loader stability.
- `Scripts/step03_validate_model_robustness.py`: once the best model is identified, run this Y-scrambling / permutation test to show the actual metric (AUC / R²) stands apart from random-label re-fits; it saves histogram/scatter figures and statistics under `models_out/validation/<model>/seed_<seed>`.

### Stage 2 – Model interpretation / SHAP workflows
- `Scripts/step11_extract_contributions.py`: exports SHAP-style tensors (fingerprints, GAT node features, ChemBERTa tokens) for every PyTorch model (MLP, GAT, ChemBERTa) so the visualizers have consistent inputs.
- `Scripts/step12_shap_interpreter_gat.py`: Captum IG on the exported GAT graphs using `Shap_config/shap_gat_runner_config.yaml`, writing per-atom weights before visualization.
- `Scripts/step13_shap_interpreter_chemberta.py`: LayerIntegratedGradients over ChemBERTa embeddings to generate token contributions, offsets, and SMILES links for downstream heatmaps.
- `Scripts/step14_compare_external_shap.py`: SHAP on the traditional classifiers (LR/RFC/SVC/XGBC/LGBMC/ETC/Ridge/RFR/ETR) that live under `models/full_dev/`; saves summary/heatmap figures and `feature_importance.csv`.
- `Scripts/step15_summarize_global_features.py`: aggregates SMARTS-based patterns (benzene, amide, hydroxyl, etc.) by mean SHAP effect and draws a styled bar chart.
- `Scripts/step32_plot_pharmacophore_maps.py`: loads `step11`/`step12`/`step13` exports to render SHAP summaries, heatmaps, and GAT/ChemBERTa similarity maps with the Times New Roman style.

### Stage 3 – Virtual screening pipeline
- `Scripts/step21_vs_inference.py`: loads every saved model for a run, reapplies fingerprint preprocessing/scalers, and writes a wide predictions table with per-model columns plus `Consensus_Sum`.
- `Scripts/step22_vs_filter_hits.py`: filters the virtual-screening table for candidates supported by at least `--min-sum` models and (optionally) one high-probability score.

### Stage 4 – Visualization
- `Scripts/step31_plot_performance_metrics.py`: aggregates metrics/predictions for publication-ready curves.
- `Scripts/step32_plot_pharmacophore_maps.py`: renders ROC/PR, similarity maps, and ChemBERTa heatmaps using the Times New Roman + `RdBu_r` palette.

## SHAP workflows

- **Traditional sklearn models (LR/RFC/SVC/XGBC/LGBMC/ETC/Ridge/RFR/ETR)**: train with `step01`, then run `Scripts/step14_compare_external_shap.py -p models_out/... -m <model> -s <seed>` to compute SHAP on the saved external split and write the summary plot, heatmap, and `feature_importance.csv`.
- **MLP**: export the saved `model.pt` via `Scripts/step11_extract_contributions.py -m MLP -s <seed>` (produces `exports/MLP/seed_<seed>/pytorch_shap_export.npz`), then visualize SHAP summaries/heatmaps with `Scripts/step32_plot_pharmacophore_maps.py -m MLP -s <seed> -i .../pytorch_shap_export.npz --max-display 25 --heatmap-samples 50`.
- **GAT**: run `Scripts/step11_extract_contributions.py -m GAT -s <seed>` to get the graph export, run `Scripts/step12_shap_interpreter_gat.py --export .../pytorch_shap_export.npz` (override runner settings via `Shap_config/shap_gat_runner_config.yaml` if needed), and finalize the visuals with `Scripts/step32_plot_pharmacophore_maps.py --gat-contributions .../gat_atom_contributions.csv`.
- **ChemBERTa**: export via `Scripts/step11_extract_contributions.py -m ChemBERTa -s <seed>`, run `Scripts/step13_shap_interpreter_chemberta.py --export .../pytorch_shap_export.npz --max-samples 64 --n-steps 32`, and draw token heatmaps with `Scripts/step32_plot_pharmacophore_maps.py --chemberta-token-contributions .../chemberta_token_contributions.npz`.

## Virtual screening pipeline

- Run `Scripts/step21_vs_inference.py` on a new compound list to create the wide prediction table (saved under `./virtual_screening/` by default), then prune with `Scripts/step22_vs_filter_hits.py --min-sum <N> [--min-score 0.7]` to shortlist high-confidence candidates.

## Typical training & inference flow

1. **Training (Stage 1)** – run `python Scripts/step01_train_qsar_models.py -c Config/test_config.yaml` or supply CLI overrides to produce `models_out/{task}_{timestamp}/split_seed_<k>/`.
2. **Smoke validation** – `python Scripts/step02_run_smoke_test.py` quickly checks NaN handling and prediction exports.
3. **Virtual screening (Stage 3)** – `python Scripts/step21_vs_inference.py --run-dir models_out/.../split_seed_<k> --input <compounds.csv>` writes `virtual_screening/<run>_<ts>.csv`; then filter via `python Scripts/step22_vs_filter_hits.py --input virtual_screening/... --min-sum 2 [--min-score 0.7]`.
4. **SHAP interpretation (Stage 2)** – export tensors with `step11`, run `step12`/`step13`/`step14` for your chosen model/seed, and aggregate/visualize via `step15` and `step32` to explain the filtered hits.

## Outputs & metrics

- **Outputs**: `models_out/.../results/` (metrics), `predictions/`, `models/`, `feature_processors/`, `exports/`, and `shape/` (SHAP artifacts + figures).  
- **Metrics**: classification (AUC, PR-AUC, MCC, ACC, F1, EF1%, Hit Rate); regression (R², RMSE, MAE).  
- **Validation**: single split for fast iteration, two-stage (external + K-fold CV) for publication-ready evaluation, optional multi-seed (`--seeds 3,29,2026`).

## Notes

- Default split is scaffold to prevent scaffold leakage; switch via `--split-method` or `--cv-split-method` as needed.  
- SHAP helpers reuse the exported fingerprints/pytorch tensors; visualizers trust the Runner’s global Max-Abs scaling so no further normalization is applied.  

## License & contact

**Academic research use only.**
If any quesion, contact TobyLo.  
luooo8961@gmail.com
