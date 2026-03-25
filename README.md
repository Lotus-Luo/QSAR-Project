# QSAR Modeling with PyTorch Deep Learning

Enhanced QSAR modeling tool integrating traditional machine learning and deep learning models for molecular property prediction.

Supports two validation modes:
- **Single Split Mode**: Simple 80/20 train/validation split with model saving and SHAP analysis
- **Two-Stage Validation Mode** (Recommended): K-Fold Cross-Validation + External Test Set evaluation for more reliable performance assessment

## Features

- **Traditional Machine Learning Models**: LR, RFC, SVC, XGBC, LGBMC, ETC
- **Deep Learning Models**: MLP, GAT (Graph Attention Network), ChemBERTa
- **Enhanced Evaluation Metrics**: AUC, PR-AUC, MCC, ACC, F1, EF1% (Early Enrichment Factor), Hit Rate
- **YAML Configuration File Support**: Flexible configuration management
- **Two-Stage Validation Workflow**: Automatically performs external test set evaluation when using K-Fold CV
- **Automatic Fingerprint Generation**: Supports automatic generation of Morgan, MACCS, RDKit, AtomPair, Torsion fingerprints
- **Feature Processing**: Variance filtering and standardization within each fold to prevent data leakage; supports frequency-based binary fingerprint filtering
- **SMILES Validation**: Automatically validates SMILES string validity and reports invalid entries
- **Version Checking**: Automatically checks version compatibility of critical libraries
- **Training Progress Display**: Real-time progress bars during deep learning model training
- **GPU Memory Management**: Automatically clears GPU cache to prevent memory leaks
- **Multiple Data Split Methods**: Supports scaffold (skeleton), stratified, and random splitting, with scaffold as default to avoid data leakage

## New Features

### v2.0 Updates

- **Binary Fingerprint Frequency Filtering**: Added `min_frequency` parameter for frequency-based filtering of binary fingerprint bits, removing features with too low frequency
- **Automatic SMILES Validation**: Automatically validates SMILES string validity when loading data, records invalid entries, and suggests data cleaning
- **Library Version Checking**: Automatically checks version compatibility of critical libraries (PyTorch, PyTorch Geometric, Transformers, RDKit) at startup
- **Training Progress Display**: Real-time progress bars during deep learning model training for improved user experience
- **GPU Memory Management**: Automatically clears GPU cache to prevent memory leaks, supporting long-duration training
- **Random Seed Improvement**: Unified random seed setting for numpy, torch, and python random to ensure full reproducibility
- **Metric Validity Checking**: Automatically validates external test set evaluation metric validity, falling back to default values for non-existent metrics
- **Feature Name Return**: Feature processing function returns filtered feature name list to support SHAP analysis
- **Model Compatibility Checking**: After command-line parameter overrides configuration, automatically verifies model compatibility with current task
- **ChemBERTa Improvements**: Uses HuggingFace's `AutoModelForSequenceClassification` following best practices
- **Enhanced Model Saving**: Saves feature processors in K-Fold CV mode, supporting same transformation on new data
- **ScaffoldKFold Improvements**: Uses greedy allocation strategy to avoid empty folds in K-Fold CV, resulting in more balanced sample distribution
- **CV Split Method Selection**: Added `--cv-split-method` parameter to independently control K-Fold cross-validation split method
- **K-Fold CV Detailed Saving for SHAP**: Introduced `--save-cv-details` parameter to save models, processed features, labels, SMILES, IDs, and feature processors for each fold and seed during K-fold cross-validation. This enables robust post-hoc SHAP analysis for all model types, including deep learning, for each fold.

## Input Data Format

Input CSV file must contain the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `id` | Molecular unique identifier | CHEMBL4523582 |
| `smiles` | SMILES string representation of the molecule | C1=CC=CC=C1C(=O)O |
| `label` | Binary classification label (0 or 1) | 1 |
| `pic50` | Continuous pIC50 value (for regression tasks) | 7.5 |

### Example Data (Data/test_data.csv)

```csv
id,smiles,label,pic50
CHEMBL4523582,C1=CC=CC=C1C(=O)O,1,7.5
CHEMBL4523583,CC(C)C1=CC=CC=C1,0,5.2
...
```

## Installing Dependencies

### Basic Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib tqdm pyyaml shap joblib
```

### GPU Support (PyTorch)
```bash
pip install torch torchvision
```

### RDKit (for fingerprint generation and GAT models)
```bash
pip install rdkit
```

### Graph Neural Network Support (GAT)
```bash
pip install torch-geometric
```

### ChemBERTa Support
```bash
pip install transformers
```

### XGBoost / LightGBM
```bash
pip install xgboost lightgbm
```

### Complete Installation (Recommended)
```bash
pip install numpy pandas scikit-learn matplotlib tqdm pyyaml shap joblib torch torchvision rdkit torch-geometric transformers xgboost lightgbm
```

### GPU Environment Installation (Optional)
```bash
# Install CUDA version of PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### 1. Using YAML Configuration File (Recommended)

Create a configuration file `config.yaml`:

```yaml
# Basic configuration
input_path: "Data/test_data.csv"
label_column: "label"
pic50_column: "pic50"
smiles_column: "smiles"
id_column: "id"

# Task type
task: "classification"  # or "regression"

# Data splitting
test_size: 0.2
folds: 5
seed: 42
seeds: [42, 123, 2025]  # Multiple seeds for robustness testing (optional, overrides seed if provided)
split_method: "scaffold"  # Options: scaffold (default), stratified, random

# CV split method (only applies when folds > 1)
cv_split_method: "scaffold"  # Options: scaffold (recommended), random

# External test set evaluation metric (classification: AUC, PR_AUC, ACC, F1, MCC; regression: R2, RMSE, MAE)
external_test_metric: "MCC"

# Model selection
selected_models: ["XGBC", "RFC", "MLP", "GAT"]

# Output
output_dir: "models_output"

# Feature filtering
variance_threshold: 0.01
min_frequency: 0.05  # Minimum occurrence frequency for binary fingerprints (5%), used to filter rare bits

# Deep learning parameters
max_epochs: 100
batch_size: 32
learning_rate: 0.001
early_stopping_patience: 10

# Early enrichment factor
ef_percentile: 1.0

# Automatic fingerprint generation
auto_generate_fingerprints: true
fingerprint_types: ["morgan"]  # Options: morgan, maccs, rdkit, atompair, torsion

# SHAP analysis
run_shap: true
shap_max_display: 20
shap_sample_size: 500

# Binary fingerprint frequency filtering (optional)
# min_frequency: 0.05  # Retain feature bits that appear in at least 5% of samples
```

Run:
```bash
python qsar_modeling_pytorch.py -c config.yaml
```

### 2. Using Command Line Arguments

#### Basic Usage
```bash
# Train traditional machine learning models
python qsar_modeling_pytorch.py \
  -i Data/test_data.csv \
  -l label \
  -m XGBC,RFC,LR \
  -o output/

# Train deep learning models
python qsar_modeling_pytorch.py \
  -i Data/test_data.csv \
  -l label \
  -s smiles \
  -m MLP,GAT \
  -o output/

# Train all available models
python qsar_modeling_pytorch.py \
  -i Data/test_data.csv \
  -l label \
  -s smiles \
  -o output/
```

#### Complete Parameter Example
```bash
python qsar_modeling_pytorch.py \
  -i Data/test_data.csv \
  -l label \
  -s smiles \
  -d id \
  -t classification \
  -m XGBC,RFC,MLP,GAT \
  -o output/ \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.001 \
  --folds 5 \
  --seed 42 \
  --seeds 42,123,2025 \
  --test-size 0.2 \
  --split-method scaffold \
  --ef 1.0 \
  --variance-threshold 0.01
```

#### Multi-Seed Training Example
```bash
# Train with multiple seeds for robustness testing
python qsar_modeling_pytorch.py \
  -i Data/test_data.csv \
  -l label \
  -s smiles \
  -m XGBC,MLP,GAT,ChemBERTa \
  --seeds 42,123,2025 \
  --folds 5 \
  --split-method scaffold \
  --external-test-metric MCC
```

### 3. Regression Tasks

```bash
python qsar_modeling_pytorch.py \
  -i Data/test_data.csv \
  -p pic50 \
  -s smiles \
  -t regression \
  -m Ridge,RFR,MLP \
  -o output_regression/
```

### 4. Two-Stage Validation Workflow (K-Fold CV Mode)

When using `--folds > 1`, the tool automatically executes a two-stage validation workflow for more reliable model performance evaluation:

#### Stage 1: External Test Set Split
- Split data into external test set (20%) and development set (80%)
- Use specified split method (scaffold/stratified/random)
- External test set is reserved for final model evaluation and does not participate in model training and tuning

#### Stage 2: K-Fold Cross-Validation
- Perform K-Fold cross-validation on the development set
- Independent feature processing (variance filtering and standardization) within each fold to prevent data leakage
- Calculate average performance and standard deviation of each model on the development set
- CV split method can be specified via `--cv-split-method` parameter:
  - `scaffold`: Use improved greedy strategy ScaffoldKFold (recommended, avoids empty folds, balanced sample distribution)
  - `random`: Use standard random stratified split, no guarantee of scaffold independence

#### Stage 3: External Test Set Evaluation
- Train all models on the complete development set
- Evaluate performance of all models on the external test set
- Select best models based on specified metric (`--external-test-metric`)

**Output Files**:
- `fold_results.csv`: Detailed results for each fold (development set)
- `cv_summary.csv`: Cross-validation summary results (mean ± std)
- `external_test_results.csv`: External test set evaluation results
- `external_test_predictions.csv`: External test set prediction results

**Example**:
```bash
python qsar_modeling_pytorch.py \
  -i Data/test_data.csv \
  -l label \
  -s smiles \
  -m XGBC,MLP,GAT \
  --folds 5 \
  --split-method scaffold \
  --cv-split-method scaffold \
  --external-test-metric MCC
```

### 5. Data Split Methods

This tool supports three data split methods, specified via `--split-method` parameter:

#### Scaffold Split (Default)
Split based on Bemis-Murcko molecular scaffolds to ensure test set contains different molecular scaffolds from training set. This is a best practice in drug discovery, avoiding data leakage and better assessing model generalization ability.

```bash
python qsar_modeling_pytorch.py \
  -i Data/test_data.csv \
  -l label \
  -s smiles \
  -m XGBC,MLP,GAT \
  --split-method scaffold
```

#### Stratified Split
Stratified split maintaining same class proportions in training and test sets. Suitable for imbalanced datasets.

```bash
python qsar_modeling_pytorch.py \
  -i Data/test_data.csv \
  -l label \
  -s smiles \
  -m XGBC,MLP,GAT \
  --split-method stratified
```

#### Random Split
Random split without considering class distribution or molecular scaffolds. Suitable for benchmarking.

```bash
python qsar_modeling_pytorch.py \
  -i Data/test_data.csv \
  -l label \
  -s smiles \
  -m XGBC,MLP,GAT \
  --split-method random
```

**Recommendation**: For drug discovery tasks, strongly recommend using `scaffold` split method as it more realistically evaluates model performance on unseen molecular scaffolds.

### 6. Multi-Seed Training for Robustness Testing

The tool supports training with multiple random seeds to assess model robustness and stability across different random initializations. This feature is particularly useful for:

- Evaluating model stability across different random initializations
- Obtaining more reliable performance estimates with confidence intervals
- Reducing the impact of random seed selection on model performance
- Providing mean ± std metrics for comprehensive performance assessment

**How to Use**:

```bash
# Using command line argument
python qsar_modeling_pytorch.py \
  -i Data/test_data.csv \
  -l label \
  -s smiles \
  -m XGBC,MLP,GAT \
  --seeds 42,123,2025 \
  --folds 5

# Or in YAML configuration
seeds: [42, 123, 2025]
```

**Output**:

When using multiple seeds, the results include:
- Individual seed results (stored internally)
- Aggregated results with mean ± std for each metric across all seeds
- Metrics are reported as `{metric}_mean` and `{metric}_std`
- In single split mode, models are saved with seed ID in filename (e.g., `ChemBERTa_single_seed42.pt`)

**Performance Considerations**:

- Training time scales linearly with the number of seeds
- For deep learning models (MLP, GAT, ChemBERTa), GPU memory is automatically cleared after each seed
- In K-Fold CV mode, models are not saved to avoid excessive storage usage
- Recommended seeds for robustness testing: 3-5 different seeds

**Example Output**:
```
Fold 1 - Training XGBC with 3 seeds: [42, 123, 2025]
Aggregating results for XGBC across 3 seeds
✓ XGBC aggregated results:
  Val AUC: 0.8542 ± 0.0123
  Val PR-AUC: 0.7891 ± 0.0156
  Val MCC: 0.6234 ± 0.0189
```

## Visualization & Post-Hoc Plots (Scripts/visualize.py)

Use `Scripts/visualize.py` (or `aggregate_split_predictions.py` in the repo root) to turn the per-run outputs into publication-ready visuals. The script assumes you already ran `qsar_modeling_pytorch.py` with `split_seeds`; it reads both `predictions/external_test_predictions_seed_*.csv` and `results/cv_predictions_fold_*.csv`, then saves the figures inside `<BASE_DIR>/figures/` (default `--output-dir`).

Key options:
* `--include-external` / `--include-cv`: enable ROC/PR curves for the external test set or the CV folds (or both).
* `--boxplot-stage`: choose `external`, `cv`, or `both` for the 3×2 metric boxplots (default metrics: MCC, F1, ACC, AUC, PR_AUC, EF5%).
* `--palette` (colorblind/deep), `--dpi` (e.g., 600), and `--metrics` allow you to tailor the styling and metrics.

Example:
```bash
python Scripts/visualize.py \
  --base-dir models_out/classification_20260325_141059 \
  --include-external \
  --include-cv \
  --boxplot-stage both
```

This produces `external_roc_pr.svg`, `cv_roc_pr.svg`, `external_metric_boxplots.svg`, and `cv_metric_boxplots.svg` inside `models_out/.../figures/`, all following the Times New Roman + Nature/Science style you asked for.

## Available Models

## Available Models

### Classification Tasks

#### Traditional Machine Learning Models

| Model Code | Model Name | Description | Dependencies |
|------------|------------|-------------|--------------|
| `LR` | Logistic Regression | Logistic Regression | scikit-learn |
| `RFC` | Random Forest Classifier | Random Forest Classifier | scikit-learn |
| `SVC` | Support Vector Classifier | Support Vector Machine | scikit-learn |
| `XGBC` | XGBoost Classifier | XGBoost Classifier | xgboost |
| `LGBMC` | LightGBM Classifier | LightGBM Classifier | lightgbm |
| `ETC` | Extra Trees Classifier | Extra Trees Classifier | scikit-learn |

#### Deep Learning Models

| Model Code | Model Name | Description | Dependencies |
|------------|------------|-------------|--------------|
| `MLP` | Multi-Layer Perceptron | Multi-Layer Perceptron | PyTorch |
| `GAT` | Graph Attention Network | Graph Attention Network | PyTorch Geometric, RDKit |
| `ChemBERTa` | ChemBERTa Transformer | Chemical Language Model | Transformers |

### Regression Tasks

#### Traditional Machine Learning Models

| Model Code | Model Name | Description | Dependencies |
|------------|------------|-------------|--------------|
| `Ridge` | Ridge Regression | Ridge Regression | scikit-learn |
| `RFR` | Random Forest Regressor | Random Forest Regressor | scikit-learn |
| `ETR` | Extra Trees Regressor | Extra Trees Regressor | scikit-learn |

#### Deep Learning Models

| Model Code | Model Name | Description | Dependencies |
|------------|------------|-------------|--------------|
| `MLP` | Multi-Layer Perceptron | Multi-Layer Perceptron | PyTorch |
| `GAT` | Graph Attention Network | Graph Attention Network | PyTorch Geometric, RDKit |
| `ChemBERTa` | ChemBERTa Transformer | Chemical Language Model | Transformers |

## Output Files

### Output Directory Structure

Each run creates a timestamped directory under the specified output directory (default: `models_out`):

```
models_out/
└── classification_20250319_143022/              # Run directory: {task}_{YYYYMMDD_HHMMSS}/
    ├── config.json                              # Configuration file
    ├── logs/                                    # Run logs
    │   └── qsar_run_20250319_143022.log        # Detailed run log
    ├── results/                                 # Performance metrics
    │   ├── summary_metrics.csv                  # Single split mode: overall metrics
    │   │                                        # K-Fold CV mode: not used
    │   ├── fold_results.csv                     # K-Fold CV mode: individual fold results
    │   ├── cv_summary.csv                       # K-Fold CV mode: cross-validation summary (mean ± std)
    │   └── external_test_results.csv            # K-Fold CV mode: external test set metrics
    ├── predictions/                             # Prediction results
    │   └── external_test_predictions.csv       # K-Fold CV mode: external test set predictions
    ├── models/                                  # Saved models
    │   ├── best_models/                        # Best models from external test set
    │   │   ├── best_traditional_XGBC.joblib
    │   │   ├── best_deep_learning_MLP.pt
    │   │   └── best_overall_LR.joblib
    │   └── seed_models/                        # Single split mode: models for each seed
    │       ├── MLP_single_seed42.pt
    │       ├── MLP_single_seed123.pt
    │       └── MLP_single_seed2025.pt
    └── feature_processors/                     # Feature processing components
        ├── feature_mask.npy                    # Global feature mask (K-Fold CV mode)
        ├── feature_names.json                  # Filtered feature names
        ├── variance_selector.joblib             # Variance threshold selector (if applicable)
        └── scaler.joblib                       # Standard scaler (if applicable)
```

### Single Split Mode (--folds 1)

After running, the following files will be generated in the output directory:

```
models_out/classification_20250319_143022/
├── config.json                              # Configuration file
├── logs/
│   └── qsar_run_20250319_143022.log         # Run log
├── results/
│   └── summary_metrics.csv                  # Performance metrics summary for all models
│                                            # Includes mean ± std across seeds when using multiple seeds
├── models/
│   ├── best_models/                        # Best models (if save_models=True)
│   │   ├── best_traditional_XGBC.joblib
│   │   └── best_deep_learning_MLP.pt
│   └── seed_models/                        # Models trained with different seeds
│       ├── MLP_single_seed42.pt
│       ├── MLP_single_seed123.pt
│       └── MLP_single_seed2025.pt
└── feature_processors/                     # Feature processing components
    ├── feature_mask.npy
    ├── feature_names.json
    ├── variance_selector.joblib
    └── scaler.joblib
```

### K-Fold CV Mode (--folds > 1)

When using two-stage validation workflow, the following files will be generated:

```
models_out/classification_20250319_143022/
├── config.json                              # Configuration file
├── logs/
│   └── qsar_run_20250319_143022.log         # Run log
├── results/
│   ├── fold_results.csv                     # Detailed results for each fold and seed (development set)
│   ├── cv_summary.csv                       # Cross-validation summary results (mean ± std across folds and seeds)
│   ├── external_test_results.csv            # External test set evaluation results (one row per model per seed)
│   └── external_test_summary.csv            # External test set summary (mean ± std across seeds)
├── predictions/
│   └── external_test_predictions.csv       # External test set prediction results (includes predictions for all models and seeds)
├── models/
│   └── best_models/                        # Best models from external test set
│       ├── best_traditional_XGBC.joblib
│       ├── best_deep_learning_MLP.pt
│       └── best_overall_LR.joblib
└── feature_processors/                     # Feature processing components
    ├── feature_mask.npy                    # Global feature mask
    └── feature_names.json                  # Filtered feature names
```

**Note**: 
- Model saving in `seed_models/` is only available in single split mode (`--folds 1`)
- `best_models/` are saved in both modes based on external test set performance
- Feature processors are saved to enable consistent preprocessing on new data

## Performance Metrics

### Classification Tasks
- **AUC**: Area Under ROC Curve
- **PR-AUC**: Area Under Precision-Recall Curve
- **ACC**: Accuracy
- **F1**: F1 Score
- **MCC**: Matthews Correlation Coefficient
- **EF1%**: Early Enrichment Factor (top 1%)
- **Hit Rate**: Hit rate at specified percentile

### Regression Tasks
- **R²**: Coefficient of Determination
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error

## Command Line Parameters

| Parameter | Short | Description | Default |
|-----------|-------|-------------|---------|
| `--config` | `-c` | YAML configuration file path | - |
| `--input` | `-i` | Input CSV file path | - |
| `--label` | `-l` | Label column name | `label` |
| `--pic50` | `-p` | pIC50 column name | `pic50` |
| `--smiles` | `-s` | SMILES column name | `smiles` |
| `--id` | `-d` | ID column name | `id` |
| `--task` | `-t` | Task type (classification/regression) | `classification` |
| `--models` | `-m` | List of models to train (comma-separated) | All available models |
| `--output` | `-o` | Output directory | `models_out` |
| `--folds` | - | Number of CV folds (set to 1 for single split) | 5 |
| `--seed` | - | Random seed | 42 |
| `--seeds` | - | Comma-separated list of random seeds for multi-seed training (e.g., 42,123,2025) | [42] |
| `--test-size` | - | Test set proportion | 0.2 |
| `--epochs` | - | Maximum epochs for deep learning | 100 |
| `--batch-size` | - | Batch size | 32 |
| `--lr` | - | Learning rate | 0.001 |
| `--ef` | - | Early enrichment factor percentile | 1.0 |
| `--variance-threshold` | - | Feature variance filtering threshold | 0.01 |
| `--min-frequency` | - | Minimum frequency for binary fingerprints | 0.05 |
| `--split-method` | - | Data split method (scaffold/stratified/random) | scaffold |
| `--cv-split-method` | - | CV split method (scaffold/random) | scaffold |
| `--external-test-metric` | - | External test set evaluation metric | MCC (classification) / R2 (regression) |
| `--no-auto-fp` | - | Disable automatic fingerprint generation | False |
| `--fp-types` | - | Fingerprint types (comma-separated) | morgan |
| `--no-shap` | - | Disable SHAP analysis | False |
| `--log-level` | - | Log level (DEBUG/INFO/WARNING/ERROR) | INFO |

## Notes

1. **Fingerprint Features**: If input data does not contain fingerprint features and selected models require fingerprints (LR, RFC, XGBC, MLP, etc.), the tool will automatically generate fingerprint features. You can specify fingerprint types via `--fp-types` parameter (morgan, maccs, rdkit, atompair, torsion). Data with generated fingerprints will be automatically saved as `{input_file}_with_fingerprints.csv`.

2. **SMILES Requirements**: Default scaffold split method requires SMILES column. Please ensure input data contains valid SMILES strings.

3. **Data Split Method Selection**:
   - **Scaffold Split** (default): Recommended for drug discovery tasks, avoids data leakage and realistically evaluates model generalization ability
   - **Stratified Split**: Suitable for imbalanced datasets
   - **Random Split**: Suitable for benchmarking, but may lead to overly optimistic results

4. **CV Split Method Selection** (K-Fold CV Mode):
   - `--split-method` controls the split method for Stage 1 (external test set)
   - `--cv-split-method` controls the split method for Stage 2 (K-Fold CV)
   - **Scaffold CV** (default): Uses improved greedy strategy, avoids empty folds, balanced sample distribution, recommended for drug discovery
   - **Random CV**: Uses standard random stratified split, suitable for small datasets with few scaffolds
   - For datasets with fewer scaffolds than folds, it's recommended to use Random CV or reduce the number of folds

5. **Two-Stage Validation**: When using `--folds > 1`, the two-stage validation workflow is automatically executed. This is the recommended approach as it provides more reliable model performance evaluation.

6. **GPU Support**: Deep learning models (MLP, GAT, ChemBERTa) will automatically use GPU if available.

7. **Feature Processing**: Variance filtering and standardization are performed independently within each fold, using training set statistics to process validation set to avoid data leakage. For binary fingerprints, you can use `min_frequency` parameter for frequency-based filtering to remove feature bits with too low frequency.

8. **SMILES Validation**: The tool automatically validates SMILES string validity. If invalid SMILES are found, warnings will be logged and invalid entry IDs will be displayed in the log. It's recommended to clean data before running.

9. **SHAP Analysis**: SHAP analysis is currently only available in single split mode (`--folds 1`). SHAP analysis can be time-consuming, especially for large datasets. You can adjust sample size via `shap_sample_size` parameter.

10. **Memory Usage**: For large datasets, it's recommended to adjust test set proportion using `--test-size` parameter or adjust batch size using `--batch-size` parameter.

11. **Dependency Installation**: Some models require additional dependencies (such as XGBoost, LightGBM, PyTorch Geometric, RDKit), please ensure they are properly installed. The tool automatically checks version compatibility of critical libraries.

12. **Model Saving**: Best model saving feature is available in single split mode (`--folds 1`), including both traditional and deep learning models. In K-Fold CV mode, feature processors (variance selector and standardizer) are saved for convenient transformation of new data.

13. **Multi-Seed Training**: Use multiple seeds for robustness testing, especially for deep learning models:
    - Recommended: 3-5 different seeds (e.g., 42, 123, 2025)
    - Provides mean ± std metrics for confidence intervals
    - Training time scales linearly with number of seeds
    - Models saved in single split mode include seed ID in filename
    - In K-Fold CV mode, models are not saved to manage storage
    - Automatically clears GPU memory after each seed to prevent OOM

## Example Workflows

### Basic Workflow (Single Split Mode)

```bash
# 1. Generate fingerprint features (optional, tool will auto-generate)
python generate_fingerprints.py -i Data/compound_data.csv -s smiles -o Data/with_fingerprints.csv

# 2. Data cleaning (optional)
python Data_cleaning.py -i Data/with_fingerprints.csv -o Data/clean_data.csv

# 3. Train models (single split mode)
python qsar_modeling_pytorch.py -i Data/clean_data.csv -l label -s smiles -m XGBC,MLP,GAT --folds 1

# 4. View results
cat models_out/classification_*/results/summary_metrics.csv
```

### Recommended Workflow (Two-Stage Validation)

```bash
# 1. Prepare data (including id, smiles, label/pic50 columns)
# Tool will auto-generate fingerprint features

# 2. Train models (K-Fold CV + external test set evaluation)
python qsar_modeling_pytorch.py \
  -i Data/test_data.csv \
  -l label \
  -s smiles \
  -m XGBC,MLP,GAT \
  --folds 5 \
  --split-method scaffold \
  --cv-split-method scaffold \
  --external-test-metric MCC

# 3. View cross-validation results
cat models_out/classification_*/results/cv_summary.csv

# 4. View external test set results
cat models_out/classification_*/results/external_test_results.csv

# 5. View external test set predictions
cat models_out/classification_*/predictions/external_test_predictions.csv
```

### Robustness Testing Workflow (Multi-Seed)

```bash
# 1. Prepare data (including id, smiles, label/pic50 columns)
# Tool will auto-generate fingerprint features

# 2. Train models with multiple seeds for robustness testing
python qsar_modeling_pytorch.py \
  -i Data/test_data.csv \
  -l label \
  -s smiles \
  -m XGBC,MLP,GAT,ChemBERTa \
  --seeds 42,123,2025 \
  --folds 5 \
  --split-method scaffold \
  --cv-split-method scaffold \
  --external-test-metric MCC

# 3. View cross-validation results (now includes mean ± std across seeds)
cat models_out/classification_*/results/cv_summary.csv

# 4. View external test set results
cat models_out/classification_*/results/external_test_results.csv

# 5. View external test set predictions
cat models_out/classification_*/predictions/external_test_predictions.csv

# 6. In single split mode, saved models include seed ID:
#    ls models_out/classification_*/models/seed_models/
#    XGBC_single_seed42.joblib    XGBC_single_seed123.joblib    XGBC_single_seed2025.joblib
#    MLP_single_seed42.pt         MLP_single_seed123.pt         MLP_single_seed2025.pt
```

### Using YAML Configuration File

```bash
# 1. Create configuration file config.yaml
# See YAML configuration file example above

# 2. Run with configuration file
python qsar_modeling_pytorch.py -c config.yaml

# 3. View log
tail -f models_out/classification_*/logs/qsar_run_*.log
```

## Usage Tips

### 1. Choosing the Right Validation Mode

- **Beginners/Quick Testing**: Use single split mode (`--folds 1`) for quick workflow verification
- **Research/Publications**: Use two-stage validation mode (`--folds 5` or higher) for reliable performance evaluation
- **Large-scale Data**: Can use `--folds 10` for more comprehensive cross-validation

### 6. Choosing the Right Seed Strategy

- **Single Seed** (`--seed 42`): Fast development and testing, suitable for:
  - Initial model development and debugging
  - Quick performance evaluation
  - When computational resources are limited
  
- **Multiple Seeds** (`--seeds 42,123,2025`): Robustness testing, recommended for:
  - Final model evaluation for publications
  - Assessing model stability and reproducibility
  - Deep learning models (sensitive to random initialization)
  - When reliable confidence intervals are needed
  - Research publications requiring rigorous statistical evaluation

### 7. Choosing the Right Data Split Method

- **Drug Discovery**: Use `scaffold` split (default) to avoid data leakage
- **Class Imbalance**: Use `stratified` split to maintain class proportions
- **Benchmarking**: Use `random` split for comparison with other studies

### 8. Choosing the Right CV Split Method (K-Fold CV Mode)

- **Scaffold CV** (default): Recommended for drug discovery, ensures scaffold non-overlap, uses greedy strategy to avoid empty folds
- **Random CV**: Suitable for small datasets with few scaffolds, or for quick benchmarking
- **Note**: If using Scaffold CV encounters errors indicating insufficient scaffolds, you can:
  - Reduce `--folds` number (e.g., from 5 to 3)
  - Or use `--cv-split-method random`

### 9. Choosing the Right Evaluation Metrics

- **Classification Tasks**:
  - Balanced datasets: Use AUC, ACC
  - Imbalanced datasets: Use PR-AUC, MCC, EF1%
  - Early screening: Use EF1%, Hit Rate
- **Regression Tasks**:
  - Use R2 to evaluate overall fit
  - Use RMSE/MAE to evaluate prediction error

### 10. Choosing the Right Models

- **Traditional Models** (fast, interpretable):
  - LR: Linear relationships, strong interpretability
  - RFC/XGBC: Non-linear relationships, excellent performance
  - SVC: Small datasets, high-dimensional features
- **Deep Learning Models** (stronger performance, require more data):
  - MLP: Representation learning, suitable for medium-scale data
  - GAT: Utilizes molecular graph structure, suitable for structure-activity relationships
  - ChemBERTa: Utilizes pre-trained knowledge, suitable for large-scale data

### 11. Tuning Recommendations

- **Small Datasets** (< 1000 samples): Use traditional models, reduce folds count (3-5)
- **Medium Datasets** (1000-10000 samples): Use traditional models + MLP, folds=5
- **Large Datasets** (> 10000 samples): Use all models including GAT and ChemBERTa, folds=5-10
- **Feature Engineering**: Try different fingerprint types (morgan, maccs, rdkit, etc.)
- **Hyperparameter Tuning**: For deep learning models, you can adjust `--epochs`, `--batch-size`, `--lr`

## Citation

If you use this code, please cite:

```bibtex
@article{???,
  title={???},
  author={???},
  year={???}
}
```

## License

This code is for academic and research use only. Modification, redistribution, or commercial use is not permitted without permission.

## Contact

For questions or suggestions, please contact Toby Lo.
