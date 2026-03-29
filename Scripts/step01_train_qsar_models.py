"""
QSAR Batch Modeling with PyTorch Deep Learning
================================================
Enhanced QSAR script with PyTorch-based deep learning models,
including MLP, GAT (Graph Attention Network), and ChemBERTa Transformer.

Key Features:
- Traditional models: LR, RFC, SVC, XGBC, LGBMC, ETC
- Deep learning models: MLP, GAT, ChemBERTa
- Enhanced metrics: MCC, PR-AUC, Early Enrichment (EF1%)
- YAML configuration file support
- Input format: id, smiles, label (binary), pic50 (continuous)
- Separate best model tracking for traditional and deep learning models
- SHAP analysis for model interpretability
- Multi-modal fusion interface (for future integration)
- K-fold Cross-Validation support for robust model evaluation

Data Preprocessing Assumptions
===============================
This script assumes that the input data has been preprocessed by a separate data cleaning script.
The preprocessing script should perform the following operations:

1. SMILES Standardization:
   - Use RDKit to standardize SMILES strings
   - Canonicalize SMILES representation
   - Remove stereochemistry if not needed
   - Convert to lowercase for consistency

2. Data Cleaning:
   - Remove duplicate molecules (based on canonical SMILES)
   - Remove molecules with invalid SMILES
   - Handle missing values
   - Remove outliers if necessary

3. Data Validation:
   - Ensure all SMILES are valid (RDKit can parse them)
   - Ensure labels are in correct format (0/1 for classification)
   - Ensure pIC50 values are numeric and in reasonable range for regression

Example preprocessing workflow:
- Input: Raw compound data from databases or experimental results
- Process: Standardize SMILES → Remove duplicates → Validate → Clean
- Output: Cleaned CSV file with columns: id, smiles, label, pic50

Recommended preprocessing tools:
- RDKit for SMILES validation and standardization
- Pandas for data cleaning
- Custom scripts for domain-specific filtering

Input CSV Format:
- id: Molecular unique identifier (e.g., ChEMBL ID)
- smiles: SMILES string representation (STANDARDIZED)
- label: Binary classification label (0 or 1)
- pic50: Continuous pIC50 value for regression tasks

Note: If the input data contains duplicate SMILES or invalid molecules,
         the preprocessing script should handle them before running this QSAR script.
"""

import os
import copy
import sys
import json
import math
import shutil
import argparse
import inspect
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, r2_score

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')

# --------- Headless matplotlib setup ---------
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib
matplotlib.use("Agg")

matplotlib.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.5,
    "lines.linewidth": 2,
})

import matplotlib.pyplot as plt

# --------- GPU setup ---------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Check GPU availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# --------- Version Checks ---------
def check_library_versions(logger: logging.Logger = None):
    """Check versions of critical libraries and log warnings if needed"""
    issues = []
    
    # Check PyTorch version
    try:
        import torch
        torch_version = torch.__version__
        torch_major = int(torch_version.split('.')[0])
        if torch_major < 2:
            issues.append(f"PyTorch version {torch_version} is outdated. Recommended: >= 2.0.0")
        elif logger:
            logger.info(f"PyTorch version: {torch_version}")
    except ImportError:
        issues.append("PyTorch is not installed")
    
    # Check PyTorch Geometric version
    try:
        import torch_geometric
        tg_version = torch_geometric.__version__
        if logger:
            logger.info(f"PyTorch Geometric version: {tg_version}")
    except ImportError:
        pass  # GAT is optional
    
    # Check Transformers version
    try:
        import transformers
        tf_version = transformers.__version__
        if logger:
            logger.info(f"Transformers version: {tf_version}")
    except ImportError:
        pass  # ChemBERTa is optional
    
    # Check RDKit version
    try:
        import rdkit
        from rdkit import Chem
        rdkit_version = Chem.rdBase.rdkitVersion
        if logger:
            logger.info(f"RDKit version: {rdkit_version}")
    except ImportError:
        pass  # RDKit is optional for some models
    
    # Log issues
    if issues and logger:
        logger.warning("=" * 60)
        logger.warning("Library Version Warnings:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        logger.warning("=" * 60)
    
    return issues

# --------- Random Seed Management ---------
def set_all_seeds(seed: int, logger: logging.Logger = None):
    """
    Set random seeds for all libraries to ensure reproducibility
    
    Args:
        seed: Random seed value
        logger: Logger instance for logging messages
    """
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if logger:
        logger.debug(f"All random seeds set to {seed}")

# --------- Configuration Dataclass ---------
@dataclass
class QSARConfig:
    """Configuration container for QSAR modeling pipeline"""
    input_path: str = ""
    label_column: str = "label"
    pic50_column: str = "pic50"
    smiles_column: str = "smiles"
    id_column: str = "id"
    task: str = "classification"
    test_size: float = 0.2
    folds: int = 5
    # Whether to run Stage 2 (K-Fold CV) on the Development Set.
    # Even when disabled, Stage 1 (Dev/External Test split) and Stage 3 (final external evaluation)
    # still run, so the Dev/External split remains leakage-safe.
    run_cv_stage2: bool = True
    save_cv_details: bool = False  # Save per-fold/seed artifacts for SHAP and diagnostics during CV

    # Whether to use Stage 2 CV results for hyperparameter tuning.
    # Tuning is performed only on the Development Set and never touches the External Test Set.
    tune: bool = False
    # Tuning mode for sklearn grid-based models: 'grid' enumerates combinations, 'random' samples.
    tune_mode: str = "grid"
    # Metric used to rank hyperparameter candidates in Stage 2 CV.
    # If empty/None, falls back to `external_test_metric` (e.g., MCC or R2).
    cv_tune_metric: Optional[str] = None
    seed: int = 42
    # Multiple seeds for robustness testing / repeated training.
    seeds: List[int] = field(default_factory=lambda: [42])
    tune_iter: int = 25
    max_epochs: int = 100
    selected_models: List[str] = field(default_factory=list)
    output_dir: str = ""
    variance_threshold: float = 0.01
    min_frequency: float = 0.05  # Minimum frequency for fingerprint filtering (5%)
    shap_max_display: int = 20
    shap_sample_size: int = 500
    run_shap: bool = True
    create_ensemble: bool = False
    ensemble_n_top: int = 3
    n_jobs: int = -1
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    config_file: str = ""
    split_seeds: List[int] = field(default_factory=lambda: [42])
    save_train_features: bool = False  # Whether to persist Development-set features/labels for downstream AD scripts
    
    # Early enrichment settings
    ef_percentile: float = 1.0  # EF1%
    
    # NOTE: Multi-modal fusion features are not currently implemented
    # The following fields are kept for potential future implementation
    # enable_multimodal: bool = False
    # multimodal_weights: Dict[str, float] = field(default_factory=dict)
    
    # Data split settings
    split_method: str = "scaffold"  # Options: scaffold, stratified, random
    cv_split_method: str = "scaffold"  # CV split method: scaffold, random
    
    # Automatic fingerprint generation settings
    auto_generate_fingerprints: bool = True  # Automatically generate fingerprints if needed
    fingerprint_types: List[str] = field(default_factory=lambda: ["morgan"])  # Types to generate
    
    # Metric for selecting best model on external test set 
    external_test_metric: str = "MCC"  # (classification: AUC, PR_AUC, ACC, F1, MCC; regression: R2, RMSE, MAE)
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> 'QSARConfig':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'QSARConfig':
        """Load configuration from YAML file"""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML configuration files. Install with: pip install pyyaml")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

# --------- Applicability Domain (AD) Calculation ---------
def calculate_ad(X_train: np.ndarray, X_test: np.ndarray, method: str = "leverage", 
                 threshold: float = 3.0, k: int = 5) -> np.ndarray:
    """
    Calculate Applicability Domain (AD) for test samples.
    
    Applicability Domain indicates whether a prediction is reliable based on
    the similarity to training data. Samples outside AD may have unreliable predictions.
    
    Args:
        X_train: Training features (n_train, n_features)
        X_test: Test features (n_test, n_features)
        method: AD calculation method - "leverage", "distance", or "density"
        threshold: Threshold for AD (default: 3.0 for leverage/distance, varies for density)
        k: Number of neighbors for density method (default: 5)
    
    Returns:
        in_ad: Boolean array (n_test,) where True indicates sample is within AD
    
    Methods:
        - leverage: Based on leverage values from hat matrix (h = diag(X(X'X)^-1X'))
        - distance: Based on distance to nearest training samples (Euclidean distance)
        - density: Based on local density using k-nearest neighbors
    
    References:
        - leverage: Wold, S. (1995). PLS in chemical practice. Chemometrics.
        - distance: Tropsha, A. (2010). Best practices in QSAR model development.
        - density: Sheridan, R.P. (2014). Using Gaussian DBSCAN clustering.
    """
    n_train, n_features = X_train.shape
    n_test = X_test.shape[0]
    
    if method == "leverage":
        # Leverage method (Williams plot)
        # h_i = x_i(X'X)^-1x_i'
        try:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Calculate leverage: h = X(X'X)^-1X'
            # Use pseudo-inverse for numerical stability
            X_pinv = np.linalg.pinv(X_train_scaled)
            H = X_train_scaled @ X_pinv  # Hat matrix
            h_train = np.diag(H)
            h_star = 3 * (n_train + 1) / n_train  # Warning leverage (3x average)
            
            # Calculate leverage for test samples
            h_test = np.sum((X_test_scaled @ X_pinv) ** 2, axis=1)
            
            # Sample is within AD if leverage < threshold * h_star
            in_ad = h_test < threshold * h_star
            
        except Exception as e:
            print(f"Warning: Leverage calculation failed ({e}). Using distance method as fallback.")
            in_ad = calculate_ad(X_train, X_test, method="distance", threshold=threshold)
    
    elif method == "distance":
        # Distance-based method (distance to nearest training sample)
        from scipy.spatial.distance import cdist
        
        # Calculate distance matrix between test and training samples
        dist_matrix = cdist(X_test, X_train, metric='euclidean')
        
        # Find distance to nearest training sample
        min_distances = np.min(dist_matrix, axis=1)
        
        # Calculate threshold as threshold * mean distance between training samples
        dist_train = cdist(X_train, X_train, metric='euclidean')
        # Set diagonal to infinity to exclude self-distance
        np.fill_diagonal(dist_train, np.inf)
        mean_dist_train = np.mean(np.min(dist_train, axis=1))
        
        ad_threshold = threshold * mean_dist_train
        
        # Sample is within AD if distance to nearest training sample < threshold
        in_ad = min_distances < ad_threshold
    
    elif method == "density":
        # Density-based method using k-nearest neighbors
        from scipy.spatial.distance import cdist
        
        # Calculate distance matrix
        dist_matrix = cdist(X_train, X_test, metric='euclidean')
        
        # For each test sample, find distance to k-th nearest training sample
        sorted_distances = np.sort(dist_matrix, axis=0)
        k_distances = sorted_distances[k, :] if k < n_train else sorted_distances[-1, :]
        
        # Calculate density for training samples (using k nearest neighbors)
        dist_train = cdist(X_train, X_train, metric='euclidean')
        np.fill_diagonal(dist_train, np.inf)
        sorted_train_distances = np.sort(dist_train, axis=0)
        k_distances_train = sorted_train_distances[k, :] if k < n_train else sorted_train_distances[-1, :]
        
        # Calculate density threshold
        mean_k_dist_train = np.mean(k_distances_train)
        ad_threshold = threshold * mean_k_dist_train
        
        # Sample is within AD if its local density (inverse of k-distance) is sufficient
        in_ad = k_distances < ad_threshold
    
    else:
        raise ValueError(f"Unknown AD method: {method}. Choose from 'leverage', 'distance', or 'density'.")
    
    return in_ad

# --------- Enhanced Metrics ---------
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, 
                     task: str = "classification", ef_percentile: float = 1.0) -> Dict[str, float]:
    """
    Calculate comprehensive metrics including MCC, PR-AUC, and Early Enrichment
    
    Args:
        y_true: True labels
        y_pred: Predicted labels (binary)
        y_proba: Predicted probabilities
        task: "classification" or "regression"
        ef_percentile: Percentile for Early Enrichment Factor (e.g., 1.0 for EF1%)
    
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, f1_score, average_precision_score,
        precision_recall_curve, confusion_matrix, matthews_corrcoef
    )
    
    metrics = {}
    import warnings
    if y_true is None or y_pred is None or y_proba is None:
        warnings.warn("calculate_metrics: received None inputs; returning NaN metrics")
        metrics['AUC'] = float('nan')
        metrics['PR_AUC'] = float('nan')
        metrics['ACC'] = float('nan')
        metrics['F1'] = float('nan')
        metrics['MCC'] = float('nan')
        metrics['R2'] = float('nan')
        metrics['RMSE'] = float('nan')
        metrics['MAE'] = float('nan')
        return metrics

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)
    
    if task == "classification":
        # Guard against empty inputs or length mismatches which can happen when
        # ChemBERTa filters out empty SMILES samples.
        if len(y_true) == 0 or len(y_pred) == 0 or len(y_proba) == 0:
            warnings.warn("calculate_metrics: empty inputs for classification; returning NaN metrics")
            metrics['AUC'] = float('nan')
            metrics['PR_AUC'] = float('nan')
            metrics['ACC'] = float('nan')
            metrics['F1'] = float('nan')
            metrics['MCC'] = float('nan')
            metrics[f'EF{ef_percentile:g}%'] = float('nan')
            metrics[f'Hit_Rate_{ef_percentile:g}%'] = float('nan')
            metrics[f'EF{ef_percentile:g}%_actual_percentile'] = float('nan')
            metrics['Precision_at_0.5'] = float('nan')
            metrics['Recall_at_0.5'] = float('nan')
            return metrics

        if not (len(y_true) == len(y_pred) == len(y_proba)):
            warnings.warn(
                f"calculate_metrics: length mismatch for classification "
                f"(len(y_true)={len(y_true)}, len(y_pred)={len(y_pred)}, len(y_proba)={len(y_proba)}); "
                "returning NaN metrics"
            )
            metrics['AUC'] = float('nan')
            metrics['PR_AUC'] = float('nan')
            metrics['ACC'] = float('nan')
            metrics['F1'] = float('nan')
            metrics['MCC'] = float('nan')
            metrics[f'EF{ef_percentile:g}%'] = float('nan')
            metrics[f'Hit_Rate_{ef_percentile:g}%'] = float('nan')
            metrics[f'EF{ef_percentile:g}%_actual_percentile'] = float('nan')
            metrics['Precision_at_0.5'] = float('nan')
            metrics['Recall_at_0.5'] = float('nan')
            return metrics

        # Standard metrics
        try:
            metrics['AUC'] = float(roc_auc_score(y_true, y_proba))
        except Exception as e:
            warnings.warn(f"calculate_metrics: roc_auc_score failed ({e}); setting AUC=NaN")
            metrics['AUC'] = float('nan')
        try:
            metrics['PR_AUC'] = float(average_precision_score(y_true, y_proba))
        except Exception as e:
            warnings.warn(f"calculate_metrics: average_precision_score failed ({e}); setting PR_AUC=NaN")
            metrics['PR_AUC'] = float('nan')
        try:
            metrics['ACC'] = float(accuracy_score(y_true, y_pred))
        except Exception as e:
            warnings.warn(f"calculate_metrics: accuracy_score failed ({e}); setting ACC=NaN")
            metrics['ACC'] = float('nan')
        try:
            metrics['F1'] = float(f1_score(y_true, y_pred))
        except Exception as e:
            warnings.warn(f"calculate_metrics: f1_score failed ({e}); setting F1=NaN")
            metrics['F1'] = float('nan')
        try:
            metrics['MCC'] = float(matthews_corrcoef(y_true, y_pred))
        except Exception as e:
            warnings.warn(f"calculate_metrics: matthews_corrcoef failed ({e}); setting MCC=NaN")
            metrics['MCC'] = float('nan')
        
        # Early Enrichment Factor (EF%)
        n_total = len(y_true)
        # Use ceil to ensure actual percentage >= target percentage
        # This ensures we don't underestimate EF by selecting too few samples
        n_percentile = int(np.ceil(n_total * ef_percentile / 100.0))
        # Ensure at least 1 sample is selected
        n_percentile = max(1, n_percentile)
        
        # Calculate actual percentage used (may differ from target for small datasets)
        actual_percentile = (n_percentile / n_total) * 100.0
        
        if n_percentile > 0:
            # Sort by predicted probability (descending)
            sorted_indices = np.argsort(y_proba)[::-1]
            top_indices = sorted_indices[:n_percentile]
            
            # Count actives in top percentile
            n_actives_top = np.sum(y_true[top_indices])
            
            # Total actives
            n_actives_total = np.sum(y_true)
            
            if n_actives_total > 0:
                # Expected actives in random selection
                n_expected = n_actives_total * (n_percentile / n_total)
                
                # EF% = (actives in top%) / (expected actives)
                metrics[f'EF{ef_percentile:g}%'] = float(n_actives_top / n_expected if n_expected > 0 else 0.0)
                
                # Hit rate in top percentile
                metrics[f'Hit_Rate_{ef_percentile:g}%'] = float(n_actives_top / n_percentile)
                
                # Store actual percentile used for transparency
                metrics[f'EF{ef_percentile:g}%_actual_percentile'] = float(actual_percentile)
            else:
                metrics[f'EF{ef_percentile:g}%'] = 0.0
                metrics[f'Hit_Rate_{ef_percentile:g}%'] = 0.0
                metrics[f'EF{ef_percentile:g}%_actual_percentile'] = float(actual_percentile)
        else:
            metrics[f'EF{ef_percentile:g}%'] = 0.0
            metrics[f'Hit_Rate_{ef_percentile:g}%'] = 0.0
            metrics[f'EF{ef_percentile:g}%_actual_percentile'] = float(actual_percentile)
        
        # Additional precision/recall at different thresholds
        try:
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
            idx = int(np.argmin(np.abs(thresholds - 0.5))) if len(thresholds) > 0 else 0
            metrics['Precision_at_0.5'] = float(precisions[idx] if len(precisions) > idx else precisions[0])
            metrics['Recall_at_0.5'] = float(recalls[idx] if len(recalls) > idx else recalls[0])
        except Exception as e:
            warnings.warn(f"calculate_metrics: precision_recall_curve failed ({e}); setting Precision/Recall at 0.5=NaN")
            metrics['Precision_at_0.5'] = float('nan')
            metrics['Recall_at_0.5'] = float('nan')
    
    else:
        # Regression metrics
        if len(y_true) == 0 or len(y_proba) == 0 or len(y_true) != len(y_proba):
            warnings.warn(
                "calculate_metrics: regression received empty or length-mismatched inputs; returning NaN metrics"
            )
            metrics['R2'] = float('nan')
            metrics['RMSE'] = float('nan')
            metrics['MAE'] = float('nan')
            return metrics
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        metrics['R2'] = float(r2_score(y_true, y_proba))
        metrics['RMSE'] = float(math.sqrt(mean_squared_error(y_true, y_proba)))
        metrics['MAE'] = float(mean_absolute_error(y_true, y_proba))
    
    return metrics

# --------- Data Processing ---------
FP_PREFIXES = ("morgan_", "maccs_", "rdkit_", "atompair_", "torsion_")

def read_table(path: Path) -> pd.DataFrame:
    """Read CSV or Parquet file"""
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported format: {ext}")

def select_fp_columns(df: pd.DataFrame) -> List[str]:
    """Select numeric fingerprint columns"""
    cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and str(c).startswith(FP_PREFIXES):
            cols.append(c)
    return cols

def filter_low_variance_features(df: pd.DataFrame, fp_cols: List[str], 
                               threshold: float = 0.01, logger: Optional[logging.Logger] = None) -> List[str]:
    from sklearn.feature_selection import VarianceThreshold
    
    X = df[fp_cols].values
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)
    kept_mask = selector.get_support() # Boolen mask
    kept_cols = [c for c, keep in zip(fp_cols, kept_mask) if keep]
    
    if logger:
        logger.info(f"Feature filtering: {len(fp_cols)} → {len(kept_cols)} features (threshold={threshold})")
    
    return kept_cols

def holdout_split(task: str, X, y, test_size=0.2, seed=42):
    """Stratified split for classification, random for regression"""
    from sklearn.model_selection import train_test_split
    if task == "classification":
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    else:
        return train_test_split(X, y, test_size=test_size, random_state=seed)

def get_scaffold(smiles: str) -> str:
    """
    Generate Bemis-Murcko scaffold from SMILES string
    
    Args:
        smiles: SMILES string
        
    Returns:
        Scaffold SMILES string, or empty string if conversion fails
    """
    from rdkit import Chem
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        from rdkit.Chem.Scaffolds import MurckoScaffold
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None:
            return ""
        return Chem.MolToSmiles(scaffold)
    except Exception as e:
        print(f"Error generating scaffold for {smiles}: {e}")
        return ""

class ScaffoldKFold:
    """
    K-Fold cross-validator based on molecular scaffolds.
    
    Splits data into k folds such that scaffolds are distributed across folds.
    This ensures that molecules with the same scaffold are always in the same fold,
    providing a more robust evaluation of model generalizability.
    
    Parameters
    ----------
    n_splits : int
        Number of folds. Must be at least 2.
    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches.
    random_state : int, default=None
        Random seed for shuffling.
    """
    
    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: int = None):
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, smiles_list: List[str], y: np.ndarray = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test sets using greedy allocation.
        
        This method uses a greedy strategy to balance sample sizes across folds,
        ensuring each fold gets a reasonable number of samples and avoiding empty folds.
        
        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings
        y : np.ndarray, optional
            Labels (not used for scaffold split, but kept for API compatibility)
        
        Yields
        ------
        train_indices : np.ndarray
            The training set indices for that split.
        test_indices : np.ndarray
            The testing set indices for that split.
        
        Raises
        ------
        ValueError
            If the number of unique scaffolds is less than n_splits.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        scaffolds = [get_scaffold(s) for s in smiles_list]
        
        # Separate scaffold-less molecules (empty strings) from scaffold-based molecules
        scaffold_less_indices = [i for i, s in enumerate(scaffolds) if s == ""]
        
        # Group indices by scaffold (excluding scaffold-less molecules)
        scaffold_dict = {}
        for idx, scaffold in enumerate(scaffolds):
            if scaffold == "":
                continue
            if scaffold not in scaffold_dict:
                scaffold_dict[scaffold] = []
            scaffold_dict[scaffold].append(idx)
        
        # Get list of scaffolds and sort by size (descending)
        scaffold_list = list(scaffold_dict.keys())
        scaffold_sizes = {scaffold: len(indices) for scaffold, indices in scaffold_dict.items()}
        sorted_scaffolds = sorted(scaffold_list, key=lambda x: scaffold_sizes[x], reverse=True)
        
        # Check if we have enough scaffolds
        if len(sorted_scaffolds) < self.n_splits:
            raise ValueError(
                f"Number of unique scaffolds ({len(sorted_scaffolds)}) is less than "
                f"n_splits ({self.n_splits}). This will result in empty folds. "
                f"Please reduce n_splits or use a different split method."
            )
        
        # Warn if there are many scaffold-less molecules
        n_scaffold_less = len(scaffold_less_indices)
        n_total = len(smiles_list)
        if n_scaffold_less > 0:
            if n_scaffold_less / n_total > 0.1:  # If more than 10% of molecules have no scaffold
                print(f"Warning: {n_scaffold_less}/{n_total} molecules ({n_scaffold_less/n_total*100:.1f}%) have no scaffold. "
                      f"These will be randomly distributed across folds.")
            else:
                print(f"Note: {n_scaffold_less}/{n_total} molecules have no scaffold. "
                      f"These will be randomly distributed across folds.")
        
        # Shuffle scaffolds of similar sizes if shuffle=True
        if self.shuffle:
            # Group scaffolds by size
            size_groups = {}
            for scaffold in sorted_scaffolds:
                size = scaffold_sizes[scaffold]
                if size not in size_groups:
                    size_groups[size] = []
                size_groups[size].append(scaffold)
            
            # Shuffle within each size group
            np.random.seed(self.random_state)
            for size in size_groups:
                np.random.shuffle(size_groups[size])
            
            # Reconstruct sorted list with shuffled groups
            sorted_scaffolds = []
            for size in sorted(size_groups.keys(), reverse=True):
                sorted_scaffolds.extend(size_groups[size])
            
            # Also shuffle scaffold-less indices
            np.random.shuffle(scaffold_less_indices)
        
        # Initialize fold assignments
        fold_scaffolds = [[] for _ in range(self.n_splits)]
        
        # Stratified round-robin allocation:
        # Divide scaffolds into size tiers (large, medium, small) and distribute each tier evenly
        # This ensures each fold gets representatives from all size tiers, avoiding structural bias
        if len(sorted_scaffolds) >= 3:
            # Determine size tiers (large: top 20%, medium: middle 60%, small: bottom 20%)
            n_large = max(1, int(len(sorted_scaffolds) * 0.2))
            n_small = max(1, int(len(sorted_scaffolds) * 0.2))
            n_medium = len(sorted_scaffolds) - n_large - n_small
            
            large_scaffolds = sorted_scaffolds[:n_large]
            medium_scaffolds = sorted_scaffolds[n_large:n_large + n_medium]
            small_scaffolds = sorted_scaffolds[n_large + n_medium:]
            
            # Distribute each tier using round-robin
            for tier_scaffolds in [large_scaffolds, medium_scaffolds, small_scaffolds]:
                fold_idx = 0
                for scaffold in tier_scaffolds:
                    fold_scaffolds[fold_idx].append(scaffold)
                    fold_idx = (fold_idx + 1) % self.n_splits
        else:
            # Fallback to simple round-robin for very few scaffolds
            fold_idx = 0
            for scaffold in sorted_scaffolds:
                fold_scaffolds[fold_idx].append(scaffold)
                fold_idx = (fold_idx + 1) % self.n_splits
        
        # Distribute scaffold-less molecules across folds
        # Assign them one by one to the fold with the smallest total count
        for idx in scaffold_less_indices:
            # Calculate current sample counts for each fold (including scaffold-less molecules)
            fold_sizes = []
            for fold_idx in range(self.n_splits):
                total_samples = sum(scaffold_sizes[s] for s in fold_scaffolds[fold_idx])
                fold_sizes.append(total_samples)
            
            # Find fold with minimum samples
            min_fold_idx = np.argmin(fold_sizes)
            # Assign this scaffold-less molecule to the minimum fold
            # We create a special scaffold for this fold to track it
            special_scaffold = f"_scaffold_less_{min_fold_idx}"
            if special_scaffold not in scaffold_sizes:
                scaffold_sizes[special_scaffold] = 0
            scaffold_sizes[special_scaffold] += 1
            fold_scaffolds[min_fold_idx].append(special_scaffold)
            # Store the mapping from special scaffold to actual index
            if special_scaffold not in scaffold_dict:
                scaffold_dict[special_scaffold] = []
            scaffold_dict[special_scaffold].append(idx)
        
        # Generate splits with validation
        for fold_idx in range(self.n_splits):
            test_indices = []
            train_indices = []
            
            # Current fold is test set
            for scaffold in fold_scaffolds[fold_idx]:
                test_indices.extend(scaffold_dict[scaffold])
            
            # All other folds are train set
            for i in range(self.n_splits):
                if i != fold_idx:
                    for scaffold in fold_scaffolds[i]:
                        train_indices.extend(scaffold_dict[scaffold])
            
            # Validate that test set is not empty
            if len(test_indices) == 0:
                raise ValueError(
                    f"Fold {fold_idx + 1} has an empty test set. "
                    f"This indicates a problem with scaffold distribution. "
                    f"Please reduce n_splits or use a different split method."
                )
            
            yield np.array(train_indices, dtype=int), np.array(test_indices, dtype=int) # yield: return a generator object
    
    def get_n_splits(self) -> int:
        """Returns the number of splitting iterations in the cross-validator"""
        return self.n_splits


def scaffold_split(smiles_list: List[str], y: np.ndarray, test_size: float = 0.2, 
                  seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split data based on molecular scaffolds to ensure test set contains
    molecules with different scaffolds from training set
    
    Args:
        smiles_list: List of SMILES strings
        y: Labels
        test_size: Fraction of data for test set
        seed: Random seed
        
    Returns:
        train_indices, test_indices
    """
    from sklearn.model_selection import train_test_split
    
    np.random.seed(seed)
    
    # Generate scaffolds for all molecules
    scaffolds = [get_scaffold(s) for s in smiles_list]
    
    # Group indices by scaffold
    scaffold_dict = {}
    for idx, scaffold in enumerate(scaffolds):
        if scaffold not in scaffold_dict:
            scaffold_dict[scaffold] = []
        scaffold_dict[scaffold].append(idx)
    
    # Sort scaffolds by size (descending) for more balanced distribution
    scaffold_sizes = {scaffold: len(indices) for scaffold, indices in scaffold_dict.items()}
    sorted_scaffolds = sorted(scaffold_sizes.keys(), key=lambda x: scaffold_sizes[x], reverse=True)
    
    # Calculate target test size
    n_samples = len(smiles_list)
    n_test_target = int(n_samples * test_size)
    
    # Use a more robust assignment algorithm that minimizes deviation from target
    # Assign scaffolds alternately to minimize imbalance
    train_indices = []
    test_indices = []
    test_size_current = 0
    
    # Try multiple iterations to find the best split
    best_train = None
    best_test = None
    best_deviation = float('inf')
    
    for iteration in range(10):  # Try 10 different random seeds
        np.random.seed(seed + iteration)
        np.random.shuffle(sorted_scaffolds)
        
        train_indices_iter = []
        test_indices_iter = []
        test_size_current_iter = 0
        
        for scaffold in sorted_scaffolds:
            indices = scaffold_dict[scaffold]
            scaffold_size = len(indices)
            
            # Decide whether to put this scaffold in test set
            # Based on whether we're still under the target
            if test_size_current_iter + scaffold_size <= n_test_target * 1.05:  # Allow 5% tolerance
                test_indices_iter.extend(indices)
                test_size_current_iter += scaffold_size
            else:
                train_indices_iter.extend(indices)
        
        # Calculate deviation from target
        deviation = abs(len(test_indices_iter) - n_test_target) / n_test_target
        
        # Keep the best split
        if deviation < best_deviation:
            best_deviation = deviation
            best_train = train_indices_iter.copy()
            best_test = test_indices_iter.copy()
    
    # Convert to numpy arrays
    train_indices = np.array(best_train, dtype=int)
    test_indices = np.array(best_test, dtype=int)
    
    return train_indices, test_indices

# --------- PyTorch Deep Learning Models ---------
class ResidualBlock(nn.Module):
    """Residual block with configurable activation and AlphaDropout regularization."""
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: str = "mish",
        dropout: float = 0.3,
        use_residual: bool = True,
        norm_type: str = "layernorm",
    ):
        super(ResidualBlock, self).__init__()
        self.activation = activation.lower()
        self.use_residual = use_residual
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm_type = norm_type.lower()
        if self.norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(out_dim)
        else:
            self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.AlphaDropout(dropout) if dropout and dropout > 0.0 else nn.Identity()
        self.shortcut = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.linear(x)
        if self.norm_type == "batchnorm" and self.training and out.size(0) < 2:
            out = F.batch_norm(
                out,
                self.norm.running_mean,
                self.norm.running_var,
                self.norm.weight,
                self.norm.bias,
                training=False,
                momentum=self.norm.momentum,
                eps=self.norm.eps,
            )
        else:
            out = self.norm(out)
        out = self._activate(out)
        out = self.dropout(out)
        if self.use_residual:
            out = out + self.shortcut(residual)
        return out

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "gelu":
            return F.gelu(x)
        if self.activation == "mish":
            return F.mish(x)
        return F.relu(x)


class ResidualMLP(nn.Module):
    """Generalized MLP backbone using configurable residual blocks."""
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 1,
        activation: str = "mish",
        dropout: float = 0.3,
        use_residual: bool = True,
        norm_type: str = "layernorm",
    ):
        super(ResidualMLP, self).__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        dims = [input_dim] + hidden_dims
        self.blocks = nn.ModuleList([
            ResidualBlock(
                in_dim,
                out_dim,
                activation=activation,
                dropout=dropout,
                use_residual=use_residual,
                norm_type=norm_type,
            )
            for in_dim, out_dim in zip(dims[:-1], dims[1:])
        ])
        final_dim = dims[-1]
        self.output_layer = nn.Linear(final_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)

# --------- GAT Model (Graph Attention Network) ---------
try:
    from torch_geometric.nn import GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader as GeometricDataLoader
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    class GATModel(nn.Module):
        """Graph Attention Network for molecular graphs"""
        def __init__(self, num_node_features: int, num_edge_features: int, hidden_dim: int = 64, 
                     num_heads: int = 4, num_layers: int = 3, dropout: float = 0.3):
            super(GATModel, self).__init__()
            
            self.num_layers = num_layers
            self.convs = nn.ModuleList()
            
            # First GAT layer
            self.convs.append(
                GATConv(
                    num_node_features,
                    hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=num_edge_features,
                )
            )
            
            # Additional GAT layers
            for _ in range(num_layers - 1):
                self.convs.append(
                    GATConv(
                        hidden_dim * num_heads,
                        hidden_dim,
                        heads=num_heads,
                        dropout=dropout,
                        edge_dim=num_edge_features,
                    )
                )
            
            # Output layer
            self.output_dim = hidden_dim * num_heads
            self.fc = nn.Linear(self.output_dim, 1)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x, edge_index, batch, edge_attr=None):
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index, edge_attr=edge_attr)
                x = F.elu(x)
                x = self.dropout(x)
            
            # Global pooling
            x = global_mean_pool(x, batch)
            
            # Final prediction (no sigmoid here - BCEWithLogitsLoss handles it)
            return self.fc(x).squeeze(dim=-1)
    
    GAT_AVAILABLE = True
except ImportError:
    GAT_AVAILABLE = False
    print("Warning: PyTorch Geometric or RDKit not available. GAT model will be disabled.")

# --------- GAT Feature Encoding ---------
# Atom feature categories (one-hot)
GAT_ATOM_ELEMENTS = ['H', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
GAT_ATOM_DEGREES = [0, 1, 2, 3, 4, 5]
GAT_ATOM_IMPLICIT_HS = [0, 1, 2, 3, 4]
GAT_FORMAL_CHARGES = [-2, -1, 0, 1, 2]
GAT_RADICAL_ELECTRONS = [0, 1, 2]
GAT_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
GAT_TOTAL_HS = [0, 1, 2, 3, 4]

# Bond feature categories (one-hot)
GAT_BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
GAT_STEREOS = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]

GAT_NUM_NODE_FEATURES = (
    (len(GAT_ATOM_ELEMENTS) + 1) +
    (len(GAT_ATOM_DEGREES) + 1) +
    (len(GAT_ATOM_IMPLICIT_HS) + 1) +
    (len(GAT_FORMAL_CHARGES) + 1) +
    (len(GAT_RADICAL_ELECTRONS) + 1) +
    (len(GAT_HYBRIDIZATIONS) + 1) +
    2 +  # aromatic boolean
    (len(GAT_TOTAL_HS) + 1)
)

GAT_NUM_EDGE_FEATURES = (
    (len(GAT_BOND_TYPES) + 1) +
    2 +  # conjugation boolean
    2 +  # ring boolean
    (len(GAT_STEREOS) + 1)
)

def _one_hot_with_unknown(value, choices) -> List[float]:
    """
    One-hot encode `value` against known `choices` with an explicit UNKNOWN bucket.
    """
    vec = [0.0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
    except ValueError:
        idx = len(choices)  # unknown bucket
    vec[idx] = 1.0
    return vec

def smiles_to_graph(smiles: str, logger: Optional[logging.Logger] = None) -> Optional[Data]:
    """Convert SMILES to PyTorch Geometric graph"""
    if not GAT_AVAILABLE:
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        mol = Chem.AddHs(mol) # Add hydrogen atoms
        
        # Generate 3D coordinates (optional, can be slow)
        # AllChem.EmbedMolecule(mol)
        # AllChem.MMFFOptimizeMolecule(mol)
        
        # Node features (one-hot atom features)
        node_features = []
        
        for atom in mol.GetAtoms():
            features = []
            features += _one_hot_with_unknown(atom.GetSymbol(), GAT_ATOM_ELEMENTS)
            features += _one_hot_with_unknown(atom.GetDegree(), GAT_ATOM_DEGREES)
            features += _one_hot_with_unknown(atom.GetNumImplicitHs(), GAT_ATOM_IMPLICIT_HS)
            features += _one_hot_with_unknown(atom.GetFormalCharge(), GAT_FORMAL_CHARGES)
            features += _one_hot_with_unknown(atom.GetNumRadicalElectrons(), GAT_RADICAL_ELECTRONS)
            features += _one_hot_with_unknown(atom.GetHybridization(), GAT_HYBRIDIZATIONS)
            features += [1.0 if atom.GetIsAromatic() else 0.0, 0.0 if atom.GetIsAromatic() else 1.0]
            features += _one_hot_with_unknown(atom.GetTotalNumHs(), GAT_TOTAL_HS)
            node_features.append(features)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Edge indices and edge attributes (one-hot bond features)
        edge_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_feat = []
            bond_feat += _one_hot_with_unknown(bond.GetBondType(), GAT_BOND_TYPES)
            bond_feat += [1.0 if bond.GetIsConjugated() else 0.0, 0.0 if bond.GetIsConjugated() else 1.0]
            bond_feat += [1.0 if bond.IsInRing() else 0.0, 0.0 if bond.IsInRing() else 1.0]
            bond_feat += _one_hot_with_unknown(bond.GetStereo(), GAT_STEREOS)
            edge_indices.append([i, j])
            edge_indices.append([j, i])  # Undirected graph
            edge_attrs.append(bond_feat)
            edge_attrs.append(bond_feat)
        
        if len(edge_indices) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, GAT_NUM_EDGE_FEATURES), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    except Exception as e:
        # Avoid stdout-only debugging; allow caller to route into the training logs.
        if logger:
            logger.warning(f"Error converting SMILES to graph: {e}")
        else:
            print(f"Error converting SMILES to graph: {e}")
        return None

class MoleculeDataset(Dataset):
    """Dataset for molecular graphs"""
    def __init__(self, smiles_list, labels, logger: Optional[logging.Logger] = None):
        self.graphs = []
        invalid_indices = []
        total = len(smiles_list) if smiles_list is not None else 0
        
        # Prefer logging over stdout; also avoid tqdm spamming in non-interactive runs.
        iterator = tqdm(smiles_list, disable=(logger is not None))
        if logger:
            logger.info("Converting SMILES to graphs (GAT)...")
        else:
            print("Converting SMILES to graphs...")
        
        for idx, smiles in enumerate(iterator):
            graph = smiles_to_graph(smiles, logger=logger)
            self.graphs.append(graph)
            if graph is None:
                invalid_indices.append(idx)
        
        # Filter out None values - track valid indices for external use
        valid_indices = [i for i, g in enumerate(self.graphs) if g is not None]
        self.valid_indices = valid_indices  # Store original indices of valid samples
        self.graphs = [self.graphs[i] for i in valid_indices]
        self.labels = [labels[i] for i in valid_indices]
        self.smiles_list = [smiles_list[i] for i in valid_indices]
        
        n_valid = len(self.graphs)
        n_invalid = len(invalid_indices)
        fail_rate = (n_invalid / total) if total else 0.0
        msg = f"GAT graph conversion: {n_valid}/{total} valid ({fail_rate*100:.2f}% failed)"
        if logger:
            logger.info(msg)
            if n_invalid > 0:
                # Log a short preview of failures to help track down data issues.
                preview = invalid_indices[:5]
                logger.warning(f"GAT invalid SMILES indices (first 5): {preview}")
        else:
            print(msg)
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        # Return a single Data object with label stored in y attribute
        # This format is compatible with PyTorch Geometric's GeometricDataLoader
        graph = self.graphs[idx]
        graph.y = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return graph

# --------- ChemBERTa Model ---------
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaForSequenceClassification
    # DeepChem/ChemBERTa-77M-MLM, DeepChem/ChemBERTa-77M-MTR, seyonec/ChemBERTa-zinc-base-v1
    class ChemBERTaModel(nn.Module):
        """
        ChemBERTa Transformer for molecular property prediction using AutoModelForSequenceClassification.
        
        Features:
        - Local-first loading strategy: Prioritizes loading locally.
        - Full fine-tuning support: Support full parameter fine-tuning
        - GPU/CPU fallback: Automatically adapts to GPU operation and falls back to CPU when there is no GPU
        
        Args:
            model_name: Hugging Face model name or local path. Default: 'DeepChem/ChemBERTa-77M-MTR'
            num_labels: Number of output labels (1 for binary classification)
        """
        def __init__(self, model_name: str = "DeepChem/ChemBERTa-77M-MLM", num_labels: int = 1):
            super(ChemBERTaModel, self).__init__()

            # Mapping from Hugging Face model names to local folder names
            # Note: Keys use standard capitalization (77M), but matching is case-insensitive
            model_name_mapping = {
                "DeepChem/ChemBERTa-77M-MLM": "chemberta_77m_mlm",
                "DeepChem/ChemBERTa-77M-MTR": "chemberta_77m_mtr",
                "seyonec/ChemBERTa-zinc-base-v1": "chemberta_zinc_v1",
            }

            # Get the corresponding local folder name (case-insensitive matching)
            local_folder_name = None
            for key, value in model_name_mapping.items():
                if key.lower() == model_name.lower():
                    local_folder_name = value
                    break

            # Fallback to generating folder name from model_name if no match found
            if local_folder_name is None:
                local_folder_name = model_name.split('/')[-1].replace("-", "_")
            local_path = os.path.join("pretrained_model", "all_chemberta_models", local_folder_name)

            # Check if local path exists and contains config file
            if os.path.exists(local_path) and os.path.exists(os.path.join(local_path, "config.json")):
                load_path = local_path
                use_local = True
                print(f"[INFO] Loading ChemBERTa from LOCAL path: {load_path}")
            else:
                load_path = model_name
                use_local = False
                print(f"[INFO] Local model not found at {local_path}. Downloading from HuggingFace: {load_path}")
            
            # Load tokenizer with local_files_only only when using local path
            # This allows automatic download from HuggingFace when local model is not available
            self.tokenizer = AutoTokenizer.from_pretrained(
                load_path,
                local_files_only=use_local
            )
            
            # Use AutoModelForSequenceClassification which already includes a classification head
            # ignore_mismatched_sizes=True allows loading models with different classification head dimensions
            # local_files_only only applies when using local path to prevent unintended downloads
            self.model = AutoModelForSequenceClassification.from_pretrained(
                load_path,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
                local_files_only=use_local
            )
        
        def forward(self, input_ids, attention_mask):
            """
            Forward pass through ChemBERTa model
            
            Args:
                input_ids: Tokenized input IDs [batch_size, seq_len]
                attention_mask: Attention mask [batch_size, seq_len]
            
            Returns:
                Logits [batch_size] for binary classification
            """
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            return logits.squeeze(-1)
    
    CHEMBERTA_AVAILABLE = True
except ImportError:
    CHEMBERTA_AVAILABLE = False
    print("Warning: Transformers library not available. ChemBERTa model will be disabled.")

class ChemBERTaDataset(Dataset):
    def __init__(self, smiles_list, labels, tokenizer, max_length=128, logger=None):
        self.smiles_list = smiles_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Indices of non-empty SMILES entries.
        # Used to align y_true with model predictions (collate_fn filters out empty SMILES).
        self.valid_indices = [i for i, s in enumerate(smiles_list) if s and str(s).strip() != ""]

        # Validate tokenizer
        self._validate_tokenizer(logger)

        # Validate SMILES strings
        self._validate_smiles(logger)

    def _validate_tokenizer(self, logger=None):
        """
        Validate that the tokenizer object is properly initialized
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer cannot be None")

        # Check for required tokenizer attributes
        required_attrs = ['vocab_size', 'model_max_length', 'pad_token_id']
        missing_attrs = [attr for attr in required_attrs if not hasattr(self.tokenizer, attr)]

        if missing_attrs:
            error_msg = f"Tokenizer is missing required attributes: {missing_attrs}"
            if logger:
                logger.error(error_msg)
            raise ValueError(error_msg)

        # Validate tokenizer can handle typical SMILES
        test_smiles = "C"
        try:
            encoding = self.tokenizer(test_smiles, return_tensors='pt')
            if encoding is None or len(encoding) == 0:
                raise ValueError("Tokenizer returned empty encoding for test SMILES")
        except Exception as e:
            error_msg = f"Tokenizer failed to encode test SMILES: {e}"
            if logger:
                logger.error(error_msg)
            raise ValueError(error_msg)

    def _validate_smiles(self, logger=None):
        """
        Validate SMILES strings for common issues that could affect tokenization
        """
        # Check for empty strings
        empty_indices = [i for i, s in enumerate(self.smiles_list) if not s or s.strip() == ""]
        if empty_indices:
            if logger:
                logger.warning(f"ChemBERTaDataset: Found {len(empty_indices)} empty SMILES strings (indices: {empty_indices[:10]}...)")
            else:
                print(f"[WARNING] ChemBERTaDataset: Found {len(empty_indices)} empty SMILES strings")

        # Check for excessively long SMILES strings
        long_indices = [i for i, s in enumerate(self.smiles_list) if len(s) > self.max_length * 2]
        if long_indices:
            if logger:
                logger.warning(f"ChemBERTaDataset: Found {len(long_indices)} SMILES strings exceeding safe length (indices: {long_indices[:10]}...)")
            else:
                print(f"[WARNING] ChemBERTaDataset: Found {len(long_indices)} SMILES strings exceeding safe length")

        # Check for SMILES with unusual characters
        import re
        unusual_chars_pattern = re.compile(r'[^a-zA-Z0-9@\+\-\[\]\(\)#=:$/\.%]', re.UNICODE)
        unusual_indices = [i for i, s in enumerate(self.smiles_list) if unusual_chars_pattern.search(s)]
        if unusual_indices:
            if logger:
                logger.warning(f"ChemBERTaDataset: Found {len(unusual_indices)} SMILES strings with unusual characters (indices: {unusual_indices[:10]}...)")
            else:
                print(f"[WARNING] ChemBERTaDataset: Found {len(unusual_indices)} SMILES strings with unusual characters")

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        label = self.labels[idx]

        # Return None for empty SMILES to be filtered by collate_fn
        if not smiles or smiles.strip() == "":
            return None
        
        # Tokenize SMILES
        encoding = self.tokenizer(
            smiles,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float32)
        }


def chemberta_collate_fn(batch, max_length: int = 128):
    """
    Custom collate function for ChemBERTa that filters out None samples (e.g., empty SMILES).
    
    Args:
        batch: List of samples from ChemBERTaDataset (may contain None values)
    
    Returns:
        Batched tensor dictionary with only valid samples, or None if all samples are invalid
    """
    # Filter out None samples
    valid_batch = [item for item in batch if item is not None]
    
    if not valid_batch:
        # Return empty batch if all samples are None
        return {
            'input_ids': torch.zeros((0, max_length), dtype=torch.long),
            'attention_mask': torch.zeros((0, max_length), dtype=torch.long),
            'label': torch.zeros((0,), dtype=torch.float32)
        }
    
    # Stack valid samples
    return {
        'input_ids': torch.stack([item['input_ids'] for item in valid_batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in valid_batch]),
        'label': torch.stack([item['label'] for item in valid_batch])
    }

# --------- Model Registry ---------
def build_model_registry(task: str = "classification", input_dim: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    """
    Build model registry with all available models
    
    Returns:
        Dictionary mapping model keys to model configurations
    """
    registry = {}
    
    # ===== Traditional Models =====
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.svm import SVC, SVR
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
    
    if task == "classification":
        # LR
        registry['LR'] = {
            'type': 'sklearn',
            'model': LogisticRegression,
            'params': {'max_iter': 2000, 'solver': 'lbfgs'},
            'grid': {'C': [0.01, 0.1, 1, 10, 100]}
        }
        
        # RFC
        registry['RFC'] = {
            'type': 'sklearn',
            'model': RandomForestClassifier,
            'params': {'n_estimators': 400, 'n_jobs': -1, 'bootstrap': True},
            'grid': {'n_estimators': [300, 500, 800], 'max_depth': [None, 20, 40]}
        }
        
        # SVC
        registry['SVC'] = {
            'type': 'sklearn',
            'model': SVC,
            'params': {'probability': True},
            'grid': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
        }
        
        # XGBC
        try:
            from xgboost import XGBClassifier
            registry['XGBC'] = {
                'type': 'sklearn',
                'model': XGBClassifier,
                # Explicit keys (None values will be overwritten by train_model_wrapper using the active seed).
                'params': {'random_state': None, 'seed': None, 'subsample': 0.8},
                'grid': {'n_estimators': [300, 600, 1000], 'max_depth': [3, 6, 9], 'learning_rate': [0.03, 0.1]}
            }
        except ImportError:
            pass
        
        # LGBMC
        try:
            from lightgbm import LGBMClassifier
            registry['LGBMC'] = {
                'type': 'sklearn',
                'model': LGBMClassifier,
                'params': {'subsample': 0.8},
                'grid': {'n_estimators': [400, 800, 1200], 'num_leaves': [31, 63, 127], 'learning_rate': [0.03, 0.1]}
            }
        except ImportError:
            pass
        
        # ETC
        registry['ETC'] = {
            'type': 'sklearn',
            'model': ExtraTreesClassifier,
            'params': {'n_estimators': 400, 'n_jobs': -1},
            'grid': {'n_estimators': [300, 600, 1000], 'max_depth': [None, 20, 40]}
        }
    
    else:
        # Regression models
        registry['Ridge'] = {
            'type': 'sklearn',
            'model': Ridge,
            'params': {},
            'grid': {'alpha': [1e-3, 1e-2, 1e-1, 1, 10, 100]}
        }
        
        registry['RFR'] = {
            'type': 'sklearn',
            'model': RandomForestRegressor,
            'params': {'n_estimators': 400, 'n_jobs': -1},
            'grid': {'n_estimators': [300, 600, 1000], 'max_depth': [None, 20, 40]}
        }
        
        registry['ETR'] = {
            'type': 'sklearn',
            'model': ExtraTreesRegressor,
            'params': {'n_estimators': 400, 'n_jobs': -1},
            'grid': {'n_estimators': [300, 600, 1000], 'max_depth': [None, 20, 40]}
        }
    
    # ===== Deep Learning Models =====
    
    # MLP (Multi-Layer Perceptron)
    # Always add MLP to registry, but input_dim will be set dynamically in train_model_wrapper
    registry['MLP'] = {
        'type': 'pytorch',
        'model_class': ResidualMLP,
        'params': {
            'input_dim': input_dim if input_dim is not None else -1,  # Placeholder -1, will be updated dynamically
            'hidden_dims': [512, 256],
            'dropout': 0.3,
            'activation': 'mish',
            'use_residual': True,
            'output_dim': 1,
            'norm_type': 'layernorm'
        }
    }
    
    # GAT (Graph Attention Network)
    if GAT_AVAILABLE:
        registry['GAT'] = {
            'type': 'pytorch_geometric',
            'model_class': GATModel,
            'params': {
                # Dynamically inferred from actual graph tensor shape in train_model_wrapper.
                'num_node_features': None,
                'num_edge_features': None,
                'hidden_dim': 64,
                'num_heads': 4,
                'num_layers': 3,
                'dropout': 0.3
            }
        }
    
    # ChemBERTa
    if CHEMBERTA_AVAILABLE:
        registry['ChemBERTa'] = {
            'type': 'transformer',
            'model_class': ChemBERTaModel,
            'params': {
                'model_name': 'DeepChem/ChemBERTa-77M-MLM',
                'num_labels': 1
            }
        }
    
    return registry

# --------- Training Functions ---------
def train_pytorch_model(model, train_loader, val_loader, config: QSARConfig, 
                       task: str = "classification", logger: logging.Logger = None,
                       model_type: str = "default") -> nn.Module:
    """
    Train PyTorch model with early stopping and model-specific training strategies.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader (can be None)
        config: QSAR configuration
        task: "classification" or "regression"
        logger: Logger instance
        model_type: Model type identifier ("default", "transformer" for ChemBERTa, "pytorch", "pytorch_geometric")
    
    Returns:
        Trained model with best weights
    """
    # Use the passed logger, or get the current module's logger if None
    if logger is None:
        logger = logging.getLogger(__name__)
    
    device = DEVICE
    model = model.to(device)
    
    # ========== Model-specific Training Strategies ==========
    # For ChemBERTa transformer models, use specialized hyperparameters and optimizer
    if model_type == "transformer":
        # AdamW optimizer with weight decay (standard for transformer fine-tuning)
        base_lr = 5e-5
        wt_decay = 0.01
        min_lr = 1e-7
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=wt_decay)
        
        # Cosine Annealing Learning Rate scheduler
        # Gradually reduces learning rate following a cosine curve
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.max_epochs,  # Number of iterations for the cosine cycle
            eta_min=min_lr  # Minimum learning rate
        )
        
        logger.info("ChemBERTa training strategy:")
        current_lr = optimizer.param_groups[0]['lr']
        current_wd = optimizer.param_groups[0]['weight_decay']
        logger.info(f"  - Optimizer: {type(optimizer).__name__} (lr={current_lr:.1e}, weight_decay={current_wd})")
        logger.info(f"  - Scheduler: {type(scheduler).__name__} (T_max={scheduler.T_max}, eta_min={scheduler.eta_min:.1e})")
        logger.info("  - Fine-tuning: Full parameter fine-tuning (no frozen layers)")
    else:
        # Default strategy for other deep learning models (MLP, GAT)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Loss function
    # Use BCEWithLogitsLoss for classification for better numerical stability
    # This combines sigmoid and BCE in a single function for improved stability
    if task == "classification":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_epoch = 0
    
    # Training loop with epoch-wise tracking
    for epoch in range(config.max_epochs):
        # Training
        model.train()
        train_loss = 0.0
        num_train_batches_processed = 0
        
        # Progress bar for training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.max_epochs} [Train]", 
                    leave=False, disable=logger is None)
        
        for batch in pbar:
            # Handle different batch formats
            if isinstance(batch, Data):  # GAT - Batch object (Data subclass) with y attribute
                batch = batch.to(device)
                labels = batch.y
                optimizer.zero_grad()
                outputs = model(batch.x, batch.edge_index, batch.batch, edge_attr=getattr(batch, "edge_attr", None))
                loss = criterion(outputs, labels)
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:  # MLP
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs).squeeze(-1)
                loss = criterion(outputs, labels)
            else:  # ChemBERTa (transformer)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                # Skip fully-empty collated batches (e.g., when all SMILES were empty)
                if input_ids.size(0) == 0:
                    continue
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_train_batches_processed += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        pbar.close()
        if num_train_batches_processed == 0:
            logger.warning("No non-empty batches processed during training; aborting training loop.")
            break
        train_loss /= num_train_batches_processed

        # Clear GPU cache after each epoch to prevent OOM, especially for transformer models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Validation (skip if val_loader is None)
        if val_loader is None:
            # No validation set - use training loss as proxy
            val_loss = train_loss
            logger.info(f"Epoch {epoch+1}/{config.max_epochs} - Train Loss: {train_loss:.4f} (No validation set)")
            
            # Update scheduler based on epoch (for CosineAnnealingLR)
            if model_type == "transformer":
                scheduler.step()
            
            # No validation set - save best model based on training loss
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
        else:
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []
            num_val_batches_processed = 0
            
            with torch.no_grad():
                # Progress bar for validation
                pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.max_epochs} [Val]", 
                              leave=False, disable=logger is None)
                
                for batch in pbar_val:
                    if isinstance(batch, Data):  # GAT - Batch object (Data subclass) with y attribute
                        batch = batch.to(device)
                        labels = batch.y
                        outputs = model(batch.x, batch.edge_index, batch.batch, edge_attr=getattr(batch, "edge_attr", None))
                        loss = criterion(outputs, labels)
                    elif isinstance(batch, (list, tuple)) and len(batch) == 2:  # MLP
                        inputs, labels = batch
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = model(inputs).squeeze(-1)
                        loss = criterion(outputs, labels)
                    else:  # ChemBERTa (transformer)
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['label'].to(device)
                        # Skip fully-empty collated batches
                        if input_ids.size(0) == 0:
                            continue
                        outputs = model(input_ids, attention_mask)
                        loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    num_val_batches_processed += 1
                    
                    # Collect predictions
                    all_preds.extend(outputs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    # Update progress bar
                    pbar_val.set_postfix({'loss': f'{loss.item():.4f}'})
                
                pbar_val.close()
            
            if num_val_batches_processed == 0:
                val_loss = float('inf')
            else:
                val_loss /= num_val_batches_processed
            
            # Calculate validation metrics
            if task == "classification":
                # Apply sigmoid to logits for metric calculation
                val_preds_proba = torch.sigmoid(torch.tensor(all_preds)).numpy()
                val_preds_binary = (val_preds_proba >= 0.5).astype(int)
                
                # Calculate comprehensive metrics
                try:
                    val_auc = float(roc_auc_score(all_labels, val_preds_proba))
                except Exception as e:
                    logger.warning(f"  Val AUC computation failed ({e}); setting AUC=NaN")
                    val_auc = float('nan')
                from sklearn.metrics import accuracy_score
                try:
                    val_acc = float(accuracy_score(all_labels, val_preds_binary))
                except Exception as e:
                    logger.warning(f"  Val ACC computation failed ({e}); setting ACC=NaN")
                    val_acc = float('nan')
                
                # Log all key metrics for each epoch
                logger.info(
                    f"Epoch {epoch+1}/{config.max_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Val AUC: {val_auc:.4f}, Val ACC: {val_acc:.4f}"
                )
            else:
                val_r2 = float(r2_score(all_labels, all_preds))
                logger.info(
                    f"Epoch {epoch+1}/{config.max_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val R2: {val_r2:.4f}"
                )
            
            # Learning rate scheduling
            if model_type == "transformer":
                # CosineAnnealingLR steps by epoch
                scheduler.step()
            else:
                # ReduceLROnPlateau steps by validation loss
                scheduler.step(val_loss)
            
            # Early stopping (skip if no validation set - train all epochs)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
    
    # Load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model weights from epoch {best_epoch}")
    
    return model

def predict_pytorch_model(model, data_loader, task: str = "classification") -> Tuple[np.ndarray, np.ndarray]:
    """Get predictions from PyTorch model"""
    # Handle empty data loader
    if data_loader is None:
        return None, None
    
    device = DEVICE
    model = model.to(device)
    model.eval()
    
    all_preds = []
    
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, Data):  # GAT - Batch object (Data subclass)
                batch = batch.to(device)
                outputs = model(batch.x, batch.edge_index, batch.batch, edge_attr=getattr(batch, "edge_attr", None))
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:  # MLP
                inputs, _ = batch
                inputs = inputs.to(device)
                outputs = model(inputs).squeeze(-1)
            else:  # ChemBERTa
                input_ids = batch['input_ids'].to(device)
                if input_ids.size(0) == 0:
                    continue
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask)
            
            all_preds.extend(outputs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    
    # Clear GPU cache to prevent OOM errors, especially for large validation sets
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if task == "classification":
        # Apply sigmoid to get probabilities (model now outputs logits)
        all_proba = torch.sigmoid(torch.tensor(all_preds)).numpy()
        all_labels = (all_proba >= 0.5).astype(int) # threshold should be modifiable if needed
        return all_labels, all_proba
    else:
        all_labels = all_preds
        return all_labels, all_preds

# --------- Model Training Wrapper ---------
def train_model_wrapper(model_key: str, model_config: Dict[str, Any], X_train, y_train, 
                       X_val, y_val, config: QSARConfig, task: str = "classification",
                       smiles_train: Optional[List[str]] = None, smiles_val: Optional[List[str]] = None,
                       logger: logging.Logger = None,
                       sklearn_random_state: Optional[int] = None) -> Dict[str, Any]:
    """
    Train a model based on its type
    
    Args:
        model_key: Model identifier
        model_config: Model configuration dictionary
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Configuration
        task: "classification" or "regression"
        smiles_train: Training SMILES (for GAT/ChemBERTa)
        smiles_val: Validation SMILES (for GAT/ChemBERTa)
        logger: Logger instance
        sklearn_random_state: If set, passed as ``random_state`` to sklearn estimators that accept it
            (RFC/ETC/XGB/LGBM/etc.). Should match the active seed from ``set_all_seeds`` in multi-seed
            loops. If None, falls back to ``config.seeds[0]`` or ``config.seed``.
    
    Returns:
        Dictionary with model and predictions
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"\n--- Training {model_key} ---")
    
    model_type = model_config.get('type', 'sklearn')
    
    # ===== Sklearn Models =====
    if model_type == 'sklearn':
        params = dict(model_config.get('params') or {})
        model_cls = model_config['model']

        # Derive a reproducibility seed for sklearn estimators.
        rs = sklearn_random_state
        if rs is None:
            seeds = getattr(config, 'seeds', None) or []
            rs = seeds[0] if seeds else getattr(config, 'seed', 42)

        # XGBoost: relying on inspect.signature may fail because sklearn wrapper can hide params
        # behind **kwargs. Explicitly set both `random_state` and `seed` when training XGBC.
        if model_key == 'XGBC' or getattr(model_cls, '__name__', '') == 'XGBClassifier':
            if params.get('random_state') is None:
                params['random_state'] = rs
            if params.get('seed') is None:
                params['seed'] = rs
            if logger:
                logger.debug(f"{model_key}: injecting XGBoost seeds random_state={rs}, seed={rs}")
        else:
            # Generic case: inject random_state when the estimator's __init__ exposes it.
            try:
                sig = inspect.signature(model_cls.__init__)
            except (TypeError, ValueError):
                sig = None

            if sig is not None and 'random_state' in sig.parameters and params.get('random_state') is None:
                params['random_state'] = rs
                if logger:
                    logger.debug(f"{model_key}: injecting random_state={rs} (sklearn estimators need it)")

        model = model_cls(**params)
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Predictions on training set
        if hasattr(model, 'predict_proba'):
            y_proba_train = model.predict_proba(X_train)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_proba_train = model.decision_function(X_train)
            # Sigmoid for classification
            if task == "classification":
                y_proba_train = 1.0 / (1.0 + np.exp(-y_proba_train)) # sigmoid to convert to probabilities
        else:
            y_proba_train = model.predict(X_train)
        
        y_pred_train = (y_proba_train >= 0.5).astype(int) if task == "classification" else y_proba_train
        
        # Predictions on validation set (if provided)
        y_proba_val = None
        y_pred_val = None
        if X_val is not None and len(X_val) > 0:
            if hasattr(model, 'predict_proba'):
                y_proba_val = model.predict_proba(X_val)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_proba_val = model.decision_function(X_val)
                # Sigmoid for classification
                if task == "classification":
                    y_proba_val = 1.0 / (1.0 + np.exp(-y_proba_val))
            else:
                y_proba_val = model.predict(X_val)
            
            y_pred_val = (y_proba_val >= 0.5).astype(int) if task == "classification" else y_proba_val
        
        return {
            'model': model,
            'y_pred_train': y_pred_train,
            'y_proba_train': y_proba_train,
            'y_pred_val': y_pred_val,
            'y_proba_val': y_proba_val
        }
    
    # ===== PyTorch MLP =====
    elif model_type == 'pytorch':
        # Dynamically update input_dim based on actual feature dimensions
        if X_train is not None:
            actual_input_dim = X_train.shape[1]
            model_config = model_config.copy()  # Create a copy to avoid modifying original
            model_config['params'] = model_config['params'].copy()
            model_config['params']['input_dim'] = actual_input_dim
            if logger:
                logger.debug(f"MLP input_dim set to {actual_input_dim}")
        
        model = model_config['model_class'](**model_config['params'])
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        
        # Create validation loader only if validation data is provided
        val_loader = None
        if X_val is not None and len(X_val) > 0:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            )
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        
        # Train model with model_type="pytorch"
        model = train_pytorch_model(model, train_loader, val_loader, config, task, logger, model_type="pytorch")
        
        # Predictions
        y_pred_train, y_proba_train = predict_pytorch_model(model, train_loader, task)
        y_pred_val, y_proba_val = predict_pytorch_model(model, val_loader, task) if val_loader is not None else (None, None)
        
        return {
            'model': model,
            'y_pred_train': y_pred_train,
            'y_proba_train': y_proba_train,
            'y_pred_val': y_pred_val,
            'y_proba_val': y_proba_val
        }
    
    # ===== PyTorch Geometric GAT =====
    elif model_type == 'pytorch_geometric':
        if smiles_train is None:
            logger.error(f"SMILES data required for {model_key}")
            return None

        # Create datasets
        train_dataset = MoleculeDataset(smiles_train, y_train, logger=logger)
        if len(train_dataset) == 0:
            logger.error(f"{model_key}: no valid training graphs after SMILES->graph conversion.")
            return None

        # Dynamically infer graph feature dimensions from converted tensors.
        inferred_node_dim = int(train_dataset.graphs[0].x.shape[1])
        inferred_edge_dim = int(train_dataset.graphs[0].edge_attr.shape[1]) if hasattr(train_dataset.graphs[0], "edge_attr") else 0
        gat_params = dict(model_config['params'])
        gat_params['num_node_features'] = inferred_node_dim
        gat_params['num_edge_features'] = inferred_edge_dim
        model = model_config['model_class'](**gat_params)
        logger.info(
            f"{model_key}: dynamic graph dims inferred from train set -> "
            f"node_features={inferred_node_dim}, edge_features={inferred_edge_dim}"
        )

        train_loader = GeometricDataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        
        # Filter y_train to match valid samples (those that could be converted to graphs)
        y_train_filtered = y_train[train_dataset.valid_indices]
        
        # Create validation loader only if validation data is provided
        val_loader = None
        y_val_filtered = None
        if smiles_val is not None and len(smiles_val) > 0:
            val_dataset = MoleculeDataset(smiles_val, y_val, logger=logger)
            val_loader = GeometricDataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
            # Filter y_val to match valid samples
            y_val_filtered = y_val[val_dataset.valid_indices]
        
        # Train model with model_type="pytorch_geometric"
        model = train_pytorch_model(model, train_loader, val_loader, config, task, logger, model_type="pytorch_geometric")
        
        # Predictions
        y_pred_train, y_proba_train = predict_pytorch_model(model, train_loader, task)
        y_pred_val, y_proba_val = predict_pytorch_model(model, val_loader, task) if val_loader is not None else (None, None)
        
        return {
            'model': model,
            'y_pred_train': y_pred_train,
            'y_proba_train': y_proba_train,
            'y_pred_val': y_pred_val,
            'y_proba_val': y_proba_val,
            'valid_indices_train': train_dataset.valid_indices,  # Return for reference
            'valid_indices_val': val_dataset.valid_indices if val_loader is not None else None
        }
    
    # ===== Transformer ChemBERTa =====
    elif model_type == 'transformer':
        if smiles_train is None:
            logger.error(f"SMILES data required for {model_key}")
            return None
        
        model = model_config['model_class'](**model_config['params'])
        
        # Create datasets
        train_dataset = ChemBERTaDataset(smiles_train, y_train, model.tokenizer)
        valid_indices_train = getattr(train_dataset, "valid_indices", None)
        max_length = getattr(train_dataset, "max_length", 128)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                                 collate_fn=lambda b, ml=max_length: chemberta_collate_fn(b, max_length=ml))
        
        # Create validation loader only if validation data is provided
        val_loader = None
        valid_indices_val = None
        if smiles_val is not None and len(smiles_val) > 0:
            val_dataset = ChemBERTaDataset(smiles_val, y_val, model.tokenizer)
            valid_indices_val = getattr(val_dataset, "valid_indices", None)
            max_length_val = getattr(val_dataset, "max_length", max_length)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                                   collate_fn=lambda b, ml=max_length_val: chemberta_collate_fn(b, max_length=ml))
        
        # Train model with model_type="transformer" for specialized ChemBERTa training strategy
        # This uses: AdamW optimizer (lr=5e-5), CosineAnnealingLR scheduler, full fine-tuning
        model = train_pytorch_model(model, train_loader, val_loader, config, task, logger, model_type="transformer")
        
        # Predictions
        y_pred_train, y_proba_train = predict_pytorch_model(model, train_loader, task)
        y_pred_val, y_proba_val = predict_pytorch_model(model, val_loader, task) if val_loader is not None else (None, None)
        
        return {
            'model': model,
            'y_pred_train': y_pred_train,
            'y_proba_train': y_proba_train,
            'y_pred_val': y_pred_val,
            'y_proba_val': y_proba_val,
            'valid_indices_train': valid_indices_train,
            'valid_indices_val': valid_indices_val
        }
    
    else:
        logger.error(f"Unknown model type: {model_type}")
        return None

# --------- Save Utilities ---------
def _save_pytorch_model(model, path: Path, model_type: str):
    """Save PyTorch model"""
    if model_type == 'pytorch':
        torch.save(model.state_dict(), path.with_suffix('.pt'))
    elif model_type == 'pytorch_geometric':
        torch.save(model.state_dict(), path.with_suffix('.pt'))
    elif model_type == 'transformer':
        # For ChemBERTa model: model.model contains the AutoModelForSequenceClassification
        # which already includes the classification head, so save everything together
        model.model.save_pretrained(str(path.with_suffix('')))
        model.tokenizer.save_pretrained(str(path.with_suffix('')))

def _save_sklearn_model(model, path: Path):
    """Save sklearn model"""
    import joblib
    joblib.dump(model, path.with_suffix('.joblib'))

def _save_fig(fig, path_noext: Path):
    """Save figure as PNG and SVG"""
    png = path_noext.with_suffix(".png")
    svg = path_noext.with_suffix(".svg")
    fig.tight_layout()
    fig.savefig(png, bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(svg, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)

def _save_feature_processors(selector, scaler, feature_mask, path: Path, logger: logging.Logger = None):
    """Save feature processors (variance selector and standardizer)"""
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold
    
    if selector is not None:
        file_path = path.parent / (path.stem + '_variance_selector.joblib')
        joblib.dump(selector, file_path)
        if logger:
            logger.info(f"  ✓ Saved variance selector: {file_path}")
    
    if scaler is not None:
        file_path = path.parent / (path.stem + '_scaler.joblib')
        joblib.dump(scaler, file_path)
        if logger:
            logger.info(f"  ✓ Saved scaler: {file_path}")
    
    # Save feature mask as numpy array
    if feature_mask is not None:
        file_path = path.parent / (path.stem + '_feature_mask.npy')
        np.save(file_path, feature_mask)
        if logger:
            logger.info(f"  ✓ Saved feature mask: {file_path}")


def _save_cv_fold_seed_details(root_dir: Path,
                               fold_idx: Optional[int],
                               seed: int,
                               model_key: str,
                               model_config: Dict[str, Any],
                               model_type: str,
                               model_obj,
                               X_train_arr,
                               scaler,
                               selector,
                               feature_mask: Optional[np.ndarray],
                               y_train,
                               smiles_train: Optional[List[str]],
                               ids_train: Optional[List[str]],
                               logger: logging.Logger = None):
    """
    Save per-fold per-seed artifacts required for post-hoc analysis.

    Args:
        root_dir: Root directory for CV artifacts
        fold_idx: Fold identifier
        seed: Random seed used for training
        model_key: Model identifier
        model_config: Registry entry for the model
        model_type: Model type string (e.g., 'pytorch', 'sklearn')
        model_obj: Trained model instance
        X_train_arr: Processed training matrix
        scaler: Fitted scaler (if any)
        selector: Fitted selector (if any)
        feature_mask: Boolean mask for features
        y_train: Training labels
        smiles_train: SMILES strings for training samples
        ids_train: Molecule IDs for training samples
        logger: Optional logger
    """
    if root_dir is None or model_obj is None:
        return

    fold_label = f"fold_{fold_idx}" if fold_idx is not None else "fold_unknown"
    seed_label = f"seed_{seed}"
    seed_dir = Path(root_dir) / fold_label / seed_label
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Save model artifact
    model_path = seed_dir / f"model_{model_key}"
    try:
        if model_type in ['sklearn']:
            _save_sklearn_model(model_obj, model_path)
        else:
            _save_pytorch_model(model_obj, model_path, model_type)
    except Exception as exc:
        if logger:
            logger.warning(f"    ✗ Failed to save CV model artifact: {exc}")
    else:
        if logger:
            logger.info(f"    ✓ Saved CV model artifact: {model_path}")

    # Save processed training data
    if X_train_arr is not None:
        try:
            X_array = np.asarray(X_train_arr)
            np.save(seed_dir / "X_train_processed.npy", X_array)
            if logger:
                logger.info(f"    ✓ Saved processed training data: {seed_dir / 'X_train_processed.npy'}")
        except Exception as exc:
            if logger:
                logger.warning(f"    ✗ Failed to save X_train_processed: {exc}")

    # Save metadata for SHAP traceability
    try:
        metadata = {'y_train': np.asarray(y_train).tolist()}
        if smiles_train is not None:
            metadata['smiles'] = list(smiles_train)
        if ids_train is not None:
            metadata['id'] = list(ids_train)
        metadata_df = pd.DataFrame(metadata)
        metadata_path = seed_dir / "metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        if logger:
            logger.info(f"    ✓ Saved metadata: {metadata_path}")
    except Exception as exc:
        if logger:
            logger.warning(f"    ✗ Failed to save metadata: {exc}")

    # Save processors
    try:
        import joblib
        if scaler is not None:
            joblib.dump(scaler, seed_dir / "scaler.joblib")
            if logger:
                logger.info(f"    ✓ Saved scaler: {seed_dir / 'scaler.joblib'}")
        if selector is not None:
            joblib.dump(selector, seed_dir / "selector.joblib")
            if logger:
                logger.info(f"    ✓ Saved selector: {seed_dir / 'selector.joblib'}")
    except Exception as exc:
        if logger:
            logger.warning(f"    ✗ Failed to save processor: {exc}")

    # Save feature mask for this fold if available
    if feature_mask is not None:
        try:
            np.save(seed_dir / "feature_mask.npy", feature_mask)
            if logger:
                logger.info(f"    ✓ Saved feature mask: {seed_dir / 'feature_mask.npy'}")
        except Exception as exc:
            if logger:
                logger.warning(f"    ✗ Failed to save feature mask: {exc}")

    # Save PyTorch model configuration for reconstruction
    if model_type == 'pytorch':
        try:
            params = dict(model_config.get('params', {})) if model_config else {}
            if X_train_arr is not None:
                params['input_dim'] = int(np.asarray(X_train_arr).shape[1])
            config_snapshot = {
                'model_key': model_key,
                'model_type': model_type,
                'fold': fold_idx,
                'seed': seed,
                'params': params
            }
            config_path = seed_dir / "model_config.json"
            with open(config_path, "w") as f:
                json.dump(config_snapshot, f, indent=2)
            if logger:
                logger.info(f"    ✓ Saved PyTorch config: {config_path}")
        except Exception as exc:
            if logger:
                logger.warning(f"    ✗ Failed to save PyTorch config: {exc}")


def _save_full_dev_model_artifacts(root_dir: Path,
                                   model_key: str,
                                   seed: int,
                                   model_config: Dict[str, Any],
                                   model_type: str,
                                   model_obj,
                                   scaler,
                                   task: str,
                                   external_data_path: Optional[Path],
                                   feature_mask_path: Optional[Path],
                                   feature_names: Optional[List[str]],
                                   input_dim: Optional[int],
                                   logger: logging.Logger = None):
    """Persist Stage 3 (full Development Set) artifacts for SHAP."""
    if root_dir is None or model_obj is None:
        return

    seed_dir = root_dir / model_key / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    model_path = seed_dir / "model"
    try:
        if model_type == 'sklearn':
            _save_sklearn_model(model_obj, model_path)
        else:
            _save_pytorch_model(model_obj, model_path, model_type)
    except Exception as exc:
        if logger:
            logger.warning(f"    ✗ Failed to save full development model: {exc}")
    else:
        if logger:
            logger.info(f"    ✓ Saved full development model: {model_path}")

    try:
        import joblib
        if scaler is not None:
            joblib.dump(scaler, seed_dir / "scaler.joblib")
            if logger:
                logger.info(f"    ✓ Saved scaler: {seed_dir / 'scaler.joblib'}")
    except Exception as exc:
        if logger:
            logger.warning(f"    ✗ Failed to save scaler: {exc}")

    metadata = {
        'model_key': model_key,
        'seed': seed,
        'task': task,
        'model_type': model_type,
        'external_data': str(external_data_path) if external_data_path is not None else None,
        'feature_mask': str(feature_mask_path) if feature_mask_path is not None else None,
        'feature_names': feature_names,
        'input_dim': input_dim,
        'saved_at': datetime.utcnow().isoformat() + 'Z'
    }
    with open(seed_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    config_snapshot = {
        'model_key': model_key,
        'model_type': model_type,
        'params': dict(model_config.get('params', {})) if model_config else {},
        'input_dim': input_dim
    }
    with open(seed_dir / "model_config.json", 'w') as f:
        json.dump(config_snapshot, f, indent=2)

# --------- SHAP Analysis ---------
def run_shap_analysis(model_path: Path, X_train: pd.DataFrame, X_val: pd.DataFrame,
                     model_key: str, out_dir: Path, task: str,
                     max_display: int = 20, sample_size: int = 500,
                     logger: logging.Logger = None) -> Dict[str, Any]:
    """
    Run SHAP analysis for model interpretability
    
    Args:
        model_path: Path to saved model
        X_train: Training features
        X_val: Validation features
        model_key: Model identifier
        out_dir: Output directory
        task: Task type
        max_display: Maximum features to display
        sample_size: Sample size for SHAP computation
        logger: Logger instance
    
    Returns:
        Dictionary with SHAP results
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        import shap
    except ImportError:
        logger.error("SHAP is not installed. Install with: pip install shap")
        return None
    
    import joblib
    
    logger.info(f"Loading model from: {model_path}")
    
    # Load model
    if model_path.suffix == '.joblib':
        model = joblib.load(model_path)
    elif model_path.suffix == '.pt':
        # PyTorch model loading
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Need to recreate model architecture (simplified)
        logger.warning("PyTorch model SHAP analysis requires model architecture recreation")
        return None
    else:
        logger.error(f"Unsupported model format: {model_path.suffix}")
        return None
    
    # Create output directory
    shap_dir = out_dir / "shap_analysis"
    shap_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'model_key': model_key,
        'plots': [],
        'feature_importance': None
    }
    
    try:
        # Sample data for efficiency
        if len(X_train) > sample_size:
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(len(X_train), sample_size, replace=False)
            X_sample = X_train.iloc[sample_idx]
        else:
            X_sample = X_train
        
        # Select appropriate explainer
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            logger.info("  → Using TreeExplainer")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
        elif hasattr(model, 'coef_'):
            # Linear models
            logger.info("  → Using LinearExplainer")
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)
            
        else:
            # KernelExplainer fallback
            logger.info("  → Using KernelExplainer (slower)")
            background = shap.kmeans(X_sample, k=min(50, len(X_sample)))
            
            def predict_fn(X):
                if task == "classification":
                    return model.predict_proba(X)[:, 1]
                return model.predict(X)
            
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_sample)
        
        # Handle classification multi-output
        if task == "classification" and isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
        
        # Feature importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_names = list(X_sample.columns)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        importance_path = shap_dir / f"feature_importance_{model_key}.csv"
        importance_df.to_csv(importance_path, index=False)
        results['feature_importance'] = str(importance_path)
        
        # Summary plot
        fig = plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                         max_display=max_display, show=False)
        plt.title(f"SHAP Summary - {model_key}", fontsize=14, fontweight='bold')
        summary_path = shap_dir / f"shap_summary_{model_key}"
        _save_fig(fig, summary_path)
        results['plots'].append(str(summary_path.with_suffix('.png')))
        
        # Waterfall plot for a sample
        # Handle base_values for binary classification (tree models return array)
        if task == "classification" and isinstance(explainer.expected_value, (list, tuple, np.ndarray)):
            base_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
        else:
            base_value = explainer.expected_value
        
        fig = plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap.Explanation(values=shap_values[0],
                                              base_values=base_value,
                                              data=X_sample.iloc[0],
                                              feature_names=feature_names),
                           max_display=max_display, show=False)
        plt.title(f"SHAP Waterfall - {model_key}", fontsize=12, fontweight='bold')
        waterfall_path = shap_dir / f"waterfall_{model_key}"
        _save_fig(fig, waterfall_path)
        results['plots'].append(str(waterfall_path.with_suffix('.png')))
        
        logger.info(f"  ✓ SHAP analysis completed: {len(results['plots'])} plots generated")
        
    except Exception as e:
        logger.error(f"  ✗ SHAP analysis failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
    return results

# --------- Feature Processing (Fold-level) ---------
def apply_feature_processing(X_train, X_val, config: QSARConfig, logger: logging.Logger = None, 
                             feature_names: Optional[List[str]] = None, 
                             model_type: Optional[str] = None):
    """
    Apply feature filtering and standardization within a fold to prevent data leakage.
    All preprocessing is fitted on training data only and then applied to validation data.
    
    Note: Standardization is only applied to models that benefit from it (LR, SVC, MLP).
    Tree-based models (RF, XGBC, LGBMC, ETC) do not require standardization.
    
    Args:
        X_train: Training features
        X_val: Validation features
        config: Configuration
        logger: Logger instance
        feature_names: List of feature names (optional, for SHAP analysis)
        model_type: Model type ('tree_based' or 'gradient_based')
                   - 'tree_based': RF, XGBC, LGBMC, ETC, RFC, RFR, ETR (no standardization)
                   - 'gradient_based': LR, SVC, MLP (with standardization)
                   - 'none': explicitly disable standardization (skip scaling)
                   - None: Auto-detect (backward compatible default) and apply standardization
    
    Returns:
        X_train_processed, X_val_processed, feature_mask, selector, scaler, feature_names_filtered
    """
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import StandardScaler
    
    X_train_processed = X_train.copy()
    X_val_processed = X_val.copy()
    
    # Step 1: Filter features (fitted on training data)
    # For binary fingerprints (values 0/1), use frequency-based filtering
    # For continuous features, use variance-based filtering
    feature_mask = np.ones(X_train.shape[1], dtype=bool)
    selector = None
    
    if config.min_frequency > 0 and np.all(np.isin(X_train, [0, 1])):
        # Binary fingerprint filtering based on frequency
        n_samples = X_train.shape[0]
        min_count = int(n_samples * config.min_frequency)
        
        # Count occurrences of 1s in each feature
        feature_counts = np.sum(X_train, axis=0)
        
        # Keep features that appear in at least min_frequency of samples
        feature_mask = (feature_counts >= min_count) & (feature_counts <= n_samples - min_count)
        
        X_train_processed = X_train_processed[:, feature_mask]
        X_val_processed = X_val_processed[:, feature_mask]
        
        if logger:
            n_features_after = X_train_processed.shape[1]
            logger.info(f"  Feature filtering (frequency): {X_train.shape[1]} → {n_features_after} features (min_frequency={config.min_frequency*100:.1f}%)")
    
    elif config.variance_threshold > 0:
        # Variance-based filtering for continuous features
        selector = VarianceThreshold(threshold=config.variance_threshold)
        X_train_processed = selector.fit_transform(X_train_processed)
        X_val_processed = selector.transform(X_val_processed)
        feature_mask = selector.get_support()
        
        if logger:
            n_features_after = X_train_processed.shape[1]
            logger.info(f"  Feature filtering (variance): {X_train.shape[1]} → {n_features_after} features (threshold={config.variance_threshold})")
    else:
        if logger:
            logger.info(f"  Feature filtering: skipped")
    
    # Filter feature names if provided
    feature_names_filtered = None
    if feature_names is not None:
        feature_names_filtered = [name for name, keep in zip(feature_names, feature_mask) if keep]
    
    # Step 2: Standardize features (fitted on training data)
    # Only standardize for gradient-based models, not tree-based models
    scaler = None
    apply_standardization = False
    
    # Tree-based models that do NOT need standardization
    tree_based_models = {'RFC', 'RFR', 'XGBC', 'LGBMC', 'ETC', 'ETR'}
    
    # Gradient-based models that benefit from standardization
    gradient_based_models = {'LR', 'Ridge', 'SVC', 'SVR', 'MLP'}
    
    if model_type == 'tree_based':
        apply_standardization = False
        if logger:
            logger.info(f"  Standardization: skipped (tree-based model)")
    elif model_type == 'gradient_based':
        apply_standardization = True
        if logger:
            logger.info(f"  Standardization: applied (gradient-based model)")
    elif model_type in {'none', 'no_standardization'}:
        apply_standardization = False
        if logger:
            logger.info(f"  Standardization: skipped (explicit model_type='{model_type}')")
    else:
        # Auto-detect: if no specific model_type provided, default to standardization
        # for backward compatibility
        apply_standardization = True
        if logger:
            logger.info(f"  Standardization: applied (default)")
    
    if apply_standardization:
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train_processed)
        X_val_processed = scaler.transform(X_val_processed)
        if logger:
            logger.info(f"  Standardization: completed (mean=0, std=1)")
    
    return X_train_processed, X_val_processed, feature_mask, selector, scaler, feature_names_filtered


def get_model_type(model_key: str) -> str:
    """
    Determine if a model is tree-based or gradient-based for preprocessing purposes.
    
    Args:
        model_key: Model identifier (e.g., 'XGBC', 'LR', 'MLP')
    
    Returns:
        'tree_based' or 'gradient_based'
    """
    # Tree-based models (no standardization needed)
    tree_based_models = {'RFC', 'RFR', 'XGBC', 'LGBMC', 'ETC', 'ETR'}
    
    # Gradient-based models (standardization beneficial)
    gradient_based_models = {'LR', 'Ridge', 'SVC', 'SVR', 'MLP'}
    
    if model_key in tree_based_models:
        return 'tree_based'
    elif model_key in gradient_based_models:
        return 'gradient_based'
    else:
        # For deep learning models (GAT, ChemBERTa), they don't use fingerprint features
        # so this won't be called for them
        return 'gradient_based'  # Default to gradient-based for safety


def apply_global_feature_filtering(X: np.ndarray, config: QSARConfig, logger: logging.Logger = None,
                                   feature_names: Optional[List[str]] = None):
    """
    Apply global feature filtering on the matrix passed in (in main_pipeline: Development Set only).
    This ensures all CV folds and External evaluation use the same feature subset.

    Leakage note: frequency / variance statistics are computed only from ``X`` (must be Dev-only).
    The returned ``feature_mask`` is then applied to the held-out External matrix without refitting
    on External labels or features — so External rows do not influence which columns are kept.

    CV optimism: using all Dev rows to choose features yields slightly optimistic CV vs strict
    nested fold-wise selection; it does not leak External labels into training.
    
    Args:
        X: Full feature matrix (Development Set rows only in the standard pipeline)
        config: Configuration
        logger: Logger instance
        feature_names: List of feature names (optional)
    
    Returns:
        X_filtered, feature_mask, feature_names_filtered
    """
    from sklearn.feature_selection import VarianceThreshold
    
    X_filtered = X.copy()
    feature_mask = np.ones(X.shape[1], dtype=bool)
    
    if config.min_frequency > 0 and np.all(np.isin(X, [0, 1])):
        # Binary fingerprint filtering based on frequency
        n_samples = X.shape[0]
        min_count = int(n_samples * config.min_frequency)
        
        # Count occurrences of 1s in each feature
        feature_counts = np.sum(X, axis=0)
        
        # Keep features that appear in at least min_frequency of samples
        feature_mask = (feature_counts >= min_count) & (feature_counts <= n_samples - min_count)
        
        X_filtered = X_filtered[:, feature_mask]
        
        if logger:
            n_features_after = X_filtered.shape[1]
            logger.info(f"Global feature filtering (frequency): {X.shape[1]} → {n_features_after} features (min_frequency={config.min_frequency*100:.1f}%)")
    
    elif config.variance_threshold > 0:
        # Variance-based filtering for continuous features
        selector = VarianceThreshold(threshold=config.variance_threshold)
        X_filtered = selector.fit_transform(X_filtered)
        feature_mask = selector.get_support()
        
        if logger:
            n_features_after = X_filtered.shape[1]
            logger.info(f"Global feature filtering (variance): {X.shape[1]} → {n_features_after} features (threshold={config.variance_threshold})")
    else:
        if logger:
            logger.info(f"Global feature filtering: skipped")
    
    # Filter feature names if provided
    feature_names_filtered = None
    if feature_names is not None:
        feature_names_filtered = [name for name, keep in zip(feature_names, feature_mask) if keep]
    
    return X_filtered, feature_mask, feature_names_filtered


# --------- Training Function for Single Split/Fold ---------
def train_single_fold(config: QSARConfig, X_train, y_train, X_val, y_val,
                     smiles_train: Optional[List[str]] = None, smiles_val: Optional[List[str]] = None,
                     ids_train: Optional[List[str]] = None, ids_val: Optional[List[str]] = None,
                     registry: Dict[str, Dict[str, Any]] = None,
                     fold_idx: Optional[int] = None,
                     save_models: bool = False,
                     model_save_dir: Optional[Path] = None,
                     global_feature_mask: Optional[np.ndarray] = None,
                     save_cv_details: bool = False,
                     cv_save_root: Optional[Path] = None,
                     cv_prediction_accumulator: Optional[List[pd.DataFrame]] = None,
                     logger: logging.Logger = None) -> List[Dict[str, Any]]:
    """
    Train models on a single train/validation split with multi-seed support
    
    Args:
        config: Configuration
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        smiles_train: Training SMILES
        smiles_val: Validation SMILES
        ids_train: Training IDs
        ids_val: Validation IDs
        registry: Model registry
        fold_idx: Fold index (for k-fold CV)
        save_models: Whether to save models for each seed
        model_save_dir: Directory to save models
        global_feature_mask: Optional mask describing the active features for this fold
        save_cv_details: Whether to persist CV artifacts (models, data, processors)
        cv_save_root: Root directory for per-fold/seed artifacts
        logger: Logger instance
    
    Returns:
        List of aggregated results for each model (mean ± std across seeds)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if registry is None:
        input_dim = X_train.shape[1] if X_train is not None else None
        registry = build_model_registry(task=config.task, input_dim=input_dim)
    
    fold_prefix = f"Fold {fold_idx} - " if fold_idx is not None else ""
    logger.info(f"\n{'='*60}")
    logger.info(f"{fold_prefix}Training samples: {len(y_train)}, Validation samples: {len(y_val)}")
    logger.info(f"{fold_prefix}Using {len(config.seeds)} seeds: {config.seeds}")
    logger.info(f"{'='*60}")
    
    # Feature processing:
    # - In CV mode, `main_pipeline()` already applied a global feature mask to `X_dev`,
    #   and `train_single_fold()` receives per-fold subsets of that same masked matrix.
    #   Re-applying filtering here would (a) unnecessarily re-fit selectors on each fold
    #   and (b) risk inconsistent feature counts across folds.
    # - In single-split mode, there is no global feature mask yet, so we apply filtering
    #   inside the fold to avoid leakage.
    fold_selector = None
    if X_train is not None and X_val is not None:
        if config.folds > 1:
            logger.info(f"{fold_prefix}Skipping fold-level feature filtering (global filtering already applied)")
            X_train_processed = X_train
            X_val_processed = X_val
            feature_mask = None
        else:
            logger.info(f"{fold_prefix}Applying fold-level feature filtering (no standardization yet)...")
            X_train_processed, X_val_processed, feature_mask, fold_selector, _, _ = apply_feature_processing(
                X_train, X_val, config, logger, model_type='none'  # No standardization
            )
    else:
        X_train_processed = X_train
        X_val_processed = X_val
        feature_mask = None
    
    # Train each model
    results = []
    valid_models = [m for m in config.selected_models if m in registry]
    
    for model_key in valid_models:
        model_config = registry[model_key]
        scaler = None
        selector = fold_selector
        X_train_final = X_train_processed
        X_val_final = X_val_processed
        
        # Determine if this is a traditional or deep learning model
        model_type_category = 'deep_learning' if model_config['type'] in ['pytorch', 'pytorch_geometric', 'transformer'] else 'traditional'
        
        logger.info(f"\n--- {fold_prefix}Training {model_key} with {len(config.seeds)} seeds ---")
        
        # Prepare data based on model type
        # Apply standardization only for models that need it (gradient-based)
        if model_config['type'] in ['sklearn']:  # Traditional models
            model_preprocessing_type = get_model_type(model_key)
            if X_train_processed is not None and X_val_processed is not None:
                if model_preprocessing_type == 'gradient_based':
                    # Apply standardization for gradient-based models
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_train_final = scaler.fit_transform(X_train_processed)
                    X_val_final = scaler.transform(X_val_processed)
                    logger.info(f"{fold_prefix}Standardization applied for {model_key}")
                else:
                    # No standardization for tree-based models
                    X_train_final = X_train_processed
                    X_val_final = X_val_processed
                    logger.info(f"{fold_prefix}Standardization skipped for {model_key} (tree-based model)")
            else:
                X_train_final = None
                X_val_final = None
        else:
            # Deep learning models (MLP, GAT, ChemBERTa)
            # MLP uses fingerprint features, GAT and ChemBERTa use SMILES
            if model_config['type'] == 'pytorch':  # MLP
                # MLP needs standardization
                if X_train_processed is not None and X_val_processed is not None:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_train_final = scaler.fit_transform(X_train_processed)
                    X_val_final = scaler.transform(X_val_processed)
                    logger.info(f"{fold_prefix}Standardization applied for {model_key}")
                else:
                    X_train_final = None
                    X_val_final = None
            else:  # GAT, ChemBERTa (use SMILES, no feature processing needed)
                X_train_final = None
                X_val_final = None
        
        # Multi-seed training loop
        seed_results = []  # Store results for each seed
        
        for seed_idx, seed in enumerate(config.seeds):
            logger.info(f"\n  {fold_prefix}{model_key} - Seed {seed_idx+1}/{len(config.seeds)} (seed={seed})")
            
            # Set random seed for reproducibility
            set_all_seeds(seed, logger)
            valid_indices_val = None
            valid_indices_train = None
            
            # Prepare data based on model type
            if model_config['type'] in ['pytorch_geometric', 'transformer']:
                if smiles_train is None or smiles_val is None:
                    logger.warning(f"{fold_prefix}SMILES data not available, skipping {model_key}")
                    break
                
                result = train_model_wrapper(
                    model_key, model_config,
                    None, y_train,  # GAT/ChemBERTa don't use X
                    None, y_val,
                    config, config.task,
                    smiles_train, smiles_val,
                    logger,
                    sklearn_random_state=seed,
                )
            else:
                result = train_model_wrapper(
                    model_key, model_config,
                    X_train_final, y_train,
                    X_val_final, y_val,
                    config, config.task,
                    None, None,
                    logger,
                    sklearn_random_state=seed,
                )
            
            if result is None:
                logger.warning(f"  {fold_prefix}{model_key} - Seed {seed} failed")
                continue
            
            # Save model if requested (with seed ID in filename)
            if save_models and model_save_dir is not None and 'model' in result:
                fold_str = f"fold{fold_idx}" if fold_idx is not None else "single"
                model_filename = f"{model_key}_{fold_str}_seed{seed}"
                model_path = model_save_dir / model_filename
                
                try:
                    if model_type_category == 'traditional':
                        _save_sklearn_model(result['model'], model_path)
                        logger.info(f"  {fold_prefix}✓ Saved model: {model_filename}.joblib")
                    else:
                        _save_pytorch_model(result['model'], model_path, model_config['type'])
                        logger.info(f"  {fold_prefix}✓ Saved model: {model_filename}.pt")
                except Exception as e:
                    logger.warning(f"  {fold_prefix}✗ Failed to save model: {e}")
            
            # Calculate metrics
            y_val_for_metrics = y_val
            y_train_for_metrics = y_train

            # Deep learning datasets may filter out invalid/empty SMILES, so we must
            # align y_true with the length/order of predictions returned by predict_pytorch_model().
            if model_config['type'] in ['pytorch_geometric', 'transformer']:
                valid_indices_val = result.get('valid_indices_val', None)
                valid_indices_train = result.get('valid_indices_train', None)

            if valid_indices_val is not None:
                y_val_for_metrics = y_val[valid_indices_val]
            if valid_indices_train is not None:
                y_train_for_metrics = y_train[valid_indices_train]

            ids_val_for_metrics = ids_val
            smiles_val_for_metrics = smiles_val
            if valid_indices_val is not None:
                if ids_val is not None:
                    ids_val_for_metrics = [ids_val[i] for i in valid_indices_val]
                if smiles_val is not None:
                    smiles_val_for_metrics = [smiles_val[i] for i in valid_indices_val]

            val_metrics = calculate_metrics(
                y_val_for_metrics, result['y_pred_val'], result['y_proba_val'],
                config.task, config.ef_percentile
            )
            
            train_metrics = calculate_metrics(
                y_train_for_metrics, result['y_pred_train'], result['y_proba_train'],
                config.task, config.ef_percentile
            )

            # Record CV predictions for the fold (needed for predictions export)
            if cv_prediction_accumulator is not None and result.get('y_proba_val') is not None:
                y_scores = np.asarray(result['y_proba_val'])
                y_trues = np.asarray(y_val_for_metrics)
                rows = []
                for idx in range(len(y_scores)):
                    rows.append({
                        'model': model_key,
                        'seed': seed,
                        'fold': fold_idx,
                        'molecule_id': ids_val_for_metrics[idx] if ids_val_for_metrics is not None else None,
                        'smiles': smiles_val_for_metrics[idx] if smiles_val_for_metrics is not None else None,
                        'true_label': float(y_trues[idx]),
                        'predicted_probability': float(y_scores[idx])
                    })
                if rows:
                    cv_prediction_accumulator.append(pd.DataFrame(rows))
            
            # Store seed-specific results
            seed_result_dict = {
                'fold': fold_idx,
                'model': model_key,
                'model_type': model_type_category,
                'seed': seed,
                **{f'{k}_val': v for k, v in val_metrics.items()},
                **{f'{k}_train': v for k, v in train_metrics.items()}
            }
            seed_results.append(seed_result_dict)
            
            if save_cv_details and cv_save_root is not None and 'model' in result:
                X_data_for_saving = X_train_final if X_train_final is not None else X_train_processed
                mask_to_save = feature_mask if feature_mask is not None else global_feature_mask
                _save_cv_fold_seed_details(
                    cv_save_root,
                    fold_idx,
                    seed,
                    model_key,
                    model_config,
                    model_config.get('type', 'sklearn'),
                    result['model'],
                    X_data_for_saving,
                    scaler,
                    selector,
                    mask_to_save,
                    y_train,
                    smiles_train,
                    ids_train,
                    logger=logger
                )
            
            # GPU memory management: clear cache after training each seed for deep learning models
            if model_type_category == 'deep_learning' and torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Delete model to free memory
                if 'model' in result:
                    del result['model']
                torch.cuda.empty_cache()
            
            # Log metrics for this seed
            if config.task == "classification":
                logger.info(f"  {fold_prefix}✓ {model_key} - Seed {seed} completed")
                logger.info(f"    Val AUC: {val_metrics.get('AUC', 'N/A'):.4f}, PR-AUC: {val_metrics.get('PR_AUC', 'N/A'):.4f}")
                logger.info(f"    Val MCC: {val_metrics.get('MCC', 'N/A'):.4f}, EF{config.ef_percentile:g}%: {val_metrics.get(f'EF{config.ef_percentile:g}%', 'N/A'):.4f}")
            else:
                logger.info(f"  {fold_prefix}✓ {model_key} - Seed {seed} completed")
                logger.info(f"    Val R2: {val_metrics.get('R2', 'N/A'):.4f}, RMSE: {val_metrics.get('RMSE', 'N/A'):.4f}")
        
        # Skip if no seeds succeeded
        if not seed_results:
            logger.warning(f"{fold_prefix}{model_key} failed on all seeds")
            continue
        
        # Aggregate results across seeds (calculate mean and std)
        logger.info(f"\n{fold_prefix}Aggregating results for {model_key} across {len(seed_results)} seeds")
        
        # Get all metric keys (excluding fold, model, model_type, seed)
        metric_keys = [k for k in seed_results[0].keys() if k not in ['fold', 'model', 'model_type', 'seed']]
        
        # Calculate mean and std for each metric
        aggregated_result = {
            'fold': fold_idx,
            'model': model_key,
            'model_type': model_type_category,
            'num_seeds': len(seed_results)
        }
        
        for metric_key in metric_keys:
            values = [r[metric_key] for r in seed_results]
            aggregated_result[f'{metric_key}_mean'] = np.mean(values)
            aggregated_result[f'{metric_key}_std'] = np.std(values)
        
        results.append(aggregated_result)
        
        # Log aggregated metrics
        if config.task == "classification":
            logger.info(f"{fold_prefix}✓ {model_key} aggregated results:")
            logger.info(f"  Val AUC: {aggregated_result.get('AUC_val_mean', 'N/A'):.4f} ± {aggregated_result.get('AUC_val_std', 'N/A'):.4f}")
            logger.info(f"  Val PR-AUC: {aggregated_result.get('PR_AUC_val_mean', 'N/A'):.4f} ± {aggregated_result.get('PR_AUC_val_std', 'N/A'):.4f}")
            logger.info(f"  Val MCC: {aggregated_result.get('MCC_val_mean', 'N/A'):.4f} ± {aggregated_result.get('MCC_val_std', 'N/A'):.4f}")
            logger.info(f"  Val EF{config.ef_percentile:g}%: {aggregated_result.get(f'EF{config.ef_percentile:g}%_val_mean', 'N/A'):.4f} ± {aggregated_result.get(f'EF{config.ef_percentile:g}%_val_std', 'N/A'):.4f}")
        else:
            logger.info(f"{fold_prefix}✓ {model_key} aggregated results:")
            logger.info(f"  Val R2: {aggregated_result.get('R2_val_mean', 'N/A'):.4f} ± {aggregated_result.get('R2_val_std', 'N/A'):.4f}")
            logger.info(f"  Val RMSE: {aggregated_result.get('RMSE_val_mean', 'N/A'):.4f} ± {aggregated_result.get('RMSE_val_std', 'N/A'):.4f}")
    
    return results

# --------- Main Training Pipeline ---------
def main_pipeline(config: QSARConfig, X: np.ndarray, y: np.ndarray, 
                 feature_df: pd.DataFrame, smiles_list: Optional[List[str]] = None,
                 ids: Optional[List[str]] = None,
                 logger: logging.Logger = None,
                 split_seed: Optional[int] = None):
    """
    Main training pipeline
    
    Args:
        config: Configuration
        X: Feature matrix
        y: Labels
        feature_df: Feature DataFrame
        smiles_list: SMILES strings (for GAT/ChemBERTa)
        ids: Molecule IDs
        logger: Logger instance
        split_seed: Seed used for Stage 1/2 splits (overrides ``config.seed`` when provided)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    stage_seed = split_seed if split_seed is not None else config.seed
    logger.info(f"Split seed in use: {stage_seed}")

    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.model_selection import train_test_split
    
    result_payload = {
        'external_summary': None,
        'cv_summary': None,
        'external_rows': None,
        'cv_rows': None
    }

    # Setup output directory (already created in main_cli)
    out_dir = Path(config.output_dir)
    
    # Create subdirectories for organized output structure
    subdirs = ['results', 'models', 'feature_processors', 'predictions']
    for subdir in subdirs:
        (out_dir / subdir).mkdir(exist_ok=True)
    
    # Save configuration
    config_dict = config.__dict__.copy()
    # Convert non-serializable types
    config_dict['selected_models'] = config.selected_models
    # Only save multimodal_weights if it exists (for backward compatibility)
    if hasattr(config, 'multimodal_weights'):
        config_dict['multimodal_weights'] = config.multimodal_weights
    with open(out_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    # Build model registry
    input_dim = X.shape[1] if X is not None else None
    registry = build_model_registry(task=config.task, input_dim=input_dim)
    
    logger.info(f"\nAvailable models: {list(registry.keys())}")
    logger.info(f"Selected models: {config.selected_models}")
    
    # Validate selected models
    valid_models = [m for m in config.selected_models if m in registry]
    if not valid_models:
        logger.error("No valid models selected!")
        return
    
    # Use the leakage-safe Dev/External split when folds > 1.
    # Stage 2 (K-Fold on Development) can optionally be disabled via config.run_cv_stage2.
    use_cv = config.folds > 1
    run_stage2_cv = use_cv and getattr(config, "run_cv_stage2", True)
    
    if use_cv:
        logger.info(f"\n{'='*60}")
        logger.info(f"K-Fold Cross-Validation: {config.folds} folds")
        logger.info(f"{'='*60}")
        if not run_stage2_cv:
            logger.info(f"(Stage 2 CV disabled; still using Dev/External Test split)")
    else:
        logger.info(f"\n{'='*60}")
        logger.info("Single Train/Validation Split")
        logger.info(f"{'='*60}")
        logger.info("Using a single 80/20 split for model training and evaluation.")
        logger.info("For cross-validation, set --folds to a value > 1.")
    
    # Train/val split based on specified method
    logger.info(f"\n--- Data Splitting ---")
    logger.info(f"Split method: {config.split_method}")
    
    all_fold_results = []  # Store results from all folds
    external_test_results = []  # Store external test results
    external_split_path: Optional[Path] = None
    
    if use_cv:
        # K-Fold Cross-Validation with two-stage split
        # Stage 1: Split into External Test Set and Development Set
        # Stage 2: Perform K-Fold CV on Development Set
        
        logger.info(f"\n--- Stage 1: External Test Set Split ---")
        logger.info(f"Using {config.split_method} split to create External Test Set")
        
        if config.split_method == "scaffold":
            if smiles_list is None:
                logger.error("SMILES data required for scaffold split!")
                return
            logger.info(f"Splitting data: {int((1 - config.test_size) * 100)}% Development Set, {int(config.test_size * 100)}% External Test Set")
            dev_indices, ext_test_indices = scaffold_split(smiles_list, y, config.test_size, stage_seed)
            
            # Count unique scaffolds
            dev_scaffolds = set([get_scaffold(smiles_list[i]) for i in dev_indices])
            ext_test_scaffolds = set([get_scaffold(smiles_list[i]) for i in ext_test_indices])
            overlap = dev_scaffolds & ext_test_scaffolds
            logger.info(f"Development Set: {len(dev_indices)} samples, {len(dev_scaffolds)} unique scaffolds")
            logger.info(f"External Test Set: {len(ext_test_indices)} samples, {len(ext_test_scaffolds)} unique scaffolds")
            logger.info(f"Scaffold overlap: {len(overlap)} scaffolds")
            
        elif config.split_method == "stratified":
            if config.task == "classification":
                dev_indices, ext_test_indices = train_test_split(
                    np.arange(len(y)), y, 
                    test_size=config.test_size, 
                    random_state=stage_seed, 
                    stratify=y
                )
            else:
                dev_indices, ext_test_indices = train_test_split(
                    np.arange(len(y)), 
                    test_size=config.test_size, 
                    random_state=stage_seed
                )
            logger.info(f"Development Set: {len(dev_indices)} samples")
            logger.info(f"External Test Set: {len(ext_test_indices)} samples")
            
        else:  # random
            dev_indices, ext_test_indices = train_test_split(
                np.arange(len(y)), 
                test_size=config.test_size, 
                random_state=stage_seed
            )
            logger.info(f"Development Set: {len(dev_indices)} samples")
            logger.info(f"External Test Set: {len(ext_test_indices)} samples")
        
        # Prepare Development Set data
        y_dev = y[dev_indices]
        X_dev = X[dev_indices] if X is not None else None
        smiles_dev = [smiles_list[i] for i in dev_indices] if smiles_list else None
        ids_dev = [ids[i] for i in dev_indices] if ids else None
        
        # Prepare External Test Set data
        y_ext_test = y[ext_test_indices]
        X_ext_test = X[ext_test_indices] if X is not None else None
        smiles_ext_test = [smiles_list[i] for i in ext_test_indices] if smiles_list else None
        ids_ext_test = [ids[i] for i in ext_test_indices] if ids else None
        
        # Always apply global feature filtering on Development Set.
        # This ensures Stage 2 (CV) and Stage 3 (final external evaluation) use the same feature set.
        X_ext_test_filtered = None
        if X_dev is not None:
            logger.info(f"\n--- Global Feature Filtering on Development Set ---")
            fp_cols = select_fp_columns(feature_df)
            X_dev_filtered, dev_feature_mask, dev_feature_names_filtered = apply_global_feature_filtering(
                X_dev, config, logger, feature_names=fp_cols
            )
            if X_ext_test is not None and dev_feature_mask is not None:
                X_ext_test_filtered = X_ext_test[:, dev_feature_mask]
            elif X_ext_test is not None:
                X_ext_test_filtered = X_ext_test
        else:
            X_dev_filtered = X_dev
            dev_feature_mask = None
            dev_feature_names_filtered = None
            if X_ext_test is not None:
                X_ext_test_filtered = X_ext_test

        if getattr(config, "save_train_features", False):
            train_dir = out_dir / "data"
            train_dir.mkdir(parents=True, exist_ok=True)
            features_to_save = X_dev_filtered if X_dev_filtered is not None else X_dev
            if features_to_save is not None:
                ids_arr = np.array(ids_dev, dtype=object) if ids_dev else np.array([], dtype=object)
                smiles_arr = np.array(smiles_dev, dtype=object) if smiles_dev else np.array([], dtype=object)
                train_features_path = train_dir / "train_features.npz"
                np.savez_compressed(
                    train_features_path,
                    features=np.asarray(features_to_save, dtype=float),
                    labels=np.asarray(y_dev, dtype=float),
                    ids=ids_arr,
                    smiles=smiles_arr,
                )
                logger.info(f"Saved Development features for AD: {train_features_path}")
            else:
                logger.warning("Could not export Development features for AD because the feature matrix is missing")

        # Persist External Test Set split for downstream SHAP analysis
        split_data_dir = out_dir / "data" / "splits"
        split_data_dir.mkdir(parents=True, exist_ok=True)
        external_split_path = split_data_dir / "external_test.npz"
        np.savez_compressed(
            external_split_path,
            features=X_ext_test_filtered if X_ext_test_filtered is not None else (np.array([], dtype=float) if X_ext_test is None else np.asarray(X_ext_test)),
            labels=y_ext_test,
            ids=np.array(ids_ext_test, dtype=object) if ids_ext_test is not None else np.array([], dtype=object),
            smiles=np.array(smiles_ext_test, dtype=object) if smiles_ext_test is not None else np.array([], dtype=object),
            feature_names=np.array(dev_feature_names_filtered if dev_feature_names_filtered is not None else [], dtype=object)
        )
        logger.info(f"External Test split saved to: {external_split_path}")

        if run_stage2_cv:
            # Stage 2: K-Fold Cross-Validation on Development Set
            logger.info(f"\n--- Stage 2: K-Fold Cross-Validation on Development Set ---")
            logger.info(f"Using {config.folds}-fold CV on Development Set ({len(y_dev)} samples)")
            cv_data_dir = out_dir / "cv_data"
            if config.save_cv_details:
                cv_data_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Detailed CV artifacts will be saved to: {cv_data_dir}")
            else:
                cv_data_dir = None

            if config.cv_split_method == "scaffold":
                logger.info("Using ScaffoldKFold: scaffolds are distributed across folds with greedy allocation")
                kfold = ScaffoldKFold(n_splits=config.folds, shuffle=True, random_state=stage_seed)
            else:  # random
                # Create k-fold splitter
                logger.info("Using standard KFold/StratifiedKFold for cross-validation")
                if config.task == "classification":
                    kfold = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=stage_seed)
                else:
                    kfold = KFold(n_splits=config.folds, shuffle=True, random_state=stage_seed)

            # Materialize folds once so Stage 2 tuning (optional) and Stage 2 training
            # use exactly the same split definitions.
            split_X = smiles_dev if smiles_dev else np.arange(len(y_dev))
            fold_splits = list(kfold.split(split_X, y_dev))

            # Optional Stage 2 hyperparameter tuning (Development Set only)
            if getattr(config, "tune", False):
                import itertools
                import copy as _copy

                tune_metric = config.cv_tune_metric or config.external_test_metric
                minimize_metrics = {"RMSE", "MAE"}
                minimize_tune = tune_metric in minimize_metrics
                tune_read_key = f"{tune_metric}_val_mean"

                tuned_models = []
                best_params_by_model = {}

                # Only tune sklearn/grid-based models (PyTorch tuning would require a different loop).
                for model_key in valid_models:
                    model_cfg = registry.get(model_key, {})
                    if model_cfg.get("type") != "sklearn":
                        continue
                    grid = model_cfg.get("grid", None)
                    if not grid:
                        continue

                    tuned_models.append(model_key)
                    base_params = model_cfg.get("params", {})
                    param_names = list(grid.keys())

                    if config.tune_mode == "grid":
                        # Enumerate cartesian product of grid values.
                        grid_values = [grid[p] for p in param_names]
                        combos = list(itertools.product(*grid_values))
                        if config.tune_iter is not None and len(combos) > config.tune_iter:
                            logger.warning(
                                f"  Tuning {model_key}: grid has {len(combos)} combos; "
                                f"capping to tune_iter={config.tune_iter}"
                            )
                            combos = combos[: config.tune_iter]
                        candidates = [dict(zip(param_names, c)) for c in combos]
                    else:
                        # Randomly sample candidates.
                        rng = np.random.default_rng(stage_seed)
                        candidates = []
                        for _ in range(int(config.tune_iter)):
                            cand = {p: rng.choice(grid[p]) for p in param_names}
                            candidates.append(cand)

                    logger.info(
                        f"\nStage2 tuning: model={model_key}, candidates={len(candidates)}, metric={tune_metric}, mode={config.tune_mode}"
                    )

                    best_score = None
                    best_candidate_params = None

                    for cand_idx, cand_params in enumerate(candidates, start=1):
                        fold_scores = []
                        fold_count = 0

                        for fold_idx, (train_indices, val_indices) in enumerate(fold_splits, start=1):
                            y_train = y_dev[train_indices]
                            y_val = y_dev[val_indices]
                            if X_dev_filtered is not None:
                                X_train = X_dev_filtered[train_indices]
                                X_val = X_dev_filtered[val_indices]
                            else:
                                X_train, X_val = None, None

                            smiles_train = [smiles_dev[i] for i in train_indices] if smiles_dev else None
                            smiles_val = [smiles_dev[i] for i in val_indices] if smiles_dev else None
                            ids_train = [ids_dev[i] for i in train_indices] if ids_dev else None
                            ids_val = [ids_dev[i] for i in val_indices] if ids_dev else None

                            # Override params for this candidate only.
                            merged_params = dict(base_params)
                            merged_params.update(cand_params)
                            registry_candidate = dict(registry)
                            registry_candidate[model_key] = dict(registry[model_key])
                            registry_candidate[model_key]["params"] = merged_params

                            config_for_model = _copy.copy(config)
                            config_for_model.selected_models = [model_key]

                            fold_results = train_single_fold(
                                config=config_for_model,
                                X_train=X_train,
                                y_train=y_train,
                                X_val=X_val,
                                y_val=y_val,
                                smiles_train=smiles_train,
                                smiles_val=smiles_val,
                                ids_train=ids_train,
                                ids_val=ids_val,
                                registry=registry_candidate,
                                fold_idx=fold_idx,
                                save_models=False,
                                global_feature_mask=dev_feature_mask,
                                logger=logger,
                            )

                            if not fold_results:
                                continue
                            val_score = fold_results[0].get(tune_read_key, np.nan)
                            if val_score is None:
                                continue
                            val_score_f = float(val_score)
                            if np.isnan(val_score_f) or np.isinf(val_score_f):
                                continue

                            # Convert to "maximize" space.
                            score = -val_score_f if minimize_tune else val_score_f
                            fold_scores.append(score)
                            fold_count += 1

                        if not fold_scores:
                            continue

                        avg_score = float(np.mean(fold_scores))
                        if best_score is None or avg_score > best_score:
                            best_score = avg_score
                            best_candidate_params = cand_params

                        if cand_idx == 1 or cand_idx % max(1, int(len(candidates) / 5)) == 0:
                            logger.info(
                                f"  {model_key} cand {cand_idx}/{len(candidates)} avg({tune_metric})={'-' if minimize_tune else ''}{avg_score:.6f}"
                            )

                    if best_candidate_params is not None:
                        merged_params_final = dict(base_params)
                        merged_params_final.update(best_candidate_params)
                        registry[model_key]["params"] = merged_params_final
                        best_params_by_model[model_key] = best_candidate_params
                        logger.info(f"  ✓ Stage2 best params for {model_key}: {best_candidate_params}")
                    else:
                        logger.warning(f"  ✗ Stage2 tuning found no valid candidate for {model_key}; keeping defaults.")

                if best_params_by_model:
                    # Persist tuning result for transparency (Development Set only).
                    tuned_path = out_dir / "results" / "tuned_stage2_params.json"
                    with open(tuned_path, "w") as f:
                        json.dump(best_params_by_model, f, indent=2, default=str)
                    logger.info(f"\nStage2 tuning complete. Saved: {tuned_path}")

            # Iterate over folds on Development Set (now with tuned registry params if enabled)
            for fold_idx, (train_indices, val_indices) in enumerate(fold_splits, start=1):
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing Fold {fold_idx}/{config.folds}")
                logger.info(f"{'='*60}")

                # Apply split indices to Development Set data
                y_train, y_val = y_dev[train_indices], y_dev[val_indices]
                if X_dev_filtered is not None:
                    X_train, X_val = X_dev_filtered[train_indices], X_dev_filtered[val_indices]
                else:
                    X_train, X_val = None, None

                if smiles_dev:
                    smiles_train = [smiles_dev[i] for i in train_indices]
                    smiles_val = [smiles_dev[i] for i in val_indices]
                else:
                    smiles_train, smiles_val = None, None

                if ids_dev:
                    ids_train = [ids_dev[i] for i in train_indices]
                    ids_val = [ids_dev[i] for i in val_indices]
                else:
                    ids_train, ids_val = None, None

                # Train models on this fold
                cv_prediction_frames = []
                fold_results = train_single_fold(
                    config=config,
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    smiles_train=smiles_train, smiles_val=smiles_val,
                    ids_train=ids_train, ids_val=ids_val,
                    registry=registry,
                    fold_idx=fold_idx,
                    save_models=False,
                    global_feature_mask=dev_feature_mask,
                    save_cv_details=config.save_cv_details,
                    cv_save_root=cv_data_dir,
                    cv_prediction_accumulator=cv_prediction_frames,
                    logger=logger
                )
                if cv_prediction_frames:
                    cv_fold_preds = pd.concat(cv_prediction_frames, ignore_index=True, sort=False)
                    cv_path = out_dir / "results" / f"cv_predictions_fold_{fold_idx}.csv"
                    cv_fold_preds.to_csv(cv_path, index=False)
                    logger.info(f"  ✓ Saved CV predictions for fold {fold_idx}: {cv_path}")

                all_fold_results.extend(fold_results)

                # GPU memory management: clear cache after each fold
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if logger:
                        logger.debug(f"Fold {fold_idx}: GPU cache cleared")
        else:
            logger.info("\n--- Stage 2: K-Fold Cross-Validation skipped (run_cv_stage2=False) ---")
        
        # Calculate average metrics across folds (Stage 2 only)
        cv_summary_df = None
        summary_df = None
        if run_stage2_cv:
            logger.info(f"\n{'='*60}")
            logger.info("Cross-Validation Results Summary (Development Set)")
            logger.info(f"{'='*60}")

            # Aggregate results by model
            model_cv_results = {}
            for result in all_fold_results:
                model_name = result['model']
                if model_name not in model_cv_results:
                    model_cv_results[model_name] = []
                model_cv_results[model_name].append(result)

            # Calculate mean and std for each metric
            cv_summary = []
            for model_name, fold_results_list in model_cv_results.items():
                summary_row = {'model': model_name, 'model_type': fold_results_list[0]['model_type']}

                # Check if results are already aggregated (contain _mean and _std suffixes)
                # This happens when train_single_fold has already aggregated across seeds
                if any('_mean' in k for k in fold_results_list[0].keys()):
                    metric_keys = [k for k in fold_results_list[0].keys() if k.endswith('_mean')]

                    for metric_key in metric_keys:
                        base_metric = metric_key.replace('_mean', '')
                        std_key = f'{base_metric}_std'
                        # Combine fold-level (mean, std, n_seeds) into a pooled estimate
                        # across all fold x seed runs:
                        #   mu = sum(n_i * mu_i) / N
                        #   var = sum(n_i * (sigma_i^2 + (mu_i - mu)^2)) / N
                        fold_stats = []
                        for r in fold_results_list:
                            mean_val = r.get(metric_key, None)
                            if mean_val is None:
                                continue
                            try:
                                mean_f = float(mean_val)
                            except Exception:
                                continue
                            if np.isnan(mean_f) or np.isinf(mean_f):
                                continue
                            std_val = r.get(std_key, 0.0)
                            try:
                                std_f = float(std_val)
                                if np.isnan(std_f) or np.isinf(std_f):
                                    std_f = 0.0
                            except Exception:
                                std_f = 0.0
                            n_i = int(r.get('num_seeds', 1) or 1)
                            fold_stats.append((mean_f, std_f, n_i))

                        if not fold_stats:
                            summary_row[f'{base_metric}_mean'] = np.nan
                            summary_row[f'{base_metric}_std'] = np.nan
                            continue

                        total_n = float(sum(n for _, _, n in fold_stats))
                        pooled_mean = sum(mu * n for mu, _, n in fold_stats) / total_n
                        pooled_var = sum(n * ((sigma ** 2) + ((mu - pooled_mean) ** 2)) for mu, sigma, n in fold_stats) / total_n
                        summary_row[f'{base_metric}_mean'] = pooled_mean
                        summary_row[f'{base_metric}_std'] = float(np.sqrt(max(pooled_var, 0.0)))
                else:
                    exclude_keys = ['fold', 'model', 'model_type', 'num_seeds', 'seed']
                    metric_keys = [k for k in fold_results_list[0].keys() if k not in exclude_keys]

                    for metric_key in metric_keys:
                        values = [r[metric_key] for r in fold_results_list]
                        summary_row[f'{metric_key}_mean'] = np.mean(values)
                        summary_row[f'{metric_key}_std'] = np.std(values)

                cv_summary.append(summary_row)

                # Log summary
                if config.task == "classification":
                    logger.info(f"\n{model_name}:")
                    logger.info(f"  AUC: {summary_row.get('AUC_val_mean', 'N/A'):.4f} ± {summary_row.get('AUC_val_std', 'N/A'):.4f}")
                    logger.info(f"  PR-AUC: {summary_row.get('PR_AUC_val_mean', 'N/A'):.4f} ± {summary_row.get('PR_AUC_val_std', 'N/A'):.4f}")
                    logger.info(f"  MCC: {summary_row.get('MCC_val_mean', 'N/A'):.4f} ± {summary_row.get('MCC_val_std', 'N/A'):.4f}")
                else:
                    logger.info(f"\n{model_name}:")
                    logger.info(f"  R2: {summary_row.get('R2_val_mean', 'N/A'):.4f} ± {summary_row.get('R2_val_std', 'N/A'):.4f}")
                    logger.info(f"  RMSE: {summary_row.get('RMSE_val_mean', 'N/A'):.4f} ± {summary_row.get('RMSE_val_std', 'N/A'):.4f}")

            # Save fold-wise results
            fold_results_df = pd.DataFrame(all_fold_results)
            fold_results_df.to_csv(out_dir / "results" / "fold_results.csv", index=False)

            # Save CV summary
            cv_summary_df = pd.DataFrame(cv_summary)
            cv_summary_df.to_csv(out_dir / "results" / "cv_summary.csv", index=False)
            result_payload['cv_summary'] = cv_summary_df
            result_payload['cv_rows'] = fold_results_df
        
        # Stage 3: Evaluate ALL models on External Test Set
        logger.info(f"\n{'='*60}")
        logger.info("Stage 3: External Test Set Evaluation for All Models (Multi-Seed)")
        logger.info(f"{'='*60}")
        logger.info(f"Training all models on full Development Set ({len(y_dev)} samples)")
        logger.info(f"Evaluating all models on External Test Set ({len(y_ext_test)} samples)")
        logger.info(f"Using {len(config.seeds)} seeds: {config.seeds}")
        logger.info(f"Best model will be selected based on: {config.external_test_metric}")
        full_dev_models_dir = out_dir / "models" / "full_dev"
        full_dev_models_dir.mkdir(parents=True, exist_ok=True)
        feature_mask_path = (out_dir / "feature_processors" / "feature_mask.npy") if dev_feature_mask is not None else None
        if run_stage2_cv:
            logger.info(
                "Split integrity: External indices are excluded from Stage 2 (CV/tuning); "
                "feature masks are fit on Development only, then applied to External (no refit on External)."
            )
            logger.info(
                "CV vs External: External metrics can exceed mean CV validation performance due to "
                "split variance, distribution shift, smaller External n, or Stage-2 hyperparameter "
                "choice — this is not, by itself, evidence of label leakage."
            )
        
        # Apply feature processing to Development Set and External Test Set
        # Use the same global filtering applied to Development Set in Stage 2
        # This ensures consistency across all evaluation stages
        if X_dev is not None and X_ext_test is not None:
            logger.info(f"\nApplying feature processing to Development Set and External Test Set...")
            # Apply the same feature mask from Stage 2
            X_dev_for_ext_test = X_dev[:, dev_feature_mask] if dev_feature_mask is not None else X_dev
            X_ext_test_for_ext_test = X_ext_test[:, dev_feature_mask] if dev_feature_mask is not None else X_ext_test
            
            logger.info(f"Using same features as K-Fold CV: {X_dev_for_ext_test.shape[1]} features")
        else:
            X_dev_for_ext_test = X_dev
            X_ext_test_for_ext_test = X_ext_test
        
        all_ext_test_results = []  # Store all seed-specific results
        all_ext_test_predictions = []  # Store all predictions
        trained_models = {}  # Store trained model objects for saving (best seed per model)
        model_best_seed = {}  # Track best seed used for each model object in trained_models
        best_traditional = None
        best_deep_learning = None

        # Metric direction for model selection (used for both best-seed tracking and model ranking)
        lower_is_better = {'RMSE', 'MAE'}
        higher_is_better = {'R2', 'AUC', 'PR_AUC', 'MCC', 'ACC', 'F1'}
        metric_name = str(config.external_test_metric)
        lower_better = metric_name in lower_is_better
        higher_better = metric_name in higher_is_better
        if not lower_better and not higher_better:
            lower_better = False  # default: higher is better
        
        # Evaluate each model on External Test Set with multiple seeds
        for model_key in valid_models:
            model_config = registry[model_key]
            model_type_category = 'deep_learning' if model_config['type'] in ['pytorch', 'pytorch_geometric', 'transformer'] else 'traditional'
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Model: {model_key}")
            logger.info(f"{'='*60}")
            
            # Multi-seed training and evaluation loop
            model_seed_results = []  # Store results for each seed
            best_seed_model_obj = None
            best_seed_value = None
            best_seed_metric_val = None
            last_success_model_obj = None
            last_success_seed = None
            
            for seed_idx, seed in enumerate(config.seeds):
                logger.info(f"\n--- Seed {seed_idx+1}/{len(config.seeds)} (seed={seed}) ---")

                # Set random seed for reproducibility
                set_all_seeds(seed, logger)
                scaler = None
                
                # Prepare data based on model type
                # Apply standardization only for models that need it (gradient-based)
                if model_config['type'] in ['sklearn']:  # Traditional models
                    model_preprocessing_type = get_model_type(model_key)
                    if X_dev_for_ext_test is not None and X_ext_test_for_ext_test is not None:
                        if model_preprocessing_type == 'gradient_based':
                            # Apply standardization for gradient-based models
                            from sklearn.preprocessing import StandardScaler
                            scaler = StandardScaler()
                            X_train_final = scaler.fit_transform(X_dev_for_ext_test)
                            X_test_final = scaler.transform(X_ext_test_for_ext_test)
                            logger.info(f"  Standardization applied for {model_key}")
                        else:
                            # No standardization for tree-based models
                            X_train_final = X_dev_for_ext_test
                            X_test_final = X_ext_test_for_ext_test
                            logger.info(f"  Standardization skipped for {model_key} (tree-based model)")
                    else:
                        X_train_final = None
                        X_test_final = None
                else:
                    # Deep learning models (MLP, GAT, ChemBERTa)
                    if model_config['type'] == 'pytorch':  # MLP
                        # MLP needs standardization
                        if X_dev_for_ext_test is not None and X_ext_test_for_ext_test is not None:
                            from sklearn.preprocessing import StandardScaler
                            scaler = StandardScaler()
                            X_train_final = scaler.fit_transform(X_dev_for_ext_test)
                            X_test_final = scaler.transform(X_ext_test_for_ext_test)
                            logger.info(f"  Standardization applied for {model_key}")
                        else:
                            X_train_final = None
                            X_test_final = None
                    else:  # GAT, ChemBERTa (use SMILES, no feature processing needed)
                        X_train_final = None
                        X_test_final = None
                
                # Train on full Development Set
                model_result = train_model_wrapper(
                    model_key=model_key,
                    model_config=model_config,
                    X_train=X_train_final, y_train=y_dev,
                    X_val=None, y_val=None,  # No validation set when training on full Dev Set
                    config=config, task=config.task,
                    smiles_train=smiles_dev, smiles_val=None,
                    logger=logger,
                    sklearn_random_state=seed,
                )
                
                if model_result is None:
                    logger.warning(f"  Failed to train {model_key} with seed {seed}")
                    continue
                
                last_success_model_obj = model_result['model']
                last_success_seed = seed
                
                logger.info(f"  ✓ {model_key} trained on full Development Set (seed={seed})")
                
                # Evaluate on External Test Set
                logger.info(f"  --- Evaluating on External Test Set (seed={seed}) ---")
                
                # Some deep learning datasets (GAT/ChemBERTa) filter out invalid/empty SMILES,
                # so we must align y_true (and ids/smiles) to the prediction outputs.
                y_ext_test_for_metrics = y_ext_test
                ids_ext_test_for_metrics = ids_ext_test
                smiles_ext_test_for_metrics = smiles_ext_test

                # Make predictions on External Test Set
                if model_config['type'] in ['pytorch_geometric', 'transformer', 'pytorch']:
                    # For GAT, ChemBERTa, and MLP, we need to create a dataset/data loader
                    if model_config['type'] == 'pytorch_geometric':
                        test_dataset = MoleculeDataset(smiles_ext_test, y_ext_test, logger=logger)
                        valid_indices = getattr(test_dataset, "valid_indices", None)
                        if valid_indices is not None:
                            y_ext_test_for_metrics = y_ext_test[valid_indices]
                            if ids_ext_test is not None:
                                ids_ext_test_for_metrics = [ids_ext_test[i] for i in valid_indices]
                            if smiles_ext_test is not None:
                                smiles_ext_test_for_metrics = [smiles_ext_test[i] for i in valid_indices]
                        test_loader = GeometricDataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
                    elif model_config['type'] == 'transformer':
                        model = model_result['model']
                        test_dataset = ChemBERTaDataset(smiles_ext_test, y_ext_test, model.tokenizer)
                        valid_indices = getattr(test_dataset, "valid_indices", None)
                        if valid_indices is not None:
                            y_ext_test_for_metrics = y_ext_test[valid_indices]
                            if ids_ext_test is not None:
                                ids_ext_test_for_metrics = [ids_ext_test[i] for i in valid_indices]
                            if smiles_ext_test is not None:
                                smiles_ext_test_for_metrics = [smiles_ext_test[i] for i in valid_indices]
                        max_length_test = getattr(test_dataset, "max_length", 128)
                        test_loader = DataLoader(
                            test_dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            collate_fn=lambda b, ml=max_length_test: chemberta_collate_fn(b, max_length=ml)
                        )
                    else:  # pytorch (MLP)
                        test_dataset = TensorDataset(torch.tensor(X_test_final, dtype=torch.float32), 
                                                    torch.tensor(y_ext_test, dtype=torch.float32))
                        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
                    
                    y_pred_ext_test, y_proba_ext_test = predict_pytorch_model(
                        model_result['model'], test_loader, config.task
                    )
                else:
                    # For traditional models
                    if hasattr(model_result['model'], 'predict_proba'):
                        y_proba_ext_test = model_result['model'].predict_proba(X_test_final)[:, 1]
                    elif hasattr(model_result['model'], 'decision_function'):
                        y_proba_ext_test = model_result['model'].decision_function(X_test_final)
                        if config.task == "classification":
                            y_proba_ext_test = 1.0 / (1.0 + np.exp(-y_proba_ext_test))
                    else:
                        y_proba_ext_test = model_result['model'].predict(X_test_final)
                    
                    y_pred_ext_test = (y_proba_ext_test >= 0.5).astype(int) if config.task == "classification" else y_proba_ext_test
                
                # Calculate metrics on External Test Set
                ext_test_metrics = calculate_metrics(
                    y_ext_test_for_metrics, y_pred_ext_test, y_proba_ext_test,
                    config.task, config.ef_percentile
                )
                
                # Store seed-specific result
                seed_result = {
                    'model': model_key,
                    'model_type': model_type_category,
                    'seed': seed,
                    'n_ext_test_samples': len(y_ext_test_for_metrics) if y_ext_test_for_metrics is not None else 0,
                    **ext_test_metrics  # All metrics without _ext_test suffix
                }
                model_seed_results.append(seed_result)

                # Track best seed model object for this model according to external_test_metric.
                metric_val = seed_result.get(config.external_test_metric, None)
                metric_val_f = None
                if metric_val is not None:
                    try:
                        metric_val_f = float(metric_val)
                        if np.isnan(metric_val_f) or np.isinf(metric_val_f):
                            metric_val_f = None
                    except Exception:
                        metric_val_f = None
                if metric_val_f is not None:
                    if best_seed_metric_val is None:
                        best_seed_metric_val = metric_val_f
                        best_seed_model_obj = model_result['model']
                        best_seed_value = seed
                    else:
                        is_better = (metric_val_f < best_seed_metric_val) if lower_better else (metric_val_f > best_seed_metric_val)
                        if is_better:
                            best_seed_metric_val = metric_val_f
                            best_seed_model_obj = model_result['model']
                            best_seed_value = seed
                
                # Store predictions for this seed
                pred_df = pd.DataFrame({
                    'model': model_key,
                    'seed': seed,
                    'id': ids_ext_test_for_metrics,
                    'smiles': smiles_ext_test_for_metrics,
                    'true_label': y_ext_test_for_metrics,
                    'predicted_label': y_pred_ext_test,
                    'predicted_probability': y_proba_ext_test
                })
                all_ext_test_predictions.append(pred_df)
                
                # Log External Test Set results for this seed
                if config.task == "classification":
                    logger.info(f"  Seed {seed} Results:")
                    logger.info(f"    AUC: {ext_test_metrics.get('AUC', 'N/A'):.4f}")
                    logger.info(f"    PR-AUC: {ext_test_metrics.get('PR_AUC', 'N/A'):.4f}")
                    logger.info(f"    MCC: {ext_test_metrics.get('MCC', 'N/A'):.4f}")
                    logger.info(f"    ACC: {ext_test_metrics.get('ACC', 'N/A'):.4f}")
                    logger.info(f"    F1: {ext_test_metrics.get('F1', 'N/A'):.4f}")
                    logger.info(f"    EF{config.ef_percentile:g}%: {ext_test_metrics.get(f'EF{config.ef_percentile:g}%', 'N/A'):.4f}")
                else:
                    logger.info(f"  Seed {seed} Results:")
                    logger.info(f"    R2: {ext_test_metrics.get('R2', 'N/A'):.4f}")
                    logger.info(f"    RMSE: {ext_test_metrics.get('RMSE', 'N/A'):.4f}")
                    logger.info(f"    MAE: {ext_test_metrics.get('MAE', 'N/A'):.4f}")
                
                _save_full_dev_model_artifacts(
                    root_dir=full_dev_models_dir,
                    model_key=model_key,
                    seed=seed,
                    model_config=model_config,
                    model_type=model_config.get('type', 'sklearn'),
                    model_obj=model_result['model'],
                    scaler=scaler,
                    task=config.task,
                    external_data_path=external_split_path,
                    feature_mask_path=feature_mask_path,
                    feature_names=dev_feature_names_filtered,
                    input_dim=X_dev_for_ext_test.shape[1] if X_dev_for_ext_test is not None else None,
                    logger=logger
                )

                # GPU memory management: clear cache after each seed
                if model_type_category == 'deep_learning' and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Skip if no seeds succeeded
            if not model_seed_results:
                logger.warning(f"{model_key} failed on all seeds")
                continue

            # Persist best-seed model object for this model.
            # Fallback to last successful model when all per-seed metric values are invalid.
            trained_models[model_key] = best_seed_model_obj if best_seed_model_obj is not None else last_success_model_obj
            model_best_seed[model_key] = best_seed_value if best_seed_value is not None else last_success_seed
            
            # Add all seed results to overall results
            all_ext_test_results.extend(model_seed_results)
            
            # Calculate mean and std across seeds
            logger.info(f"\n--- {model_key} Multi-Seed Summary ---")
            metric_keys = [k for k in model_seed_results[0].keys() if k not in ['model', 'model_type', 'seed', 'n_ext_test_samples']]
            
            for metric_key in metric_keys:
                values = [r[metric_key] for r in model_seed_results]
                mean_val = np.mean(values)
                std_val = np.std(values)
                logger.info(f"  {metric_key}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Save all External Test Set results (including individual seed results)
        ext_test_results_df = pd.DataFrame(all_ext_test_results)
        ext_test_results_df.to_csv(out_dir / "results" / "external_test_results.csv", index=False)
        
        # Save all predictions
        ext_test_predictions_df = pd.concat(all_ext_test_predictions, ignore_index=True)
        ext_test_predictions_df.to_csv(out_dir / "predictions" / "external_test_predictions.csv", index=False)

        # Save label/predictions summary for visualization scripts
        id_cols = ['id', 'smiles', 'true_label']
        pivot = (
            ext_test_predictions_df
            .groupby(id_cols + ['model'], dropna=False)['predicted_probability']
            .mean()
            .reset_index()
        )
        pivot_wide = pivot.pivot_table(
            index=id_cols,
            columns='model',
            values='predicted_probability'
        ).reset_index()
        pivot_wide.columns.name = None
        score_cols = [c for c in pivot_wide.columns if c not in id_cols]
        pivot_wide = pivot_wide.rename(columns={c: f"{c}_score" for c in score_cols})
        labels_path = out_dir / "predictions" / f"external_test_predictions_seed_{split_seed}.csv"
        pivot_wide['split_seed'] = split_seed
        pivot_wide = pivot_wide[['id', 'smiles', 'true_label', 'split_seed'] + [c for c in pivot_wide.columns if c.endswith('_score')]]
        pivot_wide.rename(columns={'id': 'molecule_id'}, inplace=True)
        pivot_wide.to_csv(labels_path, index=False)
        logger.info(f"  ✓ Saved per-seed predictions: {labels_path}")
        
        # Create summary with mean ± std across seeds
        summary_results = []
        for model_name in [r['model'] for r in all_ext_test_results]:
            model_seed_results_list = [r for r in all_ext_test_results if r['model'] == model_name]
            if not model_seed_results_list:
                continue
            
            summary_row = {'model': model_name, 'model_type': model_seed_results_list[0]['model_type'], 'num_seeds': len(model_seed_results_list)}
            metric_keys = [k for k in model_seed_results_list[0].keys() if k not in ['model', 'model_type', 'seed', 'n_ext_test_samples']]
            
            for metric_key in metric_keys:
                values = [r[metric_key] for r in model_seed_results_list]
                summary_row[f'{metric_key}_mean'] = np.mean(values)
                summary_row[f'{metric_key}_std'] = np.std(values)
            
            summary_results.append(summary_row)
        
        # Save summary results
        summary_df = pd.DataFrame(summary_results)
        summary_df.to_csv(out_dir / "results" / "external_test_summary.csv", index=False)
        result_payload['external_summary'] = summary_df
        result_payload['external_rows'] = pd.DataFrame(all_ext_test_results)
        
        # Find best models based on external test set metric (use mean across seeds)
        logger.info(f"\n{'='*60}")
        logger.info("External Test Set Results Summary")
        logger.info(f"{'='*60}")
        
        # Get metric keys for external test set summary table
        ext_test_metric_key = f'{config.external_test_metric}_mean'
        ext_test_metric_std_key = f'{config.external_test_metric}_std'

        # Separate traditional and deep learning models based on mean performance
        traditional_summary = [r for r in summary_results if r['model_type'] == 'traditional']
        deep_learning_summary = [r for r in summary_results if r['model_type'] == 'deep_learning']
        
        best_traditional = None
        best_deep_learning = None
        
        def _select_best(rows: List[Dict[str, Any]]):
            best_row = None
            best_val = None
            for r in rows:
                if ext_test_metric_key not in r:
                    continue
                val = r.get(ext_test_metric_key)
                if val is None:
                    continue
                try:
                    val_f = float(val)
                except Exception:
                    continue
                if np.isnan(val_f) or np.isinf(val_f):
                    continue
                if best_row is None:
                    best_row = r
                    best_val = val_f
                    continue
                if lower_better:
                    if val_f < best_val:
                        best_row = r
                        best_val = val_f
                else:
                    if val_f > best_val:
                        best_row = r
                        best_val = val_f
            return best_row

        def _format_metric_with_std(row: Dict[str, Any]) -> str:
            mean_v = row.get(ext_test_metric_key, np.nan)
            std_v = row.get(ext_test_metric_std_key, np.nan)
            try:
                mean_f = float(mean_v)
            except Exception:
                return "N/A"
            if np.isnan(mean_f) or np.isinf(mean_f):
                return "N/A"
            try:
                std_f = float(std_v)
                if np.isnan(std_f) or np.isinf(std_f):
                    return f"{mean_f:.4f}"
                return f"{mean_f:.4f} ± {std_f:.4f}"
            except Exception:
                return f"{mean_f:.4f}"

        if traditional_summary:
            best_traditional = _select_best(traditional_summary)
            if best_traditional is not None:
                logger.info(f"\nBest Traditional Model: {best_traditional['model']} ({config.external_test_metric}={_format_metric_with_std(best_traditional)})")
            else:
                logger.warning("\nBest Traditional Model selection failed (no valid metric values).")
        
        if deep_learning_summary:
            best_deep_learning = _select_best(deep_learning_summary)
            if best_deep_learning is not None:
                logger.info(f"Best Deep Learning Model: {best_deep_learning['model']} ({config.external_test_metric}={_format_metric_with_std(best_deep_learning)})")
            else:
                logger.warning("Best Deep Learning Model selection failed (no valid metric values).")
        
        # Save best models
        if best_traditional or best_deep_learning:
            best_model_dir = out_dir / "models" / "best_models"
            best_model_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"\n{'='*60}")
            logger.info("Saving Best Models")
            logger.info(f"{'='*60}")
            
            # Save best traditional model
            if best_traditional is not None:
                model_key = best_traditional['model']
                if model_key in trained_models:
                    model_path = best_model_dir / f"best_traditional_{model_key}"
                    _save_sklearn_model(trained_models[model_key], model_path)
                    logger.info(
                        f"  ✓ Saved best traditional model: {model_key} "
                        f"(seed={model_best_seed.get(model_key, 'N/A')}, "
                        f"{config.external_test_metric}={_format_metric_with_std(best_traditional)})"
                    )
                else:
                    logger.warning(f"  ✗ Model object not found for {model_key}")
            
            # Save best deep learning model
            if best_deep_learning is not None:
                model_key = best_deep_learning['model']
                if model_key in trained_models:
                    model_path = best_model_dir / f"best_deep_learning_{model_key}"
                    model_config = registry.get(model_key)
                    if model_config:
                        model_type = model_config.get('type', 'pytorch')
                        _save_pytorch_model(trained_models[model_key], model_path, model_type)
                        logger.info(
                            f"  ✓ Saved best deep learning model: {model_key} "
                            f"(seed={model_best_seed.get(model_key, 'N/A')}, "
                            f"{config.external_test_metric}={_format_metric_with_std(best_deep_learning)})"
                        )
                else:
                    logger.warning(f"  ✗ Model object not found for {model_key}")
            
            # If one of the branches is missing, save the overall best model as well
            if best_traditional is None or best_deep_learning is None:
                logger.info(f"\nNote: One model type branch is missing. Saving overall best model as backup.")
                # Use the mean-over-seeds summary table (it contains *_mean keys).
                overall_best = _select_best(summary_results)
                if overall_best is None:
                    logger.warning("Overall best model selection failed (no valid metric values). Skipping save.")
                    overall_best = {}
                model_key = overall_best.get('model')
                if model_key in trained_models:
                    model_path = best_model_dir / f"best_overall_{model_key}"
                    model_config = registry.get(model_key)
                    if model_config:
                        model_type = model_config.get('type', 'pytorch')
                        if model_type in ['sklearn']:
                            _save_sklearn_model(trained_models[model_key], model_path)
                        else:
                            _save_pytorch_model(trained_models[model_key], model_path, model_type)
                        logger.info(
                            f"  ✓ Saved overall best model: {model_key} "
                            f"(seed={model_best_seed.get(model_key, 'N/A')}, "
                            f"{config.external_test_metric}={_format_metric_with_std(overall_best)})"
                        )
        
        # Save feature processors
        # Save the global feature mask and feature names from Stage 2
        if dev_feature_mask is not None:
            processor_dir = out_dir / "feature_processors"
            # Save feature mask as numpy array
            np.save(processor_dir / "feature_mask.npy", dev_feature_mask)
            if logger:
                logger.info(f"  ✓ Saved global feature mask: {processor_dir / 'feature_mask.npy'}")
            # Save feature names if available
            if dev_feature_names_filtered is not None:
                with open(processor_dir / "feature_names.json", 'w') as f:
                    json.dump(dev_feature_names_filtered, f)
                if logger:
                    logger.info(f"  ✓ Saved filtered feature names: {processor_dir / 'feature_names.json'}")
        
        logger.info(f"\nExternal Test Set results saved to:")
        logger.info(f"  - {out_dir / 'results' / 'external_test_results.csv'} (metrics for all models and seeds)")
        logger.info(f"  - {out_dir / 'results' / 'external_test_summary.csv'} (mean ± std across seeds)")
        logger.info(f"  - {out_dir / 'predictions' / 'external_test_predictions.csv'} (predictions for all models and seeds)")
        logger.info(f"Full Development models (all seeds) saved to: {full_dev_models_dir}")
        if external_split_path is not None:
            logger.info(f"External Test split saved for SHAP analysis: {external_split_path}")

    else:
        # Single train/validation split
        logger.info(f"{'='*60}")
        logger.info("Single Split Mode")
        logger.info(f"{'='*60}")
        
        # Scaffold-based split using SMILES
        if config.split_method == "scaffold":
            if smiles_list is None:
                logger.error("SMILES data required for scaffold split!")
                return
            
            logger.info("Using scaffold-based split to ensure test set contains different molecular scaffolds")
            train_indices, val_indices = scaffold_split(smiles_list, y, config.test_size, stage_seed)
            
            # Apply split indices to all data
            y_train, y_val = y[train_indices], y[val_indices]
            if X is not None:
                X_train, X_val = X[train_indices], X[val_indices]
            else:
                X_train, X_val = None, None
            smiles_train = [smiles_list[i] for i in train_indices]
            smiles_val = [smiles_list[i] for i in val_indices]
            if ids:
                ids_train = [ids[i] for i in train_indices]
                ids_val = [ids[i] for i in val_indices]
            
            # Count unique scaffolds in train and val sets
            train_scaffolds = set([get_scaffold(s) for s in smiles_train])
            val_scaffolds = set([get_scaffold(s) for s in smiles_val])
            overlap = train_scaffolds & val_scaffolds
            logger.info(f"Train scaffolds: {len(train_scaffolds)}, Val scaffolds: {len(val_scaffolds)}, Overlap: {len(overlap)}")
        
        elif config.split_method == "stratified":
            # Stratified split for classification, random for regression
            logger.info("Using stratified split")
            if config.task == "classification":
                train_indices, val_indices = train_test_split(np.arange(len(y)), test_size=config.test_size, 
                                                             random_state=stage_seed, stratify=y)
            else:
                train_indices, val_indices = train_test_split(np.arange(len(y)), test_size=config.test_size, 
                                                             random_state=stage_seed)
            
            # Apply split to all data
            y_train, y_val = y[train_indices], y[val_indices]
            if X is not None:
                X_train, X_val = X[train_indices], X[val_indices]
            else:
                X_train, X_val = None, None
            if smiles_list:
                smiles_train = [smiles_list[i] for i in train_indices]
                smiles_val = [smiles_list[i] for i in val_indices]
            if ids:
                ids_train = [ids[i] for i in train_indices]
                ids_val = [ids[i] for i in val_indices]
        
        else:  # random
            # Random split
            logger.info("Using random split")
            train_indices, val_indices = train_test_split(np.arange(len(y)), y, test_size=config.test_size, 
                                                         random_state=stage_seed)
            
            # Apply split to all data
            y_train, y_val = y[train_indices], y[val_indices]
            if X is not None:
                X_train, X_val = X[train_indices], X[val_indices]
            else:
                X_train, X_val = None, None
            if smiles_list:
                smiles_train = [smiles_list[i] for i in train_indices]
                smiles_val = [smiles_list[i] for i in val_indices]
            if ids:
                ids_train = [ids[i] for i in train_indices]
                ids_val = [ids[i] for i in val_indices]
        
        # Train models on single split
        # Enable model saving in single split mode
        model_save_dir = out_dir / "models" / "seed_models"
        model_save_dir.mkdir(parents=True, exist_ok=True)
        
        all_fold_results = train_single_fold(
            config=config,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            smiles_train=smiles_train, smiles_val=smiles_val,
            ids_train=ids_train, ids_val=ids_val,
            registry=registry,
            fold_idx=None,  # Single split, no fold index
            save_models=True,
            model_save_dir=model_save_dir,
            logger=logger
        )
        
        # Save results
        summary_df = pd.DataFrame(all_fold_results)
        summary_df.to_csv(out_dir / "results" / "summary_metrics.csv", index=False)
        result_payload['cv_summary'] = summary_df
        result_payload['cv_rows'] = pd.DataFrame(all_fold_results)
    
    logger.info(f"\n{'='*60}")
    logger.info("Training completed!")
    if use_cv:
        logger.info(f"Two-stage validation results saved to:")
        logger.info(f"  - {out_dir / 'results' / 'fold_results.csv'} (individual fold results on Development Set)")
        logger.info(f"  - {out_dir / 'results' / 'cv_summary.csv'} (mean ± std across folds and seeds on Development Set)")
        logger.info(f"  - {out_dir / 'results' / 'external_test_results.csv'} (External Test Set metrics for each seed)")
        logger.info(f"  - {out_dir / 'results' / 'external_test_summary.csv'} (External Test Set mean ± std across seeds)")
        logger.info(f"  - {out_dir / 'predictions' / 'external_test_predictions.csv'} (External Test Set predictions for each seed)")
    else:
        logger.info(f"Results saved to: {out_dir / 'results' / 'summary_metrics.csv'} (mean ± std across seeds)")
    logger.info(f"{'='*60}")
    
    # Note: Model saving and SHAP analysis are not implemented for k-fold CV unless
    # `--save-cv-details` is set, which persists fold/seed artifacts under `cv_data/`.
    if not use_cv:
        # Find best models (separate for traditional and deep learning)
        # Check if results are already aggregated (contain _mean suffix)
        if all_fold_results and any('_mean' in k for k in all_fold_results[0].keys()):
            # Results are aggregated, use _mean metrics
            if config.task == "classification":
                metric = 'AUC_val_mean'
            else:
                metric = 'R2_val_mean'
        else:
            # Results are not aggregated, use regular metrics
            if config.task == "classification":
                metric = 'AUC_val'
            else:
                metric = 'R2_val'
        
        traditional_results = [r for r in all_fold_results if r['model_type'] == 'traditional']
        deep_learning_results = [r for r in all_fold_results if r['model_type'] == 'deep_learning']
        
        best_traditional = None
        best_deep_learning = None
        
        if traditional_results:
            best_traditional = max(traditional_results, key=lambda x: x.get(metric, -float('inf')))
            logger.info(f"\nBest Traditional Model: {best_traditional['model']} ({metric}={best_traditional.get(metric, 'N/A'):.4f})")
        
        if deep_learning_results:
            best_deep_learning = max(deep_learning_results, key=lambda x: x.get(metric, -float('inf')))
            logger.info(f"Best Deep Learning Model: {best_deep_learning['model']} ({metric}={best_deep_learning.get(metric, 'N/A'):.4f})")
        
        # Save best models
        best_model_dir = out_dir / "models" / "best_models"
        best_model_dir.mkdir(parents=True, exist_ok=True)
        
        if best_traditional:
            # Get the model object from the results
            for result in all_fold_results:
                if result['model'] == best_traditional['model'] and 'model' in result:
                    model_path = best_model_dir / f"best_traditional_{best_traditional['model']}"
                    _save_sklearn_model(result['model'], model_path)
                    logger.info(f"  ✓ Saved best traditional model: {best_traditional['model']}")
                    break
        
        if best_deep_learning:
            # Get the model object from the results
            for result in all_fold_results:
                if result['model'] == best_deep_learning['model'] and 'model' in result:
                    model_path = best_model_dir / f"best_deep_learning_{best_deep_learning['model']}"
                    # Determine model type for proper saving
                    model_config = registry.get(best_deep_learning['model'])
                    if model_config:
                        model_type = model_config.get('type', 'pytorch')
                        _save_pytorch_model(result['model'], model_path, model_type)
                        logger.info(f"  ✓ Saved best deep learning model: {best_deep_learning['model']}")
                    break
        
        # SHAP analysis for best models
        if config.run_shap and X is not None:
            logger.info("\n=== SHAP Analysis ===")
            logger.info("Note: SHAP analysis requires saved models. This feature is under development.")
        elif config.run_shap:
            logger.warning("\nSHAP analysis skipped: No feature matrix available (models trained on SMILES only)")
    else:
        if config.save_cv_details:
            logger.info(f"\nDetailed CV artifacts saved under: {out_dir / 'cv_data'}")
        else:
            logger.info("\nNote: Model saving and SHAP analysis are only available in single split mode (--folds 1)")
            logger.info("Use --save-cv-details to persist per-fold/seed artifacts for post-hoc SHAP analysis.")
    
    return result_payload

# --------- Automatic Fingerprint Generation ---------
def generate_fingerprints_from_csv(df: pd.DataFrame, smiles_column: str, 
                                   fingerprint_types: List[str] = ["morgan"], 
                                   logger: logging.Logger = None) -> pd.DataFrame:
    """
    Generate molecular fingerprints from SMILES column in DataFrame
    
    Args:
        df: Input DataFrame with SMILES column
        smiles_column: Name of SMILES column
        fingerprint_types: List of fingerprint types to generate
                           Options: morgan, maccs, rdkit, atompair, torsion
        logger: Logger instance
    
    Returns:
        DataFrame with added fingerprint columns
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors, RDKFingerprint
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    except ImportError:
        if logger:
            logger.error("RDKit is required for fingerprint generation. Install with: pip install rdkit")
        raise ImportError("RDKit is required for fingerprint generation")
    
    if logger:
        logger.info(f"Generating fingerprints: {', '.join(fingerprint_types)}")
    
    # Fingerprint parameters
    NBITS_HASHED = 2048
    MORGAN_RADIUS = 2
    
    df_with_fp = df.copy()
    smiles_list = df[smiles_column].tolist()
    
    # Initialize fingerprint columns with 0
    fp_columns = {}
    for fp_type in fingerprint_types:
        nbits = 167 if fp_type == "maccs" else NBITS_HASHED
        for i in range(nbits):
            col_name = f"{fp_type}_{i}"
            fp_columns[col_name] = [0] * len(df)
    
    # Generate fingerprints for each molecule
    n_ok = 0
    n_fail = 0
    
    for idx, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                n_fail += 1
                continue
            
            # Generate requested fingerprints
            fp_dict = {}
            
            if "morgan" in fingerprint_types:
                try:
                    fpgen = GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=NBITS_HASHED)
                    bv = fpgen.GetFingerprint(mol)
                    fp_dict["morgan"] = [int(bv.GetBit(i)) for i in range(NBITS_HASHED)]
                except Exception:
                    pass
            
            if "maccs" in fingerprint_types:
                try:
                    bv = MACCSkeys.GenMACCSKeys(mol)
                    fp_dict["maccs"] = [int(bv.GetBit(i)) for i in range(167)]
                except Exception:
                    pass
            
            if "rdkit" in fingerprint_types:
                try:
                    bv = RDKFingerprint.MolFingerprintAsBitVect(mol, fpSize=NBITS_HASHED)
                    fp_dict["rdkit"] = [int(bv.GetBit(i)) for i in range(NBITS_HASHED)]
                except Exception:
                    pass
            
            if "atompair" in fingerprint_types:
                try:
                    bv = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=NBITS_HASHED)
                    fp_dict["atompair"] = [int(bv.GetBit(i)) for i in range(NBITS_HASHED)]
                except Exception:
                    pass
            
            if "torsion" in fingerprint_types:
                try:
                    bv = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=NBITS_HASHED)
                    fp_dict["torsion"] = [int(bv.GetBit(i)) for i in range(NBITS_HASHED)]
                except Exception:
                    pass
            
            # Fill fingerprint columns
            for fp_type, bits in fp_dict.items():
                for i, bit_val in enumerate(bits):
                    col_name = f"{fp_type}_{i}"
                    fp_columns[col_name][idx] = bit_val
            
            n_ok += 1
            
        except Exception as e:
            if logger:
                logger.warning(f"Failed to generate fingerprints for row {idx}: {e}")
            n_fail += 1
    
    # Add fingerprint columns to DataFrame
    for col_name, values in fp_columns.items():
        df_with_fp[col_name] = values
    
    if logger:
        total_fp_cols = sum(1 for c in df_with_fp.columns if any(c.startswith(f"{fp}_") for fp in fingerprint_types))
        logger.info(f"Fingerprint generation completed: {n_ok} succeeded, {n_fail} failed")
        logger.info(f"Total fingerprint columns added: {total_fp_cols}")
    
    return df_with_fp

def validate_smiles(smiles_list: List[str], logger: logging.Logger = None) -> Tuple[List[bool], List[int]]:
    """
    Validate SMILES strings using RDKit
    
    Args:
        smiles_list: List of SMILES strings to validate
        logger: Logger instance
    
    Returns:
        Tuple of (validity_list, invalid_indices) where:
        - validity_list: List of booleans indicating if each SMILES is valid
        - invalid_indices: List of indices of invalid SMILES
    """
    try:
        from rdkit import Chem
    except ImportError:
        if logger:
            logger.warning("RDKit is not installed. SMILES validation will be skipped.")
        return [True] * len(smiles_list), []
    
    validity_list = []
    invalid_indices = []
    
    for idx, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            is_valid = mol is not None
            validity_list.append(is_valid)
            if not is_valid:
                invalid_indices.append(idx)
        except Exception:
            validity_list.append(False)
            invalid_indices.append(idx)
    
    return validity_list, invalid_indices

def check_models_require_fingerprints(selected_models: List[str]) -> bool:
    """
    Check if any selected model requires fingerprint features
    
    Args:
        selected_models: List of selected model names
    
    Returns:
        True if any model requires fingerprints
    """
    models_requiring_fingerprints = ['LR', 'RFC', 'SVC', 'XGBC', 'LGBMC', 'ETC', 
                                     'Ridge', 'RFR', 'ETR', 'MLP']
    
    for model in selected_models:
        if model in models_requiring_fingerprints:
            return True
    
    return False

def _normalize_metric_key(metric_key: str) -> str:
    """
    Normalize a metric key by removing suffixes such as '_mean' and '_val'
    so we can construct stage-prefixed column names like 'Ex_AUC_mean'.
    """
    if not metric_key.endswith('_mean'):
        return metric_key
    base = metric_key[:-len('_mean')]
    if base.endswith('_val'):
        base = base[:-len('_val')]
    return base

def _aggregate_stage_summaries(frames: List[pd.DataFrame], stage_prefix: str) -> pd.DataFrame:
    """
    Aggregate metrics from multiple split seeds, prefix columns with stage label.
    """
    combined = pd.concat(frames, ignore_index=True, sort=False)
    metric_mean_cols = [col for col in combined.columns if col.endswith('_mean')]
    aggregated_rows = []

    for model_name, group in combined.groupby('model'):
        row = {
            'model': model_name,
            'model_type': group['model_type'].iloc[0] if 'model_type' in group.columns else None,
            'split_seed_count': len(group)
        }

        for mean_col in metric_mean_cols:
            values = pd.to_numeric(group[mean_col], errors='coerce').dropna()
            base_name = _normalize_metric_key(mean_col)
            mean_name = f"{stage_prefix}_{base_name}_mean"
            std_name = f"{stage_prefix}_{base_name}_std"
            if not values.empty:
                row[mean_name] = float(values.mean())
                row[std_name] = float(values.std(ddof=0))
            else:
                row[mean_name] = float('nan')
                row[std_name] = float('nan')

        aggregated_rows.append(row)

    return pd.DataFrame(aggregated_rows)

# --------- Command Line Interface ---------
def main_cli():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description="QSAR Batch Modeling with PyTorch Deep Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use YAML configuration file
  python step01_train_qsar_models.py -c config.yaml
  
  # Train specific models
  python step01_train_qsar_models.py -i data.csv -l label -m xgbc,mlp
  
  # Train all deep learning models
  python step01_train_qsar_models.py -i data.csv -l label -s smiles -m mlp,gat,chemberta
  
  # With SMILES for GNN models
  python step01_train_qsar_models.py -i data.csv -l label -s smiles -m xgbc,gat -o output/

Input CSV Format:
  - id: Molecular unique identifier (e.g., ChEMBL ID)
  - smiles: SMILES string representation
  - label: Binary classification label (0 or 1)
  - pic50: Continuous pIC50 value (for regression)

Available models:
  Traditional: LR, RFC, SVC, XGBC, LGBMC, ETC, Ridge, RFR, ETR
  Deep Learning: MLP, GAT, ChemBERTa

External Test Set Evaluation:
  When using --folds > 1, the pipeline performs two-stage validation:
  1. Split data into External Test Set (20%) and Development Set (80%)
  2. Perform K-Fold CV on Development Set (use --cv-split-method to choose method)
  3. Train ALL models on full Development Set and evaluate on External Test Set
  4. Select best models based on specified metric (--external-test-metric)
  
  CV Split Methods (--cv-split-method):
  - scaffold: Scaffold-based split with greedy allocation (default, recommended)
  - random: Standard random/stratified KFold
  
  Default metrics:
  - Classification: MCC (Matthews Correlation Coefficient)
  - Regression: R2 (Coefficient of Determination)
        """
    )
    
    # Configuration file
    parser.add_argument('-c', '--config', help='Path to YAML configuration file')
    
    # Required arguments (if not using config file)
    parser.add_argument('-i', '--input', help='Input CSV/Parquet file path')
    parser.add_argument('-l', '--label', help='Label column name')
    
    # Optional arguments
    parser.add_argument('-p', '--pic50', help='pIC50 column name (for regression)')
    parser.add_argument('-s', '--smiles', help='SMILES column name (required for GAT/ChemBERTa)')
    parser.add_argument('-d', '--id', help='ID column name')
    parser.add_argument('-t', '--task', choices=['classification', 'regression'], 
                       help='Task type (default: classification)')
    parser.add_argument('-m', '--models', help='Comma-separated list of models to train (e.g., xgbc,mlp,gat)')
    parser.add_argument('-o', '--output', default='models_out', help='Output directory (default: models_out)')
    parser.add_argument('--folds', type=int, help='Number of CV folds (default: 5)')
    parser.add_argument('--seed', type=int, help='Random seed (default: 42)')
    parser.add_argument('--seeds', type=str, help='Comma-separated list of random seeds for multiple runs (e.g., 42,123,2025). If provided, overrides --seed.')
    parser.add_argument('--split-seeds', type=str, help='Comma-separated list of seeds to use when creating Stage 1/2 splits (multi-run split robustness).')
    parser.add_argument('--test-size', type=float, help='Test size fraction (default: 0.2)')
    parser.add_argument('--epochs', type=int, help='Max epochs for deep models (default: 100)')
    parser.add_argument('--batch-size', type=int, help='Batch size for deep models (default: 32)')
    parser.add_argument('--lr', type=float, help='Learning rate (default: 0.001)')
    parser.add_argument('--ef', type=float, help='Early Enrichment Factor percentile (default: 1.0)')
    parser.add_argument('--variance-threshold', type=float, help='Variance threshold for feature filtering (default: 0.01)')
    parser.add_argument('--split-method', choices=['scaffold', 'stratified', 'random'], 
                       help='Data split method: scaffold (default), stratified, or random')
    parser.add_argument('--cv-split-method', choices=['scaffold', 'random'], 
                       help='CV split method for K-Fold cross-validation: scaffold (default) or random')
    parser.add_argument('--external-test-metric', choices=['AUC', 'PR_AUC', 'ACC', 'F1', 'MCC', 'R2', 'RMSE', 'MAE'],
                       help='Metric for selecting best model on external test set (default: MCC for classification, R2 for regression)')
    parser.add_argument('--no-auto-fp', action='store_true', 
                       help='Disable automatic fingerprint generation')
    parser.add_argument('--fp-types', default='morgan', 
                       help='Fingerprint types to generate (comma-separated): morgan,maccs,rdkit,atompair,torsion')
    parser.add_argument('--no-shap', action='store_true', help='Disable SHAP analysis')
    parser.add_argument('--skip-cv-stage2', action='store_true',
                       help='Skip Stage 2 K-Fold CV on Development Set (Dev/External split + Stage 3 still run)')
    parser.add_argument('--tune-stage2', action='store_true',
                       help='Use Stage 2 CV to tune sklearn hyperparameters (Dev only, never touch External Test)')
    parser.add_argument('--save-cv-details', action='store_true',
                       help='Persist fold/seed models, processors, and training data for CV (useful for SHAP).')
    parser.add_argument('--tune-mode', choices=['grid', 'random'], default='grid',
                       help='Hyperparameter search mode for Stage 2 tuning')
    parser.add_argument('--cv-tune-metric', choices=['AUC', 'PR_AUC', 'ACC', 'F1', 'MCC', 'R2', 'RMSE', 'MAE'],
                       help='Metric to rank hyperparameter candidates during Stage 2 tuning')
    parser.add_argument('--tune-iter', type=int, help='Max number of candidate settings to evaluate during Stage 2 tuning')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--save-train-features', action='store_true',
                       help='Serialize Development-set features/labels to models_out/.../data/train_features.npz for AD analysis')
    
    args = parser.parse_args()
    
    # Load configuration from YAML file if provided
    config = None
    if args.config:
        if not YAML_AVAILABLE:
            sys.exit("Error: PyYAML is required for YAML configuration files. Install with: pip install pyyaml")
        
        config_path = Path(args.config)
        if config_path.suffix.lower() in ('.yaml', '.yml'):
            config = QSARConfig.from_yaml(config_path)
            print(f"Loaded configuration from: {args.config}")
        else:
            print(f"Warning: Configuration file '{args.config}' is not a YAML file, trying to load as JSON")
            config = QSARConfig.from_json(config_path)
        
        # Override with CLI arguments
        if args.input: config.input_path = args.input
        if args.label: config.label_column = args.label
        if args.pic50: config.pic50_column = args.pic50
        if args.smiles: config.smiles_column = args.smiles
        if args.id: config.id_column = args.id
        if args.task: config.task = args.task  # Command line task overrides config file
        if args.models: config.selected_models = [m.strip().upper() for m in args.models.split(',')]
        if args.output: config.output_dir = args.output
        if args.folds: config.folds = args.folds
        if args.seed: config.seed = args.seed
        if args.seeds: config.seeds = [int(s.strip()) for s in args.seeds.split(',')]
        if args.split_seeds: config.split_seeds = [int(s.strip()) for s in args.split_seeds.split(',')]
        if args.test_size is not None: config.test_size = args.test_size
        if args.epochs: config.max_epochs = args.epochs
        if args.batch_size: config.batch_size = args.batch_size
        if args.lr: config.learning_rate = args.lr
        if args.ef: config.ef_percentile = args.ef
        if args.variance_threshold is not None: config.variance_threshold = args.variance_threshold
        if args.split_method: config.split_method = args.split_method
        if args.cv_split_method: config.cv_split_method = args.cv_split_method
        if args.external_test_metric: config.external_test_metric = args.external_test_metric
        if args.no_auto_fp: config.auto_generate_fingerprints = False
        if args.fp_types: config.fingerprint_types = [fp.strip() for fp in args.fp_types.split(',')]
        if args.no_shap: config.run_shap = False
        if args.save_cv_details: config.save_cv_details = True
        if args.skip_cv_stage2:
            config.run_cv_stage2 = False
        if args.tune_stage2:
            config.tune = True
            config.tune_mode = args.tune_mode
            if args.cv_tune_metric:
                config.cv_tune_metric = args.cv_tune_metric
            if args.tune_iter is not None:
                config.tune_iter = args.tune_iter
        if args.save_train_features:
            config.save_train_features = True
        
        config.config_file = str(args.config)
    
    else:
        # Build config from CLI arguments
        if not args.input or not args.label:
            parser.error("-i/--input and -l/--label are required when not using a config file")
        
        # Parse model selection
        selected_models = []
        if args.models:
            selected_models = [m.strip().upper() for m in args.models.split(',')]
        
        # Determine task from pic50 presence (only if not explicitly specified via --task)
        # Command line --task parameter has highest priority
        # If --task is not specified, use 'classification' as default
        task = args.task if args.task is not None else 'classification'
        
        config = QSARConfig(
            input_path=args.input,
            label_column=args.label,
            pic50_column=args.pic50 or 'pic50',
            smiles_column=args.smiles or 'smiles',
            id_column=args.id or 'id',
            task=task,
            test_size=args.test_size or 0.2,
            folds=args.folds or 5,
            seed=args.seed or 42,
            seeds=[int(s.strip()) for s in args.seeds.split(',')] if args.seeds else [args.seed or 42],
            split_seeds=[int(s.strip()) for s in args.split_seeds.split(',')] if args.split_seeds else [args.seed or 42],
            selected_models=selected_models,
            output_dir=args.output or 'models_out',
            variance_threshold=args.variance_threshold or 0.01,
            run_shap=not args.no_shap,
            max_epochs=args.epochs or 100,
            batch_size=args.batch_size or 32,
            learning_rate=args.lr or 0.001,
            ef_percentile=args.ef or 1.0,
            split_method=args.split_method or 'scaffold',
            cv_split_method=args.cv_split_method or 'scaffold',
            external_test_metric=args.external_test_metric or ('MCC' if task == 'classification' else 'R2'),
            auto_generate_fingerprints=not args.no_auto_fp if hasattr(args, 'no_auto_fp') else True,
            fingerprint_types=[fp.strip() for fp in args.fp_types.split(',')] if hasattr(args, 'fp_types') and args.fp_types else ['morgan'],
            save_cv_details=args.save_cv_details,
            run_cv_stage2=not args.skip_cv_stage2,
            save_train_features=args.save_train_features,
        )
        if args.tune_stage2:
            config.tune = True
            config.tune_mode = args.tune_mode
            config.cv_tune_metric = args.cv_tune_metric
            if args.tune_iter is not None:
                config.tune_iter = args.tune_iter
    
    if not config.split_seeds:
        config.split_seeds = [config.seed]

    # Set all random seeds for reproducibility
    # Use the first seed from the seeds list for initial setup
    initial_seed = config.seeds[0] if config.seeds else config.seed
    set_all_seeds(initial_seed)
    
    # Setup output directory with timestamp and task type
    # Create a unique directory for each run: output/{task}_{YYYYMMDD_HHMMSS}/
    base_output_dir = Path(config.output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_output_dir = base_output_dir / f"{config.task}_{timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Update config with the actual output directory
    config.output_dir = str(run_output_dir)
    
    # Setup logging
    # Create logs subdirectory before setting up logging
    logs_dir = run_output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, (args.log_level if hasattr(args, 'log_level') else 'INFO').upper()),
        format='%(asctime)s | %(levelname)-7s | %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / f"qsar_run_{timestamp}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Output directory: {run_output_dir}")
    
    # Check library versions
    check_library_versions(logger)
    
    # Load data
    logger.info(f"Loading data from: {config.input_path}")
    df = read_table(Path(config.input_path))
    logger.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Validate SMILES strings
    smiles_list = df[config.smiles_column].tolist()
    validity_list, invalid_indices = validate_smiles(smiles_list, logger)
    n_invalid = len(invalid_indices)
    
    if n_invalid > 0:
        logger.warning(f"Found {n_invalid} invalid SMILES strings out of {len(smiles_list)} total.")
        logger.warning("Invalid SMILES entries will be filtered out for all models.")
        if n_invalid <= 10:
            invalid_ids = [df[config.id_column].iloc[i] for i in invalid_indices]
            logger.warning(f"Invalid IDs: {invalid_ids}")
        else:
            logger.warning(f"First 10 invalid IDs: {[df[config.id_column].iloc[i] for i in invalid_indices[:10]]}")
            logger.warning("Consider cleaning your data with a SMILES validation script before running.")
        
        # Filter out invalid SMILES and corresponding data for ALL models
        valid_mask = np.array(validity_list)
        df = df.loc[valid_mask].reset_index(drop=True)
        if X is not None:
            X = X[valid_mask]
        y = y[valid_mask]
        smiles_list = [smiles_list[i] for i, valid in enumerate(validity_list) if valid]
        ids = [ids[i] for i, valid in enumerate(validity_list) if valid]
        logger.info(f"Filtered out {n_invalid} invalid SMILES samples. Remaining samples: {len(df)}")
    else:
        logger.info("All SMILES strings are valid.")
    
    # Validate required columns
    required_cols = [config.id_column, config.smiles_column]
    if config.task == "classification":
        required_cols.append(config.label_column)
    else:
        required_cols.append(config.pic50_column)
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        logger.info(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # Select fingerprint columns (optional for GAT/ChemBERTa, required for traditional models/MLP)
    fp_cols = select_fp_columns(df)
    
    if not fp_cols:
        logger.warning("No fingerprint columns detected!")
        
        # Check if selected models require fingerprints
        requires_fingerprints = check_models_require_fingerprints(config.selected_models)
        
        if requires_fingerprints:
            logger.info("Selected models require fingerprints. Generating fingerprints automatically...")
            logger.info(f"Using {', '.join(config.fingerprint_types)} fingerprints")
            
            try:
                df = generate_fingerprints_from_csv(
                    df, 
                    config.smiles_column, 
                    fingerprint_types=config.fingerprint_types,
                    logger=logger
                )
                
                # Save DataFrame with fingerprints to a new file
                input_path = Path(config.input_path)
                output_with_fp_path = input_path.parent / f"{input_path.stem}_with_fingerprints{input_path.suffix}"
                df.to_csv(output_with_fp_path, index=False)
                logger.info(f"Data with fingerprints saved to: {output_with_fp_path}")
                
                # Re-select fingerprint columns
                fp_cols = select_fp_columns(df)
                logger.info(f"Generated {len(fp_cols)} fingerprint columns")
                
                # Prepare features from generated fingerprints
                filtered_cols = fp_cols
                X = df[filtered_cols].to_numpy(dtype=float)
                
            except Exception as e:
                logger.error(f"Failed to generate fingerprints: {e}")
                logger.info("Please install RDKit: pip install rdkit")
                logger.info("Or manually generate fingerprints using: python generate_fingerprints.py")
                sys.exit(1)
        else:
            logger.info("GAT and ChemBERTa models can still be trained using SMILES only.")
            logger.info("Traditional models (LR, RFC, XGBC, etc.) and MLP require fingerprints.")
            logger.info("Use generate_fingerprints.py to add fingerprint features if needed.")
            
            # Set X to None if no fingerprints
            X = None
            filtered_cols = []
    else:
        logger.info(f"Detected {len(fp_cols)} fingerprint columns")
        logger.info(f"Feature filtering and standardization will be performed within each fold to prevent data leakage")
        
        # Prepare features (no filtering here, will be done in each fold)
        filtered_cols = fp_cols
        X = df[filtered_cols].to_numpy(dtype=float)
    
    # Prepare labels based on task
    if config.task == "classification":
        y_series = pd.Series(df[config.label_column]).astype(str).str.strip().str.lower()
        mapper = {"1": 1, "0": 0, "true": 1, "false": 0, "active": 1, "inactive": 0}
        
        if set(y_series.unique()) - set(mapper.keys()):
            try:
                y = pd.Series(df[config.label_column]).astype(int).to_numpy()
            except:
                logger.error(f"Cannot convert label column '{config.label_column}' to binary")
                sys.exit(1)
        else:
            y = y_series.map(mapper).astype(int).to_numpy()
    else:
        y = pd.to_numeric(df[config.pic50_column], errors="coerce").astype(float).to_numpy()
        # Remove NaN
        valid_mask = ~np.isnan(y)
        # Keep all downstream arrays (X, y, smiles_list, ids) aligned.
        # Filtering only X/y causes length mismatches later during splits/evaluation.
        df = df.loc[valid_mask].reset_index(drop=True)
        if X is not None:
            X = X[valid_mask]
        y = y[valid_mask]
        logger.info(f"Removed {np.sum(~valid_mask)} samples with missing pIC50 values")
    
    # Prepare SMILES
    smiles_list = df[config.smiles_column].tolist()
    logger.info(f"SMILES column found: {config.smiles_column}")
    
    # Prepare IDs
    ids = df[config.id_column].tolist()
    
    # Determine models to train
    if not config.selected_models:
        input_dim = X.shape[1] if X is not None else None
        registry = build_model_registry(task=config.task, input_dim=input_dim)
        config.selected_models = list(registry.keys())
    else:
        # Validate that selected models are compatible with the current task
        input_dim = X.shape[1] if X is not None else None
        registry = build_model_registry(task=config.task, input_dim=input_dim)
        available_models = set(registry.keys())
        selected_models_set = set(config.selected_models)
        
        # Find incompatible models
        incompatible_models = selected_models_set - available_models
        
        if incompatible_models:
            logger.warning(f"Warning: The following models are not compatible with {config.task} task:")
            for model in incompatible_models:
                logger.warning(f"  - {model}")
            logger.warning("These models will be filtered out.")
            
            # Filter out incompatible models
            config.selected_models = [m for m in config.selected_models if m in available_models]
        
        if not config.selected_models:
            logger.error("Error: No compatible models selected for the current task!")
            sys.exit(1)
    
    logger.info(f"Selected models: {config.selected_models}")
    
    split_seeds = config.split_seeds or [config.seed]
    aggregate_results_dir = run_output_dir / "results"
    aggregate_results_dir.mkdir(parents=True, exist_ok=True)

    split_ext_summary_frames = []
    split_cv_summary_frames = []
    split_ext_row_frames = []
    split_cv_row_frames = []

    for split_seed in split_seeds:
        split_dir = run_output_dir / f"split_seed_{split_seed}"
        split_dir.mkdir(parents=True, exist_ok=True)

        config_split = copy.copy(config)
        config_split.output_dir = str(split_dir)

        logger.info(f"\n=== Running pipeline for split seed {split_seed} ===")
        summary_dict = main_pipeline(
            config_split, X, y, df[filtered_cols], smiles_list, ids, logger, split_seed=split_seed
        )
        if not summary_dict:
            continue

        ext_summary = summary_dict.get('external_summary')
        if ext_summary is not None and not ext_summary.empty:
            ext_copy = ext_summary.copy()
            ext_copy['split_seed'] = split_seed
            split_ext_summary_frames.append(ext_copy)

        cv_summary = summary_dict.get('cv_summary')
        if cv_summary is not None and not cv_summary.empty:
            cv_copy = cv_summary.copy()
            cv_copy['split_seed'] = split_seed
            split_cv_summary_frames.append(cv_copy)

        ext_rows = summary_dict.get('external_rows')
        if ext_rows is not None and not ext_rows.empty:
            ext_rows_copy = ext_rows.copy()
            ext_rows_copy['split_seed'] = split_seed
            ext_rows_copy['stage'] = 'external'
            split_ext_row_frames.append(ext_rows_copy)

        cv_rows = summary_dict.get('cv_rows')
        if cv_rows is not None and not cv_rows.empty:
            cv_rows_copy = cv_rows.copy()
            cv_rows_copy['split_seed'] = split_seed
            cv_rows_copy['stage'] = 'cv'
            split_cv_row_frames.append(cv_rows_copy)

    summary_messages = []
    stage_aggregated_frames = []
    stage_configs = [
        ('external', split_ext_summary_frames, 'Ex', "all_split_external_summary.csv"),
        ('cv', split_cv_summary_frames, 'CV', "all_split_cv_summary.csv"),
    ]

    for stage_name, frames, prefix, filename in stage_configs:
        if not frames:
            continue
        agg_df = _aggregate_stage_summaries(frames, prefix)
        agg_df.insert(0, 'stage', stage_name)
        stage_path = aggregate_results_dir / filename
        agg_df.to_csv(stage_path, index=False)
        summary_messages.append(f"  - {stage_path} (aggregated {stage_name} metrics)")
        stage_aggregated_frames.append(agg_df)

    if stage_aggregated_frames:
        combined_df = pd.concat(stage_aggregated_frames, ignore_index=True, sort=False)
        cols = ['stage'] + [col for col in combined_df.columns if col != 'stage']
        combined_df = combined_df[cols]
        combined_summary_path = aggregate_results_dir / "all_split_summaries.csv"
        combined_df.to_csv(combined_summary_path, index=False)
        summary_messages.append(f"  - {combined_summary_path} (per-stage summary across splits)")

    if split_ext_row_frames or split_cv_row_frames:
        row_frames = []
        if split_ext_row_frames:
            row_frames.append(pd.concat(split_ext_row_frames, ignore_index=True, sort=False))
        if split_cv_row_frames:
            row_frames.append(pd.concat(split_cv_row_frames, ignore_index=True, sort=False))

        if row_frames:
            all_rows_df = pd.concat(row_frames, ignore_index=True, sort=False)
            cols = ['stage'] + [col for col in all_rows_df.columns if col != 'stage']
            all_rows_df = all_rows_df[cols]
            all_rows_path = aggregate_results_dir / "all_split_row_data.csv"
            all_rows_df.to_csv(all_rows_path, index=False)
            summary_messages.append(f"  - {all_rows_path} (all row data across splits)")

    if summary_messages:
        logger.info("\nCombined split summaries saved to:")
        for msg in summary_messages:
            logger.info(msg)

if __name__ == "__main__":
    main_cli()
