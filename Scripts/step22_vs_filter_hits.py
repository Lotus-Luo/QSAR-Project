#!/usr/bin/env python3
"""Filter virtual hits from a QSAR prediction matrix.
# Example usage:
python Scripts/step22_vs_filter_hits.py \
    -i ./models_out/classification_20260326_213228/split_seed_3/predictions/predictions_20260328_211826.csv

python Scripts/step22_vs_filter_hits.py \
  -i virtual_screening/VS_prediction_test_data_fingerprints_20260328_213415.csv \
  --min-sum 2 \
  --min-score 0.8 
"""

# %%
from pathlib import Path

FILTER_CONFIG = {
    "input_path": Path("models_out/classification_20260326_213228/split_seed_3/predictions/predictions_20260328_211826.csv"),
    "min_sum": 1,
    "min_score": 0.8,
    "output_path": Path("models_out/classification_20260326_213228/split_seed_3/predictions/predictions_20260328_211826_filtered_hits.csv"),
    "log_level": "INFO",
}

# %%
import logging
import sys
from typing import List

import pandas as pd



def setup_logger(level: str) -> logging.Logger:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)-7s | %(message)s")
    return logging.getLogger("filter_virtual_hits")


def find_predicted_columns(columns: List[str]) -> List[str]:
    return [c for c in columns if c.endswith("_predicted")]


def find_score_columns(columns: List[str]) -> List[str]:
    return [c for c in columns if c.endswith("_predicted_score")]


def main():
    logger = setup_logger(FILTER_CONFIG["log_level"])
    input_path = FILTER_CONFIG["input_path"].resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    df = pd.read_csv(input_path)
    predicted_cols = find_predicted_columns(df.columns.tolist())
    if not predicted_cols:
        raise SystemExit("No *_predicted columns found in the input file")
    score_cols = find_score_columns(df.columns.tolist())

    consensus = df[predicted_cols].fillna(0).sum(axis=1)
    df["Consensus_Sum"] = consensus.astype(int)
    mask = df["Consensus_Sum"] >= FILTER_CONFIG["min_sum"]

    if FILTER_CONFIG["min_score"] is not None:
        if not score_cols:
            logger.warning("--min-score provided but no *_predicted_score columns were found; ignoring this filter")
        else:
            mask_score = df[score_cols].fillna(-float("inf")).ge(FILTER_CONFIG["min_score"]).any(axis=1)
            mask &= mask_score

    filtered = df[mask]
    output_path = FILTER_CONFIG["output_path"] or (input_path.parent / f"{input_path.stem}_filtered_hits.csv")
    filtered.to_csv(output_path, index=False)
    logger.info(f"Saved {len(filtered)} out of {len(df)} candidates to {output_path}")


if __name__ == "__main__":
    main()
