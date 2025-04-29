"""
Author: Tim Frenzel
Version: 1.00
Usage:  python src/baselines/logit.py --eval_path data/eval.arrow [--output results/logit_metrics.json]

Objective of the Code:
------------
Implements a simple baseline model for the medical diagnosis task using
Term Frequency-Inverse Document Frequency (TF-IDF) features extracted from
the patient prompts and a Logistic Regression classifier. It trains on a
portion of the evaluation data and evaluates on the rest, providing basic
performance metrics (Accuracy, F1-score) for comparison against the MoA pipeline.
"""

import os
import sys
import argparse
import pathlib
import json
import logging

import pyarrow.feather as feather
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# --- Add project root to sys.path --- 
ROOT = pathlib.Path(__file__).resolve().parents[2] # Go up two levels from src/baselines
sys.path.append(str(ROOT))

# --- Configuration & Paths ---
DEFAULT_OUTPUT_DIR = ROOT / "results"
DEFAULT_OUTPUT_FILENAME = "logit_baseline_metrics.json"

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(eval_path: str) -> pd.DataFrame:
    """Loads the evaluation data from an Arrow Feather file."""
    eval_file = pathlib.Path(eval_path)
    if not eval_file.exists():
        logger.error(f"Evaluation file not found: {eval_path}")
        sys.exit(1)
    
    try:
        logger.info(f"Loading evaluation data from: {eval_path}")
        df = feather.read_feather(eval_file)
        logger.info(f"Loaded {len(df)} records.")
        # Ensure required columns are present
        if 'prompt' not in df.columns or 'gold_icd10' not in df.columns:
             logger.error(f"Missing required columns ('prompt', 'gold_icd10') in {eval_path}")
             sys.exit(1)
        # Handle potential missing values if necessary
        df = df.dropna(subset=['prompt', 'gold_icd10'])
        logger.info(f"{len(df)} records remaining after dropping NA.")
        return df
    except Exception as e:
        logger.error(f"Failed to load or process data from {eval_path}: {e}")
        sys.exit(1)

def train_and_evaluate(df: pd.DataFrame, output_path: str):
    """Trains and evaluates the TF-IDF + Logistic Regression baseline."""
    logger.info("Starting baseline model training and evaluation...")
    
    # 1. Prepare Data
    X = df['prompt']
    y_raw = df['gold_icd10']
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    num_classes = len(label_encoder.classes_)
    logger.info(f"Encoded {num_classes} unique diagnosis codes.")

    # Split data (using stratify for potentially imbalanced classes)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        logger.info(f"Split data: {len(X_train)} train, {len(X_test)} test samples.")
    except ValueError as e:
         logger.warning(f"Could not stratify due to insufficient class samples: {e}. Splitting without stratification.")
         X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size=0.25, random_state=42
         )

    # 2. Define Model Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000)), # Limit feature size
        ('clf', LogisticRegression(solver='liblinear', multi_class='auto', random_state=42, C=1.0))
        # liblinear is good for smaller datasets, auto handles multiclass
        # C is regularization strength
    ])

    # 3. Train Model
    logger.info("Training the Logistic Regression pipeline...")
    pipeline.fit(X_train, y_train)
    logger.info("Training complete.")

    # 4. Evaluate Model
    logger.info("Evaluating the model on the test set...")
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0)
    
    logger.info(f"Test Set Accuracy: {accuracy:.4f}")
    # Log F1-scores for clarity
    logger.info(f"Test Set F1-score (macro): {report['macro avg']['f1-score']:.4f}")
    logger.info(f"Test Set F1-score (weighted): {report['weighted avg']['f1-score']:.4f}")

    # 5. Save Results
    results = {
        'model': 'Logistic Regression (TF-IDF)',
        'accuracy': accuracy,
        'classification_report': report,
        'num_classes': num_classes,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': pipeline.named_steps['tfidf'].max_features
    }
    
    output_file = pathlib.Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True) # Create results dir if needed
    
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Baseline metrics saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save metrics to {output_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a Logistic Regression baseline.")
    parser.add_argument("--eval_path", type=str, required=True, 
                        help="Path to the evaluation data file (e.g., data/eval.arrow)")
    parser.add_argument("--output", type=str, 
                        default=str(DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_FILENAME),
                        help=f"Path to save the output metrics JSON file (default: {DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_FILENAME})")
    
    args = parser.parse_args()
    
    # Load data
    data_df = load_data(args.eval_path)
    
    # Train and evaluate
    if data_df is not None:
        train_and_evaluate(data_df, args.output) 