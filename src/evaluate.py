"""
evaluate.py
------------
Loads the trained ensemble pipeline and evaluates it on the held-out ISOT test
split (same random_state=42 used during training, so the split is reproducible).
Writes a full classification report and confusion matrix to evaluation_results.txt.
"""

import joblib
import logging
import gc
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = "models/ensemble_pipeline.pkl"
TRUE_CSV     = "data/True.csv"
FAKE_CSV     = "data/Fake.csv"
RESULTS_PATH = "evaluation_results.txt"


def load_isot_test_split() -> tuple:
    """Reconstruct the identical 20 % test split used during training."""
    logger.info("Reading ISOT dataset …")
    true_df = pd.read_csv(TRUE_CSV)
    fake_df = pd.read_csv(FAKE_CSV)

    true_df["Label"] = 1
    fake_df["Label"] = 0

    df = pd.concat([true_df, fake_df], ignore_index=True)
    del true_df, fake_df
    gc.collect()

    df.dropna(subset=["title", "text", "Label"], inplace=True)
    df["original_text"] = df["title"].astype(str) + " " + df["text"].astype(str)

    X = df["original_text"]
    y = df["Label"].astype(int)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Test split size: {len(X_test):,} samples")
    return X_test, y_test


def format_confusion_matrix(cm: np.ndarray, labels: list) -> str:
    """Return a neat text-based confusion matrix."""
    col_w = max(len(str(cm.max())), max(len(l) for l in labels)) + 2
    header = " " * (col_w + 2) + "".join(l.center(col_w) for l in labels)
    rows = [header, "-" * len(header)]
    for i, label in enumerate(labels):
        row = label.ljust(col_w) + " |" + "".join(str(cm[i][j]).center(col_w) for j in range(len(labels)))
        rows.append(row)
    return "\n".join(rows)


def main() -> None:
    # ── Load model ───────────────────────────────────────────────────────────
    logger.info(f"Loading model from {MODEL_PATH} …")
    sys.path.insert(0, os.path.abspath("."))   # so train.SpacyPreprocessor resolves
    from src.train import SpacyPreprocessor    # noqa – needed for joblib unpickling
    sys.modules["__main__"].SpacyPreprocessor = SpacyPreprocessor  # noqa

    pipeline = joblib.load(MODEL_PATH)
    logger.info("Model loaded.")

    # ── Reproduce test split ─────────────────────────────────────────────────
    X_test, y_test = load_isot_test_split()

    # ── Predict ──────────────────────────────────────────────────────────────
    logger.info("Running predictions on test set …")
    y_pred = pipeline.predict(X_test)

    # ── Metrics ──────────────────────────────────────────────────────────────
    acc   = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["FAKE (0)", "TRUE (1)"])
    cm    = confusion_matrix(y_test, y_pred)
    cm_text = format_confusion_matrix(cm, labels=["FAKE (0)", "TRUE (1)"])

    # ── Write report ─────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 65,
        "  FAKE NEWS DETECTION — ENSEMBLE PIPELINE EVALUATION REPORT",
        f"  Generated : {timestamp}",
        f"  Dataset   : ISOT (True.csv + Fake.csv) — 20% held-out test split",
        f"  Test size : {len(y_test):,} samples",
        "=" * 65,
        "",
        f"  Overall Accuracy : {acc * 100:.2f} %",
        "",
        "— Classification Report —".center(65),
        "",
        report,
        "",
        "— Confusion Matrix —".center(65),
        "",
        cm_text,
        "",
        "  Rows = Actual label  |  Columns = Predicted label",
        "",
        "=" * 65,
        "  Model : Soft-Voting Ensemble",
        "    • MultinomialNB",
        "    • Logistic Regression",
        "    • SVC (linear kernel)",
        "    • SGDClassifier (log-loss)",
        "    • RandomForestClassifier (100 trees)",
        "  Features : CountVectorizer + TF-IDF (bigrams) via FeatureUnion",
        "  NLP      : SpaCy en_core_web_sm — lemmatisation + stopword removal",
        "=" * 65,
    ]

    report_text = "\n".join(lines)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"Report saved → {RESULTS_PATH}")
    print("\n" + report_text)


if __name__ == "__main__":
    main()
