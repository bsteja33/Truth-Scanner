"""
src/robustness_test.py
----------------------
Four-part robustness audit for the Fake News Detection ensemble pipeline.

  Test 1 – Byline Ablation       : strip Reuters/agency datelines, re-score
  Test 2 – Adversarial Perturbation: inflammatory / social-spam modifications
  Test 3 – Cross-Domain (2024-26) : modern article snippets not in ISOT
  Test 4 – Error Probability Dist.: confidence on minority misclassified samples

Run from the project root:
    python src/robustness_test.py
"""

import gc
import os
import re
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ─── logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─── Bootstrap: allow importing SpacyPreprocessor ───────────────────────────
sys.path.insert(0, os.path.abspath("."))
import joblib
from src.train import SpacyPreprocessor          # noqa
sys.modules["__main__"].SpacyPreprocessor = SpacyPreprocessor  # noqa

MODEL_PATH = "models/ensemble_pipeline.pkl"
REPORT_PATH = "robustness_report.md"

# ─── Byline regex (same as preprocessing step) ───────────────────────────────
_BYLINE_RE = [
    re.compile(r'\b[A-Z][A-Z\s,]{2,30}\([^)]{1,30}\)\s*[-–—]\s*'),   # CITY (Agency) -
    re.compile(r'\([^)]{1,30}\)\s*[-–—]\s*'),                          # (Agency) -
]

def strip_bylines(text: str) -> str:
    for pat in _BYLINE_RE:
        text = pat.sub('', text)
    return text.strip()

# ─── Synonym map for adversarial test ────────────────────────────────────────
_INFLAMMATORY = {
    r'\bSHOCKING\b': 'notable',
    r'\bBREAKING\b': 'developing',
    r'\bEXPLOSIVE\b': 'significant',
    r'\bSCOOP\b': 'report',
    r'\bBOMBSHELL\b': 'finding',
    r'\bEXCLUSIVE\b': 'special',
    r'\bTERRIFYING\b': 'concerning',
    r'\bOUTRAGEOUS\b': 'contentious',
    r'\bSTUNNING\b': 'unexpected',
    r'\bCRUSHED\b': 'defeated',
}

def neutralise_text(text: str) -> str:
    for pattern, replacement in _INFLAMMATORY.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

# ─── Cross-domain modern samples (2024-2026) ─────────────────────────────────
MODERN_SAMPLES = [
    # ---- expected TRUE ----
    (1, "OpenAI released GPT-4o in May 2024, a multimodal model capable of real-time audio and visual reasoning. "
        "The model represents a significant step in AI assistant capabilities, integrating voice and vision seamlessly."),
    (1, "The Federal Reserve held interest rates steady in its March 2025 meeting, citing persistent inflation above "
        "its 2 percent target despite slowing economic growth in the fourth quarter of 2024."),
    (1, "NASA's Artemis III mission, targeting a crewed lunar landing in 2025, completed a critical parachute system "
        "test at the White Sands Missile Range. The agency confirmed all deployment sequences functioned nominally."),
    (1, "The European Union formally enacted the AI Act in August 2024, establishing the world's first comprehensive "
        "legal framework for artificial intelligence systems, classifying them by risk levels and imposing obligations."),
    (1, "Apple unveiled the iPhone 16 lineup in September 2024 at its annual event in Cupertino, featuring the A18 "
        "chip and a dedicated hardware button for its new AI assistant built on Apple Intelligence."),
    # ---- expected FAKE ----
    (0, "BOMBSHELL: Secret documents PROVE the moon landing was staged inside a Hollywood studio. "
        "NASA insiders are now coming forward with undeniable proof the entire Apollo program was a hoax!"),
    (0, "SHARE THIS BEFORE IT'S DELETED: Bill Gates admits microchips in vaccines have been tracking "
        "your location since 2021. Mainstream media REFUSES to cover this explosive revelation."),
    (0, "BREAKING: George Soros caught funding violent protests in 47 cities simultaneously using a secret "
        "network of nonprofits. Click here to see the proof they don't want you to find!"),
    (0, "SHOCKING truth revealed: 5G towers are emitting mind-control frequencies that cause people to "
        "support globalist policies. Scientists who discovered this have been silenced by Big Tech."),
    (0, "EXCLUSIVE: The deep state has been replacing world leaders with AI clones since 2023. "
        "Multiple insiders confirm the real presidents are being held in underground bunkers."),
]

# ─── Helpers ─────────────────────────────────────────────────────────────────
def load_model():
    logger.info("Loading model …")
    pipeline = joblib.load(MODEL_PATH)
    logger.info("Model ready.")
    return pipeline

def load_isot_test_split():
    logger.info("Reconstructing ISOT test split …")
    true_df = pd.read_csv("data/True.csv")
    fake_df = pd.read_csv("data/Fake.csv")
    true_df["Label"] = 1
    fake_df["Label"] = 0
    df = pd.concat([true_df, fake_df], ignore_index=True)
    del true_df, fake_df; gc.collect()
    df.dropna(subset=["title", "text", "Label"], inplace=True)
    df["original_text"] = df["title"].astype(str) + " " + df["text"].astype(str)
    _, X_test, _, y_test = train_test_split(
        df["original_text"], df["Label"].astype(int),
        test_size=0.2, random_state=42, stratify=df["Label"]
    )
    df_test = pd.DataFrame({"text": X_test, "label": y_test}).reset_index(drop=True)
    logger.info(f"Test split: {len(df_test):,} samples")
    return df_test

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1 – Byline Ablation
# ═══════════════════════════════════════════════════════════════════════════════
def test1_byline_ablation(pipeline, df_test: pd.DataFrame) -> dict:
    logger.info("TEST 1: Byline Ablation …")
    true_samples = df_test[df_test["label"] == 1].head(100).copy()

    original_preds  = pipeline.predict(true_samples["text"].tolist())
    baseline_acc    = accuracy_score(true_samples["label"], original_preds)

    stripped_texts  = true_samples["text"].apply(strip_bylines).tolist()
    stripped_preds  = pipeline.predict(stripped_texts)
    stripped_acc    = accuracy_score(true_samples["label"], stripped_preds)

    drop            = baseline_acc - stripped_acc
    bias_flag       = drop > 0.10

    return {
        "baseline_acc": baseline_acc,
        "stripped_acc": stripped_acc,
        "accuracy_drop": drop,
        "bias_detected": bias_flag,
        "n": 100,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2 – Adversarial Perturbation
# ═══════════════════════════════════════════════════════════════════════════════
def test2_adversarial(pipeline, df_test: pd.DataFrame) -> dict:
    logger.info("TEST 2: Adversarial Perturbation …")
    fake_50 = df_test[df_test["label"] == 0].head(50).copy()
    true_50 = df_test[df_test["label"] == 1].head(50).copy()

    # Neutralise fake samples
    fake_perturbed = fake_50["text"].apply(neutralise_text).tolist()
    fake_orig_preds = pipeline.predict(fake_50["text"].tolist())
    fake_pert_preds = pipeline.predict(fake_perturbed)
    fake_flip = int(np.sum(fake_orig_preds != fake_pert_preds))

    # Add social-spam noise to true samples
    true_spammed = (true_50["text"] + " Click here! Share now! You won't believe this!").tolist()
    true_orig_preds = pipeline.predict(true_50["text"].tolist())
    true_pert_preds = pipeline.predict(true_spammed)
    true_flip = int(np.sum(true_orig_preds != true_pert_preds))

    return {
        "fake_flipped": fake_flip,
        "fake_total": 50,
        "true_flipped": true_flip,
        "true_total": 50,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3 – Cross-Domain (Modern 2024-2026)
# ═══════════════════════════════════════════════════════════════════════════════
def test3_cross_domain(pipeline) -> dict:
    logger.info("TEST 3: Cross-Domain Modern Samples …")
    texts  = [s[1] for s in MODERN_SAMPLES]
    labels = [s[0] for s in MODERN_SAMPLES]
    preds  = pipeline.predict(texts)
    probas = pipeline.predict_proba(texts)

    results = []
    for i, (text, true_lbl, pred, prob) in enumerate(zip(texts, labels, preds, probas)):
        correct = (true_lbl == pred)
        confidence = float(prob[pred])
        results.append({
            "idx": i + 1,
            "expected": "TRUE" if true_lbl == 1 else "FAKE",
            "predicted": "TRUE" if pred == 1 else "FAKE",
            "correct": correct,
            "confidence": confidence,
            "snippet": text[:80] + "…",
        })

    n_correct = sum(r["correct"] for r in results)
    return {"samples": results, "accuracy": n_correct / len(results), "n": len(results)}

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4 – Error Probability Distribution
# ═══════════════════════════════════════════════════════════════════════════════
def test4_error_distribution(pipeline, df_test: pd.DataFrame) -> dict:
    logger.info("TEST 4: Error Probability Distribution …")
    sample = df_test.sample(n=min(500, len(df_test)), random_state=42)
    texts  = sample["text"].tolist()
    y_true = sample["label"].values

    preds  = pipeline.predict(texts)
    probas = pipeline.predict_proba(texts)

    errors = []
    for i, (true_lbl, pred, prob) in enumerate(zip(y_true, preds, probas)):
        if true_lbl != pred:
            confidence = float(prob[pred])
            errors.append({
                "true": "TRUE" if true_lbl == 1 else "FAKE",
                "predicted": "TRUE" if pred == 1 else "FAKE",
                "conf_wrong": confidence,
            })

    if not errors:
        return {"n_errors": 0, "avg_conf": None, "confident_errors": 0, "close_calls": 0}

    confs = [e["conf_wrong"] for e in errors]
    confident_errors = sum(1 for c in confs if c > 0.80)
    close_calls      = sum(1 for c in confs if c <= 0.60)

    return {
        "n_errors": len(errors),
        "avg_conf": float(np.mean(confs)),
        "min_conf": float(np.min(confs)),
        "max_conf": float(np.max(confs)),
        "confident_errors": confident_errors,
        "close_calls": close_calls,
        "sample_errors": errors[:5],
    }

# ═══════════════════════════════════════════════════════════════════════════════
# Report writer
# ═══════════════════════════════════════════════════════════════════════════════
def write_report(t1: dict, t2: dict, t3: dict, t4: dict) -> str:
    sep  = "=" * 68
    sep2 = "-" * 68
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    bias_verdict = "⚠️  BIAS DETECTED — byline leakage confirmed" if t1["bias_detected"] else "✅ PASS — No significant byline bias"
    adv_verdict  = "✅ ROBUST" if (t2["fake_flipped"] + t2["true_flipped"]) <= 5 else "⚠️  SENSITIVE to surface-level wording"
    cd_verdict   = "✅ GENERALISES" if t3["accuracy"] >= 0.7 else "⚠️  DOMAIN-SPECIFIC (ISOT era only)"

    lines = [
        sep, f"  FAKE NEWS DETECTION — ROBUSTNESS REPORT",
        f"  Generated : {ts}", sep, "",
        "## TEST 1 — Byline Ablation (Reuters Litmus Test)",
        sep2,
        f"  Samples tested    : {t1['n']} True news articles",
        f"  Baseline accuracy : {t1['baseline_acc']*100:.1f}% (with bylines present)",
        f"  Post-strip acc.   : {t1['stripped_acc']*100:.1f}% (bylines removed)",
        f"  Accuracy drop     : {t1['accuracy_drop']*100:.1f}%",
        f"  Verdict           : {bias_verdict}", "",
        "## TEST 2 — Adversarial Perturbation",
        sep2,
        f"  Fake samples: {t2['fake_flipped']}/{t2['fake_total']} flipped after neutralising inflammatory words",
        f"  True samples: {t2['true_flipped']}/{t2['true_total']} flipped after appending social-spam suffix",
        f"  Verdict     : {adv_verdict}", "",
        "## TEST 3 — Cross-Domain Generalisation (2024-2026 News)",
        sep2,
        f"  Samples : {t3['n']}  |  Accuracy : {t3['accuracy']*100:.1f}%",
        f"  Verdict : {cd_verdict}", "",
        "  Sample breakdown:",
    ]

    for r in t3["samples"]:
        tick = "✓" if r["correct"] else "✗"
        lines.append(f"  [{tick}] Exp:{r['expected']:4s} Got:{r['predicted']:4s} "
                     f"({r['confidence']*100:.1f}%) — {r['snippet']}")

    lines += [
        "", "## TEST 4 — Error Probability Distribution",
        sep2,
        f"  Sample size         : 500 test records",
        f"  Errors found        : {t4['n_errors']}",
    ]

    if t4['avg_conf'] is not None:
        lines += [
            f"  Avg. confidence     : {t4['avg_conf']*100:.1f}% on wrong predictions",
            f"  Confident errors    : {t4['confident_errors']} (>80% confidence, wrong label)",
            f"  Close calls         : {t4['close_calls']} (≤60% confidence, wrong label)",
            "",
            "  Sample misclassifications:"
        ]
        for e in t4["sample_errors"]:
            lines.append(f"  True:{e['true']:4s} → Predicted:{e['predicted']:4s} "
                         f"with {e['conf_wrong']*100:.1f}% confidence")
    else:
        lines.append("  No errors found in sample — perfect window.")

    lines += ["", sep,
              "## Summary & Recommendations", sep2]

    recs = []
    if t1["bias_detected"]:
        recs.append("• RETRAIN with byline-stripping regex in preprocessing (already added to v2 pipeline).")
    if t2["fake_flipped"] + t2["true_flipped"] > 5:
        recs.append("• Model is surface-sensitive. Consider data augmentation with paraphrase variants.")
    if t3["accuracy"] < 0.7:
        recs.append("• Supplement ISOT with a modern dataset (e.g. FakeNewsNet or LIAR-Plus) for temporal robustness.")
    if t4.get("confident_errors", 0) > 5:
        recs.append("• High-confidence errors suggest calibration issues. Consider Platt scaling or temperature tuning.")
    if not recs:
        recs.append("• No critical issues found. Pipeline is robust across all four test dimensions.")

    lines += recs + ["", sep]
    return "\n".join(lines)


# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    pipeline = load_model()
    df_test  = load_isot_test_split()

    t1 = test1_byline_ablation(pipeline, df_test)
    t2 = test2_adversarial(pipeline, df_test)
    t3 = test3_cross_domain(pipeline)
    t4 = test4_error_distribution(pipeline, df_test)

    report = write_report(t1, t2, t3, t4)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Report saved → {REPORT_PATH}")
    print("\n" + report)


if __name__ == "__main__":
    main()
