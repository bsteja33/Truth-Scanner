import os
import gc
import re
import logging
import joblib
import numpy as np
import pandas as pd
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import logging
import gc
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    logger.warning("SpaCy model not found. Downloading en_core_web_sm...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


class SpacyPreprocessor(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer that cleans and lemmatises text."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.Series) -> List[str]:
        return [self._clean(doc) for doc in X]

    def _clean(self, text: str) -> str:
        """Return a lemmatised, stopword-free string from raw text."""
        try:
            if not isinstance(text, str):
                text = str(text)
            text = text.lower()
            # Strip bylines: "CITY (Agency) -" and "(Agency) -"
            text = re.sub(r'[A-Z\s,]+\s*\([^)]{1,20}\)\s*[-\u2013\u2014]\s*', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\([^)]{1,20}\)\s*[-\u2013\u2014]\s*', '', text)
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            text = re.sub(r'<.*?>', '', text)
            text = re.sub(r'[^\w\s]', '', text)
            doc = nlp(text)
            tokens = [t.lemma_ for t in doc if not t.is_stop and t.lemma_.strip()]
            return " ".join(tokens)
        except Exception as e:
            logger.error("Preprocessing error: %s", e)
            return ""


def main() -> None:
    """Train the ensemble pipeline on the ISOT dataset and serialise it."""
    logger.info("Loading ISOT dataset...")
    true_df = pd.read_csv("data/True.csv")
    fake_df = pd.read_csv("data/Fake.csv")
    true_df["Label"] = 1
    fake_df["Label"] = 0

    logger.info("Merging datasets...")
    df = pd.concat([true_df, fake_df], ignore_index=True)
    del true_df, fake_df
    gc.collect()

    df.dropna(subset=["title", "text", "Label"], inplace=True)
    df["original_text"] = df["title"].astype(str) + " " + df["text"].astype(str)

    X = df["original_text"]
    y = df["Label"].astype(int)

    logger.info("Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    features = FeatureUnion([
        ("count", CountVectorizer(max_features=5000)),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
    ])

    clf1 = MultinomialNB()
    clf2 = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf3 = SVC(kernel="linear", probability=True)
    clf4 = SGDClassifier(loss="log_loss", max_iter=1000, random_state=42)
    clf5 = RandomForestClassifier(n_estimators=100, random_state=42)

    ensemble = VotingClassifier(
        estimators=[
            ("mnb", clf1),
            ("lr",  clf2),
            ("svc", clf3),
            ("sgd", clf4),
            ("rf",  clf5),
        ],
        voting="soft",
    )

    pipeline = Pipeline([
        ("preprocessor", SpacyPreprocessor()),
        ("features",     features),
        ("classifier",   ensemble),
    ])

    logger.info("Training pipeline...")
    pipeline.fit(X_train, y_train)

    logger.info("Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    logger.info("\n%s", classification_report(y_test, y_pred))
    logger.info("Confusion matrix:\n%s", confusion_matrix(y_test, y_pred))

    logger.info("Saving pipeline to models/ensemble_pipeline.pkl")
    joblib.dump(pipeline, "models/ensemble_pipeline.pkl")
    logger.info("Done.")


if __name__ == "__main__":
    main()
