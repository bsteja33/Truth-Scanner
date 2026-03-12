import os
import sys
import logging
from typing import Any, Optional

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.train import SpacyPreprocessor  # noqa: E402

app = FastAPI(title="Fake News Detection API")

_MAX_CHARS = 50_000

model: Optional[Any] = None


@app.on_event("startup")
def load_model() -> None:
    """Load the serialised pipeline into memory on server start."""
    global model
    try:
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "models", "ensemble_pipeline.pkl",
        )
        logger.info("Loading model from %s", model_path)
        model = joblib.load(model_path)
        logger.info("Model loaded.")
    except Exception as e:
        logger.error("Model load failed: %s", e)
        raise RuntimeError("Model loading failed.") from e


class NewsRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_must_be_non_empty(cls, v: str) -> str:
        """Reject blank or whitespace-only payloads with HTTP 422."""
        if not v or not v.strip():
            raise ValueError("text field must not be empty or whitespace.")
        return v[:_MAX_CHARS]


class NewsResponse(BaseModel):
    prediction: str
    confidence_score: float


@app.post("/predict", response_model=NewsResponse)
def predict(request: NewsRequest) -> NewsResponse:
    """Return a prediction and confidence score for the supplied text."""
    if model is None:
        logger.error("Prediction requested but model is not loaded.")
        raise HTTPException(status_code=503, detail="Model not available.")

    text = request.text
    try:
        pred       = model.predict([text])[0]
        proba      = model.predict_proba([text])[0]
        confidence = float(proba[pred])
        return NewsResponse(
            prediction="TRUE" if pred == 1 else "FALSE",
            confidence_score=confidence,
        )
    except Exception as e:
        logger.error("Prediction error for text '%s...': %s", text[:40], e)
        raise HTTPException(status_code=500, detail="Prediction failed internally.")
