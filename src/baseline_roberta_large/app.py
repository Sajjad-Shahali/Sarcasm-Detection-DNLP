"""
Minimal FastAPI application for serving the BESSTIE sarcasm/sentiment detector.  ssss

This API exposes a single endpoint, ``/predict``, which accepts a JSON
payload containing a list of texts and returns the predicted labels (0 or
1).  The model and tokenizer are loaded at startup to avoid overhead on
each request.  To run the server locally::

    uvicorn app:app --host 0.0.0.0 --port 8000

Make sure to install ``fastapi`` and ``uvicorn``.  You can then query the
endpoint via HTTP POST:

    curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" \
         -d '{"texts": ["I love this", "This is awful"]}'
"""

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel


class PredictRequest(BaseModel):
    texts: List[str]


class PredictResponse(BaseModel):
    predictions: List[int]


def create_app(checkpoint_dir: str) -> FastAPI:
    """Factory to create a FastAPI app with a pre‑loaded model.

    The model and tokenizer are both loaded from the ``checkpoint_dir``.  This
    mirrors the behaviour of the inference helper and ensures that any
    special tokens added during training are correctly handled.

    Parameters
    ----------
    checkpoint_dir : str
        Directory containing the fine‑tuned model and tokenizer.

    Returns
    -------
    FastAPI
        Configured application.
    """
    app = FastAPI(title="BESSTIE Figurative Language Detection API")
    # Pre‑load model and tokenizer once.  Use closure to capture them in the endpoint.
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    device_str = os.environ.get("BESSTIE_DEVICE", "auto")
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    model.to(device)
    model.eval()

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest) -> PredictResponse:
        # Tokenise and predict.  Use CPU for inference to avoid GPU dependency.
        inputs = tokenizer(request.texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
        preds = logits.argmax(dim=-1).tolist()
        return PredictResponse(predictions=preds)

    return app


# Instantiate default app using environment variables or a default checkpoint directory
import os
CHECKPOINT_DIR = os.environ.get("BESSTIE_CHECKPOINT_DIR", "./model_output")
app = create_app(CHECKPOINT_DIR)
