"""
app.py

FastAPI server for BESSTIE.
Supports both:
- Standard HuggingFace checkpoints
- Custom-decoder checkpoints (CNN / attention pooling)
"""

from __future__ import annotations

import os
from typing import List

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from model_io import load_model_and_tokenizer


class PredictRequest(BaseModel):
    texts: List[str]


class PredictResponse(BaseModel):
    predictions: List[int]


def create_app(checkpoint_dir: str) -> FastAPI:
    app = FastAPI(title="BESSTIE Figurative Language Detection API")

    device_str = os.environ.get("BESSTIE_DEVICE", "auto")
    model, tokenizer = load_model_and_tokenizer(checkpoint_dir, device=device_str)

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest) -> PredictResponse:
        inputs = tokenizer(request.texts, padding=True, truncation=True, return_tensors="pt")
        # move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
        preds = logits.argmax(dim=-1).tolist()
        return PredictResponse(predictions=preds)

    return app


CHECKPOINT_DIR = os.environ.get("BESSTIE_CHECKPOINT_DIR", "./model_output")
app = create_app(CHECKPOINT_DIR)
