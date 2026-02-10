"""
Minimal FastAPI application for serving the BESSTIE decoder baseline.
"""

import os
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
import torch

from inference import load_model_and_tokenizer
from utils import build_prompt_text


class PredictRequest(BaseModel):
    texts: List[str]


class PredictResponse(BaseModel):
    predictions: List[int]


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y"}


def _parse_label(text: str) -> int:
    for ch in text:
        if ch == "0":
            return 0
        if ch == "1":
            return 1
    return 0


def create_app(checkpoint_dir: str) -> FastAPI:
    app = FastAPI(title="BESSTIE Decoder Baseline API")

    task = os.environ.get("BESSTIE_TASK", "Sarcasm")
    prompt_sentiment = os.environ.get(
        "BESSTIE_PROMPT_SENTIMENT",
        "Generate the sentiment of the given text. 1 for positive sentiment, and 0 for negative sentiment. Do not give an explanation.",
    )
    prompt_sarcasm = os.environ.get(
        "BESSTIE_PROMPT_SARCASM",
        "Predict if the given text is sarcastic. 1 if the text is sarcastic, and 0 if the text is not sarcastic. Do not give an explanation.",
    )
    prompt_template = os.environ.get("BESSTIE_PROMPT_TEMPLATE", "{prompt}\n{text}\n")

    device = os.environ.get("BESSTIE_DEVICE", "auto")
    use_qlora = _env_bool("BESSTIE_USE_QLORA", True)
    bnb_4bit_quant_type = os.environ.get("BESSTIE_BNB_4BIT_QUANT_TYPE", "nf4")
    bnb_4bit_use_double_quant = _env_bool("BESSTIE_BNB_4BIT_USE_DOUBLE_QUANT", True)
    compute_dtype = os.environ.get("BESSTIE_COMPUTE_DTYPE", "bf16")
    fp16 = _env_bool("BESSTIE_FP16", False)
    bf16 = _env_bool("BESSTIE_BF16", True)
    max_length = int(os.environ.get("BESSTIE_MAX_LENGTH", "256"))
    max_new_tokens = int(os.environ.get("BESSTIE_MAX_NEW_TOKENS", "1"))
    batch_size = int(os.environ.get("BESSTIE_BATCH_SIZE", "4"))

    model, tokenizer, device_t = load_model_and_tokenizer(
        checkpoint_dir,
        device=device,
        use_qlora=use_qlora,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        compute_dtype=compute_dtype,
    )

    autocast_dtype = None
    if device_t.type == "cuda" and fp16:
        autocast_dtype = torch.float16
    elif device_t.type == "cuda" and bf16:
        autocast_dtype = torch.bfloat16
    use_autocast = autocast_dtype is not None

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest) -> PredictResponse:
        prompts = [
            build_prompt_text(task, t, prompt_sentiment, prompt_sarcasm, prompt_template)
            for t in request.texts
        ]
        preds: List[int] = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            inputs = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device_t, non_blocking=device_t.type == "cuda") for k, v in inputs.items()}
            with torch.inference_mode():
                with torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                    )
            prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()
            for row, prompt_len in zip(outputs, prompt_lens):
                gen_ids = row[int(prompt_len):]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                preds.append(_parse_label(text))
        return PredictResponse(predictions=preds)

    return app


CHECKPOINT_DIR = os.environ.get("BESSTIE_CHECKPOINT_DIR", "./model_output")
app = create_app(CHECKPOINT_DIR)
