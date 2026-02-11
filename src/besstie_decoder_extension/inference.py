"""
inference.py

Inference script for BESSTIE.
Supports:
- Standard HF checkpoints (AutoModelForSequenceClassification)
- Custom-decoder checkpoints (encoder + CNN/attention head)

If a checkpoint folder contains:
  - decoder_config.json
  - decoder_head.pt
it will automatically load the custom decoder head.
"""

from __future__ import annotations

import argparse
import os
from typing import List

import pandas as pd
import torch

from model_io import load_model_and_tokenizer


def predict_binary(
    checkpoint_dir: str,
    texts: List[str],
    device: str = "auto",
    batch_size: int = 32,
    max_length: int = 256,
    fp16: bool = False,
    bf16: bool = False,
) -> List[int]:
    if fp16 and bf16:
        raise ValueError("Choose only one of fp16 or bf16.")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)
    use_cuda = device_t.type == "cuda"

    model, tokenizer = load_model_and_tokenizer(checkpoint_dir, device=device)

    from torch.utils.data import DataLoader

    def collate(batch_texts: List[str]):
        return tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    loader = DataLoader(texts, batch_size=batch_size, shuffle=False, collate_fn=collate)

    autocast_dtype = None
    if use_cuda and fp16:
        autocast_dtype = torch.float16
    elif use_cuda and bf16:
        autocast_dtype = torch.bfloat16
    use_autocast = autocast_dtype is not None

    preds: List[int] = []
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device_t, non_blocking=use_cuda) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
                logits = model(**batch).logits
            preds.extend(logits.argmax(dim=-1).tolist())

    return preds


def main():
    parser = argparse.ArgumentParser(description="Generate predictions using a trained BESSTIE model.")
    parser.add_argument("--checkpoint_dir", type=str, default="./model_output/")
    parser.add_argument("--input_file", type=str, default=None, help="CSV with a 'text' column")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--text", type=str, nargs="*", default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    if args.input_file:
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(args.input_file)

        df = pd.read_csv(args.input_file)
        if "text" not in df.columns:
            raise ValueError("Input CSV must contain a 'text' column")

        texts = df["text"].astype(str).tolist()
        preds = predict_binary(
            args.checkpoint_dir,
            texts,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            fp16=args.fp16,
            bf16=args.bf16,
        )
        df["prediction"] = preds

        out_path = args.output_file or os.path.splitext(args.input_file)[0] + "_predictions.csv"
        df.to_csv(out_path, index=False)
        print(f"Predictions written to {out_path}")
        return

    if not args.text:
        raise ValueError("Provide either --input_file or --text")

    preds = predict_binary(
        args.checkpoint_dir,
        args.text,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        fp16=args.fp16,
        bf16=args.bf16,
    )
    for t, p in zip(args.text, preds):
        print(f"{p}\t{t}")


if __name__ == "__main__":
    main()
