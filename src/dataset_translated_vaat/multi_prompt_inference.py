#!/usr/bin/env python3
"""
multi_prompt_inference.py

Multi-prompt (multi-template) inference for YOUR trained sarcasm classifier.
Works with:
- Standard HF checkpoints
- Custom-decoder checkpoints (CNN / attention pooling)

It wraps the SAME input text with multiple templates and aggregates probs.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model_io import load_model_and_tokenizer

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


# ----------------------------
# Templates
# ----------------------------

@dataclass
class Template:
    name: str
    prefix: str
    suffix: str = ""
    weight: float = 1.0


DEFAULT_TEMPLATES = [
    Template(
        name="direct",
        prefix="Task: Sarcasm detection. Decide if the text is sarcastic.\nText: ",
        weight=1.0,
    ),
    Template(
        name="definition",
        prefix=("Task: Sarcasm detection.\n"
                "Definition: Sarcasm often means the opposite of the literal words; it can be ironic or mocking.\n"
                "Text: "),
        weight=1.0,
    ),
    Template(
        name="context_cross_variety",
        prefix=("Task: Sarcasm detection across English varieties (UK/IN/AU).\n"
                "Focus on pragmatic cues (irony, exaggeration, implicit criticism), not only literal meaning.\n"
                "Text: "),
        weight=1.0,
    ),
    Template(
        name="evidence_based",
        prefix=("Task: Sarcasm detection.\n"
                "Consider cues like contradiction, exaggeration, or implicit criticism.\n"
                "Text: "),
        weight=1.0,
    ),
]


# ----------------------------
# Metrics
# ----------------------------

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1s = []
    for cls in [0, 1]:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        f1s.append(f1)
    return float(np.mean(f1s))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def precision_recall_f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(precision), float(recall), float(f1)


# ----------------------------
# Core inference
# ----------------------------

@torch.no_grad()
def predict_probs_for_texts(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 256,
) -> np.ndarray:
    """
    Returns probs: (N, 2) for labels [0,1]
    """
    def collate(batch_texts: List[str]) -> Dict[str, torch.Tensor]:
        return tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    loader = DataLoader(texts, batch_size=batch_size, shuffle=False, collate_fn=collate)

    all_probs = []
    iterator = tqdm(loader, desc="Inference", leave=False) if tqdm is not None else loader

    for batch in iterator:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs.detach().cpu().numpy())

    return np.vstack(all_probs)


def aggregate_probs(
    per_template_probs: List[np.ndarray],   # list of (N,2)
    templates: List[Template],
    method: str = "weighted_mean",          # weighted_mean | mean | majority
) -> Tuple[np.ndarray, np.ndarray]:
    N = per_template_probs[0].shape[0]
    K = len(per_template_probs)

    if method == "majority":
        votes = np.stack([p.argmax(axis=1) for p in per_template_probs], axis=1)  # (N,K)
        ones = votes.sum(axis=1)
        zeros = K - ones
        final_pred = (ones > zeros).astype(int)
        p1 = ones / K
        p0 = 1.0 - p1
        final_prob = np.stack([p0, p1], axis=1)
        return final_pred, final_prob

    if method == "mean":
        avg = np.mean(np.stack(per_template_probs, axis=0), axis=0)
        final_pred = avg.argmax(axis=1)
        return final_pred, avg

    # weighted_mean
    weights = np.array([t.weight for t in templates], dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)
    stacked = np.stack(per_template_probs, axis=0)  # (K,N,2)
    avg = np.tensordot(weights, stacked, axes=(0, 0))  # (N,2)
    final_pred = avg.argmax(axis=1)
    return final_pred, avg


def run_multi_prompt_classifier(
    checkpoint_dir: str,
    df: pd.DataFrame,
    templates: List[Template],
    aggregation: str,
    device: torch.device,
    batch_size: int,
    max_length: int,
    task_filter: str = "Sarcasm",
) -> pd.DataFrame:
    if "task" in df.columns:
        df = df[df["task"].astype(str) == task_filter].copy()

    if df.empty:
        raise ValueError(f"No rows after filtering task='{task_filter}'.")

    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column.")

    model, tokenizer = load_model_and_tokenizer(checkpoint_dir, device=str(device))
    base_texts = df["text"].astype(str).tolist()

    per_template_probs = []
    for t in templates:
        wrapped = [t.prefix + x + t.suffix for x in base_texts]
        probs = predict_probs_for_texts(
            model=model,
            tokenizer=tokenizer,
            texts=wrapped,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        per_template_probs.append(probs)

        df[f"p0_{t.name}"] = probs[:, 0]
        df[f"p1_{t.name}"] = probs[:, 1]
        df[f"pred_{t.name}"] = probs.argmax(axis=1)

    final_pred, final_prob = aggregate_probs(per_template_probs, templates, method=aggregation)
    df["final_pred"] = final_pred
    df["final_p0"] = final_prob[:, 0]
    df["final_p1"] = final_prob[:, 1]
    return df


def evaluate(df: pd.DataFrame, label_col: str = "label", group_col: Optional[str] = None) -> None:
    if label_col not in df.columns:
        print(f"[warn] No ground-truth column '{label_col}'. Skipping evaluation.")
        return

    y_true = df[label_col].astype(int).to_numpy()
    y_pred = df["final_pred"].astype(int).to_numpy()

    acc = accuracy(y_true, y_pred)
    prec, rec, f1_bin = precision_recall_f1_binary(y_true, y_pred)
    f1_mac = macro_f1(y_true, y_pred)

    print("\n=== Overall Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (pos=1): {prec:.4f}")
    print(f"Recall    (pos=1): {rec:.4f}")
    print(f"F1 (binary): {f1_bin:.4f}")
    print(f"F1 (macro):  {f1_mac:.4f}")

    if group_col and group_col in df.columns:
        print(f"\n=== Macro F1 by {group_col} ===")
        for g, sub in df.groupby(group_col):
            yt = sub[label_col].astype(int).to_numpy()
            yp = sub["final_pred"].astype(int).to_numpy()
            print(f"{g}: macroF1={macro_f1(yt, yp):.4f}  n={len(sub)}")


def main():
    parser = argparse.ArgumentParser(description="Multi-prompt inference for BESSTIE sarcasm classifier.")
    parser.add_argument("--checkpoint_dir", type=str, default="./model_output/")
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default="multi_prompt_predictions.csv")
    parser.add_argument("--aggregation", type=str, choices=["weighted_mean", "mean", "majority"], default="weighted_mean")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--task_filter", type=str, default="Sarcasm")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--group_col", type=str, default="variety")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if not args.input_file or not os.path.exists(args.input_file):
        raise FileNotFoundError("Provide --input_file pointing to valid/test CSV.")

    df = pd.read_csv(args.input_file)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"[info] device = {device}")

    templates = DEFAULT_TEMPLATES

    out_df = run_multi_prompt_classifier(
        checkpoint_dir=args.checkpoint_dir,
        df=df,
        templates=templates,
        aggregation=args.aggregation,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        task_filter=args.task_filter,
    )

    evaluate(out_df, label_col=args.label_col, group_col=args.group_col)

    out_df.to_csv(args.output_file, index=False)
    print(f"\nSaved predictions to: {args.output_file}")


if __name__ == "__main__":
    main()
