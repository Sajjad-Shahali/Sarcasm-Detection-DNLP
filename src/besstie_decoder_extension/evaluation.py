# evaluation.py
# Evaluate multiple trained models and save results to CSV
# Outputs:
#   - per_variety_results.csv           (as before)
#   - per_source_results.csv            (new)
#   - per_variety_source_results.csv    (new)

import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import time
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from model_io import load_model_and_tokenizer


def _get_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def predict_in_batches(
    model: torch.nn.Module,
    tokenizer,
    texts: list,
    batch_size: int = 4,
    max_length: int = 256,
):
    device = _get_model_device(model)
    model.eval()

    all_preds = []
    start_time = time.time()

    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting", unit="batch"):
        batch_texts = texts[i : i + batch_size]

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.inference_mode():
            out = model(**enc)
            logits = out.logits if hasattr(out, "logits") else out[0]
            preds = logits.argmax(dim=-1).detach().cpu().numpy()

        all_preds.append(preds)

        del enc, out, logits
        if device.type == "cuda":
            torch.cuda.empty_cache()

    end_time = time.time()
    latency_ms = (end_time - start_time) / max(len(texts), 1) * 1000
    return np.concatenate(all_preds, axis=0), latency_ms


def _compute_group_metrics(
    df: pd.DataFrame,
    group_cols,
    model_name: str,
    min_group_size: int = 1,
):
    """
    Compute metrics for each group in df grouped by group_cols.
    Returns list of dict rows.
    """
    results = []

    for keys, g in df.groupby(group_cols):
        if len(g) < min_group_size:
            continue

        # normalize keys into a tuple
        if not isinstance(keys, tuple):
            keys = (keys,)

        y_true = g["label"].to_numpy()
        y_pred = g["prediction"].to_numpy()

        # Confusion matrix safe (binary)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        row = {
            "model": model_name,
            "num_samples": int(len(g)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }

        # attach group columns
        for col, val in zip(group_cols, keys):
            row[col] = val

        results.append(row)

    return results


def evaluate_model(
    model_dir: str,
    data: pd.DataFrame,
    device: str = "auto",
    batch_size: int = 4,
    max_length: int = 256,
    min_group_size: int = 1,
):
    model, tokenizer = load_model_and_tokenizer(model_dir, device=device)
    model_name = os.path.basename(model_dir)

    # keep needed columns (source optional)
    keep_cols = ["text", "label", "variety"]
    if "source" in data.columns:
        keep_cols.append("source")

    df = data[keep_cols].copy()

    preds, latency_ms = predict_in_batches(
        model=model,
        tokenizer=tokenizer,
        texts=df["text"].tolist(),
        batch_size=batch_size,
        max_length=max_length,
    )

    df["prediction"] = preds

    # -----------------------------
    # Error samples (per-variety) - keep as before
    # -----------------------------
    errors = df[df["label"] != df["prediction"]].copy()
    errors["model"] = model_name
    errors = errors.groupby("variety").head(3)
    errors.to_csv(f"errors_{model_name}.csv", index=False)

    # -----------------------------
    # 1) Per-variety metrics (same as before, but now richer columns)
    # -----------------------------
    per_variety = _compute_group_metrics(
        df=df,
        group_cols=["variety"],
        model_name=model_name,
        min_group_size=min_group_size,
    )

    # Robustness gap over varieties
    f1s = [r["f1_macro"] for r in per_variety]
    if len(f1s) > 0:
        per_variety.append(
            {
                "model": model_name,
                "variety": "__ROBUSTNESS_GAP__",
                "num_samples": int(df.shape[0]),
                "accuracy": None,
                "f1_macro": float(max(f1s) - min(f1s)),
                "precision": None,
                "recall": None,
                "tp": None,
                "fp": None,
                "fn": None,
                "tn": None,
            }
        )

    # Latency row (kept same convention: stored in f1_macro column)
    per_variety.append(
        {
            "model": model_name,
            "variety": "__LATENCY_MS_PER_SAMPLE__",
            "num_samples": int(df.shape[0]),
            "accuracy": None,
            "f1_macro": float(latency_ms),
            "precision": None,
            "recall": None,
            "tp": None,
            "fp": None,
            "fn": None,
            "tn": None,
        }
    )

    # -----------------------------
    # 2) Per-source metrics (new)
    # -----------------------------
    per_source = []
    if "source" in df.columns:
        per_source = _compute_group_metrics(
            df=df,
            group_cols=["source"],
            model_name=model_name,
            min_group_size=min_group_size,
        )

    # -----------------------------
    # 3) Per-variety × per-source metrics (new)
    # -----------------------------
    per_variety_source = []
    if "source" in df.columns:
        per_variety_source = _compute_group_metrics(
            df=df,
            group_cols=["variety", "source"],
            model_name=model_name,
            min_group_size=min_group_size,
        )

    return per_variety, per_source, per_variety_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_root", required=True, help="Folder containing model subfolders")
    parser.add_argument("--validation_csv", required=True, help="CSV with columns: text, label, variety (and optionally source)")
    parser.add_argument("--output_prefix", default="results", help="Prefix for output CSV files")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--batch_size", type=int, default=4, help="CPU: 2-8; GPU: 16-64")
    parser.add_argument("--max_length", type=int, default=256, help="Try 128 if RAM is tight on CPU")
    parser.add_argument("--min_group_size", type=int, default=1, help="Skip groups with fewer than this many samples")
    args = parser.parse_args()

    data = pd.read_csv(args.validation_csv)

    required = {"text", "label", "variety"}
    if not required.issubset(set(data.columns)):
        raise ValueError(f"validation_csv must contain columns {required}. Found: {list(data.columns)}")

    model_folders = [d for d in os.listdir(args.models_root) if os.path.isdir(os.path.join(args.models_root, d))]
    if len(model_folders) == 0:
        raise ValueError(f"No model folders found in: {args.models_root}")

    all_variety = []
    all_source = []
    all_variety_source = []

    for model_name in model_folders:
        model_dir = os.path.join(args.models_root, model_name)
        print(f"\nEvaluating model: {model_name}")

        per_variety, per_source, per_variety_source = evaluate_model(
            model_dir=model_dir,
            data=data,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            min_group_size=args.min_group_size,
        )

        all_variety.extend(per_variety)
        all_source.extend(per_source)
        all_variety_source.extend(per_variety_source)

    # Save outputs
    out_variety = pd.DataFrame(all_variety).sort_values(["model", "variety"], ascending=[True, True])
    out_variety.to_csv(f"{args.output_prefix}_per_variety.csv", index=False)
    print(f"\nSaved: {args.output_prefix}_per_variety.csv")

    if len(all_source) > 0:
        out_source = pd.DataFrame(all_source).sort_values(["model", "source"], ascending=[True, True])
        out_source.to_csv(f"{args.output_prefix}_per_source.csv", index=False)
        print(f"Saved: {args.output_prefix}_per_source.csv")
    else:
        print("No 'source' column found -> skipping per-source outputs.")

    if len(all_variety_source) > 0:
        out_vs = pd.DataFrame(all_variety_source).sort_values(["model", "variety", "source"], ascending=[True, True, True])
        out_vs.to_csv(f"{args.output_prefix}_per_variety_source.csv", index=False)
        print(f"Saved: {args.output_prefix}_per_variety_source.csv")
    else:
        print("No 'source' column found -> skipping per-variety×source outputs.")


if __name__ == "__main__":
    main()
