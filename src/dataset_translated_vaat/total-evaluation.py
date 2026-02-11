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
    variety_ids: list = None,
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
        #  VAAT conditioning
        if variety_ids is not None:
            batch_var_ids = variety_ids[i : i + batch_size]
            enc["variety_ids"] = torch.tensor(batch_var_ids, dtype=torch.long, device=device)
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
    save_all: bool = False,
):
    model, tokenizer = load_model_and_tokenizer(model_dir, device=device)
    model_name = os.path.basename(model_dir)

    # keep needed columns (source optional)
    keep_cols = ["text", "label", "variety"]
    if "source" in data.columns:
        keep_cols.append("source")

    df = data[keep_cols].copy()

    # Adding model name to DataFrame to avoid KeyError when using groupby
    df["model"] = model_name

    # VAAT: if the loaded model exposes a variety_to_id mapping, build variety_ids for conditioning.
    variety_ids = None
    if hasattr(model, "variety_to_id"):
        if "variety" not in df.columns:
            raise ValueError("Model requires variety conditioning but 'variety' column is missing.")
        variety_ids = [int(model.variety_to_id.get(str(v), 0)) for v in df["variety"].tolist()]
    preds, latency_ms = predict_in_batches(
        model=model,
        tokenizer=tokenizer,
        texts=df["text"].tolist(),
        variety_ids=variety_ids,
        batch_size=batch_size,
        max_length=max_length,
    )

    df["prediction"] = preds

    # -----------------------------
    # 1) General evaluation metrics (no grouping)
    # -----------------------------
    results = _compute_group_metrics(
        df=df,
        group_cols=["model"],
        model_name=model_name,
        min_group_size=min_group_size,
    )

    # Latency row
    results.append({
        "model": model_name,
        "num_samples": int(df.shape[0]),
        "accuracy": None,
        "f1_macro": float(latency_ms),
        "precision": None,
        "recall": None,
        "tp": None,
        "fp": None,
        "fn": None,
        "tn": None,
    })

    # Save final result to a file
    if save_all:
        result_df = pd.DataFrame(results)
        result_df.to_csv(f"results\evaluation_{model_name}_summary.csv", index=False)
        print(f"Saved: evaluation_{model_name}_summary.csv")
    else:
        print("Evaluation complete, but no file saved.")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_root",  help="Folder containing model subfolders", default=r".\model_output")
    parser.add_argument("--validation_csv",  help="CSV with columns: text, label, variety (and optionally source)", default=r".\dataset\valid-new.csv")
    parser.add_argument("--output_prefix", help="Prefix for output CSV files",default="tdata_vat")
    parser.add_argument("--device", help="auto | cpu | cuda",default="auto")
    parser.add_argument("--batch_size", type=int,  help="CPU: 2-8; GPU: 16-64",default=16)
    parser.add_argument("--max_length", type=int, default=256, help="Try 128 if RAM is tight on CPU")
    parser.add_argument("--min_group_size", type=int, default=1, help="Skip groups with fewer than this many samples")
    parser.add_argument("--save_all", type=bool, default=True, help="Whether to save all output results")

    args = parser.parse_args()

    data = pd.read_csv(args.validation_csv)

    required = {"text", "label", "variety"}
    if not required.issubset(set(data.columns)):
        raise ValueError(f"validation_csv must contain columns {required}. Found: {list(data.columns)}")

    model_folders = [d for d in os.listdir(args.models_root) if os.path.isdir(os.path.join(args.models_root, d))]
    if len(model_folders) == 0:
        raise ValueError(f"No model folders found in: {args.models_root}")

    for model_name in model_folders:
        model_dir = os.path.join(args.models_root, model_name)
        print(f"\nEvaluating model: {model_name}")

        evaluate_model(
            model_dir=model_dir,
            data=data,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            min_group_size=args.min_group_size,
            save_all=args.save_all,
        )


if __name__ == "__main__":
    main()
