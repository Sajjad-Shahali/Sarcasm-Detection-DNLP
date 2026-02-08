"""
Compute macro F1 (and basic metrics) from a CSV with labels and predictions.

Usage examples:
  python eval_macro_f1.py --input_file path/to/valid_predictions.csv
  python eval_macro_f1.py --input_file preds.csv --label_col label --pred_col final_pred
  python eval_macro_f1.py --input_file preds.csv --task Sarcasm --task_col task
  python eval_macro_f1.py --input_file preds.csv --group_col variety
"""

import argparse
import pandas as pd
import numpy as np


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


def precision_recall_f1_binary(y_true: np.ndarray, y_pred: np.ndarray):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(precision), float(recall), float(f1)


def main():
    parser = argparse.ArgumentParser(description="Compute macro F1 from a CSV.")
    parser.add_argument("--input_file", type=str, required=True, help="CSV with labels and predictions")
    parser.add_argument("--label_col", type=str, default="label", help="Ground-truth label column")
    parser.add_argument("--pred_col", type=str, default="prediction", help="Prediction column")
    parser.add_argument("--task", type=str, help="Optional task filter (e.g., Sarcasm)")
    parser.add_argument("--task_col", type=str, default="task", help="Task column name")
    parser.add_argument("--group_col", type=str, help="Optional group-by column (e.g., variety)")
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)

    if args.task:
        if args.task_col not in df.columns:
            raise ValueError(f"task_col '{args.task_col}' not found in CSV")
        df = df[df[args.task_col].astype(str).str.lower() == args.task.lower()].copy()

    if args.label_col not in df.columns:
        raise ValueError(f"label_col '{args.label_col}' not found in CSV")
    if args.pred_col not in df.columns:
        raise ValueError(f"pred_col '{args.pred_col}' not found in CSV")

    y_true = df[args.label_col].astype(int).to_numpy()
    y_pred = df[args.pred_col].astype(int).to_numpy()

    acc = accuracy(y_true, y_pred)
    prec, rec, f1_bin = precision_recall_f1_binary(y_true, y_pred)
    f1_mac = macro_f1(y_true, y_pred)

    print("=== Overall Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (pos=1): {prec:.4f}")
    print(f"Recall    (pos=1): {rec:.4f}")
    print(f"F1 (binary): {f1_bin:.4f}")
    print(f"F1 (macro):  {f1_mac:.4f}")

    if args.group_col:
        if args.group_col not in df.columns:
            raise ValueError(f"group_col '{args.group_col}' not found in CSV")
        print(f"\n=== Macro F1 by {args.group_col} ===")
        for g, sub in df.groupby(args.group_col):
            yt = sub[args.label_col].astype(int).to_numpy()
            yp = sub[args.pred_col].astype(int).to_numpy()
            print(f"{g}: macroF1={macro_f1(yt, yp):.4f}  n={len(sub)}")


if __name__ == "__main__":
    main()
