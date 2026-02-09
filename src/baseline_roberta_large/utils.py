"""
Utility functions for the BESSTIE figurative language project.

This module centralises common helpers such as dataset loading, tokenisation,
class‑weight computation and metric calculation.  Separating these utilities
allows ``train.py`` and ``inference.py`` to stay concise.

The functions here are general enough to be reused for both binary and
multi‑task classification settings.wwww
"""

import os
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np

try:
    from datasets import Dataset, DatasetDict, load_dataset  # type: ignore
except ImportError:
    # ``datasets`` is an optional dependency.  When not installed the functions
    # that rely on it will raise informative errors.
    Dataset = None  # type: ignore
    DatasetDict = None  # type: ignore
    load_dataset = None  # type: ignore

from transformers import AutoTokenizer, DataCollatorWithPadding
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def load_besstie_from_csv(train_file: str, valid_file: str, task: str) -> DatasetDict:
    """Load BESSTIE data from CSV files and filter by task.

    Parameters
    ----------
    train_file : str
        Path to the training CSV file.  Must contain columns ``text``, ``label``,
        ``variety``, ``source`` and ``task``.
    valid_file : str
        Path to the validation CSV file with the same column structure.
    task : str
        Task name to filter on (``"Sentiment"`` or ``"Sarcasm"``).

    Returns
    -------
    DatasetDict
        A dictionary with ``train`` and ``validation`` splits containing only rows
        for the specified task.

    Raises
    ------
    FileNotFoundError
        If either CSV file does not exist.
    ValueError
        If the required columns are missing or if no rows match the task.
    """
    if not os.path.exists(train_file) or not os.path.exists(valid_file):
        raise FileNotFoundError(
            "CSV files for BESSTIE not found. Please download train.csv and valid.csv from the dataset repository."
        )
    train_df = pd.read_csv(train_file)
    valid_df = pd.read_csv(valid_file)
    required_cols = {"text", "label", "variety", "source", "task"}
    if not required_cols.issubset(train_df.columns) or not required_cols.issubset(valid_df.columns):
        raise ValueError(f"CSV files must contain columns {required_cols}")
    train_df = train_df[train_df["task"].str.lower() == task.lower()].reset_index(drop=True)
    valid_df = valid_df[valid_df["task"].str.lower() == task.lower()].reset_index(drop=True)
    if train_df.empty or valid_df.empty:
        raise ValueError(f"No rows found for task '{task}'. Check that the CSV files include this task.")
    return DatasetDict({"train": Dataset.from_pandas(train_df), "validation": Dataset.from_pandas(valid_df)})


def load_besstie_from_hf(task: str) -> DatasetDict:
    """Load the BESSTIE dataset from Hugging Face and filter by task.

    This function requires the optional ``datasets`` library.  When available,
    it downloads the BESSTIE dataset from the Hugging Face hub and returns
    a ``DatasetDict`` containing only examples for the specified task.

    Parameters
    ----------
    task : str
        Task name to filter on (``"Sentiment"`` or ``"Sarcasm"``).

    Returns
    -------
    DatasetDict
        A dictionary with ``train`` and ``validation`` splits.

    Raises
    ------
    ImportError
        If the ``datasets`` library is not installed.
    """
    if load_dataset is None:
        raise ImportError(
            "The 'datasets' library is not installed. Install it or use local CSV files instead."
        )
    ds = load_dataset("unswnlporg/BESSTIE")
    train_ds = ds["train"].filter(lambda ex: ex["task"].lower() == task.lower())
    valid_ds = ds["validation"].filter(lambda ex: ex["task"].lower() == task.lower())
    return DatasetDict({"train": train_ds, "validation": valid_ds})


def prepare_dataset(
    tokenizer: AutoTokenizer,
    dataset: DatasetDict,
    text_column: str = "text",
    label_column: str = "label",
    max_length: Optional[int] = None,
) -> DatasetDict:
    """Tokenise the text column and cast labels to integers.

    The function maps over the dataset splits and adds ``input_ids`` and
    ``attention_mask`` fields while ensuring labels are of integer type.  All
    other columns are dropped to reduce memory usage.

    Parameters
    ----------
    tokenizer : AutoTokenizer
        The tokenizer associated with the chosen pre‑trained model.
    dataset : DatasetDict
        A dataset with ``train`` and ``validation`` splits and at least the
        ``text`` and ``label`` columns.
    text_column : str, optional
        Name of the column containing the raw text.
    label_column : str, optional
        Name of the column containing the label.
    max_length : int, optional
        Maximum sequence length for truncation.  If ``None``, the tokenizer's
        default max length is used.

    Returns
    -------
    DatasetDict
        Tokenised version of the dataset.
    """
    def tokenize_function(examples):
        return tokenizer(examples[text_column], truncation=True, max_length=max_length)
    remove_cols = [c for c in dataset["train"].column_names if c not in {text_column, label_column}]
    tokenised = dataset.map(tokenize_function, batched=True, remove_columns=remove_cols)
    # Ensure the label is an integer.  Casting via cast_column may fail when the
    # underlying feature type is a plain string【180316227421938†L563-L575】.  Instead, we
    # convert the label explicitly using a map operation.  This avoids the
    # TypeError: `'str' object is not callable` seen during schema generation.
    def cast_label(example):
        example[label_column] = int(example[label_column])
        return example
    tokenised = tokenised.map(cast_label)
    return tokenised


def compute_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """Compute simple inverse‐frequency class weights for binary labels.

    The weight for each class ``i`` is calculated as
    ``len(labels) / (2 * count_i)``, ensuring that the sum of weights
    equals the number of classes.  This helps counteract class imbalance.

    Parameters
    ----------
    labels : np.ndarray
        Array of integer labels (0 or 1).

    Returns
    -------
    Dict[int, float]
        Mapping from class index to weight.
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = {}
    for u, c in zip(unique, counts):
        weights[int(u)] = total / (len(unique) * c)
    return weights


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute classification metrics for a binary task.

    This function returns both micro‑averaged (binary) and macro‑averaged
    precision, recall and F1 scores, along with overall accuracy.  The
    macro‑averaged F1 score is used in the BESSTIE paper to evaluate
    performance across classes without weighting by their support【180316227421938†L605-L606】.

    Parameters
    ----------
    eval_pred : Tuple[np.ndarray, np.ndarray]
        A tuple ``(logits, labels)`` containing model outputs and ground‑truth labels.

    Returns
    -------
    Dict[str, float]
        Metrics including ``accuracy``, ``precision_micro``, ``recall_micro``, ``f1_micro``,
        ``precision_macro``, ``recall_macro`` and ``f1_macro``.
    """
    logits, labels = eval_pred
    # Convert logits to predicted class indices
    preds = np.argmax(logits, axis=-1)
    # Micro (binary) metrics weight classes by support
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    # Macro metrics treat each class equally regardless of support
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
    }
