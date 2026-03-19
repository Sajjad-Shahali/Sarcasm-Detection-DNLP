#!/usr/bin/env python3
"""
train2.py

Training script for BESSTIE (Sarcasm/Sentiment) that supports:
- Baseline (HF head): AutoModelForSequenceClassification
- Custom heads over encoder outputs:
    - CNN over tokens
    - Attention pooling

And it keeps the "good stuff" from your baseline train.py:
- Learning-rate scheduling with warmup (linear)
- Dropout control (encoder + head)
- Early stopping (monitor f1_macro / accuracy)
- Optional encoder freezing
- FP16/BF16 handling + BF16->numpy safety

Run:
    python train2.py --config config2.yaml

Config format is intentionally flexible and keeps compatibility with your config2.

Head selection (any one):
    train.decoder_type: baseline | attn_pool | cnn_tok | ...
    train.head_type: baseline | attention_pooling | cnn_over_tokens | ...
    train.head: (legacy) can also be used as a string head type

Dropout controls (all optional):
    train.hidden_dropout / train.attention_dropout  -> encoder dropouts
    train.decoder_dropout (preferred) OR train.head_dropout -> custom head dropout

Attention pooling optional scorer MLP:
    train.attn_mlp_hidden: null (Linear scorer) OR 256 (2-layer MLP scorer)

Anything missing falls back to sane defaults.
"""

from __future__ import annotations
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt

import argparse
import json
import os
import pprint
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, List

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from utils import (
    load_besstie_from_csv,
    load_besstie_from_hf,
    prepare_dataset,
    compute_class_weights,
    compute_metrics,
)

# ---------------------------
# Config helpers (yaml/json)
# ---------------------------

def load_config(path: str) -> Dict[str, Any]:
    if not path:
        raise ValueError("You must provide --config <path to .yaml/.yml/.json>")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    _, ext = os.path.splitext(path.lower())
    text = open(path, "r", encoding="utf-8").read()

    if ext == ".json":
        return json.loads(text)

    if ext in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "YAML config requested but PyYAML is not installed. "
                "Install with: pip install pyyaml  OR use a .json config instead."
            ) from e
        return yaml.safe_load(text) or {}

    raise ValueError(f"Unsupported config extension '{ext}'. Use .yaml/.yml or .json.")


def cfg_get(cfg: Dict[str, Any], section: str, key: str, default=None):
    return cfg.get(section, {}).get(key, default)


def cfg_get_nested(cfg: Dict[str, Any], section: str, *keys: str, default=None):
    cur = cfg.get(section, {})
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return default if cur is None else cur


# ---------------------------
# Device
# ---------------------------

def _pick_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


# ---------------------------
# Custom Heads
# ---------------------------

class CNNOverTokensHead(nn.Module):
    """
    Simple CNN over tokens:
    - input: last_hidden_state [B, T, H]
    - output: logits [B, num_labels]
    """
    def __init__(
        self,
        hidden_size: int,
        num_labels: int = 2,
        kernel_sizes: Sequence[int] = (2, 3, 4),
        out_channels: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_size, out_channels=out_channels, kernel_size=int(k))
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(out_channels * len(self.convs), num_labels)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, T, H] -> [B, H, T]
        x = x.transpose(1, 2)
        feats: List[torch.Tensor] = []
        for conv in self.convs:
            # conv output: [B, C, T']
            c = torch.relu(conv(x))
            # global max pool over time: [B, C]
            feats.append(torch.max(c, dim=2).values)
        h = torch.cat(feats, dim=1)
        h = self.dropout(h)
        return self.classifier(h)


class AttentionPoolingHead(nn.Module):
    """
    Attention pooling over tokens:
    - learn scores per token, do masked softmax, take weighted sum, then classify

    Config-compatible with your `attn_mlp_hidden`:
      - if mlp_hidden is None -> Linear scorer: score = W x
      - else -> 2-layer MLP scorer: Linear -> Tanh -> Linear
    """
    def __init__(
        self,
        hidden_size: int,
        num_labels: int = 2,
        mlp_hidden: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if mlp_hidden is None:
            self.scorer: nn.Module = nn.Linear(hidden_size, 1)
        else:
            self.scorer = nn.Sequential(
                nn.Linear(hidden_size, int(mlp_hidden)),
                nn.Tanh(),
                nn.Linear(int(mlp_hidden), 1),
            )

        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, T, H]
        scores = self.scorer(x).squeeze(-1)  # [B, T]
        if attention_mask is not None:
            # mask padding tokens with -inf before softmax
            scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, T, 1]
        pooled = torch.sum(weights * x, dim=1)  # [B, H]
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


@dataclass
class CustomModelBundle:
    encoder: nn.Module
    head: nn.Module


def build_custom_head(
    head_type: str,
    hidden_size: int,
    head_dropout: float,
    cfg: Dict[str, Any],
) -> nn.Module:
    """
    head_type: 'cnn_over_tokens' | 'attention_pooling'
    Reads optional head hyperparams from config under train.head.* (if present).
    """
    head_type = (head_type or "").lower().strip()

    if head_type == "cnn_over_tokens":
        # Accept both flat and nested style configs
        kernel_sizes = cfg_get(cfg, "train", "cnn_kernel_sizes", None)
        if kernel_sizes is None:
            kernel_sizes = cfg_get_nested(cfg, "train", "head", "cnn_kernel_sizes", default=(2, 3, 4))

        out_channels = cfg_get(cfg, "train", "cnn_out_channels", None)
        if out_channels is None:
            out_channels = cfg_get_nested(cfg, "train", "head", "cnn_out_channels", default=128)
        return CNNOverTokensHead(
            hidden_size=hidden_size,
            num_labels=2,
            kernel_sizes=kernel_sizes,
            out_channels=int(out_channels),
            dropout=float(head_dropout),
        )

    if head_type == "attention_pooling":
        # Preferred flat key (matches your config2): train.attn_mlp_hidden
        mlp_hidden = cfg_get(cfg, "train", "attn_mlp_hidden", None)
        # Also accept nested variants if you ever move head params under train.head
        if mlp_hidden is None:
            mlp_hidden = cfg_get_nested(cfg, "train", "head", "attn_mlp_hidden", default=None)

        # Backward-compatible alias: train.head.attn_hidden
        # (treat it as the MLP hidden size)
        if mlp_hidden is None:
            mlp_hidden = cfg_get_nested(cfg, "train", "head", "attn_hidden", default=None)

        return AttentionPoolingHead(
            hidden_size=hidden_size,
            num_labels=2,
            mlp_hidden=(int(mlp_hidden) if mlp_hidden is not None else None),
            dropout=float(head_dropout),
        )

    raise ValueError(f"Unknown head_type='{head_type}'. Use baseline | cnn_over_tokens | attention_pooling")


# ---------------------------
# Saving / Loading custom head checkpoints
# ---------------------------

def save_custom_checkpoint(
    output_dir: str,
    tokenizer,
    encoder,
    head: nn.Module,
    head_type: str,
    head_cfg: Dict[str, Any],
) -> None:
    """
    Saves:
    - encoder + tokenizer in output_dir (HF format)
    - decoder_head.pt (state_dict)
    - decoder_config.json (head_type + params)
    """
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    encoder.save_pretrained(output_dir)

    torch.save(head.state_dict(), os.path.join(output_dir, "decoder_head.pt"))
    cfg_out = {
        "head_type": head_type,
        "head_cfg": head_cfg,
    }
    with open(os.path.join(output_dir, "decoder_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_out, f, indent=2)
    print(f"[Done] Custom checkpoint saved to {output_dir} (encoder+tokenizer + decoder_head.pt + decoder_config.json)")


# ---------------------------
# Training
# ---------------------------

def train_binary_model(
    cfg: Dict[str, Any],
    # explicit overrides (optional)
    config_device: Optional[str] = None,
) -> None:
    """
    A config-driven trainer that keeps the baseline good practices.
    """
    # --- core train params ---
    model_name = cfg_get(cfg, "train", "model_name")
    task = cfg_get(cfg, "train", "task", "Sarcasm")
    output_dir = cfg_get(cfg, "train", "output_dir", "./model_output")
    train_file = cfg_get(cfg, "train", "train_file", None)
    valid_file = cfg_get(cfg, "train", "valid_file", None)

    learning_rates = cfg_get(cfg, "train", "learning_rates", None)
    if learning_rates is None:
        # accept a single `learning_rate` if user prefers that name
        learning_rates = cfg_get(cfg, "train", "learning_rate", [2e-5])

    # normalize: allow float OR list/tuple
    if isinstance(learning_rates, (int, float, str)):
        learning_rates = [float(learning_rates)]
    elif isinstance(learning_rates, (list, tuple)):
        learning_rates = [float(x) for x in learning_rates]
    else:
        # last resort: try to cast single object to float
        learning_rates = [float(learning_rates)]
    batch_size = int(cfg_get(cfg, "train", "batch_size", 8))
    eval_batch_size = cfg_get(cfg, "train", "eval_batch_size", None)
    eval_batch_size = int(eval_batch_size) if eval_batch_size is not None else batch_size

    num_epochs = int(cfg_get(cfg, "train", "num_epochs", 30))
    weight_decay = float(cfg_get(cfg, "train", "weight_decay", 0.01))
    seed = int(cfg_get(cfg, "train", "seed", 42))
    use_class_weights = bool(cfg_get(cfg, "train", "use_class_weights", True))

    num_workers = int(cfg_get(cfg, "train", "num_workers", 2))
    pin_memory = cfg_get(cfg, "train", "pin_memory", None)
    grad_accum_steps = int(cfg_get(cfg, "train", "grad_accum_steps", 1))
    max_length = cfg_get(cfg, "train", "max_length", None)
    max_length = int(max_length) if max_length is not None else None

    fp16 = bool(cfg_get(cfg, "train", "fp16", False))
    bf16 = bool(cfg_get(cfg, "train", "bf16", False))
    tf32 = bool(cfg_get(cfg, "train", "tf32", False))

    # --- early stopping ---
    early_stopping = bool(cfg_get(cfg, "train", "early_stopping", True))
    patience = int(cfg_get(cfg, "train", "patience", 3))
    min_delta = float(cfg_get(cfg, "train", "min_delta", 0.0))
    monitor = str(cfg_get(cfg, "train", "monitor", "f1_macro"))

    # --- "good stuff" ---
    warmup_ratio = float(cfg_get(cfg, "train", "warmup_ratio", 0.1))
    hidden_dropout = float(cfg_get(cfg, "train", "hidden_dropout", 0.1))
    attention_dropout = float(cfg_get(cfg, "train", "attention_dropout", 0.1))
    freeze_encoder = bool(cfg_get(cfg, "train", "freeze_encoder", False))

    # --- head selection ---
    head_type_raw = (
        cfg_get(cfg, "train", "decoder_type", None)
        or cfg_get(cfg, "train", "head_type", None)
        or cfg_get(cfg, "train", "head", None)
        or "baseline"
    )
    head_type_raw = str(head_type_raw).lower().strip()

    # Canonicalize head names (your config2 uses attn_pool)
    head_aliases = {
        # baseline aliases
        "baseline": "baseline",
        "hf": "baseline",
        "hf_head": "baseline",
        "huggingface": "baseline",
        "sequence_classification": "baseline",
        # attention pooling aliases
        "attn_pool": "attention_pooling",
        "attn_pooling": "attention_pooling",
        "attention_pool": "attention_pooling",
        "attention_pooling": "attention_pooling",
        # cnn aliases
        "cnn_tok": "cnn_over_tokens",
        "cnn_tokens": "cnn_over_tokens",
        "cnn_over_tokens": "cnn_over_tokens",
        "cnn": "cnn_over_tokens",
    }
    head_type = head_aliases.get(head_type_raw, head_type_raw)

    # optional head dropout (separate from encoder dropout)
    # Preferred key (matches your config2): decoder_dropout
    decoder_dropout = cfg_get(cfg, "train", "decoder_dropout", None)
    if decoder_dropout is not None:
        head_dropout = float(decoder_dropout)
    else:
        head_dropout = float(cfg_get(cfg, "train", "head_dropout", hidden_dropout))

    # device
    common_device = cfg_get(cfg, "common", "device", "auto")
    device_str = config_device or cfg_get(cfg, "train", "device", common_device) or "auto"
    dev = _pick_device(device_str)
    use_cuda = dev.type == "cuda"

    if fp16 and bf16:
        raise ValueError("Choose only one of fp16 or bf16.")

    if pin_memory is None:
        pin_memory = use_cuda

    if tf32 and use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ---------------------------
    # Data
    # ---------------------------
    print("[info] Loading dataset...")
    if train_file and valid_file:
        dataset = load_besstie_from_csv(train_file, valid_file, task=task)
    else:
        dataset = load_besstie_from_hf(task=task)

    print(f"[info] Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    print("[info] Tokenizing...")
    tokenised_dataset = prepare_dataset(tokenizer, dataset, max_length=max_length)

    # Class weights
    class_weights = None
    if use_class_weights:
        labels_arr = np.array(tokenised_dataset["train"]["label"])
        class_weights = compute_class_weights(labels_arr)

    # Keep only needed columns
    tokenised_dataset = tokenised_dataset.remove_columns([
        c for c in tokenised_dataset["train"].column_names if c not in {"input_ids", "attention_mask", "label"}
    ])
    tokenised_dataset["train"].set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    tokenised_dataset["validation"].set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # DataLoaders
    from torch.utils.data import DataLoader
    loader_kwargs = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": int(num_workers) > 0,
    }
    if int(num_workers) > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        tokenised_dataset["train"],
        batch_size=int(batch_size),
        shuffle=True,
        collate_fn=collator,
        **loader_kwargs,
    )
    eval_loader = DataLoader(
        tokenised_dataset["validation"],
        batch_size=int(eval_batch_size),
        shuffle=False,
        collate_fn=collator,
        **loader_kwargs,
    )

    # tqdm optional
    try:
        from tqdm.auto import tqdm
    except Exception:
        def tqdm(x, *args, **kwargs): return x  # type: ignore

    # Reproducibility
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if use_cuda:
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.benchmark = True

    # Mixed Precision
    autocast_dtype = None
    if use_cuda and fp16:
        autocast_dtype = torch.float16
    elif use_cuda and bf16:
        autocast_dtype = torch.bfloat16
    use_autocast = autocast_dtype is not None
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda and fp16)

    # Loss
    if class_weights is not None:
        weight_tensor = torch.tensor(
            [class_weights.get(0, 1.0), class_weights.get(1, 1.0)],
            dtype=torch.float,
        ).to(dev)
        loss_fct = CrossEntropyLoss(weight=weight_tensor)
    else:
        loss_fct = CrossEntropyLoss()

    # -------------------------------------------------------
    # Utilities for best model tracking (RAM state)
    # -------------------------------------------------------
    def _state_dict_cpu(m: nn.Module) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in m.state_dict().items()}

    def _maybe_update_best_state(
        current: float,
        best: float,
        min_delta_: float,
        baseline_model: Optional[nn.Module] = None,
        encoder: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        best_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Optional[Dict[str, Any]], bool]:
        if current > (best + float(min_delta_)):
            state: Dict[str, Any] = {}
            if baseline_model is not None:
                state["baseline"] = _state_dict_cpu(baseline_model)
            if encoder is not None:
                state["encoder"] = _state_dict_cpu(encoder)
            if head is not None:
                state["head"] = _state_dict_cpu(head)
            return current, state, True
        return best, best_state, False

    # -------------------------------------------------------
    # Train loop (grid over LR)
    # -------------------------------------------------------
    from torch.optim import AdamW

    best_metric_global = -float("inf")
    best_lr_global: Optional[float] = None
    best_bundle_global: Optional[Dict[str, Any]] = None  # stores state dicts
    best_is_baseline_global: Optional[bool] = None

    for lr in learning_rates:
        lr = float(lr)
        print(
            f"\n*** Training head={head_type} | LR={lr} | warmup={warmup_ratio} | "
            f"enc_drop={hidden_dropout} | head_drop={head_dropout} ***"
        )
        # --- NEW: per-epoch history logging ---
        history_path = os.path.join(output_dir, f"training_history_lr{lr}.csv")
        history_f = open(history_path, "w", newline="", encoding="utf-8")
        history_w = csv.DictWriter(
            history_f,
            fieldnames=["epoch", "train_loss", "val_accuracy", "val_f1_macro", "lr_end"]
        )
        history_w.writeheader()
        # 1) Build model(s)
        # Encoder config with custom dropout (works for both baseline & encoder-only)
        enc_config = AutoConfig.from_pretrained(
            model_name,
            num_labels=2,
            hidden_dropout_prob=hidden_dropout,
            attention_probs_dropout_prob=attention_dropout,
        )

        # If you set decoder_dropout in config and you are in baseline mode,
        # also try to apply it to the HF classification head (when supported).
        if head_type == "baseline":
            cls_drop = cfg_get(cfg, "train", "classifier_dropout", None)
            if cls_drop is None:
                cls_drop = cfg_get(cfg, "train", "decoder_dropout", None)
            if cls_drop is not None and hasattr(enc_config, "classifier_dropout"):
                try:
                    enc_config.classifier_dropout = float(cls_drop)
                except Exception:
                    pass

        baseline_model: Optional[nn.Module] = None
        bundle: Optional[CustomModelBundle] = None

        if head_type == "baseline":
            baseline_model = AutoModelForSequenceClassification.from_pretrained(model_name, config=enc_config)
            # Freeze encoder if requested
            if freeze_encoder:
                print("[info] Freezing encoder layers, training classifier only.")
                base_model_prefix = baseline_model.base_model_prefix
                base_model = getattr(baseline_model, base_model_prefix)
                for p in base_model.parameters():
                    p.requires_grad = False
            baseline_model.to(dev)
            params = baseline_model.parameters()
        else:
            encoder = AutoModel.from_pretrained(model_name, config=enc_config)
            # Build head using encoder hidden size
            hidden_size = int(getattr(enc_config, "hidden_size", 768))
            head = build_custom_head(head_type, hidden_size, head_dropout, cfg)
            if freeze_encoder:
                print("[info] Freezing encoder layers, training head only.")
                for p in encoder.parameters():
                    p.requires_grad = False
            encoder.to(dev)
            head.to(dev)
            bundle = CustomModelBundle(encoder=encoder, head=head)
            params = list(encoder.parameters()) + list(head.parameters())

        optimizer = AdamW(params, lr=lr, weight_decay=float(weight_decay))

        # 2) Scheduler (linear with warmup)
        num_update_steps_per_epoch = max(1, len(train_loader) // max(1, grad_accum_steps))
        max_train_steps = int(num_epochs * num_update_steps_per_epoch)
        num_warmup_steps = int(max_train_steps * warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )

        # early-stopping state per LR
        best_metric_this_lr = -float("inf")
        best_state_this_lr: Optional[Dict[str, Any]] = None
        best_epoch_this_lr = 0
        bad_epochs = 0
        # ---- NEW: history for plots ----
        hist_epoch = []
        hist_train_loss = []
        hist_lr = []
        hist_val_f1 = []
        hist_val_acc = []
        # ---------------------------
        # Epochs
        # ---------------------------
        for epoch in range(int(num_epochs)):
            # --- TRAIN ---
            if baseline_model is not None:
                baseline_model.train()
            else:
                assert bundle is not None
                bundle.encoder.train()
                bundle.head.train()

            epoch_loss = 0.0
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(
                tqdm(train_loader, desc=f"LR {lr} Ep {epoch + 1}/{num_epochs} [Train]"),
                start=1,
            ):
                batch = {k: v.to(dev, non_blocking=use_cuda) for k, v in batch.items()}

                if "label" in batch:
                    labels = batch.pop("label")
                elif "labels" in batch:
                    labels = batch.pop("labels")
                else:
                    raise KeyError("Missing label in batch")

                with torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
                    if baseline_model is not None:
                        outputs = baseline_model(**batch)
                        logits = outputs.logits
                    else:
                        assert bundle is not None
                        enc_out = bundle.encoder(**batch)
                        # last_hidden_state: [B, T, H]
                        x = enc_out.last_hidden_state
                        logits = bundle.head(x, attention_mask=batch.get("attention_mask"))

                    loss = loss_fct(logits.view(-1, 2), labels.view(-1))
                    loss = loss / max(1, int(grad_accum_steps))

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                epoch_loss += loss.item() * max(1, int(grad_accum_steps))

                if step % max(1, int(grad_accum_steps)) == 0:
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            # flush last partial grad accumulation
            if len(train_loader) % max(1, int(grad_accum_steps)) != 0:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            avg_loss = epoch_loss / max(1, len(train_loader))

            # --- VALIDATE ---
            if baseline_model is not None:
                baseline_model.eval()
            else:
                assert bundle is not None
                bundle.encoder.eval()
                bundle.head.eval()

            all_logits = []
            all_labels = []

            with torch.inference_mode():
                for batch in tqdm(eval_loader, desc=f"LR {lr} Ep {epoch + 1}/{num_epochs} [Val]"):
                    batch = {k: v.to(dev, non_blocking=use_cuda) for k, v in batch.items()}
                    if "label" in batch:
                        val_labels = batch.pop("label")
                    elif "labels" in batch:
                        val_labels = batch.pop("labels")
                    else:
                        raise KeyError("Missing label in batch")

                    with torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
                        if baseline_model is not None:
                            outputs = baseline_model(**batch)
                            logits = outputs.logits
                        else:
                            assert bundle is not None
                            enc_out = bundle.encoder(**batch)
                            x = enc_out.last_hidden_state
                            logits = bundle.head(x, attention_mask=batch.get("attention_mask"))

                    # IMPORTANT: float() avoids BF16 -> numpy issues
                    all_logits.append(logits.detach().float().cpu().numpy())
                    all_labels.append(val_labels.detach().cpu().numpy())

            all_logits_np = np.concatenate(all_logits, axis=0)
            all_labels_np = np.concatenate(all_labels, axis=0)
            metrics = compute_metrics((all_logits_np, all_labels_np))
            # ---- NEW: collect history for plots ----
            lr_end = float(scheduler.get_last_lr()[0]) if scheduler is not None else float(lr)
            hist_epoch.append(epoch + 1)
            hist_train_loss.append(float(avg_loss))
            hist_lr.append(lr_end)
            hist_val_f1.append(float(metrics["f1_macro"]))
            hist_val_acc.append(float(metrics["accuracy"]))
            print(
                f"LR {lr} Ep {epoch + 1}/{num_epochs} | loss={avg_loss:.4f} | "
                f"Acc={metrics['accuracy']:.4f} | F1_macro={metrics['f1_macro']:.4f}"
            )
            # --- NEW: write epoch row (loss + lr + val metrics) ---
            
            history_w.writerow({
                "epoch": epoch + 1,
                "train_loss": float(avg_loss),
                "val_accuracy": float(metrics["accuracy"]),
                "val_f1_macro": float(metrics["f1_macro"]),
                "lr_end": lr_end,
            })
            history_f.flush()
            # --- EARLY STOPPING ---
            current_metric = float(metrics.get(monitor, metrics["f1_macro"]))
            best_metric_this_lr, best_state_this_lr, improved = _maybe_update_best_state(
                current=current_metric,
                best=best_metric_this_lr,
                min_delta_=min_delta,
                baseline_model=baseline_model,
                encoder=(bundle.encoder if bundle is not None else None),
                head=(bundle.head if bundle is not None else None),
                best_state=best_state_this_lr,
            )

            if improved:
                best_epoch_this_lr = epoch + 1
                bad_epochs = 0
            else:
                bad_epochs += 1

            if early_stopping and bad_epochs >= int(patience):
                print(
                    f"[early-stop] epoch {epoch+1}. Best was epoch {best_epoch_this_lr} "
                    f"({monitor}={best_metric_this_lr:.4f})"
                )
                break
        history_f.close()
        print(f"[info] Saved training history to: {history_path}")    
        # ---- NEW: save plots for this LR ----
        try:
            # Loss
            plt.figure()
            plt.plot(hist_epoch, hist_train_loss)
            plt.xlabel("Epoch")
            plt.ylabel("Train Loss")
            plt.title(f"Training Loss (lr={lr})")
            plt.savefig(os.path.join(output_dir, f"curve_train_loss_lr{lr}.png"), dpi=200, bbox_inches="tight")
            plt.close()

            # LR
            plt.figure()
            plt.plot(hist_epoch, hist_lr)
            plt.xlabel("Epoch")
            plt.ylabel("Learning Rate")
            plt.title(f"Learning Rate Schedule (lr={lr})")
            plt.savefig(os.path.join(output_dir, f"curve_learning_rate_lr{lr}.png"), dpi=200, bbox_inches="tight")
            plt.close()

            # Val F1
            plt.figure()
            plt.plot(hist_epoch, hist_val_f1)
            plt.xlabel("Epoch")
            plt.ylabel("Val F1_macro")
            plt.title(f"Val F1_macro (lr={lr})")
            plt.savefig(os.path.join(output_dir, f"curve_val_f1_macro_lr{lr}.png"), dpi=200, bbox_inches="tight")
            plt.close()

            # Val Acc
            plt.figure()
            plt.plot(hist_epoch, hist_val_acc)
            plt.xlabel("Epoch")
            plt.ylabel("Val Accuracy")
            plt.title(f"Val Accuracy (lr={lr})")
            plt.savefig(os.path.join(output_dir, f"curve_val_accuracy_lr{lr}.png"), dpi=200, bbox_inches="tight")
            plt.close()

            print(f"[info] Saved plots for lr={lr} to: {output_dir}")

        except Exception as e:
            print(f"[warn] Could not save plots for lr={lr}: {e}")        
                # end epochs for this LR
        final_metric_for_lr = best_metric_this_lr

        if final_metric_for_lr > best_metric_global:
            best_metric_global = final_metric_for_lr
            best_lr_global = lr
            best_bundle_global = best_state_this_lr
            best_is_baseline_global = (baseline_model is not None)
            print(f"-> New Global Best (LR={lr}, {monitor}={best_metric_global:.4f})")

        # free memory between LRs
        if baseline_model is not None:
            del baseline_model
        if bundle is not None:
            del bundle
        if use_cuda:
            torch.cuda.empty_cache()

    # ---------------------------
    # Save best checkpoint
    # ---------------------------
    os.makedirs(output_dir, exist_ok=True)

    if best_bundle_global is None:
        raise RuntimeError("Training finished but no best state was stored.")

    print("\n========== BEST RESULT ==========")
    print(f"head_type: {head_type}")
    print(f"best_lr: {best_lr_global}")
    print(f"best_{monitor}: {best_metric_global:.4f}")
    print("================================\n")

    # rebuild model(s) once, load best weights, then save
    enc_config = AutoConfig.from_pretrained(
        model_name,
        num_labels=2,
        hidden_dropout_prob=hidden_dropout,
        attention_probs_dropout_prob=attention_dropout,
    )

    if best_is_baseline_global:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=enc_config)
        model.load_state_dict(best_bundle_global["baseline"])
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"[Done] Baseline model saved to {output_dir}")
    else:
        encoder = AutoModel.from_pretrained(model_name, config=enc_config)
        hidden_size = int(getattr(enc_config, "hidden_size", 768))
        head = build_custom_head(head_type, hidden_size, head_dropout, cfg)
        encoder.load_state_dict(best_bundle_global["encoder"])
        head.load_state_dict(best_bundle_global["head"])
        # Save custom checkpoint
        # Save any head hyperparams we can find, regardless of whether they are flat or nested.
        nested_head_cfg = cfg_get(cfg, "train", "head", {})
        if not isinstance(nested_head_cfg, dict):
            nested_head_cfg = {}

        flat_head_cfg = {
            "decoder_type": cfg_get(cfg, "train", "decoder_type", None),
            "decoder_dropout": cfg_get(cfg, "train", "decoder_dropout", None),
            "attn_mlp_hidden": cfg_get(cfg, "train", "attn_mlp_hidden", None),
            "cnn_kernel_sizes": cfg_get(cfg, "train", "cnn_kernel_sizes", None),
            "cnn_out_channels": cfg_get(cfg, "train", "cnn_out_channels", None),
        }

        head_cfg = {
            **nested_head_cfg,
            **{k: v for k, v in flat_head_cfg.items() if v is not None},
            "head_dropout": head_dropout,
            "hidden_dropout": hidden_dropout,
            "attention_dropout": attention_dropout,
        }
        save_custom_checkpoint(output_dir, tokenizer, encoder, head, head_type, head_cfg)


# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Train BESSTIE with Baseline / CNN-over-tokens / Attention-pooling heads")
    parser.add_argument("--config", type=str, required=True, help="Path to config .yaml/.yml/.json")
    parser.add_argument("--device", type=str, default=None, help="Override device (auto|cuda|cpu)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    print("\n========== TRAIN2 ARGS (from config) ==========")
    # print a compact view
    pprint.pprint(cfg.get("train", {}), sort_dicts=False)
    print("==============================================\n")

    train_binary_model(cfg, config_device=args.device)
    
    
if __name__ == "__main__":
    main()
