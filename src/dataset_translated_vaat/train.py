"""
train.py

Training script for BESSTIE (Sarcasm/Sentiment) with:
- Standard HuggingFace head (baseline): AutoModelForSequenceClassification
- Custom decoder heads (extension):
    - CNN-over-tokens head
    - Attention pooling head

Custom decoder checkpoints are saved in the SAME folder as the encoder/tokenizer,
plus:
  - decoder_config.json
  - decoder_head.pt
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence, Dict, Any

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from transformers import (
    AutoModelForSequenceClassification,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from utils import (
    load_besstie_from_csv,
    load_besstie_from_hf,
    prepare_dataset,
    compute_class_weights,
    compute_metrics,
)

from decoders import DecoderConfig, build_head
from model_io import EncoderWithCustomHead, save_custom_decoder_checkpoint


def _pick_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _snapshot_state_dict_to_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    Create a *safe* snapshot of model weights on CPU.

    Why not just v.detach().cpu()?
      - If training on CPU, .cpu() returns the SAME tensor (no copy),
        so the "best" state would keep changing as training continues.
      - We therefore clone() when already on CPU.
    """
    snap: Dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        t = v.detach()
        if t.device.type == "cpu":
            snap[k] = t.clone()
        else:
            snap[k] = t.cpu()
    return snap


def train_binary_model(
    model_name: str,
    task: str,
    output_dir: str,
    train_file: Optional[str] = None,
    valid_file: Optional[str] = None,
    learning_rates: Optional[Sequence[float]] = None,
    batch_size: int = 8,
    eval_batch_size: Optional[int] = None,
    num_epochs: int = 30,
    weight_decay: float = 0.01,
    seed: int = 42,
    use_class_weights: bool = True,
    device: str = "auto",
    num_workers: int = 2,
    pin_memory: Optional[bool] = None,
    grad_accum_steps: int = 1,
    max_length: Optional[int] = None,
    fp16: bool = False,
    bf16: bool = False,
    tf32: bool = False,
    # ---- Best-epoch + Early stopping ----
    early_stopping: bool = True,
    patience: int = 3,
    min_delta: float = 0.0,
    monitor: str = "f1_macro",   # "f1_macro" or "accuracy"
    # ---- Extension knobs ----
    decoder_type: str = "hf_default",  # "hf_default" | "cnn" | "attn_pool"
    decoder_dropout: float = 0.1,
    cnn_num_filters: int = 128,
    cnn_kernel_sizes: Optional[Sequence[int]] = None,
    attn_mlp_hidden: Optional[int] = None,

    # ---- VAAT (Variety-Aware Adapter Tuning) ----
    vaat_adapter_dim: int = 64,
    vaat_freeze_encoder: bool = True,
) -> None:
    if learning_rates is None:
        learning_rates = (1e-5, 2e-5, 3e-5)
    if cnn_kernel_sizes is None:
        cnn_kernel_sizes = (2, 3, 4)

    if fp16 and bf16:
        raise ValueError("Choose only one of fp16 or bf16.")
    if monitor not in {"f1_macro", "accuracy"}:
        raise ValueError("monitor must be either 'f1_macro' or 'accuracy'.")

    # ----------------------------
    # Data
    # ----------------------------
    if train_file and valid_file:
        dataset = load_besstie_from_csv(train_file, valid_file, task=task)
    else:
        dataset = load_besstie_from_hf(task=task)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # VAAT needs the 'variety' column preserved to build variety_id conditioning.
    extra_keep_cols = ["variety"] if str(decoder_type) == "vaat" else None
    tokenised_dataset = prepare_dataset(tokenizer, dataset, max_length=max_length, extra_keep_columns=extra_keep_cols)

    # ----------------------------
    # VAAT: map variety string -> variety_id
    # ----------------------------
    vaat_varieties = None
    if str(decoder_type) == "vaat":
        if "variety" not in tokenised_dataset["train"].column_names:
            raise ValueError("decoder_type=vaat requires a 'variety' column in the dataset.")
        all_vars = list(tokenised_dataset["train"]["variety"]) + list(tokenised_dataset["validation"]["variety"])
        vaat_varieties = sorted(list(set(map(str, all_vars))))
        variety_to_id = {v: i for i, v in enumerate(vaat_varieties)}

        def _add_variety_id(batch):
            vs = [str(v) for v in batch["variety"]]
            return {"variety_id": [variety_to_id.get(v, 0) for v in vs]}

        tokenised_dataset = tokenised_dataset.map(_add_variety_id, batched=True)
        tokenised_dataset = tokenised_dataset.remove_columns(["variety"])

    # Class weights (optional)
    class_weights = None
    if use_class_weights:
        labels_arr = np.array(tokenised_dataset["train"]["label"])
        class_weights = compute_class_weights(labels_arr)

    # Keep only needed columns
    tokenised_dataset = tokenised_dataset.remove_columns([
        c for c in tokenised_dataset["train"].column_names if c not in {"input_ids", "attention_mask", "label", "variety_id"}
    ])
    train_cols = [c for c in ["input_ids","attention_mask","label","variety_id"] if c in tokenised_dataset["train"].column_names]
    val_cols   = [c for c in ["input_ids","attention_mask","label","variety_id"] if c in tokenised_dataset["validation"].column_names]
    tokenised_dataset["train"].set_format(type="torch", columns=train_cols)
    tokenised_dataset["validation"].set_format(type="torch", columns=val_cols)

    base_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def collator(features):
        # Optional VAAT conditioning: include variety_ids if present.
        has_variety = (len(features) > 0 and "variety_id" in features[0])
        variety_ids = [int(f["variety_id"]) for f in features] if has_variety else None

        model_feats = [{k: f[k] for k in ("input_ids", "attention_mask") if k in f} for f in features]
        batch = base_collator(model_feats)

        if "label" in features[0]:
            batch["label"] = torch.tensor([int(f["label"]) for f in features], dtype=torch.long)

        if variety_ids is not None:
            batch["variety_ids"] = torch.tensor(variety_ids, dtype=torch.long)
        return batch

    from torch.utils.data import DataLoader

    if eval_batch_size is None:
        eval_batch_size = batch_size

    dev = _pick_device(device)
    use_cuda = dev.type == "cuda"
    if pin_memory is None:
        pin_memory = use_cuda

    if tf32 and use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

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
        from tqdm.auto import tqdm  # type: ignore
    except Exception:
        def tqdm(x, *args, **kwargs):  # type: ignore
            return x

    # Reproducibility
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if use_cuda:
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.benchmark = True

    # Mixed precision
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

    # Track best across learning rates (final saved model)
    best_f1_macro_global = -float("inf")
    best_lr_global = None
    best_model_global = None

    def build_model_for_lr() -> torch.nn.Module:
        if decoder_type == "hf_default":
            return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        # Custom head extension
        encoder = AutoModel.from_pretrained(model_name)
        hidden_size = getattr(encoder.config, "hidden_size", getattr(encoder.config, "d_model", None))
        if hidden_size is None:
            raise ValueError("Cannot infer hidden size from encoder config.")
        hidden_size = int(hidden_size)

        dec_cfg = DecoderConfig(
            decoder_type=str(decoder_type),
            num_labels=2,
            dropout=float(decoder_dropout),
            cnn_num_filters=int(cnn_num_filters),
            cnn_kernel_sizes=list(map(int, cnn_kernel_sizes)),
            attn_mlp_hidden=attn_mlp_hidden,
            vaat_adapter_dim=int(vaat_adapter_dim),
            vaat_varieties=vaat_varieties,
        )
        head = build_head(hidden_size=hidden_size, cfg=dec_cfg)
        model = EncoderWithCustomHead(encoder=encoder, head=head)
        if str(decoder_type) == "vaat" and bool(vaat_freeze_encoder):
            for p in model.encoder.parameters():
                p.requires_grad = False
        return model

    # ----------------------------
    # Train over LRs
    # ----------------------------
    from torch.optim import AdamW

    for lr in learning_rates:
        lr = float(lr)
        print(f"\n*** Training with learning rate {lr} | decoder={decoder_type}")

        model = build_model_for_lr()
        model.to(dev)

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=float(weight_decay))

        # Best-epoch tracking (within this LR)
        best_metric_this_lr = -float("inf")
        best_state_this_lr = None
        best_epoch_this_lr = 0
        best_metrics_this_lr: Optional[Dict[str, Any]] = None
        bad_epochs = 0

        for epoch in range(int(num_epochs)):
            model.train()
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
                    raise KeyError(f"Missing label key in batch. Keys={list(batch.keys())}")

                with torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
                    outputs = model(**batch)
                    logits = outputs.logits
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
                    optimizer.zero_grad(set_to_none=True)

            # flush remaining grads
            if len(train_loader) % max(1, int(grad_accum_steps)) != 0:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            avg_loss = epoch_loss / max(1, len(train_loader))

            # ----------------------------
            # Validation
            # ----------------------------
            model.eval()
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
                        raise KeyError("Missing label key during validation.")

                    with torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
                        outputs = model(**batch)
                        logits_val = outputs.logits

                    all_logits.append(logits_val.detach().cpu().numpy())
                    all_labels.append(val_labels.detach().cpu().numpy())

            all_logits_np = np.concatenate(all_logits, axis=0)
            all_labels_np = np.concatenate(all_labels, axis=0)
            metrics = compute_metrics((all_logits_np, all_labels_np))

            if monitor not in metrics:
                raise KeyError(f"monitor='{monitor}' not found in metrics keys={list(metrics.keys())}")

            print(
                f"LR {lr} Ep {epoch + 1}/{num_epochs} | loss={avg_loss:.4f} | "
                f"acc={metrics['accuracy']:.4f} | f1_macro={metrics['f1_macro']:.4f}"
            )

            current_metric = float(metrics[monitor])

            # Update best-epoch snapshot (in-memory, no disk IO)
            if current_metric > (best_metric_this_lr + float(min_delta)):
                best_metric_this_lr = current_metric
                best_epoch_this_lr = epoch + 1
                best_metrics_this_lr = dict(metrics)
                best_state_this_lr = _snapshot_state_dict_to_cpu(model)
                bad_epochs = 0
            else:
                bad_epochs += 1

            if early_stopping and bad_epochs >= int(patience):
                print(
                    f"Early stopping at epoch {epoch + 1}. "
                    f"Best epoch was {best_epoch_this_lr} with {monitor}={best_metric_this_lr:.4f}"
                )
                break

        # Restore best epoch weights for this LR (so the saved model is best-epoch)
        if best_state_this_lr is not None:
            model.load_state_dict(best_state_this_lr)
            # free snapshot (optional)
            del best_state_this_lr

        # Choose which model to keep globally (by best-epoch macro-F1)
        # If you monitor accuracy, we still rank LRs by macro-F1 at the (best) monitored epoch.
        if best_metrics_this_lr is not None:
            f1_macro_this_lr = float(best_metrics_this_lr.get("f1_macro", -float("inf")))
        else:
            # fallback (shouldn't happen)
            f1_macro_this_lr = float(metrics["f1_macro"])

        if f1_macro_this_lr > best_f1_macro_global:
            if best_model_global is not None:
                del best_model_global
                if use_cuda:
                    torch.cuda.empty_cache()

            best_f1_macro_global = f1_macro_this_lr
            best_lr_global = lr
            best_model_global = model
            best_epoch_global = best_epoch_this_lr
            best_monitor_val_global = best_metric_this_lr
            best_monitor_global = monitor
        else:
            del model
            if use_cuda:
                torch.cuda.empty_cache()

    # ----------------------------
    # Save best (across LRs)
    # ----------------------------
    os.makedirs(output_dir, exist_ok=True)
    if best_model_global is None:
        raise RuntimeError("Training produced no model (unexpected).")

    if decoder_type == "hf_default":
        best_model_global.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(
            f"Saved HF default model to {output_dir} | best_lr={best_lr_global} | "
            f"best_epoch={best_epoch_global} | f1_macro={best_f1_macro_global:.4f}"
        )
        return

    # Custom decoder checkpoint
    dec_cfg = DecoderConfig(
        decoder_type=str(decoder_type),
        num_labels=2,
        dropout=float(decoder_dropout),
        cnn_num_filters=int(cnn_num_filters),
        cnn_kernel_sizes=list(map(int, cnn_kernel_sizes)),
        attn_mlp_hidden=attn_mlp_hidden,
        vaat_adapter_dim=int(vaat_adapter_dim),
        vaat_varieties=vaat_varieties,
    ).to_dict()

    save_custom_decoder_checkpoint(
        output_dir=output_dir,
        encoder=best_model_global.encoder,
        tokenizer=tokenizer,
        decoder_cfg=dec_cfg,
        head=best_model_global.head,
        extra_metadata={
            "best_lr": best_lr_global,
            "best_epoch": best_epoch_global,
            "monitor": best_monitor_global,
            "best_monitor_value": best_monitor_val_global,
            "best_f1_macro": best_f1_macro_global,
        },
    )
    print(
        f"Saved custom-decoder model to {output_dir} | best_lr={best_lr_global} | "
        f"best_epoch={best_epoch_global} | f1_macro={best_f1_macro_global:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Train BESSTIE classifier with optional custom decoder heads.")
    parser.add_argument("--model_name", type=str, default="roberta-large")
    parser.add_argument("--task", type=str, choices=["Sentiment", "Sarcasm"], default="Sarcasm")
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--valid_file", type=str)
    parser.add_argument("--output_dir", type=str, default="./model_output")
    parser.add_argument("--learning_rates", type=float, nargs="+", default=[1e-5, 2e-5, 3e-5])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tf32", action="store_true")

    # ---- Early stopping / best epoch ----
    parser.add_argument("--early_stopping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--monitor", type=str, default="f1_macro", choices=["f1_macro", "accuracy"])

    # extension flags
    parser.add_argument("--decoder_type", type=str, default="hf_default", choices=["hf_default", "cnn", "attn_pool", "vaat"])
    parser.add_argument("--decoder_dropout", type=float, default=0.1)
    parser.add_argument("--cnn_num_filters", type=int, default=128)
    parser.add_argument("--cnn_kernel_sizes", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--attn_mlp_hidden", type=int, default=None)

    # VAAT
    parser.add_argument("--vaat_adapter_dim", type=int, default=64)
    parser.add_argument("--vaat_freeze_encoder", action="store_true")

    args = parser.parse_args()

    train_binary_model(
        model_name=args.model_name,
        task=args.task,
        output_dir=args.output_dir,
        train_file=args.train_file,
        valid_file=args.valid_file,
        learning_rates=args.learning_rates,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        seed=args.seed,
        use_class_weights=not args.no_class_weights,
        device=args.device,
        num_workers=args.num_workers,
        pin_memory=False if args.no_pin_memory else None,
        grad_accum_steps=args.grad_accum_steps,
        max_length=args.max_length,
        fp16=args.fp16,
        bf16=args.bf16,
        tf32=args.tf32,

        early_stopping=bool(args.early_stopping),
        patience=int(args.patience),
        min_delta=float(args.min_delta),
        monitor=str(args.monitor),

        decoder_type=args.decoder_type,
        decoder_dropout=args.decoder_dropout,
        cnn_num_filters=args.cnn_num_filters,
        cnn_kernel_sizes=args.cnn_kernel_sizes,
        attn_mlp_hidden=args.attn_mlp_hidden,
    )


if __name__ == "__main__":
    main()
