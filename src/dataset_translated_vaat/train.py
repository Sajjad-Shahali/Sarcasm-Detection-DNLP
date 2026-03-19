"""
train.py

Config-first trainer used by main.py.

Supports:
- Baseline HF head (decoder_type="hf_default")
- Custom decoder heads on top of an encoder (decoder_type="cnn" | "attn_pool" | "vaat")

And adds the "good stuff" you asked for (while keeping VAAT logic intact):
- LR scheduling with warmup (linear by default; optional cosine)
- Encoder + head dropout control
- Freeze encoder option (works for baseline + custom heads)
- Early stopping + best-epoch restore (per LR) and best-LR selection
- FP16/BF16 + TF32 handling
- Keeps the "variety" -> variety_id conditioning required by VAAT

main.py calls: train_binary_model({"common":..., "train":...}, config_device=...)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_scheduler,
)

from utils import (
    load_besstie_from_csv,
    load_besstie_from_hf,
    prepare_dataset,
    compute_class_weights,
    compute_metrics,
)

from decoders import DecoderConfig, build_head
from model_io import EncoderWithCustomHead


def _pick_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _snapshot_state_dict_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Safe snapshot of model weights on CPU.

    Important detail:
      - If training on CPU, .cpu() returns the SAME tensor (no copy),
        so the "best" snapshot would keep changing.
      - We clone() on CPU to freeze the snapshot.
    """
    snap: Dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        t = v.detach()
        if t.device.type == "cpu":
            snap[k] = t.clone()
        else:
            snap[k] = t.cpu()
    return snap


def save_custom_decoder_checkpoint(
    output_dir: str,
    encoder: nn.Module,
    tokenizer,
    decoder_cfg: Dict[str, Any],
    head: nn.Module,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a custom-head checkpoint in the format expected by model_io.load_model_and_tokenizer():

      - encoder + tokenizer saved in HF format into output_dir
      - decoder_head.pt           (state_dict of the head)
      - decoder_config.json       (decoder_cfg dict; must include "decoder_type")
      - (optional) decoder_metadata.json for extra run info

    NOTE: We keep this here (instead of in model_io.py) to avoid touching inference/loading code.
    """
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    encoder.save_pretrained(output_dir)

    torch.save(head.state_dict(), os.path.join(output_dir, "decoder_head.pt"))

    cfg_out = dict(decoder_cfg)
    if "decoder_type" not in cfg_out:
        raise ValueError("decoder_cfg must contain 'decoder_type'.")
    with open(os.path.join(output_dir, "decoder_config.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(cfg_out, f, indent=2)

    if extra_metadata:
        with open(os.path.join(output_dir, "decoder_metadata.json"), "w", encoding="utf-8") as f:
            import json
            json.dump(extra_metadata, f, indent=2)

def _normalize_learning_rates(lrs: Any) -> Sequence[float]:
    if lrs is None:
        return (2e-5,)
    if isinstance(lrs, (int, float, str)):
        return (float(lrs),)
    if isinstance(lrs, (list, tuple)):
        return tuple(float(x) for x in lrs)
    return (float(lrs),)


def _extract_train_cfg(cfg: Dict[str, Any], config_device: Optional[str]) -> Dict[str, Any]:
    t = cfg.get("train", {}) or {}
    c = cfg.get("common", {}) or {}

    out: Dict[str, Any] = {}

    # Required-ish (main.py already ensures these exist; we keep safe defaults)
    out["model_name"] = t.get("model_name")
    out["task"] = t.get("task", "Sarcasm")
    out["output_dir"] = t.get("output_dir", "./model_output")
    out["train_file"] = t.get("train_file", None)
    out["valid_file"] = t.get("valid_file", None)

    # Optimization
    out["learning_rates"] = _normalize_learning_rates(t.get("learning_rates", t.get("learning_rate", None)))
    out["batch_size"] = int(t.get("batch_size", 8))
    out["eval_batch_size"] = t.get("eval_batch_size", None)
    out["eval_batch_size"] = int(out["eval_batch_size"]) if out["eval_batch_size"] is not None else out["batch_size"]
    out["num_epochs"] = int(t.get("num_epochs", 30))
    out["weight_decay"] = float(t.get("weight_decay", 0.01))
    out["seed"] = int(t.get("seed", 42))
    out["use_class_weights"] = bool(t.get("use_class_weights", True))

    # Loader
    out["num_workers"] = int(t.get("num_workers", 0))
    out["pin_memory"] = t.get("pin_memory", None)
    out["grad_accum_steps"] = int(t.get("grad_accum_steps", 1))
    out["max_length"] = t.get("max_length", None)
    out["max_length"] = int(out["max_length"]) if out["max_length"] is not None else None

    # Precision
    out["fp16"] = bool(t.get("fp16", False))
    out["bf16"] = bool(t.get("bf16", False))
    out["tf32"] = bool(t.get("tf32", False))

    # Early stopping
    out["early_stopping"] = bool(t.get("early_stopping", True))
    out["patience"] = int(t.get("patience", 3))
    out["min_delta"] = float(t.get("min_delta", 0.0))
    out["monitor"] = str(t.get("monitor", "f1_macro"))

    # ✅ "Good stuff"
    out["warmup_ratio"] = float(t.get("warmup_ratio", 0.1))
    out["num_warmup_steps"] = t.get("num_warmup_steps", None)  # if set, overrides warmup_ratio
    out["scheduler_type"] = str(t.get("scheduler_type", "linear")).lower()  # linear | cosine | ...
    out["lr_scheduler"] = str(t.get("lr_scheduler", out["scheduler_type"])).lower()  # accept either key

    out["hidden_dropout"] = float(t.get("hidden_dropout", 0.1))
    out["attention_dropout"] = float(t.get("attention_dropout", 0.1))
    # head dropout: prefer decoder_dropout (existing key), else head_dropout, else hidden_dropout
    out["decoder_dropout"] = float(t.get("decoder_dropout", t.get("head_dropout", out["hidden_dropout"])))
    out["freeze_encoder"] = bool(t.get("freeze_encoder", False))

    # Decoder / head
    out["decoder_type"] = str(t.get("decoder_type", "hf_default"))
    out["cnn_num_filters"] = int(t.get("cnn_num_filters", 128))
    out["cnn_kernel_sizes"] = t.get("cnn_kernel_sizes", [2, 3, 4])
    out["attn_mlp_hidden"] = t.get("attn_mlp_hidden", None)

    # VAAT
    out["vaat_adapter_dim"] = int(t.get("vaat_adapter_dim", 64))
    out["vaat_freeze_encoder"] = bool(t.get("vaat_freeze_encoder", False))

    # device
    common_device = c.get("device", "auto")
    out["device"] = config_device or t.get("device", common_device) or common_device or "auto"

    return out


def _build_enc_config(model_name: str, hidden_dropout: float, attention_dropout: float, classifier_dropout: Optional[float] = None):
    enc_cfg = AutoConfig.from_pretrained(
        model_name,
        num_labels=2,
        hidden_dropout_prob=float(hidden_dropout),
        attention_probs_dropout_prob=float(attention_dropout),
    )
    # Some models support classifier_dropout
    if classifier_dropout is not None and hasattr(enc_cfg, "classifier_dropout"):
        try:
            enc_cfg.classifier_dropout = float(classifier_dropout)
        except Exception:
            pass
    return enc_cfg

def train_binary_model(**kwargs):
    """
    Compatibility wrapper so old main.py (which passes flat kwargs)
    still works with the new config-based trainer.
    """

    # Rebuild config dict from flat kwargs
    cfg = {
        "common": {
            "device": kwargs.get("device", "auto")
        },
        "train": kwargs
    }

    return train_binary_model_config(cfg)
def train_binary_model_config(cfg: Dict[str, Any], config_device: Optional[str] = None) -> None:
    tr = _extract_train_cfg(cfg, config_device=config_device)

    model_name = tr["model_name"]
    if not model_name:
        raise ValueError("train.model_name is required.")
    task = tr["task"]
    output_dir = tr["output_dir"]

    # ----------------------------
    # Device / seeds
    # ----------------------------
    dev = _pick_device(str(tr["device"]))
    use_cuda = dev.type == "cuda"

    if tr["fp16"] and tr["bf16"]:
        raise ValueError("Choose only one of fp16 or bf16.")
    if tr["monitor"] not in {"f1_macro", "accuracy"}:
        raise ValueError("monitor must be either 'f1_macro' or 'accuracy'.")

    if tr["pin_memory"] is None:
        tr["pin_memory"] = use_cuda

    if tr["tf32"] and use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    torch.manual_seed(int(tr["seed"]))
    np.random.seed(int(tr["seed"]))
    if use_cuda:
        torch.cuda.manual_seed_all(int(tr["seed"]))
        torch.backends.cudnn.benchmark = True

    # ----------------------------
    # Data
    # ----------------------------
    if tr["train_file"] and tr["valid_file"]:
        dataset = load_besstie_from_csv(tr["train_file"], tr["valid_file"], task=task)
    else:
        dataset = load_besstie_from_hf(task=task)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # VAAT needs the 'variety' column preserved to build variety_id conditioning.
    extra_keep_cols = ["variety"] if tr["decoder_type"] == "vaat" else None
    tokenised = prepare_dataset(tokenizer, dataset, max_length=tr["max_length"], extra_keep_columns=extra_keep_cols)

    # ----------------------------
    # VAAT: map variety string -> variety_id
    # ----------------------------
    vaat_varieties = None
    if tr["decoder_type"] == "vaat":
        if "variety" not in tokenised["train"].column_names:
            raise ValueError("decoder_type=vaat requires a 'variety' column in the dataset.")
        all_vars = list(tokenised["train"]["variety"]) + list(tokenised["validation"]["variety"])
        vaat_varieties = sorted(list(set(map(str, all_vars))))
        variety_to_id = {v: i for i, v in enumerate(vaat_varieties)}

        def _add_variety_id(batch):
            vs = [str(v) for v in batch["variety"]]
            return {"variety_id": [variety_to_id.get(v, 0) for v in vs]}

        tokenised = tokenised.map(_add_variety_id, batched=True)
        tokenised = tokenised.remove_columns(["variety"])

    # Class weights (optional)
    class_weights = None
    if tr["use_class_weights"]:
        labels_arr = np.array(tokenised["train"]["label"])
        class_weights = compute_class_weights(labels_arr)

    # Keep only needed columns
    keep = {"input_ids", "attention_mask", "label", "variety_id"}
    tokenised = tokenised.remove_columns([c for c in tokenised["train"].column_names if c not in keep])

    train_cols = [c for c in ["input_ids", "attention_mask", "label", "variety_id"] if c in tokenised["train"].column_names]
    val_cols   = [c for c in ["input_ids", "attention_mask", "label", "variety_id"] if c in tokenised["validation"].column_names]
    tokenised["train"].set_format(type="torch", columns=train_cols)
    tokenised["validation"].set_format(type="torch", columns=val_cols)

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

    loader_kwargs = {
        "num_workers": int(tr["num_workers"]),
        "pin_memory": bool(tr["pin_memory"]),
        "persistent_workers": int(tr["num_workers"]) > 0,
    }
    if int(tr["num_workers"]) > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        tokenised["train"],
        batch_size=int(tr["batch_size"]),
        shuffle=True,
        collate_fn=collator,
        **loader_kwargs,
    )
    eval_loader = DataLoader(
        tokenised["validation"],
        batch_size=int(tr["eval_batch_size"]),
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

    # Mixed precision
    autocast_dtype = None
    if use_cuda and tr["fp16"]:
        autocast_dtype = torch.float16
    elif use_cuda and tr["bf16"]:
        autocast_dtype = torch.bfloat16
    use_autocast = autocast_dtype is not None
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda and tr["fp16"])

    # Loss
    if class_weights is not None:
        weight_tensor = torch.tensor(
            [class_weights.get(0, 1.0), class_weights.get(1, 1.0)],
            dtype=torch.float,
        ).to(dev)
        loss_fct = CrossEntropyLoss(weight=weight_tensor)
    else:
        loss_fct = CrossEntropyLoss()

    # ----------------------------
    # Helpers: build model per LR
    # ----------------------------
    
    def build_model() -> Tuple[nn.Module, Optional[nn.Module], Optional[nn.Module], Optional[Dict[str, Any]]]:
        """
        Returns:
          model_for_training,
          encoder (if custom head) else None,
          head (if custom head) else None,
          decoder_cfg_dict (if custom head) else None
        """
        # Encoder config with dropout control
        enc_cfg = _build_enc_config(
            model_name=model_name,
            hidden_dropout=tr["hidden_dropout"],
            attention_dropout=tr["attention_dropout"],
            classifier_dropout=(tr["decoder_dropout"] if tr["decoder_type"] == "hf_default" else None),
        )

        if tr["decoder_type"] == "hf_default":
            m = AutoModelForSequenceClassification.from_pretrained(model_name, config=enc_cfg)
            # Freeze encoder if requested
            if tr["freeze_encoder"]:
                base_prefix = m.base_model_prefix
                base_model = getattr(m, base_prefix)
                for p in base_model.parameters():
                    p.requires_grad = False
            return m, None, None, None

        # Custom head: encoder + head
        encoder = AutoModel.from_pretrained(model_name, config=enc_cfg)

        hidden_size = getattr(enc_cfg, "hidden_size", getattr(enc_cfg, "d_model", None))
        if hidden_size is None:
            raise ValueError("Cannot infer hidden size from encoder config.")
        hidden_size = int(hidden_size)

        dec_cfg = DecoderConfig(
            decoder_type=str(tr["decoder_type"]),
            num_labels=2,
            dropout=float(tr["decoder_dropout"]),
            cnn_num_filters=int(tr["cnn_num_filters"]),
            cnn_kernel_sizes=list(map(int, tr["cnn_kernel_sizes"])),
            attn_mlp_hidden=tr["attn_mlp_hidden"],
            vaat_adapter_dim=int(tr["vaat_adapter_dim"]),
            vaat_varieties=vaat_varieties,
        )
        head = build_head(hidden_size=hidden_size, cfg=dec_cfg)

        model = EncoderWithCustomHead(encoder=encoder, head=head)

        # Freeze encoder option for custom head
        if tr["freeze_encoder"] or (tr["decoder_type"] == "vaat" and tr["vaat_freeze_encoder"]):
            for p in model.encoder.parameters():
                p.requires_grad = False

        return model, encoder, head, dec_cfg.to_dict()

    # ----------------------------
    # Train over LRs
    # ----------------------------
    from torch.optim import AdamW

    best_f1_macro_global = -float("inf")
    best_lr_global: Optional[float] = None
    best_state_global: Optional[Dict[str, torch.Tensor]] = None
    best_is_custom_global: bool = False
    best_decoder_cfg_global: Optional[Dict[str, Any]] = None
    best_epoch_global: int = 0
    best_monitor_val_global: float = -float("inf")

    # We'll rebuild and re-load best weights before saving (keeps memory sane)
    for lr in tr["learning_rates"]:
        lr = float(lr)
        print(f"\n*** Training lr={lr} | decoder={tr['decoder_type']} | warmup={tr['warmup_ratio']} ***")

        model, encoder_obj, head_obj, decoder_cfg_dict = build_model()
        model.to(dev)

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=float(tr["weight_decay"]))

        # Scheduler (warmup + chosen type)
        num_update_steps_per_epoch = max(1, len(train_loader) // max(1, int(tr["grad_accum_steps"])))
        max_train_steps = int(tr["num_epochs"] * num_update_steps_per_epoch)

        if tr["num_warmup_steps"] is not None:
            num_warmup_steps = int(tr["num_warmup_steps"])
        else:
            num_warmup_steps = int(max_train_steps * float(tr["warmup_ratio"]))

        sched_name = str(tr["lr_scheduler"] or tr["scheduler_type"] or "linear").lower()
        # transformers.get_scheduler expects canonical names like "linear", "cosine"
        if sched_name not in {"linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"}:
            # fallback to linear (safe)
            sched_name = "linear"

        scheduler = get_scheduler(
            name=sched_name,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )

        # Early stopping / best-epoch snapshot for THIS LR
        best_metric_this_lr = -float("inf")
        best_state_this_lr: Optional[Dict[str, torch.Tensor]] = None
        best_epoch_this_lr = 0
        bad_epochs = 0

        for epoch in range(int(tr["num_epochs"])):
            model.train()
            epoch_loss = 0.0
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(
                tqdm(train_loader, desc=f"LR {lr} Ep {epoch + 1}/{tr['num_epochs']} [Train]"),
                start=1,
            ):
                batch = {k: v.to(dev, non_blocking=use_cuda) for k, v in batch.items()}

                labels = batch.pop("label")
                with torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
                    outputs = model(**batch)
                    logits = outputs.logits
                    loss = loss_fct(logits.view(-1, 2), labels.view(-1))
                    loss = loss / max(1, int(tr["grad_accum_steps"]))

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                epoch_loss += loss.item() * max(1, int(tr["grad_accum_steps"]))

                if step % max(1, int(tr["grad_accum_steps"])) == 0:
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            # flush remaining grads
            if len(train_loader) % max(1, int(tr["grad_accum_steps"])) != 0:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            avg_loss = epoch_loss / max(1, len(train_loader))

            # ----------------------------
            # Validation
            # ----------------------------
            model.eval()
            all_logits = []
            all_labels = []
            with torch.inference_mode():
                for batch in tqdm(eval_loader, desc=f"LR {lr} Ep {epoch + 1}/{tr['num_epochs']} [Val]"):
                    batch = {k: v.to(dev, non_blocking=use_cuda) for k, v in batch.items()}
                    val_labels = batch.pop("label")

                    with torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
                        outputs = model(**batch)
                        logits_val = outputs.logits

                    # IMPORTANT: float() avoids BF16 -> numpy issues
                    all_logits.append(logits_val.detach().float().cpu().numpy())
                    all_labels.append(val_labels.detach().cpu().numpy())

            logits_np = np.concatenate(all_logits, axis=0)
            labels_np = np.concatenate(all_labels, axis=0)
            metrics = compute_metrics((logits_np, labels_np))

            current_metric = float(metrics[tr["monitor"]])
            print(
                f"LR {lr} Ep {epoch + 1}/{tr['num_epochs']} | loss={avg_loss:.4f} | "
                f"acc={metrics['accuracy']:.4f} | f1_macro={metrics['f1_macro']:.4f} | "
                f"{tr['monitor']}={current_metric:.4f}"
            )

            # Best-epoch snapshot (in-memory)
            if current_metric > (best_metric_this_lr + float(tr["min_delta"])):
                best_metric_this_lr = current_metric
                best_epoch_this_lr = epoch + 1
                best_state_this_lr = _snapshot_state_dict_to_cpu(model)
                bad_epochs = 0
            else:
                bad_epochs += 1

            if tr["early_stopping"] and bad_epochs >= int(tr["patience"]):
                print(
                    f"Early stopping at epoch {epoch + 1}. "
                    f"Best epoch was {best_epoch_this_lr} with {tr['monitor']}={best_metric_this_lr:.4f}"
                )
                break

        # Restore best weights for this LR
        if best_state_this_lr is not None:
            model.load_state_dict(best_state_this_lr)

        # Rank LRs by macro-F1 (stable and paper-aligned)
        f1_macro_this_lr = float(metrics.get("f1_macro", -float("inf")))

        if f1_macro_this_lr > best_f1_macro_global:
            best_f1_macro_global = f1_macro_this_lr
            best_lr_global = lr
            best_state_global = _snapshot_state_dict_to_cpu(model)
            best_is_custom_global = (tr["decoder_type"] != "hf_default")
            best_decoder_cfg_global = decoder_cfg_dict
            best_epoch_global = best_epoch_this_lr
            best_monitor_val_global = best_metric_this_lr

        # cleanup
        del model
        if use_cuda:
            torch.cuda.empty_cache()

    if best_lr_global is None or best_state_global is None:
        raise RuntimeError("Training produced no model (unexpected).")

    # ----------------------------
    # Save best (across LRs)
    # ----------------------------
    os.makedirs(output_dir, exist_ok=True)

    # Rebuild a fresh model and load best weights (so we can save cleanly)
    model, encoder_obj, head_obj, decoder_cfg_dict = build_model()
    model.load_state_dict(best_state_global)
    model.to(dev)

    if tr["decoder_type"] == "hf_default":
        assert isinstance(model, AutoModelForSequenceClassification) or hasattr(model, "save_pretrained")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(
            f"Saved HF model to {output_dir} | best_lr={best_lr_global} | "
            f"best_epoch={best_epoch_global} | f1_macro={best_f1_macro_global:.4f}"
        )
        return

    # Custom decoder checkpoint
    if encoder_obj is None or head_obj is None or best_decoder_cfg_global is None:
        # Fallback: split model if needed
        if hasattr(model, "encoder") and hasattr(model, "head"):
            encoder_obj = model.encoder
            head_obj = model.head
        else:
            raise RuntimeError("Custom-head save requested but encoder/head not found.")

    extra_metadata = {
        "best_lr": float(best_lr_global),
        "best_epoch": int(best_epoch_global),
        "best_f1_macro": float(best_f1_macro_global),
        "monitor": str(tr["monitor"]),
        "best_monitor_val": float(best_monitor_val_global),
    }

    save_custom_decoder_checkpoint(
        output_dir=output_dir,
        encoder=encoder_obj,
        tokenizer=tokenizer,
        decoder_cfg=best_decoder_cfg_global,
        head=head_obj,
        extra_metadata=extra_metadata,
    )
    print(
        f"Saved custom decoder checkpoint to {output_dir} | best_lr={best_lr_global} | "
        f"best_epoch={best_epoch_global} | f1_macro={best_f1_macro_global:.4f}"
    )
