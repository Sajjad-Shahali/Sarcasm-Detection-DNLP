"""
Training script for the BESSTIE figurative language project (BASELINE).
Updated with: LR Scheduling, Dropout Configuration, Layer Freezing, and BF16 fixes.
"""

import argparse
import os
import pprint
from typing import Optional, Sequence

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from transformers import (
    AutoConfig,
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

def _pick_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)

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
    # ---- Early Stopping Logic ----
    early_stopping: bool = True,
    patience: int = 3,
    min_delta: float = 0.0,
    monitor: str = "f1_macro",
    # ---- NEW ARGS FOR OPTIMIZATION ----
    warmup_ratio: float = 0.1,         # % of steps for warmup
    hidden_dropout: float = 0.1,       # Control dropout (default 0.1)
    attention_dropout: float = 0.1,    # Control attn dropout
    freeze_encoder: bool = False,      # Freeze base model option
) -> None:
    
    # 1. Setup Defaults
    if learning_rates is None:
        learning_rates = (1e-5, 2e-5, 3e-5)
    
    if fp16 and bf16:
        raise ValueError("Choose only one of fp16 or bf16.")

    # 2. Data Loading
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

    # Formatting columns
    tokenised_dataset = tokenised_dataset.remove_columns([
        c for c in tokenised_dataset["train"].column_names if c not in {"input_ids", "attention_mask", "label"}
    ])
    tokenised_dataset["train"].set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    tokenised_dataset["validation"].set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 3. DataLoaders
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

    # TQDM
    try:
        from tqdm.auto import tqdm
    except Exception:
        def tqdm(x, *args, **kwargs): return x

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

    # Global Tracking Variables
    best_f1_macro_global = -float("inf")
    best_lr_global = None
    best_model_global = None

    def _maybe_update_best(model, current, best, best_state, min_delta):
        """Update best state in RAM if metric improves."""
        if current > (best + float(min_delta)):
            new_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            return current, new_state, True
        return best, best_state, False

    # -------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------
    from torch.optim import AdamW

    for lr in learning_rates:
        lr = float(lr)
        print(f"\n*** [Baseline] Training with LR={lr} | Warmup={warmup_ratio} | Drop={hidden_dropout} ***")

        # 1. Config with Custom Dropout
        config = AutoConfig.from_pretrained(
            model_name, 
            num_labels=2,
            hidden_dropout_prob=hidden_dropout,
            attention_probs_dropout_prob=attention_dropout
        )

        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
        
        # 2. Freezing Logic
        if freeze_encoder:
            print("[Info] Freezing encoder layers, training classifier only.")
            base_model_prefix = model.base_model_prefix 
            base_model = getattr(model, base_model_prefix)
            for param in base_model.parameters():
                param.requires_grad = False
        
        model.to(dev)

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=float(weight_decay))

        # 3. Learning Rate Scheduler
        num_update_steps_per_epoch = len(train_loader) // grad_accum_steps
        max_train_steps = int(num_epochs * num_update_steps_per_epoch)
        num_warmup_steps = int(max_train_steps * warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps
        )

        best_metric_this_lr = -float("inf")
        best_state_this_lr = None
        best_epoch_this_lr = 0
        bad_epochs = 0

        for epoch in range(int(num_epochs)):
            # --- TRAIN ---
            model.train()
            epoch_loss = 0.0
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(
                tqdm(train_loader, desc=f"LR {lr} Ep {epoch + 1}/{num_epochs} [Train]"),
                start=1,
            ):
                batch = {k: v.to(dev, non_blocking=use_cuda) for k, v in batch.items()}
                
                if "label" in batch: labels = batch.pop("label")
                elif "labels" in batch: labels = batch.pop("labels")
                else: raise KeyError("Missing label in batch")

                with torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
                    outputs = model(**batch)
                    loss = loss_fct(outputs.logits.view(-1, 2), labels.view(-1))
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
                    
                    # Scheduler Step (After optimizer step)
                    scheduler.step()
                    
                    # Log LR occasionally
                    if step % 100 == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        # print(f"   [Step {step}] Current LR: {current_lr:.8f}")

                    optimizer.zero_grad(set_to_none=True)

            # Flush remaining gradients
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
            model.eval()
            all_logits = []
            all_labels = []
            with torch.inference_mode():
                for batch in tqdm(eval_loader, desc=f"LR {lr} Ep {epoch + 1}/{num_epochs} [Val]"):
                    batch = {k: v.to(dev, non_blocking=use_cuda) for k, v in batch.items()}
                    if "label" in batch: val_labels = batch.pop("label")
                    elif "labels" in batch: val_labels = batch.pop("labels")
                    
                    with torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
                        outputs = model(**batch)
                    
                    # FIXED: Added .float() to prevent BFloat16/Numpy error
                    all_logits.append(outputs.logits.detach().float().cpu().numpy())
                    all_labels.append(val_labels.detach().cpu().numpy())

            all_logits_np = np.concatenate(all_logits, axis=0)
            all_labels_np = np.concatenate(all_labels, axis=0)
            metrics = compute_metrics((all_logits_np, all_labels_np))

            print(
                f"LR {lr} Ep {epoch + 1}/{num_epochs} | loss={avg_loss:.4f} | "
                f"Acc={metrics['accuracy']:.4f} | F1_macro={metrics['f1_macro']:.4f}"
            )

            # --- EARLY STOPPING CHECK ---
            current_metric = float(metrics[monitor])
            best_metric_this_lr, best_state_this_lr, improved = _maybe_update_best(
                model=model,
                current=current_metric,
                best=best_metric_this_lr,
                best_state=best_state_this_lr,
                min_delta=min_delta,
            )

            if improved:
                best_epoch_this_lr = epoch + 1
                bad_epochs = 0
            else:
                bad_epochs += 1

            if early_stopping and bad_epochs >= int(patience):
                print(f"Early stopping at epoch {epoch+1}. Best was epoch {best_epoch_this_lr} ({monitor}={best_metric_this_lr:.4f})")
                break
        
        # End of Epochs for this LR
        if best_state_this_lr is not None:
            model.load_state_dict(best_state_this_lr)
            final_f1_for_lr = best_metric_this_lr if monitor == "f1_macro" else metrics['f1_macro']
        else:
            final_f1_for_lr = metrics['f1_macro']

        if final_f1_for_lr > best_f1_macro_global:
            if best_model_global is not None:
                del best_model_global
                if use_cuda: torch.cuda.empty_cache()
            
            best_f1_macro_global = final_f1_for_lr
            best_lr_global = lr
            best_model_global = model 
            print(f"-> New Global Best Model Found (LR={lr}, F1={best_f1_macro_global:.4f})")
        else:
            del model
            if use_cuda: torch.cuda.empty_cache()

    # Save Final
    os.makedirs(output_dir, exist_ok=True)
    if best_model_global is not None:
        best_model_global.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"\n[Done] Baseline model saved to {output_dir}")
        print(f"Best LR: {best_lr_global}")
        print(f"Best F1 Macro: {best_f1_macro_global:.4f}")
    else:
        print("Error: No model was trained.")

def main():
    parser = argparse.ArgumentParser(description="Train Baseline BESSTIE classifier (Standard HF Model).")
    
    # Model & Data
    parser.add_argument("--model_name", type=str, default="roberta-large", help="Baseline model (default: roberta-large)")
    parser.add_argument("--task", type=str, choices=["Sentiment", "Sarcasm"], default="Sarcasm")
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--valid_file", type=str)
    parser.add_argument("--output_dir", type=str, default="./model_output")
    
    # Hyperparameters
    parser.add_argument("--learning_rates", type=float, nargs="+", default=[1e-5, 2e-5, 3e-5])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_class_weights", action="store_true")
    
    # Hardware
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    
    # Early Stopping Config
    parser.add_argument("--no_early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--monitor", type=str, default="f1_macro", choices=["f1_macro", "accuracy"])

    # --- NEW ARGS FOR OVERFITTING & SCHEDULER ---
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of total steps for warmup")
    parser.add_argument("--hidden_dropout", type=float, default=0.1, help="Dropout prob for hidden layers")
    parser.add_argument("--attention_dropout", type=float, default=0.1, help="Dropout prob for attention")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze the base model and train head only")

    args = parser.parse_args()
    
    print("\n========== BASELINE TRAINING ARGS ==========")
    pprint.pprint(vars(args), sort_dicts=False)
    print("==========================================\n")

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
        early_stopping=not args.no_early_stopping,
        patience=args.patience,
        min_delta=args.min_delta,
        monitor=args.monitor,
        # New args
        warmup_ratio=args.warmup_ratio,
        hidden_dropout=args.hidden_dropout,
        attention_dropout=args.attention_dropout,
        freeze_encoder=args.freeze_encoder,
    )

if __name__ == "__main__":
    main()