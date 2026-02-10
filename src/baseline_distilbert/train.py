"""
Training script for the BESSTIE figurative language project.

This module contains functions for fine‑tuning pre‑trained encoder models on
the BESSTIE dataset.  It leverages utility functions from ``utils.py`` to
load and preprocess the data, compute class weights and evaluation metrics,
and implements a manual training loop using PyTorch.  This avoids the
dependency on the Hugging Face ``Trainer`` and the associated ``accelerate``
library, which may not be available in all environments.

Example usage from the command line::

    # Train a sarcasm detector using local CSV files
    python train.py --model_name roberta-base --task Sarcasm \
        --train_file train.csv --valid_file valid.csv --output_dir ./sarcasm_model

    # Train a sentiment classifier loaded from the Hugging Face hub
    python train.py --model_name bert-base-uncased --task Sentiment \
        --output_dir ./sentiment_model

To reproduce the settings used in the BESSTIE paper, set ``--num_epochs 30``
and perform a grid search over the learning rate values {1e‑5, 2e‑5, 3e‑5}【180316227421938†L563-L571】.
"""

import argparse
import time
from typing import Optional

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from transformers import (
    AutoModelForSequenceClassification,
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


def train_binary_model(
    model_name: str,
    task: str,
    output_dir: str,
    train_file: Optional[str] = None,
    valid_file: Optional[str] = None,
    learning_rates: Optional[tuple] = None,
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
    log_every_seconds: float = 1.0,
    checkpoint_every_epochs: int = 2,
    resume_from_checkpoint: bool = True,
    checkpoint_dir: Optional[str] = None,
) -> None:
    """Fine‑tune a pre‑trained encoder model on BESSTIE for a binary task.

    This function implements a grid search over learning rates when
    multiple values are provided.  It closely follows the training
    protocol of the BESSTIE paper: 30 epochs, batch size 8, class‑
    weighted cross‑entropy loss【180316227421938†L563-L575】 and learning rate
    selection over {1e‑5, 2e‑5, 3e‑5}【180316227421938†L563-L571】.  After
    training on each candidate learning rate, the model is evaluated
    on the validation split and the model achieving the highest
    macro‑averaged F1 score is saved to ``output_dir``.

    Parameters
    ----------
    model_name : str
        Name or path of a pre‑trained Hugging Face model.  The paper
        uses "roberta-large" as one of its encoder baselines【180316227421938†L540-L552】.
    task : str
        Either ``"Sentiment"`` or ``"Sarcasm"``.
    output_dir : str
        Directory to save the fine‑tuned model and checkpoints.
    train_file : str, optional
        Path to ``train.csv``.  If ``None``, the dataset is loaded from
        Hugging Face using ``load_besstie_from_hf``.
    valid_file : str, optional
        Path to ``valid.csv``.  Required if ``train_file`` is provided.
    learning_rates : tuple, optional
        A tuple of candidate learning rates.  If more than one value is
        provided, a grid search is performed and the best model is
        selected based on macro F1.  If ``None``, defaults to
        ``(1e-5, 2e-5, 3e-5)``.
    batch_size : int, optional
        Batch size per device (default 8).
    eval_batch_size : int, optional
        Batch size for evaluation.  Defaults to ``batch_size`` if not set.
    num_epochs : int, optional
        Number of training epochs (default 30 to match the paper【180316227421938†L563-L571】).
    weight_decay : float, optional
        Weight decay applied during optimisation.
    seed : int, optional
        Random seed for reproducibility.
    use_class_weights : bool, optional
        Whether to apply class weights to mitigate imbalance【180316227421938†L573-L575】.
    device : str, optional
        ``"auto"``, ``"cuda"``, or ``"cpu"``.  ``"auto"`` selects CUDA when available.
    num_workers : int, optional
        DataLoader worker processes (set to 0 to disable multiprocessing).
    pin_memory : bool, optional
        Whether to pin CPU memory for faster host-to-device transfer.  Defaults
        to ``True`` when using CUDA.
    grad_accum_steps : int, optional
        Gradient accumulation steps to increase effective batch size.
    max_length : int, optional
        Maximum sequence length for tokenisation.
    fp16 : bool, optional
        Use FP16 mixed precision on CUDA.
    bf16 : bool, optional
        Use BF16 mixed precision on CUDA (requires hardware support).
    tf32 : bool, optional
        Enable TF32 matmul for faster training on Ampere+ GPUs.
    log_every_seconds : float, optional
        Log training progress every N seconds (default 1.0). Set to 0 to disable.
    checkpoint_every_epochs : int, optional
        Save a training checkpoint every N epochs (default 2).
    resume_from_checkpoint : bool, optional
        If True, resume from the latest checkpoint found for each learning rate.
    checkpoint_dir : str, optional
        Directory to store checkpoints. Defaults to ``<output_dir>/checkpoints``.

    Returns
    -------
    None
        The function trains models and saves the best one to ``output_dir``.
    """
    # Default candidate learning rates if none provided
    if learning_rates is None:
        # Use the three values specified in the BESSTIE paper
        learning_rates = (1e-5, 2e-5, 3e-5)
    # Load dataset either from CSV files or Hugging Face hub once
    print("[info] loading dataset...")
    if train_file and valid_file:
        dataset = load_besstie_from_csv(train_file, valid_file, task=task)
    else:
        dataset = load_besstie_from_hf(task=task)
    # Initialise tokenizer once (reused across learning rate runs)
    print("[info] loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Preprocess dataset once
    print("[info] tokenizing dataset...")
    tokenised_dataset = prepare_dataset(tokenizer, dataset, max_length=max_length)
    # Compute class weights if requested
    class_weights = None
    if use_class_weights:
        labels_arr = np.array(tokenised_dataset["train"]["label"])
        class_weights = compute_class_weights(labels_arr)
    # Prepare data loaders (reuse for all runs)
    print("[info] building dataloaders...")
    tokenised_dataset = tokenised_dataset.remove_columns([
        c for c in tokenised_dataset["train"].column_names if c not in {"input_ids", "attention_mask", "label"}
    ])
    tokenised_dataset["train"].set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    tokenised_dataset["validation"].set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    from torch.utils.data import DataLoader
    if eval_batch_size is None:
        eval_batch_size = batch_size
    if fp16 and bf16:
        raise ValueError("Choose only one of fp16 or bf16.")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    use_cuda = device.type == "cuda"
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
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(
        tokenised_dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        **loader_kwargs,
    )
    eval_loader = DataLoader(
        tokenised_dataset["validation"],
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        **loader_kwargs,
    )
    # Try to import tqdm for progress bars.  If unavailable, define a dummy function.
    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:
        def tqdm(iterable, *args, **kwargs):  # type: ignore
            return iterable
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    # Track the best model and its metric
    best_f1_macro = -float("inf")
    best_model = None
    best_lr = None
    autocast_dtype = None
    if use_cuda and fp16:
        autocast_dtype = torch.float16
    elif use_cuda and bf16:
        autocast_dtype = torch.bfloat16
    use_autocast = autocast_dtype is not None

    import os

    if checkpoint_every_epochs < 0:
        raise ValueError("checkpoint_every_epochs must be >= 0")
    checkpoint_root = checkpoint_dir or os.path.join(output_dir, "checkpoints")

    def _lr_tag(val: float) -> str:
        return str(val).replace(".", "_").replace("-", "m")

    def _find_latest_checkpoint(lr_dir: str):
        if not os.path.isdir(lr_dir):
            return None, None
        best_epoch = None
        best_path = None
        for name in os.listdir(lr_dir):
            if not name.startswith("epoch_"):
                continue
            try:
                epoch_idx = int(name.split("_", 1)[1])
            except ValueError:
                continue
            ckpt_path = os.path.join(lr_dir, name)
            if best_epoch is None or epoch_idx > best_epoch:
                best_epoch = epoch_idx
                best_path = ckpt_path
        return best_path, best_epoch

    def _save_checkpoint(model, optimizer, scaler, lr_dir: str, epoch_idx: int):
        ckpt_dir = os.path.join(lr_dir, f"epoch_{epoch_idx}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
        if scaler is not None and scaler.is_enabled():
            torch.save(scaler.state_dict(), os.path.join(ckpt_dir, "scaler.pt"))
        torch.save({"epoch": epoch_idx}, os.path.join(ckpt_dir, "training_state.pt"))

    def _evaluate_model(model):
        model.eval()
        all_logits = []
        all_labels = []
        eval_start = time.time()
        last_log = eval_start
        with torch.inference_mode():
            for step, batch in enumerate(tqdm(eval_loader, desc="Validation"), start=1):
                batch = {k: v.to(device, non_blocking=use_cuda) for k, v in batch.items()}
                if "label" in batch:
                    val_labels = batch.pop("label")
                elif "labels" in batch:
                    val_labels = batch.pop("labels")
                else:
                    raise KeyError(
                        "Neither 'label' nor 'labels' found in batch during evaluation"
                    )
                with torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
                    outputs = model(**batch)
                    logits_val = outputs.logits
                all_logits.append(logits_val.cpu().numpy())
                all_labels.append(val_labels.cpu().numpy())
                if log_every_seconds and (time.time() - last_log) >= log_every_seconds:
                    elapsed = time.time() - eval_start
                    pct = (step / len(eval_loader)) * 100
                    print(f"[eval] step {step}/{len(eval_loader)} ({pct:.1f}%) elapsed {elapsed:.1f}s")
                    last_log = time.time()
        all_logits_np = np.concatenate(all_logits, axis=0)
        all_labels_np = np.concatenate(all_labels, axis=0)
        return compute_metrics((all_logits_np, all_labels_np))

    # Iterate over candidate learning rates
    for lr in learning_rates:
        print(f"\n*** Training with learning rate {lr}" )
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda and fp16)
        start_epoch = 0
        lr_dir = os.path.join(checkpoint_root, f"lr_{_lr_tag(lr)}")
        ckpt_path, ckpt_epoch = (None, None)
        if resume_from_checkpoint:
            ckpt_path, ckpt_epoch = _find_latest_checkpoint(lr_dir)
        if ckpt_path:
            model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
            start_epoch = int(ckpt_epoch)
            print(f"[resume] Using checkpoint {ckpt_path} (next epoch {start_epoch + 1}/{num_epochs})")
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        model.to(device)
        # Configure optimiser
        from torch.optim import AdamW
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        if ckpt_path:
            opt_path = os.path.join(ckpt_path, "optimizer.pt")
            if os.path.exists(opt_path):
                optimizer.load_state_dict(torch.load(opt_path, map_location=device))
            scaler_path = os.path.join(ckpt_path, "scaler.pt")
            if scaler.is_enabled() and os.path.exists(scaler_path):
                scaler.load_state_dict(torch.load(scaler_path))
        # Set up loss function (weights on device)
        if class_weights is not None:
            weight_tensor = torch.tensor([
                class_weights.get(0, 1.0),
                class_weights.get(1, 1.0),
            ], dtype=torch.float).to(device)
            loss_fct = CrossEntropyLoss(weight=weight_tensor)
        else:
            loss_fct = CrossEntropyLoss()
        # Training loop
        metrics = None
        for epoch in range(start_epoch, num_epochs):
            model.train()
            epoch_loss = 0.0
            epoch_start = time.time()
            last_log = epoch_start
            samples_seen = 0
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(
                tqdm(train_loader, desc=f"LR {lr} Epoch {epoch + 1}/{num_epochs} [Training]"),
                start=1,
            ):
                batch = {k: v.to(device, non_blocking=use_cuda) for k, v in batch.items()}
                if "label" in batch:
                    labels = batch.pop("label")
                elif "labels" in batch:
                    labels = batch.pop("labels")
                else:
                    raise KeyError(
                        "Neither 'label' nor 'labels' found in batch: keys=" + str(list(batch.keys()))
                    )
                with torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
                    outputs = model(**batch)
                    logits = outputs.logits
                    loss = loss_fct(logits.view(-1, 2), labels.view(-1))
                    loss = loss / max(1, grad_accum_steps)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                epoch_loss += loss.item() * max(1, grad_accum_steps)
                samples_seen += labels.size(0)
                if step % max(1, grad_accum_steps) == 0:
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if log_every_seconds and (time.time() - last_log) >= log_every_seconds:
                    elapsed = time.time() - epoch_start
                    pct = (step / len(train_loader)) * 100
                    avg_loss = epoch_loss / max(1, step)
                    speed = samples_seen / max(1e-6, elapsed)
                    print(
                        f"[train] lr={lr} epoch {epoch + 1}/{num_epochs} "
                        f"step {step}/{len(train_loader)} ({pct:.1f}%) "
                        f"loss {avg_loss:.4f} samples/s {speed:.1f} elapsed {elapsed:.1f}s"
                    )
                    last_log = time.time()
            if len(train_loader) % max(1, grad_accum_steps) != 0:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            avg_loss = epoch_loss / len(train_loader)
            metrics = _evaluate_model(model)
            print(
                f"LR {lr} Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f} - "
                f"Acc: {metrics['accuracy']:.4f}, F1_macro: {metrics['f1_macro']:.4f}, "
                f"F1_micro: {metrics['f1_micro']:.4f}"
            )
            if checkpoint_every_epochs and (epoch + 1) % checkpoint_every_epochs == 0:
                _save_checkpoint(model, optimizer, scaler, lr_dir, epoch + 1)
        if metrics is None:
            metrics = _evaluate_model(model)
        if checkpoint_every_epochs and (num_epochs % checkpoint_every_epochs) != 0:
            _save_checkpoint(model, optimizer, scaler, lr_dir, num_epochs)
        # After full training at this learning rate, evaluate final performance
        # Use the last computed metrics (macro F1)
        f1_macro = metrics["f1_macro"]
        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            best_model = model
            best_lr = lr  # type: ignore
        else:
            # Free memory for the model that's not selected
            del model
            if use_cuda:
                torch.cuda.empty_cache()
    # Save the best model and tokenizer
    import os
    os.makedirs(output_dir, exist_ok=True)
    if best_model is not None:
        best_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Best model (LR={best_lr}) saved to {output_dir} with F1_macro={best_f1_macro:.4f}")
    else:
        # This should not happen; but handle gracefully
        print("No model was trained. Please check the training configuration.")


def main():
    parser = argparse.ArgumentParser(description="Fine‑tune a model for sarcasm or sentiment detection using BESSTIE.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="roberta-large",
        help=(
            "Pre‑trained Hugging Face model to fine‑tune (default 'roberta-large' to match the paper"
        ),
    )
    parser.add_argument("--task", type=str, choices=["Sentiment", "Sarcasm"], default="Sarcasm", help="Which task to train on")
    parser.add_argument("--train_file", type=str, help="Path to train.csv (optional)")
    parser.add_argument("--valid_file", type=str, help="Path to valid.csv (optional)")
    parser.add_argument("--output_dir", type=str, default="./model_output", help="Directory to save the trained model")
    # Accept one or more learning rates.  When multiple are provided, a grid
    # search is performed and the best model is selected based on macro F1.
    parser.add_argument(
        "--learning_rates",
        type=float,
        nargs="+",
        default=[1e-5, 2e-5, 3e-5],
        help="One or more learning rates to evaluate. Defaults to the three values used in the BESSTIE paper",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--eval_batch_size", type=int, help="Batch size for evaluation (defaults to --batch_size)")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=30,
        help="Number of training epochs (default 30 to match the BESSTIE paper)",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_class_weights", action="store_true", help="Disable class weights for the loss function")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, or cpu")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument("--no_pin_memory", action="store_true", help="Disable pin_memory in DataLoader")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_length", type=int, help="Max sequence length for tokenization")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision on CUDA")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision on CUDA")
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 matmul on Ampere+ GPUs")
    parser.add_argument("--log_every_seconds", type=float, default=1.0, help="Log progress every N seconds (0 to disable)")
    parser.add_argument("--checkpoint_every_epochs", type=int, default=2, help="Checkpoint every N epochs")
    parser.add_argument("--no_resume_from_checkpoint", action="store_true", help="Disable auto-resume")
    parser.add_argument("--checkpoint_dir", type=str, help="Optional checkpoint root directory")
    args = parser.parse_args()
    # Train the model.  The function handles saving the best model to ``output_dir``.
    train_binary_model(
        model_name=args.model_name,
        task=args.task,
        output_dir=args.output_dir,
        train_file=args.train_file,
        valid_file=args.valid_file,
        learning_rates=tuple(args.learning_rates),
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        seed=args.seed,
        use_class_weights=not args.no_class_weights,
        eval_batch_size=args.eval_batch_size,
        device=args.device,
        num_workers=args.num_workers,
        pin_memory=False if args.no_pin_memory else None,
        grad_accum_steps=args.grad_accum_steps,
        max_length=args.max_length,
        fp16=args.fp16,
        bf16=args.bf16,
        tf32=args.tf32,
        log_every_seconds=args.log_every_seconds,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        resume_from_checkpoint=not args.no_resume_from_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
    )
    print(f"Training complete. Best model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
