"""
Training script for decoder baselines (QLoRA) on the BESSTIE dataset.

This follows the paper's decoder setup:
- Instruction prompt + text input
- Target output token "0" or "1"
- QLoRA (4-bit) fine-tuning over all linear layers
- 30 epochs, batch size 8, LR grid {1e-5, 2e-5, 3e-5}
"""

import argparse
import time
from typing import Optional, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from utils import (
    load_besstie_from_csv,
    load_besstie_from_hf,
    prepare_decoder_dataset,
    compute_metrics_from_preds,
    build_prompt_text,
)


def _find_all_linear_names(model) -> List[str]:
    linear_cls_names = {"Linear", "Linear4bit", "Linear8bitLt"}
    lora_module_names = set()
    for name, module in model.named_modules():
        if module.__class__.__name__ in linear_cls_names:
            lora_module_names.add(name.split(".")[-1])
    lora_module_names.discard("lm_head")
    return sorted(lora_module_names)


def _collate_causal_lm(features, pad_token_id: int):
    max_len = max(len(f["input_ids"]) for f in features)
    input_ids = []
    attention_mask = []
    labels = []
    for f in features:
        pad_len = max_len - len(f["input_ids"])
        input_ids.append(f["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(f["attention_mask"] + [0] * pad_len)
        labels.append(f["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _get_quant_dtype(compute_dtype: str) -> torch.dtype:
    compute_dtype = (compute_dtype or "bf16").lower()
    if compute_dtype == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if compute_dtype == "fp16":
        return torch.float16
    raise ValueError("compute_dtype must be 'bf16' or 'fp16'")


def _parse_label(text: str) -> int:
    for ch in text:
        if ch == "0":
            return 0
        if ch == "1":
            return 1
    return 0


def _predict_labels(
    model,
    tokenizer,
    texts: List[str],
    task: str,
    prompt_sentiment: str,
    prompt_sarcasm: str,
    prompt_template: str,
    device: torch.device,
    batch_size: int,
    max_length: int,
    fp16: bool,
    bf16: bool,
    log_every_seconds: float = 1.0,
    log_prefix: str = "[eval]",
) -> List[int]:
    prompts = [
        build_prompt_text(task, t, prompt_sentiment, prompt_sarcasm, prompt_template)
        for t in texts
    ]
    loader = DataLoader(prompts, batch_size=batch_size, shuffle=False)
    preds: List[int] = []
    autocast_dtype = None
    if device.type == "cuda" and fp16:
        autocast_dtype = torch.float16
    elif device.type == "cuda" and bf16:
        autocast_dtype = torch.bfloat16
    use_autocast = autocast_dtype is not None

    start = time.time()
    last_log = start
    with torch.inference_mode():
        for step, batch_prompts in enumerate(loader, start=1):
            inputs = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device, non_blocking=device.type == "cuda") for k, v in inputs.items()}
            with torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
                outputs = model(**inputs)
                next_token_logits = outputs.logits[:, -1, :]
            next_ids = next_token_logits.argmax(dim=-1).tolist()
            for tok_id in next_ids:
                tok = tokenizer.decode([tok_id]).strip()
                preds.append(_parse_label(tok))
            if log_every_seconds and (time.time() - last_log) >= log_every_seconds:
                elapsed = time.time() - start
                pct = (step / len(loader)) * 100
                print(f"{log_prefix} step {step}/{len(loader)} ({pct:.1f}%) elapsed {elapsed:.1f}s")
                last_log = time.time()
    return preds


def train_decoder_model(
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
    device: str = "auto",
    num_workers: int = 2,
    pin_memory: Optional[bool] = None,
    grad_accum_steps: int = 1,
    max_length: Optional[int] = None,
    use_qlora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "all-linear",
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
    compute_dtype: str = "bf16",
    prompt_sentiment: str = "",
    prompt_sarcasm: str = "",
    prompt_template: str = "{prompt}\n{text}\n",
    fp16: bool = False,
    bf16: bool = False,
    tf32: bool = False,
    log_every_seconds: float = 1.0,
    checkpoint_every_epochs: int = 2,
    resume_from_checkpoint: bool = True,
    checkpoint_dir: Optional[str] = None,
) -> None:
    if learning_rates is None:
        learning_rates = (1e-5, 2e-5, 3e-5)

    print("[info] loading dataset...")
    if train_file and valid_file:
        dataset = load_besstie_from_csv(train_file, valid_file, task=task)
    else:
        dataset = load_besstie_from_hf(task=task)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    use_cuda = device.type == "cuda"
    if use_qlora and not use_cuda:
        raise ValueError("QLoRA requires CUDA. Set device=cuda or disable use_qlora.")

    if pin_memory is None:
        pin_memory = use_cuda

    if tf32 and use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    print("[info] loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[info] tokenizing dataset for decoder...")
    tokenised_dataset = prepare_decoder_dataset(
        tokenizer=tokenizer,
        dataset=dataset,
        task=task,
        prompt_sentiment=prompt_sentiment,
        prompt_sarcasm=prompt_sarcasm,
        prompt_template=prompt_template,
        max_length=max_length,
    )

    if eval_batch_size is None:
        eval_batch_size = batch_size

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    print("[info] building dataloaders...")
    train_loader = DataLoader(
        tokenised_dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda f: _collate_causal_lm(f, tokenizer.pad_token_id),
        **loader_kwargs,
    )

    # Validation data for generation-based metrics
    val_texts = [str(x) for x in dataset["validation"]["text"]]
    val_labels = np.array([int(x) for x in dataset["validation"]["label"]])

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

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

    def _load_model_from_checkpoint(ckpt_path: str):
        adapter_path = os.path.join(ckpt_path, "adapter_config.json")
        if os.path.exists(adapter_path):
            from peft import PeftConfig, PeftModel

            peft_config = PeftConfig.from_pretrained(ckpt_path)
            base_model_name = peft_config.base_model_name_or_path
            if use_qlora:
                quant_dtype = _get_quant_dtype(compute_dtype)
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                    bnb_4bit_compute_dtype=quant_dtype,
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                )
            else:
                base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
                base_model.to(device)
            model = PeftModel.from_pretrained(base_model, ckpt_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(ckpt_path)
            model.to(device)
        return model

    # Autocast config
    autocast_dtype = None
    if use_cuda and fp16:
        autocast_dtype = torch.float16
    elif use_cuda and bf16:
        autocast_dtype = torch.bfloat16
    use_autocast = autocast_dtype is not None
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda and fp16)

    # Model init (per LR)
    best_f1_macro = -float("inf")
    best_model = None
    best_lr = None

    for lr in learning_rates:
        print(f"\n*** Training with learning rate {lr}")
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda and fp16)
        start_epoch = 0
        lr_dir = os.path.join(checkpoint_root, f"lr_{_lr_tag(lr)}")
        ckpt_path, ckpt_epoch = (None, None)
        if resume_from_checkpoint:
            ckpt_path, ckpt_epoch = _find_latest_checkpoint(lr_dir)
        if ckpt_path:
            model = _load_model_from_checkpoint(ckpt_path)
            start_epoch = int(ckpt_epoch)
            print(f"[resume] Using checkpoint {ckpt_path} (next epoch {start_epoch + 1}/{num_epochs})")
        else:
            if use_qlora:
                from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

                quant_dtype = _get_quant_dtype(compute_dtype)
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                    bnb_4bit_compute_dtype=quant_dtype,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                )
                model = prepare_model_for_kbit_training(model)
                if lora_target_modules == "all-linear":
                    target_modules = _find_all_linear_names(model)
                else:
                    target_modules = [m.strip() for m in lora_target_modules.split(",") if m.strip()]
                lora_cfg = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target_modules,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, lora_cfg)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name)
                model.to(device)

        model.config.use_cache = False
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        if ckpt_path:
            opt_path = os.path.join(ckpt_path, "optimizer.pt")
            if os.path.exists(opt_path):
                optimizer.load_state_dict(torch.load(opt_path, map_location=device))
            scaler_path = os.path.join(ckpt_path, "scaler.pt")
            if scaler.is_enabled() and os.path.exists(scaler_path):
                scaler.load_state_dict(torch.load(scaler_path))

        metrics = None
        for epoch in range(start_epoch, num_epochs):
            model.train()
            epoch_loss = 0.0
            epoch_start = time.time()
            last_log = epoch_start
            samples_seen = 0
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(train_loader, start=1):
                batch = {k: v.to(device, non_blocking=use_cuda) for k, v in batch.items()}
                with torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss = loss / max(1, grad_accum_steps)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                epoch_loss += loss.item() * max(1, grad_accum_steps)
                samples_seen += batch["input_ids"].size(0)
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
            model.eval()
            preds = _predict_labels(
                model=model,
                tokenizer=tokenizer,
                texts=val_texts,
                task=task,
                prompt_sentiment=prompt_sentiment,
                prompt_sarcasm=prompt_sarcasm,
                prompt_template=prompt_template,
                device=device,
                batch_size=eval_batch_size,
                max_length=max_length or 256,
                fp16=fp16,
                bf16=bf16,
                log_every_seconds=log_every_seconds,
                log_prefix="[eval]",
            )
            metrics = compute_metrics_from_preds(np.array(preds), val_labels)
            print(
                f"LR {lr} Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f} - "
                f"Acc: {metrics['accuracy']:.4f}, F1_macro: {metrics['f1_macro']:.4f}, "
                f"F1_micro: {metrics['f1_micro']:.4f}"
            )
            if checkpoint_every_epochs and (epoch + 1) % checkpoint_every_epochs == 0:
                _save_checkpoint(model, optimizer, scaler, lr_dir, epoch + 1)

        if metrics is None:
            preds = _predict_labels(
                model=model,
                tokenizer=tokenizer,
                texts=val_texts,
                task=task,
                prompt_sentiment=prompt_sentiment,
                prompt_sarcasm=prompt_sarcasm,
                prompt_template=prompt_template,
                device=device,
                batch_size=eval_batch_size,
                max_length=max_length or 256,
                fp16=fp16,
                bf16=bf16,
                log_every_seconds=log_every_seconds,
                log_prefix="[eval]",
            )
            metrics = compute_metrics_from_preds(np.array(preds), val_labels)
        if checkpoint_every_epochs and (num_epochs % checkpoint_every_epochs) != 0:
            _save_checkpoint(model, optimizer, scaler, lr_dir, num_epochs)

        f1_macro = metrics["f1_macro"]
        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            best_model = model
            best_lr = lr
        else:
            del model
            if use_cuda:
                torch.cuda.empty_cache()

    import os
    os.makedirs(output_dir, exist_ok=True)
    if best_model is not None:
        best_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Best model (LR={best_lr}) saved to {output_dir} with F1_macro={best_f1_macro:.4f}")
    else:
        print("No model was trained. Please check the training configuration.")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a decoder model using QLoRA on BESSTIE.")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-Small-Instruct-2409")
    parser.add_argument("--task", type=str, choices=["Sentiment", "Sarcasm"], default="Sarcasm")
    parser.add_argument("--train_file", type=str, help="Path to train.csv (optional)")
    parser.add_argument("--valid_file", type=str, help="Path to valid.csv (optional)")
    parser.add_argument("--output_dir", type=str, default="./model_output")
    parser.add_argument("--learning_rates", type=float, nargs="+", default=[1e-5, 2e-5, 3e-5])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="all-linear")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true")
    parser.add_argument("--compute_dtype", type=str, default="bf16")
    parser.add_argument("--prompt_sentiment", type=str, required=True)
    parser.add_argument("--prompt_sarcasm", type=str, required=True)
    parser.add_argument("--prompt_template", type=str, default="{prompt}\n{text}\n")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--log_every_seconds", type=float, default=1.0)
    parser.add_argument("--checkpoint_every_epochs", type=int, default=2)
    parser.add_argument("--no_resume_from_checkpoint", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    args = parser.parse_args()

    train_decoder_model(
        model_name=args.model_name,
        task=args.task,
        output_dir=args.output_dir,
        train_file=args.train_file,
        valid_file=args.valid_file,
        learning_rates=tuple(args.learning_rates),
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        pin_memory=False if args.no_pin_memory else None,
        grad_accum_steps=args.grad_accum_steps,
        max_length=args.max_length,
        use_qlora=args.use_qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        compute_dtype=args.compute_dtype,
        prompt_sentiment=args.prompt_sentiment,
        prompt_sarcasm=args.prompt_sarcasm,
        prompt_template=args.prompt_template,
        fp16=args.fp16,
        bf16=args.bf16,
        tf32=args.tf32,
        log_every_seconds=args.log_every_seconds,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        resume_from_checkpoint=not args.no_resume_from_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
