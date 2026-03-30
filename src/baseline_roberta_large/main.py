# main.py

import argparse
import json
import os
import sys
from typing import Any, Dict, Set, Tuple, Optional

from train import train_binary_model
from inference import predict_binary


# ---------------------------
# Config
# ---------------------------

def load_config(path: str) -> Dict[str, Any]:
    if not path:
        raise ValueError("You must provide --config <path to .yaml/.yml/.json>")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    _, ext = os.path.splitext(path.lower())
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

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


def allowed_keys() -> Dict[str, Set[str]]:
    return {
        "common": {"device"},
        "train": {
            "model_name", "task", "train_file", "valid_file", "output_dir",
            "learning_rates", "batch_size", "eval_batch_size", "num_epochs",
            "weight_decay", "seed", "use_class_weights",
            "num_workers", "pin_memory", "grad_accum_steps", "max_length",
            "fp16", "bf16", "tf32",
            # Early Stopping Keys
            "early_stopping", "patience", "min_delta", "monitor",
            # NEW KEYS (Anti-Overfitting & Scheduler) <--- ADDED HERE
            "warmup_ratio", "hidden_dropout", "attention_dropout", "freeze_encoder"
        },
        "predict": {
            "checkpoint_dir", "input_file", "output_file", "text",
            "device", "batch_size", "max_length", "fp16", "bf16",
        },
    }


def validate_config(cfg: Dict[str, Any]) -> None:
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a dict at the top level.")

    allowed = allowed_keys()

    for section in cfg.keys():
        if section not in allowed:
            raise ValueError(
                f"Unknown top-level section '{section}'. Allowed: {sorted(allowed.keys())}"
            )

    for section, val in cfg.items():
        if val is None:
            continue
        if not isinstance(val, dict):
            raise ValueError(f"Section '{section}' must be a mapping/dict.")
        unknown = set(val.keys()) - allowed[section]
        if unknown:
            raise ValueError(
                f"Unknown keys in section '{section}': {sorted(unknown)}. "
                f"Allowed: {sorted(allowed[section])}"
            )


def require(cfg: Dict[str, Any], section: str, key: str) -> Any:
    val = cfg.get(section, {}).get(key)
    if val is None:
        raise ValueError(f"Missing required config value: {section}.{key}")
    return val


def get(cfg: Dict[str, Any], section: str, key: str, default=None) -> Any:
    return cfg.get(section, {}).get(key, default)


def coalesce(cli_val, cfg_val):
    """CLI overrides config when CLI value is not None."""
    return cfg_val if cli_val is None else cli_val


# ---------------------------
# CLI parsing (config anywhere)
# ---------------------------

def extract_config_path(argv: list[str]) -> Tuple[Optional[str], list[str]]:
    """
    Pull --config PATH out of argv no matter where it appears.
    Returns: (config_path, argv_without_config_tokens)
    """
    if "--config" not in argv:
        return None, argv

    idx = argv.index("--config")
    if idx == len(argv) - 1:
        raise ValueError("--config provided without a path")

    cfg_path = argv[idx + 1]
    # remove the two tokens
    new_argv = argv[:idx] + argv[idx + 2 :]
    return cfg_path, new_argv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BESSTIE Baseline (Standard HF Model)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    t = subparsers.add_parser("train", help="Fine-tune a baseline model")
    t.add_argument("--model_name", type=str, default=None)
    t.add_argument("--task", type=str, choices=["Sentiment", "Sarcasm"], default=None)
    t.add_argument("--train_file", type=str, default=None)
    t.add_argument("--valid_file", type=str, default=None)
    t.add_argument("--output_dir", type=str, default=None)
    t.add_argument("--learning_rates", type=float, nargs="+", default=None)
    t.add_argument("--batch_size", type=int, default=None)
    t.add_argument("--eval_batch_size", type=int, default=None)
    t.add_argument("--num_epochs", type=int, default=None)
    t.add_argument("--weight_decay", type=float, default=None)
    t.add_argument("--seed", type=int, default=None)
    t.add_argument("--no_class_weights", action="store_true", help="Disable class weights")
    t.add_argument("--device", type=str, default=None, help="auto, cuda, or cpu")
    t.add_argument("--num_workers", type=int, default=None)
    t.add_argument("--no_pin_memory", action="store_true")
    t.add_argument("--grad_accum_steps", type=int, default=None)
    t.add_argument("--max_length", type=int, default=None)
    t.add_argument("--fp16", action="store_true")
    t.add_argument("--bf16", action="store_true")
    t.add_argument("--tf32", action="store_true")
    
    # Early Stopping CLI overrides
    t.add_argument("--no_early_stopping", action="store_true")
    t.add_argument("--patience", type=int, default=None)
    t.add_argument("--min_delta", type=float, default=None)
    t.add_argument("--monitor", type=str, default=None)

    # NEW ARGS FOR OVERFITTING & SCHEDULER <--- ADDED HERE
    t.add_argument("--warmup_ratio", type=float, default=None)
    t.add_argument("--hidden_dropout", type=float, default=None)
    t.add_argument("--attention_dropout", type=float, default=None)
    t.add_argument("--freeze_encoder", action="store_true")


    # Predict
    p = subparsers.add_parser("predict", help="Run inference")
    p.add_argument("--checkpoint_dir", type=str, default=None)
    p.add_argument("--input_file", type=str, default=None)
    p.add_argument("--output_file", type=str, default=None)
    p.add_argument("--text", type=str, nargs="*", default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--max_length", type=int, default=None)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")

    return parser


# ---------------------------
# Run
# ---------------------------

def main():
    cfg_path, argv_wo_cfg = extract_config_path(sys.argv[1:])

    if not cfg_path:
        # Default behavior: if config not provided, try to find one or error out
        # Here we enforce providing it to be explicit.
        raise ValueError("Missing --config. Example: python main.py train --config config.yaml")

    cfg = load_config(cfg_path)
    validate_config(cfg)

    args = build_parser().parse_args(argv_wo_cfg)
    common_device = get(cfg, "common", "device", None)

    if args.command == "train":
        # Required config values
        model_name = require(cfg, "train", "model_name")
        task = require(cfg, "train", "task")
        train_file = require(cfg, "train", "train_file")
        valid_file = require(cfg, "train", "valid_file")
        output_dir = require(cfg, "train", "output_dir")

        learning_rates = require(cfg, "train", "learning_rates")
        batch_size = require(cfg, "train", "batch_size")
        num_epochs = require(cfg, "train", "num_epochs")
        weight_decay = require(cfg, "train", "weight_decay")
        seed = require(cfg, "train", "seed")
        use_class_weights = require(cfg, "train", "use_class_weights")
        num_workers = require(cfg, "train", "num_workers")
        pin_memory = require(cfg, "train", "pin_memory")
        grad_accum_steps = require(cfg, "train", "grad_accum_steps")
        max_length = require(cfg, "train", "max_length")
        fp16 = require(cfg, "train", "fp16")
        bf16 = require(cfg, "train", "bf16")
        tf32 = require(cfg, "train", "tf32")
        
        # Early Stopping Configs
        early_stopping = get(cfg, "train", "early_stopping", True)
        patience = get(cfg, "train", "patience", 3)
        min_delta = get(cfg, "train", "min_delta", 0.0)
        monitor = get(cfg, "train", "monitor", "f1_macro")

        # New params for Overfitting & Scheduler
        warmup_ratio = get(cfg, "train", "warmup_ratio", 0.1)
        hidden_dropout = get(cfg, "train", "hidden_dropout", 0.1)
        attention_dropout = get(cfg, "train", "attention_dropout", 0.1)
        freeze_encoder = get(cfg, "train", "freeze_encoder", False)

        eval_batch_size = get(cfg, "train", "eval_batch_size", None)

        # CLI overrides
        model_name = coalesce(args.model_name, model_name)
        task = coalesce(args.task, task)
        train_file = coalesce(args.train_file, train_file)
        valid_file = coalesce(args.valid_file, valid_file)
        output_dir = coalesce(args.output_dir, output_dir)
        learning_rates = coalesce(args.learning_rates, learning_rates)
        batch_size = coalesce(args.batch_size, batch_size)
        eval_batch_size = coalesce(args.eval_batch_size, eval_batch_size)
        num_epochs = coalesce(args.num_epochs, num_epochs)
        weight_decay = coalesce(args.weight_decay, weight_decay)
        seed = coalesce(args.seed, seed)
        num_workers = coalesce(args.num_workers, num_workers)
        grad_accum_steps = coalesce(args.grad_accum_steps, grad_accum_steps)
        max_length = coalesce(args.max_length, max_length)
        
        # Early Stopping overrides
        patience = coalesce(args.patience, patience)
        min_delta = coalesce(args.min_delta, min_delta)
        monitor = coalesce(args.monitor, monitor)

        # New param overrides
        warmup_ratio = coalesce(args.warmup_ratio, warmup_ratio)
        hidden_dropout = coalesce(args.hidden_dropout, hidden_dropout)
        attention_dropout = coalesce(args.attention_dropout, attention_dropout)
        # For boolean flags, coalesce logic is slightly different if using store_true in CLI
        # but here we use get() for config and action=store_true for CLI.
        # If CLI flag is set, it becomes True. If not set, we respect config.
        if args.freeze_encoder: freeze_encoder = True

        device = coalesce(args.device, get(cfg, "train", "device", common_device))

        # Boolean overrides
        if args.no_class_weights: use_class_weights = False
        if args.no_pin_memory: pin_memory = False
        if args.fp16: fp16 = True
        if args.bf16: bf16 = True
        if args.tf32: tf32 = True
        if args.no_early_stopping: early_stopping = False

        train_binary_model(
            model_name=model_name,
            task=task,
            output_dir=output_dir,
            train_file=train_file,
            valid_file=valid_file,
            learning_rates=tuple(learning_rates),
            batch_size=int(batch_size),
            num_epochs=int(num_epochs),
            weight_decay=float(weight_decay),
            seed=int(seed),
            use_class_weights=bool(use_class_weights),
            eval_batch_size=eval_batch_size,
            device=device,
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            grad_accum_steps=int(grad_accum_steps),
            max_length=int(max_length),
            fp16=bool(fp16),
            bf16=bool(bf16),
            tf32=bool(tf32),
            # New params passed to train.py
            early_stopping=bool(early_stopping),
            patience=int(patience),
            min_delta=float(min_delta),
            monitor=str(monitor),
            warmup_ratio=float(warmup_ratio),
            hidden_dropout=float(hidden_dropout),
            attention_dropout=float(attention_dropout),
            freeze_encoder=bool(freeze_encoder),
        )
        # Note: best model is saved by train.py automatically

    elif args.command == "predict":
        checkpoint_dir = require(cfg, "predict", "checkpoint_dir")
        input_file = get(cfg, "predict", "input_file", None)
        output_file = get(cfg, "predict", "output_file", None)
        text_list = get(cfg, "predict", "text", []) or []
        batch_size = require(cfg, "predict", "batch_size")
        max_length = require(cfg, "predict", "max_length")
        fp16 = require(cfg, "predict", "fp16")
        bf16 = require(cfg, "predict", "bf16")

        checkpoint_dir = coalesce(args.checkpoint_dir, checkpoint_dir)
        input_file = coalesce(args.input_file, input_file)
        output_file = coalesce(args.output_file, output_file)
        if args.text is not None: text_list = args.text
        batch_size = coalesce(args.batch_size, batch_size)
        max_length = coalesce(args.max_length, max_length)
        device = coalesce(args.device, get(cfg, "predict", "device", common_device))
        if args.fp16: fp16 = True
        if args.bf16: bf16 = True

        if input_file:
            import pandas as pd
            if not os.path.exists(input_file):
                print(f"Input file {input_file} does not exist", file=sys.stderr)
                sys.exit(1)
            df = pd.read_csv(input_file)
            if "text" not in df.columns:
                print("Input CSV must contain a 'text' column", file=sys.stderr)
                sys.exit(1)
            texts = df["text"].astype(str).tolist()
            preds = predict_binary(
                checkpoint_dir, texts, device=device, batch_size=int(batch_size),
                max_length=int(max_length), fp16=bool(fp16), bf16=bool(bf16),
            )
            df["prediction"] = preds
            out_path = output_file or os.path.splitext(input_file)[0] + "_predictions.csv"
            df.to_csv(out_path, index=False)
            print(f"Predictions written to {out_path}")
        else:
            if not text_list:
                print("Either predict.input_file or predict.text must be set.", file=sys.stderr)
                sys.exit(1)
            preds = predict_binary(
                checkpoint_dir, text_list, device=device, batch_size=int(batch_size),
                max_length=int(max_length), fp16=bool(fp16), bf16=bool(bf16),
            )
            for t, p in zip(text_list, preds):
                print(f"{p}\t{t}")


if __name__ == "__main__":
    main()