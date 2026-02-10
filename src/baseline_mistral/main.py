import argparse
import json
import os
import sys
from typing import Any, Dict, Set, Tuple, Optional

from train import train_decoder_model
from inference import predict_decoder


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
            "weight_decay", "seed",
            "num_workers", "pin_memory", "grad_accum_steps", "max_length",
            "use_qlora", "lora_r", "lora_alpha", "lora_dropout", "lora_target_modules",
            "bnb_4bit_quant_type", "bnb_4bit_use_double_quant", "compute_dtype",
            "prompt_sentiment", "prompt_sarcasm", "prompt_template",
            "fp16", "bf16", "tf32",
            "log_every_seconds",
            "checkpoint_every_epochs", "resume_from_checkpoint", "checkpoint_dir",
        },
        "predict": {
            "checkpoint_dir", "task", "input_file", "output_file", "text",
            "device", "batch_size", "max_length", "max_new_tokens", "fp16", "bf16",
            "use_qlora", "bnb_4bit_quant_type", "bnb_4bit_use_double_quant", "compute_dtype",
            "prompt_sentiment", "prompt_sarcasm", "prompt_template",
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
    return cfg_val if cli_val is None else cli_val


def extract_config_path(argv: list[str]) -> Tuple[Optional[str], list[str]]:
    if "--config" not in argv:
        return None, argv

    idx = argv.index("--config")
    if idx == len(argv) - 1:
        raise ValueError("--config provided without a path")

    cfg_path = argv[idx + 1]
    new_argv = argv[:idx] + argv[idx + 2 :]
    return cfg_path, new_argv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BESSTIE decoder baseline (config-first)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    t = subparsers.add_parser("train", help="Fine-tune a decoder with QLoRA on BESSTIE")
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
    t.add_argument("--device", type=str, default=None)
    t.add_argument("--num_workers", type=int, default=None)
    t.add_argument("--no_pin_memory", action="store_true")
    t.add_argument("--grad_accum_steps", type=int, default=None)
    t.add_argument("--max_length", type=int, default=None)
    t.add_argument("--use_qlora", action="store_true")
    t.add_argument("--no_qlora", action="store_true")
    t.add_argument("--lora_r", type=int, default=None)
    t.add_argument("--lora_alpha", type=int, default=None)
    t.add_argument("--lora_dropout", type=float, default=None)
    t.add_argument("--lora_target_modules", type=str, default=None)
    t.add_argument("--bnb_4bit_quant_type", type=str, default=None)
    t.add_argument("--bnb_4bit_use_double_quant", action="store_true")
    t.add_argument("--no_bnb_4bit_use_double_quant", action="store_true")
    t.add_argument("--compute_dtype", type=str, default=None)
    t.add_argument("--prompt_sentiment", type=str, default=None)
    t.add_argument("--prompt_sarcasm", type=str, default=None)
    t.add_argument("--prompt_template", type=str, default=None)
    t.add_argument("--fp16", action="store_true")
    t.add_argument("--bf16", action="store_true")
    t.add_argument("--tf32", action="store_true")
    t.add_argument("--log_every_seconds", type=float, default=None)
    t.add_argument("--checkpoint_every_epochs", type=int, default=None)
    t.add_argument("--no_resume_from_checkpoint", action="store_true")
    t.add_argument("--checkpoint_dir", type=str, default=None)

    p = subparsers.add_parser("predict", help="Run inference with a decoder baseline")
    p.add_argument("--checkpoint_dir", type=str, default=None)
    p.add_argument("--task", type=str, choices=["Sentiment", "Sarcasm"], default=None)
    p.add_argument("--input_file", type=str, default=None)
    p.add_argument("--output_file", type=str, default=None)
    p.add_argument("--text", type=str, nargs="*", default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--max_length", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--use_qlora", action="store_true")
    p.add_argument("--no_qlora", action="store_true")
    p.add_argument("--bnb_4bit_quant_type", type=str, default=None)
    p.add_argument("--bnb_4bit_use_double_quant", action="store_true")
    p.add_argument("--no_bnb_4bit_use_double_quant", action="store_true")
    p.add_argument("--compute_dtype", type=str, default=None)
    p.add_argument("--prompt_sentiment", type=str, default=None)
    p.add_argument("--prompt_sarcasm", type=str, default=None)
    p.add_argument("--prompt_template", type=str, default=None)

    return parser


def main():
    cfg_path, argv_wo_cfg = extract_config_path(sys.argv[1:])

    if not cfg_path:
        raise ValueError("Missing --config. Example: python main.py train --config config.yaml")

    cfg = load_config(cfg_path)
    validate_config(cfg)

    args = build_parser().parse_args(argv_wo_cfg)

    common_device = get(cfg, "common", "device", None)

    if args.command == "train":
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
        num_workers = require(cfg, "train", "num_workers")
        pin_memory = require(cfg, "train", "pin_memory")
        grad_accum_steps = require(cfg, "train", "grad_accum_steps")
        max_length = require(cfg, "train", "max_length")
        use_qlora = require(cfg, "train", "use_qlora")
        lora_r = require(cfg, "train", "lora_r")
        lora_alpha = require(cfg, "train", "lora_alpha")
        lora_dropout = require(cfg, "train", "lora_dropout")
        lora_target_modules = require(cfg, "train", "lora_target_modules")
        bnb_4bit_quant_type = require(cfg, "train", "bnb_4bit_quant_type")
        bnb_4bit_use_double_quant = require(cfg, "train", "bnb_4bit_use_double_quant")
        compute_dtype = require(cfg, "train", "compute_dtype")
        prompt_sentiment = require(cfg, "train", "prompt_sentiment")
        prompt_sarcasm = require(cfg, "train", "prompt_sarcasm")
        prompt_template = require(cfg, "train", "prompt_template")
        fp16 = require(cfg, "train", "fp16")
        bf16 = require(cfg, "train", "bf16")
        tf32 = require(cfg, "train", "tf32")
        log_every_seconds = require(cfg, "train", "log_every_seconds")
        checkpoint_every_epochs = require(cfg, "train", "checkpoint_every_epochs")
        resume_from_checkpoint = require(cfg, "train", "resume_from_checkpoint")
        checkpoint_dir = get(cfg, "train", "checkpoint_dir", None)
        eval_batch_size = get(cfg, "train", "eval_batch_size", None)

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
        lora_r = coalesce(args.lora_r, lora_r)
        lora_alpha = coalesce(args.lora_alpha, lora_alpha)
        lora_dropout = coalesce(args.lora_dropout, lora_dropout)
        lora_target_modules = coalesce(args.lora_target_modules, lora_target_modules)
        bnb_4bit_quant_type = coalesce(args.bnb_4bit_quant_type, bnb_4bit_quant_type)
        compute_dtype = coalesce(args.compute_dtype, compute_dtype)
        prompt_sentiment = coalesce(args.prompt_sentiment, prompt_sentiment)
        prompt_sarcasm = coalesce(args.prompt_sarcasm, prompt_sarcasm)
        prompt_template = coalesce(args.prompt_template, prompt_template)
        log_every_seconds = coalesce(args.log_every_seconds, log_every_seconds)
        checkpoint_every_epochs = coalesce(args.checkpoint_every_epochs, checkpoint_every_epochs)
        checkpoint_dir = coalesce(args.checkpoint_dir, checkpoint_dir)

        device = coalesce(args.device, get(cfg, "train", "device", common_device))

        if args.no_pin_memory:
            pin_memory = False
        if args.use_qlora:
            use_qlora = True
        if args.no_qlora:
            use_qlora = False
        if args.bnb_4bit_use_double_quant:
            bnb_4bit_use_double_quant = True
        if args.no_bnb_4bit_use_double_quant:
            bnb_4bit_use_double_quant = False
        if args.fp16:
            fp16 = True
        if args.bf16:
            bf16 = True
        if args.tf32:
            tf32 = True
        if args.no_resume_from_checkpoint:
            resume_from_checkpoint = False

        train_decoder_model(
            model_name=model_name,
            task=task,
            output_dir=output_dir,
            train_file=train_file,
            valid_file=valid_file,
            learning_rates=tuple(learning_rates),
            batch_size=int(batch_size),
            eval_batch_size=eval_batch_size,
            num_epochs=int(num_epochs),
            weight_decay=float(weight_decay),
            seed=int(seed),
            device=device,
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            grad_accum_steps=int(grad_accum_steps),
            max_length=int(max_length),
            use_qlora=bool(use_qlora),
            lora_r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            lora_target_modules=str(lora_target_modules),
            bnb_4bit_quant_type=str(bnb_4bit_quant_type),
            bnb_4bit_use_double_quant=bool(bnb_4bit_use_double_quant),
            compute_dtype=str(compute_dtype),
            prompt_sentiment=str(prompt_sentiment),
            prompt_sarcasm=str(prompt_sarcasm),
            prompt_template=str(prompt_template),
            fp16=bool(fp16),
            bf16=bool(bf16),
            tf32=bool(tf32),
            log_every_seconds=float(log_every_seconds),
            checkpoint_every_epochs=int(checkpoint_every_epochs),
            resume_from_checkpoint=bool(resume_from_checkpoint),
            checkpoint_dir=checkpoint_dir,
        )
        print(f"Training complete. Best model saved to {output_dir}")

    elif args.command == "predict":
        checkpoint_dir = require(cfg, "predict", "checkpoint_dir")
        task = require(cfg, "predict", "task")
        input_file = get(cfg, "predict", "input_file", None)
        output_file = get(cfg, "predict", "output_file", None)
        text_list = get(cfg, "predict", "text", []) or []
        batch_size = require(cfg, "predict", "batch_size")
        max_length = require(cfg, "predict", "max_length")
        max_new_tokens = require(cfg, "predict", "max_new_tokens")
        fp16 = require(cfg, "predict", "fp16")
        bf16 = require(cfg, "predict", "bf16")
        use_qlora = require(cfg, "predict", "use_qlora")
        bnb_4bit_quant_type = require(cfg, "predict", "bnb_4bit_quant_type")
        bnb_4bit_use_double_quant = require(cfg, "predict", "bnb_4bit_use_double_quant")
        compute_dtype = require(cfg, "predict", "compute_dtype")
        prompt_sentiment = require(cfg, "predict", "prompt_sentiment")
        prompt_sarcasm = require(cfg, "predict", "prompt_sarcasm")
        prompt_template = require(cfg, "predict", "prompt_template")

        checkpoint_dir = coalesce(args.checkpoint_dir, checkpoint_dir)
        task = coalesce(args.task, task)
        input_file = coalesce(args.input_file, input_file)
        output_file = coalesce(args.output_file, output_file)
        if args.text is not None:
            text_list = args.text
        batch_size = coalesce(args.batch_size, batch_size)
        max_length = coalesce(args.max_length, max_length)
        max_new_tokens = coalesce(args.max_new_tokens, max_new_tokens)
        bnb_4bit_quant_type = coalesce(args.bnb_4bit_quant_type, bnb_4bit_quant_type)
        compute_dtype = coalesce(args.compute_dtype, compute_dtype)
        prompt_sentiment = coalesce(args.prompt_sentiment, prompt_sentiment)
        prompt_sarcasm = coalesce(args.prompt_sarcasm, prompt_sarcasm)
        prompt_template = coalesce(args.prompt_template, prompt_template)

        device = coalesce(args.device, get(cfg, "predict", "device", common_device))

        if args.use_qlora:
            use_qlora = True
        if args.no_qlora:
            use_qlora = False
        if args.bnb_4bit_use_double_quant:
            bnb_4bit_use_double_quant = True
        if args.no_bnb_4bit_use_double_quant:
            bnb_4bit_use_double_quant = False
        if args.fp16:
            fp16 = True
        if args.bf16:
            bf16 = True

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
            preds = predict_decoder(
                checkpoint_dir,
                texts,
                task=task,
                prompt_sentiment=prompt_sentiment,
                prompt_sarcasm=prompt_sarcasm,
                prompt_template=prompt_template,
                device=device,
                batch_size=int(batch_size),
                max_length=int(max_length),
                max_new_tokens=int(max_new_tokens),
                fp16=bool(fp16),
                bf16=bool(bf16),
                use_qlora=bool(use_qlora),
                bnb_4bit_quant_type=str(bnb_4bit_quant_type),
                bnb_4bit_use_double_quant=bool(bnb_4bit_use_double_quant),
                compute_dtype=str(compute_dtype),
            )
            df["prediction"] = preds
            out_path = output_file or os.path.splitext(input_file)[0] + "_predictions.csv"
            df.to_csv(out_path, index=False)
            print(f"Predictions written to {out_path}")
        else:
            if not text_list:
                print("Either predict.input_file or predict.text must be set (config or CLI).", file=sys.stderr)
                sys.exit(1)

            preds = predict_decoder(
                checkpoint_dir,
                text_list,
                task=task,
                prompt_sentiment=prompt_sentiment,
                prompt_sarcasm=prompt_sarcasm,
                prompt_template=prompt_template,
                device=device,
                batch_size=int(batch_size),
                max_length=int(max_length),
                max_new_tokens=int(max_new_tokens),
                fp16=bool(fp16),
                bf16=bool(bf16),
                use_qlora=bool(use_qlora),
                bnb_4bit_quant_type=str(bnb_4bit_quant_type),
                bnb_4bit_use_double_quant=bool(bnb_4bit_use_double_quant),
                compute_dtype=str(compute_dtype),
            )
            for t, p in zip(text_list, preds):
                print(f"{p}\t{t}")


if __name__ == "__main__":
    main()
