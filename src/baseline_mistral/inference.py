"""
Inference script for decoder baselines (QLoRA) on BESSTIE.
"""

import argparse
import os
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from utils import build_prompt_text


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


def load_model_and_tokenizer(
    checkpoint_dir: str,
    device: str = "auto",
    use_qlora: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
    compute_dtype: str = "bf16",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    adapter_path = os.path.join(checkpoint_dir, "adapter_config.json")
    if os.path.exists(adapter_path):
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(checkpoint_dir)
        base_model_name = peft_config.base_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        if use_qlora:
            quant_dtype = _get_quant_dtype(compute_dtype)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=quant_dtype,
            )
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(base_model_name)
            model.to(device)

        model = PeftModel.from_pretrained(model, checkpoint_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
        model.to(device)

    model.eval()
    return model, tokenizer, device


def predict_decoder(
    checkpoint_dir: str,
    texts: List[str],
    task: str,
    prompt_sentiment: str,
    prompt_sarcasm: str,
    prompt_template: str,
    device: str = "auto",
    batch_size: int = 4,
    max_length: int = 256,
    max_new_tokens: int = 1,
    fp16: bool = False,
    bf16: bool = False,
    use_qlora: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
    compute_dtype: str = "bf16",
) -> List[int]:
    model, tokenizer, device = load_model_and_tokenizer(
        checkpoint_dir,
        device=device,
        use_qlora=use_qlora,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        compute_dtype=compute_dtype,
    )
    prompts = [
        build_prompt_text(task, t, prompt_sentiment, prompt_sarcasm, prompt_template)
        for t in texts
    ]

    preds: List[int] = []
    autocast_dtype = None
    if device.type == "cuda" and fp16:
        autocast_dtype = torch.float16
    elif device.type == "cuda" and bf16:
        autocast_dtype = torch.bfloat16
    use_autocast = autocast_dtype is not None

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device, non_blocking=device.type == "cuda") for k, v in inputs.items()}
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                )
        prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()
        for row, prompt_len in zip(outputs, prompt_lens):
            gen_ids = row[int(prompt_len):]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            preds.append(_parse_label(text))
    return preds


def main():
    parser = argparse.ArgumentParser(description="Run inference with a decoder baseline (QLoRA).")
    parser.add_argument("--checkpoint_dir", type=str, default="./model_output/")
    parser.add_argument("--task", type=str, choices=["Sentiment", "Sarcasm"], default="Sarcasm")
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--text", type=str, nargs="*", default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true")
    parser.add_argument("--compute_dtype", type=str, default="bf16")
    parser.add_argument("--prompt_sentiment", type=str, required=True)
    parser.add_argument("--prompt_sarcasm", type=str, required=True)
    parser.add_argument("--prompt_template", type=str, default="{prompt}\n{text}\n")
    args = parser.parse_args()

    if args.input_file:
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(args.input_file)
        import pandas as pd
        df = pd.read_csv(args.input_file)
        if "text" not in df.columns:
            raise ValueError("Input CSV must contain a 'text' column")
        texts = df["text"].astype(str).tolist()
        preds = predict_decoder(
            checkpoint_dir=args.checkpoint_dir,
            texts=texts,
            task=args.task,
            prompt_sentiment=args.prompt_sentiment,
            prompt_sarcasm=args.prompt_sarcasm,
            prompt_template=args.prompt_template,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            fp16=args.fp16,
            bf16=args.bf16,
            use_qlora=args.use_qlora,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            compute_dtype=args.compute_dtype,
        )
        df["prediction"] = preds
        out_path = args.output_file or os.path.splitext(args.input_file)[0] + "_predictions.csv"
        df.to_csv(out_path, index=False)
        print(f"Predictions written to {out_path}")
    else:
        if not args.text:
            raise ValueError("Either --input_file or --text must be provided")
        preds = predict_decoder(
            checkpoint_dir=args.checkpoint_dir,
            texts=args.text,
            task=args.task,
            prompt_sentiment=args.prompt_sentiment,
            prompt_sarcasm=args.prompt_sarcasm,
            prompt_template=args.prompt_template,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            fp16=args.fp16,
            bf16=args.bf16,
            use_qlora=args.use_qlora,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            compute_dtype=args.compute_dtype,
        )
        for t, p in zip(args.text, preds):
            print(f"{p}\t{t}")


if __name__ == "__main__":
    main()
