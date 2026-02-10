# Mistral-Small-Instruct Baseline (Decoder)

This folder contains the decoder baseline using QLoRA and instruction prompts.

## Train
```
python main.py train --config config.yaml
```

## Predict
```
python main.py predict --config config.yaml --input_file <csv>
```

Notes:
- Uses QLoRA (4-bit) by default; requires `bitsandbytes` and `peft`.
- Logs progress every second by default (config: `train.log_every_seconds`).
- Checkpoints are saved every 2 epochs under `output_dir/checkpoints/`.
- Training auto-resumes from the latest checkpoint unless disabled.
