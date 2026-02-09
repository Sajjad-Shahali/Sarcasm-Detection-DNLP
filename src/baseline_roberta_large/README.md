# RoBERTa-Large Baseline

This folder contains the encoder baseline for RoBERTa-large.

## Train
```
python main.py train --config config.yaml
```

## Predict
```
python main.py predict --config config.yaml --input_file <csv>
```

Notes:
- Logs progress every second by default (config: `train.log_every_seconds`).
- Checkpoints are saved every 2 epochs under `output_dir/checkpoints/`.
- Training auto-resumes from the latest checkpoint unless disabled.
