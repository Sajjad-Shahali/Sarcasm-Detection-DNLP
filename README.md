# Sarcasm Detection NLP (BESSTIE-based)

This repository contains multiple training and inference pipelines for **binary sarcasm/sentiment classification** on BESSTIE-style data (text, label, variety, source, task).  
It includes:

- Encoder baselines (`roberta-large`, multilingual `distilbert`)
- Decoder baseline (`Mistral-Small-Instruct`) with QLoRA
- RoBERTa extensions with custom heads (CNN, attention pooling)
- VAAT (Variety-Aware Adapter Tuning) extension
- Evaluation, plotting, error analysis, and multi-prompt inference scripts

---

## 1) Project goals

- Detect sarcasm (`task=Sarcasm`) and sentiment (`task=Sentiment`)
- Compare robustness across English varieties (`en-AU`, `en-IN`, `en-UK`) and sources (`Google`, `Reddit`)
- Support several model families and decoder/head variants under a unified config-first CLI

---

## 2) Repository layout

```text
.
|- src/
|  |- baseline_roberta_large/      # RoBERTa-large encoder baseline
|  |- baseline_distilbert/         # DistilBERT multilingual baseline
|  |- baseline_mistral/            # Mistral decoder baseline (QLoRA)
|  |- dataset_translated/          # RoBERTa + custom heads (cnn/attn_pool)
|  |- dataset_translated_vaat/     # RoBERTa + custom heads + VAAT
|  |- dataset/                     # main train/valid CSVs
|  |- eval_macro_f1.py             # generic metrics script
|  `- multi_prompt_inference.py    # classifier-level multi-template inference
|- plots/                          # generated charts and summaries
|- output_models/                  # example saved checkpoints
|- requirements.txt
`- requirements-colab.txt
```

---

## 3) Dataset format and task setup

Expected CSV columns:

```csv
text,label,variety,source,task
```

Label semantics:

- `task=Sarcasm`: `1 = sarcastic`, `0 = not sarcastic`
- `task=Sentiment`: `1 = positive`, `0 = negative`

Current dataset stats in `src/dataset`:

- Train rows: `17,760`
- Valid rows: `2,428`
- Train tasks: `Sentiment=8,866`, `Sarcasm=8,894`
- Valid tasks: `Sentiment=1,212`, `Sarcasm=1,216`
- Varieties: `en-AU`, `en-IN`, `en-UK`
- Sources: `Google`, `Reddit`

---

## 4) Environment setup

### Prerequisites

- Python `3.10+` recommended
- `python3` available on PATH
- For GPU runs: CUDA-capable PyTorch install
- For Mistral QLoRA: CUDA + `bitsandbytes` + `peft` (CPU-only is not practical for this path)

### Install

```bash
cd /path/to/sarcasm-detection-nlp
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Optional API serving dependency:

```bash
pip install uvicorn
```

For Colab:

```bash
pip install -r requirements-colab.txt
```

---

## 5) How the training pipeline works

All major model folders use a **config-first** entrypoint (`main.py`):

1. Load and validate `config.yaml` (`--config` can be passed before or after subcommand)
2. Merge CLI overrides over config values
3. Load BESSTIE data (from CSV paths in config, optionally HF dataset loader in some scripts)
4. Filter by `task` (`Sarcasm` or `Sentiment`)
5. Tokenize and build dataloaders
6. Train across one or more learning rates
7. Select best run (typically by macro-F1 on validation)
8. Save final checkpoint
9. Run prediction from `predict` subcommand or `inference.py`

Common training features across folders:

- Mixed precision flags (`fp16`, `bf16`, `tf32`)
- `device` selection (`auto|cuda|cpu`)
- `batch_size`, `max_length`, dataloader workers
- Class weighting for imbalanced labels (encoder pipelines)

Folder-specific additions:

- `baseline_distilbert`, `baseline_mistral`: periodic checkpoints + resume
- `dataset_translated`: custom heads (`hf_default`, `cnn`, `attn_pool`) + early stopping
- `dataset_translated_vaat`: VAAT head (`decoder_type=vaat`) + early stopping + variety conditioning

---

## 6) Quick start (recommended baseline)

Train and predict with RoBERTa-large baseline:

```bash
cd src/baseline_roberta_large

# Important: set train.valid_file to ./dataset/valid-nottranslated.csv in config.yaml
python3 main.py train --config config.yaml

python3 main.py predict --config config.yaml \
  --input_file ./dataset/valid-nottranslated.csv \
  --output_file ./valid_predictions.csv
```

---

## 7) Run each model family

### A) RoBERTa-large encoder baseline

```bash
cd src/baseline_roberta_large
python3 main.py train --config config.yaml
python3 main.py predict --config config.yaml --input_file ./dataset/valid-nottranslated.csv
```

Outputs: `./model_output/` (best model/tokenizer), predictions CSV.

---

### B) DistilBERT multilingual baseline

```bash
cd src/baseline_distilbert
python3 main.py train --config config.yaml
python3 main.py predict --config config.yaml --input_file ../dataset/valid.csv
```

Adds: checkpointing every N epochs and resume support.

---

### C) Mistral decoder baseline (QLoRA)

```bash
cd src/baseline_mistral
python3 main.py train --config config.yaml
python3 main.py predict --config config.yaml --input_file ../dataset/valid.csv
```

Notes:

- Requires CUDA for normal QLoRA workflow (`use_qlora: true`)
- Uses prompt-based label generation ("0"/"1") for classification

---

### D) RoBERTa with custom heads (translated folder)

```bash
cd src/dataset_translated
python3 main.py train --config config.yaml
python3 main.py predict --config config.yaml --input_file ./dataset/valid-new.csv
```

Head options in config:

- `decoder_type: hf_default`
- `decoder_type: cnn`
- `decoder_type: attn_pool`

Custom-head checkpoints include:

- encoder/tokenizer files
- `decoder_config.json`
- `decoder_head.pt`

---

### E) RoBERTa + VAAT extension

```bash
cd src/dataset_translated_vaat
python3 main.py train --config config.yaml
```

Key config fields:

- `decoder_type: vaat`
- `vaat_adapter_dim`
- `vaat_freeze_encoder`

Important caveat:

- `src/dataset_translated_vaat/inference.py` currently loads `AutoModelForSequenceClassification` directly, so it does **not** properly load saved custom VAAT head checkpoints.  
  For VAAT evaluation/inference, use `model_io.py`-based scripts (for example `evaluation.py`) or update inference to use `load_model_and_tokenizer`.

---

## 8) Evaluation and analysis

### Generic macro-F1 evaluator

From repo root:

```bash
python3 src/eval_macro_f1.py \
  --input_file path/to/predictions.csv \
  --label_col label \
  --pred_col prediction \
  --task Sarcasm \
  --task_col task \
  --group_col variety
```

### Multi-prompt classifier inference

```bash
python3 src/multi_prompt_inference.py \
  --checkpoint_dir output_models/1 \
  --input_file src/dataset/valid.csv \
  --output_file src/multi_prompt_predictions.csv \
  --aggregation weighted_mean \
  --task_filter Sarcasm
```

### Grouped evaluation (variety/source) + plots

For translated or VAAT folders:

```bash
cd src/dataset_translated
mkdir -p results
python3 evaluation.py --models_root ./model_output --validation_csv ./dataset/valid-new.csv
python3 plots.py --results_dir ./results --output_prefix tdata_ --plots_dir ../../plots
```

Historical outputs tracked in repo are under uppercase `Results/`; new script outputs default to lowercase `results/`.

---

## 9) API serving (FastAPI)

Each model folder contains `app.py`. Example:

```bash
cd src/dataset_translated
export BESSTIE_CHECKPOINT_DIR=./model_output
export BESSTIE_DEVICE=auto
uvicorn app:app --host 0.0.0.0 --port 8000
```

Request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"texts":["Great, another Monday morning."]}'
```

---

## 10) Common pitfalls and fixes

1. **Wrong working directory**
   Run each pipeline from its own folder (`cd src/<pipeline>`), because config paths are relative.

2. **`python` not found**
   Use `python3` explicitly.

3. **RoBERTa baseline config uses train file as valid file**
   In `src/baseline_roberta_large/config.yaml`, set:
   - `train.train_file: ./dataset/train-nottranslated.csv`
   - `train.valid_file: ./dataset/valid-nottranslated.csv`

4. **Missing `results/` dir for evaluation scripts**
   Create it first: `mkdir -p results`

5. **Mistral QLoRA on CPU**
   Not recommended; use CUDA and keep `use_qlora: true` for intended workflow.

6. **VAAT inference mismatch**
   Custom VAAT checkpoints require `model_io.py` loading path (see caveat in section 7E).

---

## 11) Reproducibility notes

- Seeds are configurable (`train.seed`)
- Best model selection is based on validation metrics (usually macro-F1)
- Several training scripts support checkpoint resume
- `run_config.json` is saved in some pipelines to record effective settings

---

## 12) Included assets

- Research PDF: `original article/2025.findings-acl.441.pdf`
- Example outputs and plots:
  - `plots/`
  - `src/dataset_translated/Results/`
  - `src/dataset_translated_vaat/Results/`
  - `errors_Robertalarge_VAAT.csv`
  - `src/multi_prompt_predictions.csv`
