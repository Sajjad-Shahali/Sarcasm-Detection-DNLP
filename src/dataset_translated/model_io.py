# model_io.py
# Utilities for saving/loading baseline HF models and encoder+custom-head checkpoints.

from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from decoders import DecoderConfig, build_head


class EncoderWithCustomHead(nn.Module):
    """
    Wraps an encoder (AutoModel) + a custom head (CNN/AttnPool) to behave like HF classifiers.
    Returns an object with `.logits`.
    """
    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        x = out.last_hidden_state  # (B, T, H)
        logits = self.head(x, attention_mask=attention_mask)

        class _Out:
            def __init__(self, logits):
                self.logits = logits

        return _Out(logits)


def _canonicalize_decoder_type(t: str) -> str:
    """
    Map newer names into decoders.py canonical names: 'cnn' | 'attn_pool'
    """
    t = (t or "").lower().strip()
    aliases = {
        # attention pooling
        "attn_pool": "attn_pool",
        "attention_pooling": "attn_pool",
        "attention_pool": "attn_pool",
        "attn_pooling": "attn_pool",
        # cnn
        "cnn": "cnn",
        "cnn_over_tokens": "cnn",
        "cnn_tok": "cnn",
        "cnn_tokens": "cnn",
    }
    return aliases.get(t, t)


def _normalize_decoder_cfg(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accept both:
      Old format: {"decoder_type": "...", "dropout": ..., ...}
      New train2 format: {"head_type": "...", "head_cfg": {...}}
    Return a dict compatible with DecoderConfig.from_dict (must include decoder_type).
    """
    # Old / expected
    if "decoder_type" in cfg_dict:
        out = dict(cfg_dict)
        out["decoder_type"] = _canonicalize_decoder_type(str(out["decoder_type"]))
        return out

    # New train2 format
    if "head_type" in cfg_dict:
        head_type = _canonicalize_decoder_type(str(cfg_dict["head_type"]))
        head_cfg = cfg_dict.get("head_cfg", {}) or {}

        dropout = head_cfg.get("dropout", None)
        if dropout is None:
            dropout = head_cfg.get("head_dropout", None)
        if dropout is None:
            dropout = head_cfg.get("decoder_dropout", None)
        if dropout is None:
            dropout = 0.1

        # map CNN out_channels -> cnn_num_filters
        cnn_num_filters = head_cfg.get("cnn_num_filters", None)
        if cnn_num_filters is None:
            cnn_num_filters = head_cfg.get("cnn_out_channels", None)
        if cnn_num_filters is None:
            cnn_num_filters = 128

        out = {
            "decoder_type": head_type,
            "num_labels": int(head_cfg.get("num_labels", 2)),
            "dropout": float(dropout),
            "cnn_num_filters": int(cnn_num_filters),
            "cnn_kernel_sizes": list(head_cfg.get("cnn_kernel_sizes", [2, 3, 4])),
            "attn_mlp_hidden": head_cfg.get("attn_mlp_hidden", None),
        }
        return out

    # If neither exists, let it error clearly
    return cfg_dict


def load_model_and_tokenizer(checkpoint_dir: str, device: str = "auto") -> Tuple[nn.Module, Any]:
    """
    Loads either:
    - Baseline HF classifier (has pytorch_model.bin / model.safetensors for AutoModelForSequenceClassification)
    OR
    - Custom encoder + head checkpoint (encoder in HF format + decoder_head.pt + decoder_config.json)
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Tokenizer always from checkpoint_dir
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)

    # If it looks like a baseline classifier checkpoint, load it directly.
    # (HF classifier checkpoints typically have config.json and model weights; custom ones also do,
    # so we detect custom by the presence of decoder_head.pt)
    custom_head_path = os.path.join(checkpoint_dir, "decoder_head.pt")
    custom_cfg_path = os.path.join(checkpoint_dir, "decoder_config.json")

    if os.path.exists(custom_head_path) and os.path.exists(custom_cfg_path):
        # Custom path
        encoder = AutoModel.from_pretrained(checkpoint_dir)

        with open(custom_cfg_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)

        cfg_dict = _normalize_decoder_cfg(cfg_dict)
        decoder_cfg = DecoderConfig.from_dict(cfg_dict)

        hidden_size = int(getattr(encoder.config, "hidden_size", 768))
        head = build_head(hidden_size, decoder_cfg)

        state = torch.load(custom_head_path, map_location="cpu")
        # --- Backward/compat: rename train2 head keys to decoders.py head keys ---
        if isinstance(state, dict):
            # AttentionPoolingHead + CNN heads from train2 use "classifier.*"
            if "classifier.weight" in state and "out.weight" not in state:
                state["out.weight"] = state.pop("classifier.weight")
            if "classifier.bias" in state and "out.bias" not in state:
                state["out.bias"] = state.pop("classifier.bias")
        head.load_state_dict(state)

        model = EncoderWithCustomHead(encoder=encoder, head=head)
        model.to(dev)
        return model, tokenizer

    # Baseline HF classifier
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.to(dev)
    return model, tokenizer
