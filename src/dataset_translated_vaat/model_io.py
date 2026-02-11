# model_io.py
# Save/load utilities that support:
# 1) Standard HuggingFace checkpoints (AutoModelForSequenceClassification)
# 2) Custom decoder checkpoints (AutoModel encoder + custom head)

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from decoders import DecoderConfig, build_head

DECODER_CONFIG_FILE = "decoder_config.json"
DECODER_WEIGHTS_FILE = "decoder_head.pt"


def _get_hidden_size(encoder_config) -> int:
    # Most encoders: hidden_size; T5/others: d_model
    if hasattr(encoder_config, "hidden_size"):
        return int(getattr(encoder_config, "hidden_size"))
    if hasattr(encoder_config, "d_model"):
        return int(getattr(encoder_config, "d_model"))
    raise ValueError("Cannot determine encoder hidden size from config.")


class EncoderWithCustomHead(nn.Module):
    """
    Encoder (AutoModel) + custom decoder head.
    Returns SequenceClassifierOutput(logits=...) to match HF interface.
    """
    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, input_ids=None, attention_mask=None, **kwargs) -> SequenceClassifierOutput:
        # Custom heads (e.g., VAAT) may need extra inputs like variety_ids.
        variety_ids = kwargs.pop("variety_ids", None)
        if variety_ids is None:
            variety_ids = kwargs.pop("variety_id", None)

        # Only pass encoder-relevant kwargs to the encoder (after popping head-only args).
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        last_hidden = out.last_hidden_state  # (B, T, H)

        head_kwargs = {"attention_mask": attention_mask}
        if variety_ids is not None:
            head_kwargs["variety_ids"] = variety_ids

        logits = self.head(last_hidden, **head_kwargs)
        return SequenceClassifierOutput(logits=logits)


def is_custom_decoder_checkpoint(checkpoint_dir: str) -> bool:
    return os.path.exists(os.path.join(checkpoint_dir, DECODER_CONFIG_FILE)) and os.path.exists(
        os.path.join(checkpoint_dir, DECODER_WEIGHTS_FILE)
    )


def save_custom_decoder_checkpoint(
    output_dir: str,
    encoder: nn.Module,
    tokenizer,
    decoder_cfg: Dict[str, Any],
    head: nn.Module,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Save encoder + tokenizer like a normal HF checkpoint
    encoder.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save head weights + config
    torch.save(head.state_dict(), os.path.join(output_dir, DECODER_WEIGHTS_FILE))

    payload = dict(decoder_cfg)
    if extra_metadata:
        payload["metadata"] = extra_metadata

    with open(os.path.join(output_dir, DECODER_CONFIG_FILE), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_model_and_tokenizer(
    checkpoint_dir: str,
    device: str = "auto",
) -> Tuple[nn.Module, Any]:
    """
    Loads either:
      - HF classification model (if no custom decoder files exist), or
      - Encoder+custom head (if decoder_config.json + decoder_head.pt exist)
    Returns: (model, tokenizer)
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    # ----------------------------
    # 1) HF default checkpoint
    # ----------------------------
    if not is_custom_decoder_checkpoint(checkpoint_dir):
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        model.to(device_t)
        model.eval()
        return model, tokenizer

    # ----------------------------
    # 2) Custom decoder checkpoint
    # ----------------------------
    with open(os.path.join(checkpoint_dir, DECODER_CONFIG_FILE), "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    decoder_cfg = DecoderConfig.from_dict(cfg_dict)

    encoder = AutoModel.from_pretrained(checkpoint_dir)
    hidden_size = _get_hidden_size(encoder.config)

    head = build_head(hidden_size=hidden_size, cfg=decoder_cfg)
    head.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, DECODER_WEIGHTS_FILE), map_location="cpu")
    )

    model = EncoderWithCustomHead(encoder=encoder, head=head)

    # ✅ attach mapping for VAAT (so evaluation.py can build variety_ids)
    if getattr(decoder_cfg, "decoder_type", None) == "vaat" and getattr(decoder_cfg, "vaat_varieties", None):
        model.variety_to_id = {v: i for i, v in enumerate(decoder_cfg.vaat_varieties)}
        model.id_to_variety = list(decoder_cfg.vaat_varieties)

    model.to(device_t)
    model.eval()
    return model, tokenizer

