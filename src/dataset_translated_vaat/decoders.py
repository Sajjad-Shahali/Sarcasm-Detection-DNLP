# decoders.py
# Custom "decoder heads" (classification heads) for BESSTIE.
# These heads sit on top of a Transformer encoder's token embeddings.

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

# Local lightweight adapters (VAAT)
from adapters import Adapter
import torch.nn.functional as F


def _mask_tokens(x: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    x: (B, T, H)
    attention_mask: (B, T) with 1 for real tokens, 0 for padding
    """
    if attention_mask is None:
        return x
    mask = attention_mask.unsqueeze(-1).to(dtype=x.dtype)  # (B, T, 1)
    return x * mask


@dataclass
class DecoderConfig:
    decoder_type: str  # "cnn" | "attn_pool"
    num_labels: int = 2

    # Common
    dropout: float = 0.1

    # CNN
    cnn_num_filters: int = 128
    cnn_kernel_sizes: List[int] = None  # type: ignore

    # Attention pooling
    attn_mlp_hidden: Optional[int] = None  # if None => single linear scorer

    # Variety-Aware Adapter Tuning (VAAT)
    vaat_adapter_dim: int = 64
    vaat_varieties: Optional[list[str]] = None  # e.g., ["en-AU","en-IN","en-UK"]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d.get("cnn_kernel_sizes") is None:
            d["cnn_kernel_sizes"] = [2, 3, 4]
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DecoderConfig":
        cfg = DecoderConfig(
            decoder_type=str(d["decoder_type"]),
            num_labels=int(d.get("num_labels", 2)),
            dropout=float(d.get("dropout", 0.1)),
            cnn_num_filters=int(d.get("cnn_num_filters", 128)),
            cnn_kernel_sizes=list(d.get("cnn_kernel_sizes", [2, 3, 4])),
            attn_mlp_hidden=d.get("attn_mlp_hidden", None),
            vaat_adapter_dim=int(d.get("vaat_adapter_dim", 64)),
            vaat_varieties=d.get("vaat_varieties", None),
        )
        return cfg


class CNNOverTokensHead(nn.Module):
    """
    TextCNN-style head on top of token embeddings.
    - Apply multiple Conv1d filters over time
    - Max pool over time
    - Concatenate and classify
    """
    def __init__(
        self,
        hidden_size: int,
        num_labels: int = 2,
        num_filters: int = 128,
        kernel_sizes: Sequence[int] = (2, 3, 4),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_labels = int(num_labels)
        self.num_filters = int(num_filters)
        self.kernel_sizes = list(map(int, kernel_sizes))
        self.dropout = float(dropout)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.hidden_size, out_channels=self.num_filters, kernel_size=k)
            for k in self.kernel_sizes
        ])
        self.drop = nn.Dropout(self.dropout)
        self.out = nn.Linear(self.num_filters * len(self.kernel_sizes), self.num_labels)

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        last_hidden_state: (B, T, H)
        attention_mask: (B, T)
        returns logits: (B, num_labels)
        """
        x = _mask_tokens(last_hidden_state, attention_mask)  # zero-out padding tokens
        x = x.transpose(1, 2)  # (B, H, T)

        pooled = []
        for conv in self.convs:
            # conv -> (B, F, T-k+1)
            h = F.relu(conv(x))
            # global max pool over time -> (B, F)
            h = F.max_pool1d(h, kernel_size=h.size(-1)).squeeze(-1)
            pooled.append(h)

        h_cat = torch.cat(pooled, dim=1)  # (B, F * K)
        h_cat = self.drop(h_cat)
        logits = self.out(h_cat)
        return logits


class AttentionPoolingHead(nn.Module):
    """
    Learn attention weights over tokens, then weighted-sum and classify.
    This often beats CLS-only heads for sarcasm because cues can be spread across tokens.
    """
    def __init__(
        self,
        hidden_size: int,
        num_labels: int = 2,
        dropout: float = 0.1,
        attn_mlp_hidden: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_labels = int(num_labels)
        self.dropout = float(dropout)
        self.attn_mlp_hidden = attn_mlp_hidden if attn_mlp_hidden is None else int(attn_mlp_hidden)

        if self.attn_mlp_hidden is None:
            self.scorer = nn.Linear(self.hidden_size, 1)
        else:
            self.scorer = nn.Sequential(
                nn.Linear(self.hidden_size, self.attn_mlp_hidden),
                nn.Tanh(),
                nn.Linear(self.attn_mlp_hidden, 1),
            )

        self.drop = nn.Dropout(self.dropout)
        self.out = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        last_hidden_state: (B, T, H)
        attention_mask: (B, T) 1=real token, 0=pad
        returns logits: (B, num_labels)
        """
        # scores: (B, T)
        scores = self.scorer(last_hidden_state).squeeze(-1)

        if attention_mask is not None:
            # mask pads by setting scores to a very negative number
            scores = scores.masked_fill(attention_mask == 0, -1e9)

        # stable softmax
        attn = torch.softmax(scores, dim=-1)  # (B, T)

        # weighted sum
        pooled = torch.bmm(attn.unsqueeze(1), last_hidden_state).squeeze(1)  # (B, H)

        pooled = self.drop(pooled)
        logits = self.out(pooled)
        return logits



class VarietyAdapterHead(nn.Module):
    """Variety-aware adapter head.

    - Applies a small bottleneck Adapter to the [CLS] embedding,
      conditioned on variety_id (one adapter per variety).
    - Then runs a standard linear classifier.

    Inputs:
        last_hidden: (B, T, H)
        attention_mask: (B, T) [unused here, kept for API consistency]
        variety_ids: (B,) LongTensor with values in [0, num_varieties-1]
    """

    def __init__(self, hidden_size: int, num_labels: int, adapter_dim: int, varieties: list[str], dropout: float = 0.1):
        super().__init__()
        if not varieties:
            raise ValueError("VAAT requires a non-empty list of varieties in cfg.vaat_varieties.")
        self.varieties = list(varieties)
        self.num_varieties = len(self.varieties)

        self.adapters = nn.ModuleList([
            Adapter(input_dim=hidden_size, bottleneck_dim=adapter_dim) for _ in range(self.num_varieties)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, last_hidden: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, variety_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # CLS pool
        x = last_hidden[:, 0, :]  # (B, H)
        if variety_ids is None:
            x = self.dropout(x)
            return self.classifier(x)

        if variety_ids.dim() != 1 or variety_ids.shape[0] != x.shape[0]:
            raise ValueError(f"variety_ids must be shape (B,), got {tuple(variety_ids.shape)} for batch size {x.shape[0]}")

        variety_ids = variety_ids.to(dtype=torch.long, device=x.device)

        out = torch.empty_like(x)
        # group by variety id
        for vid in torch.unique(variety_ids).tolist():
            if vid < 0 or vid >= self.num_varieties:
                raise ValueError(f"Unknown variety_id={vid}. Expected in [0, {self.num_varieties-1}].")
            mask = (variety_ids == vid)
            out[mask] = self.adapters[vid](x[mask])

        out = self.dropout(out)
        return self.classifier(out)


def build_head(hidden_size: int, cfg: DecoderConfig) -> nn.Module:
    cfg = DecoderConfig.from_dict(cfg.to_dict())
    if cfg.decoder_type == "vaat":
        if cfg.vaat_varieties is None:
            raise ValueError("decoder_type=vaat requires cfg.vaat_varieties (list of variety strings).")
        return VarietyAdapterHead(
            hidden_size=int(hidden_size),
            num_labels=int(cfg.num_labels),
            adapter_dim=int(cfg.vaat_adapter_dim),
            varieties=list(cfg.vaat_varieties),
            dropout=float(cfg.dropout),
        )

    if cfg.decoder_type == "cnn":
        return CNNOverTokensHead(
            hidden_size=hidden_size,
            num_labels=cfg.num_labels,
            num_filters=cfg.cnn_num_filters,
            kernel_sizes=cfg.cnn_kernel_sizes,
            dropout=cfg.dropout,
        )
    if cfg.decoder_type == "attn_pool":
        return AttentionPoolingHead(
            hidden_size=hidden_size,
            num_labels=cfg.num_labels,
            dropout=cfg.dropout,
            attn_mlp_hidden=cfg.attn_mlp_hidden,
        )
    raise ValueError(f"Unknown decoder_type: {cfg.decoder_type!r}. Use 'cnn' or 'attn_pool'.")
