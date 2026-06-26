# Copyright 2026 Matt Dyer / Dyer-Tech
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Attention visualization for transformer models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AttentionResult:
    layers: int
    heads: int
    seq_len: int
    weights: list[np.ndarray]  # [layers, heads, seq, seq]


class AttentionVisualizer:
    def capture(self, model, tokenizer, text: str) -> AttentionResult | None:
        try:
            import torch
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model(**inputs, output_attentions=True)
            atts = outputs.attentions  # tuple of [1, heads, seq, seq]
            weights = [a[0].cpu().detach().numpy() for a in atts]
            n_layers = len(weights)
            n_heads = weights[0].shape[0]
            seq_len = weights[0].shape[-1]
            return AttentionResult(layers=n_layers, heads=n_heads, seq_len=seq_len, weights=weights)
        except Exception:
            return None

    def save_heatmap(self, att: AttentionResult, layer: int = -1, head: int = 0, tokens: list[str] | None = None, path: str = "attn.png") -> str:
        try:
            import matplotlib.pyplot as plt
            w = att.weights[layer][head]
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(w, cmap="viridis")
            if tokens:
                ax.set_xticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=90)
                ax.set_yticks(range(len(tokens)))
                ax.set_yticklabels(tokens)
            ax.set_title(f"Layer {layer} Head {head}")
            fig.tight_layout()
            fig.savefig(path, dpi=120)
            plt.close(fig)
            return path
        except Exception as e:
            return f"failed: {e}"
