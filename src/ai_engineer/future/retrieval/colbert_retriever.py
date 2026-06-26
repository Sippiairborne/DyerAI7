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

"""ColBERT-style late interaction retrieval — token-level embeddings + MaxSim."""
from __future__ import annotations

import numpy as np
import torch

from ai_engineer.core.llm import LLMClient
from ai_engineer.utils.errors import AIEngineerError
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


class ColBERTRetriever:
    """Late interaction retriever: encodes every token, scores with MaxSim."""

    def __init__(self, model_name: str = "colbert-ir/colbertv2.0", dim: int = 128) -> None:
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError as e:
            raise AIEngineerError("Install transformers") from e
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dim = dim
        self.doc_token_embs: list[torch.Tensor] = []
        self.doc_masks: list[torch.Tensor] = []
        self.doc_texts: list[str] = []

    def _encode(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        inp = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            out = self.encoder(**inp)
        # Linear projection to dim (ColBERT uses a projection head; for simplicity use CLS-context)
        emb = out.last_hidden_state[:, :, : self.dim]
        if emb.shape[-1] < self.dim:
            pad = torch.zeros(*emb.shape[:-1], self.dim - emb.shape[-1])
            emb = torch.cat([emb, pad], dim=-1)
        return emb, inp.attention_mask

    def index(self, documents: list[str]) -> None:
        embs, masks = self._encode(documents)
        for i in range(len(documents)):
            self.doc_token_embs.append(embs[i])
            self.doc_masks.append(masks[i])
            self.doc_texts.append(documents[i])

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        q_emb, q_mask = self._encode([query])
        q_emb = q_emb[0][q_mask[0].bool()]
        scores = []
        for i, (d_emb, d_mask) in enumerate(zip(self.doc_token_embs, self.doc_masks)):
            d = d_emb[d_mask.bool()]
            sim = q_emb @ d.T  # [Q, D]
            maxsim = sim.max(dim=1).values.sum().item()
            scores.append((maxsim, i))
        scores.sort(key=lambda x: -x[0])
        return [{"score": s, "text": self.doc_texts[i], "index": i} for s, i in scores[:top_k]]
