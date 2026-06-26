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

"""Late interaction reranker (ColBERT-style) using cross-encoder fine-tuning-ready scorer."""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ai_engineer.utils.errors import AIEngineerError


class CrossEncoderReranker:
    """Cross-encoder reranker for top-k refinement."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str | None = None) -> None:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        except Exception as e:
            raise AIEngineerError(f"Failed to load reranker: {e}") from e

    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[dict]:
        if not documents:
            return []
        pairs = [[query, d] for d in documents]
        with torch.no_grad():
            inp = self.tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            scores = self.model(**inp).logits.squeeze(-1).cpu().numpy()
        order = np.argsort(-scores)[:top_k]
        return [{"text": documents[i], "score": float(scores[i]), "index": int(i)} for i in order]
