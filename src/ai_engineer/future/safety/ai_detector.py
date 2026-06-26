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

"""AI-text detection using statistical and perplexity-based methods."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DetectionResult:
    is_ai: bool
    confidence: float
    perplexity: float
    burstiness: float
    entropy: float


class AITextDetector:
    """Multi-signal AI text detector: perplexity, burstiness, entropy, n-gram patterns."""

    def __init__(self) -> None:
        try:
            import torch
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            self.tok = GPT2Tokenizer.from_pretrained("gpt2")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")
            self.model.eval()
            self._has_lm = True
        except Exception:
            self._has_lm = False

    def detect(self, text: str) -> DetectionResult:
        ppl = self._perplexity(text)
        burst = self._burstiness(text)
        ent = self._entropy(text)
        ngram_ai_score = self._ngram_ai_score(text)
        # Weighted ensemble
        score = 0.4 * (1 if ppl < 50 else 0) + 0.3 * (1 if burst < 0.4 else 0) + 0.2 * (1 if ent < 5.5 else 0) + 0.1 * ngram_ai_score
        is_ai = score > 0.5
        return DetectionResult(is_ai=is_ai, confidence=float(score), perplexity=float(ppl), burstiness=float(burst), entropy=float(ent))

    def _perplexity(self, text: str) -> float:
        if not self._has_lm or len(text) < 10:
            return 100.0
        import torch
        ids = self.tok(text, return_tensors="pt", truncation=True, max_length=1024).input_ids
        with torch.no_grad():
            out = self.model(ids, labels=ids)
        return float(torch.exp(out.loss))

    def _burstiness(self, text: str) -> float:
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 5]
        if len(sentences) < 3:
            return 0.5
        lengths = [len(s.split()) for s in sentences]
        if not lengths:
            return 0.5
        return float(np.std(lengths) / (np.mean(lengths) + 1e-9))

    def _entropy(self, text: str) -> float:
        from collections import Counter
        words = text.split()
        if not words:
            return 0.0
        counts = Counter(words)
        probs = np.array(list(counts.values())) / len(words)
        return float(-np.sum(probs * np.log(probs + 1e-9)))

    def _ngram_ai_score(self, text: str) -> float:
        # Bigram repetition pattern (AI tends to have lower bigram diversity)
        words = text.split()
        if len(words) < 10:
            return 0.5
        bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
        unique = len(set(bigrams))
        return float(1.0 - unique / max(len(bigrams), 1))
