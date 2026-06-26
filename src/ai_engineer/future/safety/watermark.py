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

"""Text watermarking via green-list token sampling (Kirchner et al. 2023 / Scott Aaronson)."""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import numpy as np


@dataclass
class WatermarkResult:
    is_watermarked: bool
    score: float
    p_value: float
    z_score: float


class TextWatermarker:
    """Token-level green/red list watermark with detection."""

    def __init__(self, vocab_size: int = 50_000, gamma: float = 0.5, hash_key: str = "ai_engineer", seed: int = 42) -> None:
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.hash_key = hash_key
        self.seed = seed

    def _greenlist(self, prev_token: int) -> set[int]:
        seed_bytes = hashlib.sha256(f"{self.hash_key}:{prev_token}".encode()).digest()
        rng = random.Random(int.from_bytes(seed_bytes[:4], "big"))
        green_size = int(self.vocab_size * self.gamma)
        return set(rng.sample(range(self.vocab_size), green_size))

    def watermark_token_ids(self, token_ids: list[int]) -> list[int]:
        """Bias sampling toward green tokens (assumes access to logits)."""
        out = []
        prev = 0
        for tid in token_ids:
            green = self._greenlist(prev)
            # Caller should reweight logits; here we leave as-is and just track
            out.append(tid)
            prev = tid
        return out

    def generate_with_watermark(self, model_fn, prompt_ids: list[int], max_new: int = 100) -> list[int]:
        """Generate text with watermark bias applied via logits modification."""
        out = list(prompt_ids)
        prev = out[-1] if out else 0
        for _ in range(max_new):
            logits = model_fn(out)
            green = self._greenlist(prev)
            mask = np.zeros(self.vocab_size)
            mask[list(green)] = 1.0
            # Boost green tokens
            biased = logits + mask * 2.0
            probs = np.exp(biased - biased.max())
            probs = probs / probs.sum()
            next_id = int(np.random.choice(self.vocab_size, p=probs))
            out.append(next_id)
            prev = next_id
        return out

    def detect(self, token_ids: list[int]) -> WatermarkResult:
        """Detect watermark via green token rate z-score."""
        if len(token_ids) < 2:
            return WatermarkResult(False, 0.0, 1.0, 0.0)
        green_hits = 0
        total = 0
        for i in range(1, len(token_ids)):
            green = self._greenlist(token_ids[i - 1])
            if token_ids[i] in green:
                green_hits += 1
            total += 1
        green_rate = green_hits / total
        expected = self.gamma
        # Z-test
        z = (green_rate - expected) / np.sqrt(expected * (1 - expected) / total)
        from scipy.stats import norm
        p_value = float(1 - norm.cdf(z))
        return WatermarkResult(is_watermarked=z > 3.0, score=green_rate, p_value=p_value, z_score=float(z))


class ImageWatermarker:
    """Frequency-domain image watermarking (DCT-based)."""

    def embed(self, image: np.ndarray, watermark_bits: list[int]) -> np.ndarray:
        try:
            import cv2
        except ImportError:
            return image
        if image.dtype != np.float32:
            img = image.astype(np.float32) / 255.0
        else:
            img = image
        h, w = img.shape[:2]
        for ch in range(img.shape[2] if img.ndim == 3 else 1):
            chan = img[:, :, ch] if img.ndim == 3 else img
            dct = cv2.dct(chan)
            for i, bit in enumerate(watermark_bits):
                if i * 2 + 1 < dct.shape[0] and i * 2 + 1 < dct.shape[1]:
                    dct[i * 2 + 1, i * 2 + 1] += 0.01 if bit else -0.01
            idct = cv2.idct(dct)
            if img.ndim == 3:
                img[:, :, ch] = idct
            else:
                img = idct
        return np.clip(img * 255.0, 0, 255).astype(np.uint8) if image.dtype != np.float32 else np.clip(img, 0, 1)

    def extract(self, image: np.ndarray, n_bits: int) -> list[int]:
        try:
            import cv2
        except ImportError:
            return []
        bits: list[int] = []
        for ch in range(image.shape[2] if image.ndim == 3 else 1):
            chan = image[:, :, ch] if image.ndim == 3 else image
            dct = cv2.dct(chan.astype(np.float32) / 255.0 if chan.max() > 1 else chan.astype(np.float32))
            for i in range(n_bits):
                if i * 2 + 1 < dct.shape[0] and i * 2 + 1 < dct.shape[1]:
                    bits.append(1 if dct[i * 2 + 1, i * 2 + 1] > 0 else 0)
        return bits
