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

"""Robustness evaluation: noise, corruption, perturbation tests."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RobustnessResult:
    base_accuracy: float
    noise_accuracy: dict[str, float]
    corruption_accuracy: dict[str, float]
    certified_radius: float | None


class RobustnessTester:
    def test_noise(self, predict_fn, X: np.ndarray, y: np.ndarray, noise_levels: list[float] | None = None) -> dict[str, float]:
        if noise_levels is None:
            noise_levels = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
        base = (predict_fn(X) == y).mean()
        results: dict[str, float] = {"clean": float(base)}
        rng = np.random.default_rng(42)
        for s in noise_levels:
            Xn = X + rng.normal(0, s, X.shape).astype(X.dtype)
            Xn = np.clip(Xn, X.min(), X.max())
            results[f"gaussian_{s}"] = float((predict_fn(Xn) == y).mean())
        return results

    def test_image_corruptions(self, predict_fn, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        results: dict[str, float] = {}
        for kind in ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"]:
            try:
                from imagecorruptions import corrupt
                rng = np.random.default_rng(hash(kind) % (2**32))
                severity = 3
                Xc = np.stack([corrupt(x.transpose(1, 2, 0) if x.ndim == 3 else x, corruption_name=kind, severity=severity, seed=int(rng.integers(0, 10000))) for x in X])
                if Xc.shape[1:] != X.shape[1:]:
                    Xc = Xc.transpose(0, -1, 1, 2) if Xc.ndim == 4 else Xc
                results[kind] = float((predict_fn(Xc) == y).mean())
            except Exception as e:
                results[kind] = -1.0
        return results

    def certify(self, predict_fn, X: np.ndarray, y: np.ndarray, sigma: float = 0.1, n: int = 1000, alpha: float = 0.001) -> float:
        try:
            from scipy.stats import norm
            rng = np.random.default_rng(42)
            counts = []
            for _ in range(n):
                Xn = X + rng.normal(0, sigma, X.shape).astype(X.dtype)
                counts.append(int((predict_fn(Xn) == y).sum()))
            p = np.mean([c / len(X) for c in counts])
            if p <= alpha:
                return 0.0
            return float(sigma * (norm.ppf(p) - norm.ppf(alpha)))
        except Exception:
            return 0.0
