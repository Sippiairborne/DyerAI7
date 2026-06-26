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

"""Data and concept drift detection: KS, PSI, MMD, classifier-based."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai_engineer.utils.errors import AIEngineerError
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)

Method = Literal := __import__("typing").Literal["ks", "psi", "mmd", "classifier"]  # type: ignore


@dataclass
class DriftResult:
    method: str
    feature_scores: dict[str, float]
    drift_detected: bool
    threshold: float
    p_value_or_statistic: dict[str, float]


class DriftDetector:
    def __init__(self, method: str = "ks", threshold: float = 0.05) -> None:
        self.method = method
        self.threshold = threshold

    def detect(self, reference: np.ndarray, current: np.ndarray, feature_names: list[str] | None = None) -> DriftResult:
        if self.method == "ks":
            return self._ks(reference, current, feature_names)
        if self.method == "psi":
            return self._psi(reference, current, feature_names)
        if self.method == "mmd":
            return self._mmd(reference, current, feature_names)
        if self.method == "classifier":
            return self._classifier(reference, current, feature_names)
        raise AIEngineerError(f"Unknown method: {self.method}")

    def _ks(self, ref: np.ndarray, cur: np.ndarray, names: list[str] | None) -> DriftResult:
        from scipy.stats import ks_2samp
        scores: dict[str, float] = {}
        p_values: dict[str, float] = {}
        for i in range(ref.shape[1]):
            stat, p = ks_2samp(ref[:, i], cur[:, i])
            scores[names[i] if names else f"f{i}"] = float(stat)
            p_values[names[i] if names else f"f{i}"] = float(p)
        drift = any(p < self.threshold for p in p_values.values())
        return DriftResult("ks", scores, drift, self.threshold, p_values)

    def _psi(self, ref: np.ndarray, cur: np.ndarray, names: list[str] | None) -> DriftResult:
        scores: dict[str, float] = {}
        for i in range(ref.shape[1]):
            ref_col = ref[:, i]
            cur_col = cur[:, i]
            try:
                bins = np.percentile(ref_col, np.linspace(0, 100, 11))
                bins = np.unique(bins)
                ref_counts, _ = np.histogram(ref_col, bins=bins)
                cur_counts, _ = np.histogram(cur_col, bins=bins)
                ref_pct = (ref_counts + 1) / (ref_counts.sum() + len(bins))
                cur_pct = (cur_counts + 1) / (cur_counts.sum() + len(bins))
                psi = float(((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)).sum())
            except Exception:
                psi = 0.0
            scores[names[i] if names else f"f{i}"] = psi
        drift = any(s > 0.2 for s in scores.values())
        return DriftResult("psi", scores, drift, 0.2, scores)

    def _mmd(self, ref: np.ndarray, cur: np.ndarray, names: list[str] | None) -> DriftResult:
        try:
            from torchtwoist import MMD
        except ImportError:
            try:
                from squarify import MMD
            except ImportError:
                # Simple linear MMD fallback
                return self._mmd_simple(ref, cur, names)
        scores: dict[str, float] = {}
        for i in range(ref.shape[1]):
            x = ref[:, i:i + 1]
            y = cur[:, i:i + 1]
            try:
                mmd = MMD(x, y)
                scores[names[i] if names else f"f{i}"] = float(mmd.compute())
            except Exception:
                scores[names[i] if names else f"f{i}"] = 0.0
        drift = any(s > self.threshold for s in scores.values())
        return DriftResult("mmd", scores, drift, self.threshold, scores)

    def _mmd_simple(self, ref: np.ndarray, cur: np.ndarray, names: list[str] | None) -> DriftResult:
        scores: dict[str, float] = {}
        for i in range(ref.shape[1]):
            x = ref[:, i]
            y = cur[:, i]
            mu_x, mu_y = x.mean(), y.mean()
            var_x, var_y = x.var() + 1e-8, y.var() + 1e-8
            mmd = float((mu_x - mu_y) ** 2 + var_x + var_y - 2 * (var_x * var_y) ** 0.5)
            scores[names[i] if names else f"f{i}"] = mmd
        drift = any(s > self.threshold for s in scores.values())
        return DriftResult("mmd", scores, drift, self.threshold, scores)

    def _classifier(self, ref: np.ndarray, cur: np.ndarray, names: list[str] | None) -> DriftResult:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        X = np.vstack([ref, cur])
        y = np.array([0] * len(ref) + [1] * len(cur))
        try:
            scores = cross_val_score(RandomForestClassifier(n_estimators=50), X, y, cv=3, scoring="roc_auc")
            score = float(scores.mean())
        except Exception:
            score = 0.5
        drift = score > 0.7
        return DriftResult("classifier", {"_auc": score}, drift, 0.7, {"_auc": score})
