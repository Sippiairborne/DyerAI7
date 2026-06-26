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

"""Calibration analysis: ECE, MCE, Brier, reliability diagrams, temperature scaling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class CalibrationResult:
    ece: float
    mce: float
    brier: float
    nll: float
    optimal_temperature: float
    raw_reliability: list[list[float]]  # [bin_centers, accuracies, counts]
    calibrated_reliability: list[list[float]]


class CalibrationAnalyzer:
    def analyze(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> CalibrationResult:
        ece, mce, raw_rel = self._expected_calibration(y_true, y_prob, n_bins)
        brier = float(np.mean((y_prob - np.eye(y_prob.shape[1])[y_true]) ** 2))
        nll = float(-np.log(np.clip(y_prob[np.arange(len(y_true)), y_true], 1e-12, 1)).mean())
        T = self._find_temperature(y_true, y_prob)
        y_prob_cal = self._apply_temperature(y_prob, T)
        ece_c, mce_c, cal_rel = self._expected_calibration(y_true, y_prob_cal, n_bins)
        return CalibrationResult(
            ece=ece, mce=mce, brier=brier, nll=nll, optimal_temperature=T,
            raw_reliability=raw_rel, calibrated_reliability=cal_rel,
        )

    def _expected_calibration(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> tuple[float, float, list[list[float]]]:
        confidence = y_prob.max(axis=1)
        predictions = y_prob.argmax(axis=1)
        accuracies = (predictions == y_true).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers, bin_acc, bin_conf, bin_count = [], [], [], []
        ece = 0.0
        mce = 0.0
        n = len(y_true)
        for i in range(n_bins):
            mask = (confidence >= bins[i]) & (confidence < bins[i + 1])
            if mask.sum() == 0:
                continue
            acc = accuracies[mask].mean()
            conf = confidence[mask].mean()
            count = int(mask.sum())
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            bin_acc.append(acc)
            bin_conf.append(conf)
            bin_count.append(count)
            ece += (count / n) * abs(acc - conf)
            mce = max(mce, abs(acc - conf))
        return ece, mce, [bin_centers, bin_acc, bin_conf, bin_count]

    def _find_temperature(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        logit = np.log(np.clip(y_prob, 1e-12, 1.0))
        y = np.eye(y_prob.shape[1])[y_true]
        t = torch.nn.Parameter(torch.tensor(1.0))
        opt = torch.optim.LBFGS([t], lr=0.01, max_iter=50)

        def closure():
            opt.zero_grad()
            scale = t.abs()
            loss = F.cross_entropy(torch.tensor(logit) / scale, torch.tensor(y))
            loss.backward()
            return loss

        try:
            opt.step(closure)
        except Exception:
            return 1.0
        return float(t.abs().item())

    def _apply_temperature(self, y_prob: np.ndarray, T: float) -> np.ndarray:
        if T == 1.0:
            return y_prob
        logit = np.log(np.clip(y_prob, 1e-12, 1.0))
        scaled = logit / max(T, 1e-6)
        scaled -= scaled.max(axis=1, keepdims=True)
        exp = np.exp(scaled)
        return exp / exp.sum(axis=1, keepdims=True)

    def save_reliability_plot(self, calibration: CalibrationResult, path: str | Path) -> None:
        import matplotlib.pyplot as plt

        centers, acc, conf, counts = calibration.raw_reliability
        centers_c, acc_c, conf_c, _ = calibration.calibrated_reliability
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot([0, 1], [0, 1], "k:")
        ax[0].bar(centers, acc, width=1 / len(centers), alpha=0.5, label="acc")
        ax[0].plot(centers, conf, "r-", label="conf")
        ax[0].set_title(f"Raw (ECE={calibration.ece:.3f})")
        ax[0].legend()
        ax[1].plot([0, 1], [0, 1], "k:")
        ax[1].bar(centers_c, acc_c, width=1 / len(centers_c), alpha=0.5)
        ax[1].plot(centers_c, conf_c, "r-")
        ax[1].set_title(f"Calibrated (T={calibration.optimal_temperature:.2f})")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
