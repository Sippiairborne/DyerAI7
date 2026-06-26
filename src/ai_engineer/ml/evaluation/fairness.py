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

"""Fairness & bias metrics: demographic parity, equalized odds, disparate impact."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai_engineer.utils.errors import AIEngineerError


@dataclass
class FairnessResult:
    demographic_parity_diff: float
    equalized_odds_diff: float
    disparate_impact: float
    per_group_metrics: dict[str, dict[str, float]]
    warnings: list[str]


class FairnessAuditor:
    """Audit predictions for fairness across sensitive groups."""

    def audit(self, y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray, positive_label: int = 1) -> FairnessResult:
        groups = np.unique(sensitive)
        per_group: dict[str, dict[str, float]] = {}
        rates = {}
        tprs = {}
        fprs = {}
        for g in groups:
            mask = sensitive == g
            yt = y_true[mask]
            yp = y_pred[mask]
            p_pred = (yp == positive_label).mean()
            p_true = (yt == positive_label).mean()
            tp = ((yp == positive_label) & (yt == positive_label)).sum()
            fn = ((yp != positive_label) & (yt == positive_label)).sum()
            fp = ((yp == positive_label) & (yt != positive_label)).sum()
            tn = ((yp != positive_label) & (yt != positive_label)).sum()
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            rates[str(g)] = float(p_pred)
            tprs[str(g)] = float(tpr)
            fprs[str(g)] = float(fpr)
            per_group[str(g)] = {
                "n": int(mask.sum()),
                "selection_rate": float(p_pred),
                "positive_rate": float(p_true),
                "tpr": float(tpr),
                "fpr": float(fpr),
                "accuracy": float((yp == yt).mean()),
            }
        dp_diff = float(max(rates.values()) - min(rates.values()))
        eo_diff = float(max(max(tprs.values()) - min(tprs.values()), max(fprs.values()) - min(fprs.values())))
        base = min(rates.values())
        di = float(min(rates.values()) / max(max(rates.values()), 1e-9))
        warnings: list[str] = []
        if di < 0.8:
            warnings.append(f"Disparate impact {di:.3f} < 0.8 (the 80% rule)")
        if dp_diff > 0.1:
            warnings.append(f"Demographic parity diff {dp_diff:.3f} > 0.1")
        if eo_diff > 0.1:
            warnings.append(f"Equalized odds diff {eo_diff:.3f} > 0.1")
        return FairnessResult(
            demographic_parity_diff=dp_diff,
            equalized_odds_diff=eo_diff,
            disparate_impact=di,
            per_group_metrics=per_group,
            warnings=warnings,
        )
