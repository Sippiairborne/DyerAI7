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

"""LIME explainer."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LimeResult:
    features: list[tuple[str, float]]
    intercept: float
    score: float
    local_pred: float


class LIMEExplainer:
    def __init__(self, training_data: np.ndarray, feature_names: list[str] | None = None, class_names: list[str] | None = None, mode: str = "classification") -> None:
        try:
            from lime.lime_tabular import LimeTabularExplainer
            self.exp = LimeTabularExplainer(training_data, feature_names=feature_names, class_names=class_names, mode=mode)
        except ImportError:
            self.exp = None

    def explain(self, instance: np.ndarray, predict_fn, num_features: int = 10) -> LimeResult | None:
        if self.exp is None:
            return None
        try:
            e = self.exp.explain_instance(instance, predict_fn, num_features=num_features)
            return LimeResult(features=[(f, w) for f, w in e.as_list()], intercept=float(e.intercept[1]) if hasattr(e, "intercept") else 0.0, score=float(e.score), local_pred=float(e.local_pred[0]) if e.local_pred is not None else 0.0)
        except Exception:
            return None
