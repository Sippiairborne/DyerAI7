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

"""Counterfactual explanations using DiCE or simple search."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Counterfactual:
    original: np.ndarray
    counterfactual: np.ndarray
    changes: dict[str, float]
    distance: float
    feasible: bool


class CounterfactualExplorer:
    def __init__(self, model, feature_names: list[str] | None = None) -> None:
        self.model = model
        self.feature_names = feature_names or [f"f{i}" for i in range(100)]

    def find(self, instance: np.ndarray, desired_class: int, n_trials: int = 100, step_size: float = 0.1) -> Counterfactual:
        x = instance.copy().astype(np.float64)
        best_x = x.copy()
        best_dist = float("inf")
        rng = np.random.default_rng(42)
        for _ in range(n_trials):
            x_try = x.copy()
            for i in range(len(x_try)):
                if rng.random() < 0.3:
                    x_try[i] += rng.normal(0, step_size)
            try:
                pred = self.model.predict(x_try.reshape(1, -1))[0]
                dist = float(np.linalg.norm(x_try - x))
                if pred == desired_class and dist < best_dist:
                    best_x = x_try
                    best_dist = dist
            except Exception:
                pass
        changes = {self.feature_names[i]: float(best_x[i] - x[i]) for i in range(len(x)) if abs(best_x[i] - x[i]) > 1e-6}
        return Counterfactual(original=x, counterfactual=best_x, changes=changes, distance=best_dist, feasible=best_dist < float("inf"))
