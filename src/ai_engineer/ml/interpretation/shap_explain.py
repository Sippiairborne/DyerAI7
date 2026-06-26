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

"""SHAP explainer wrapper."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ShapResult:
    values: np.ndarray
    base_value: float
    feature_names: list[str]
    expected_value: np.ndarray  # for multi-class


class SHAPExplainer:
    def __init__(self, model, background: np.ndarray | None = None) -> None:
        import shap
        self.model = model
        try:
            if hasattr(model, "predict_proba"):
                self.explainer = shap.Explainer(model.predict_proba, background)
            else:
                self.explainer = shap.Explainer(model, background)
        except Exception:
            self.explainer = None

    def explain(self, X: np.ndarray, feature_names: list[str] | None = None) -> ShapResult | None:
        if self.explainer is None:
            return None
        try:
            sv = self.explainer(X)
            if hasattr(sv, "values"):
                return ShapResult(values=np.array(sv.values), base_value=float(sv.base_values[0]) if hasattr(sv, "base_values") else 0.0, feature_names=feature_names or [], expected_value=np.array(sv.base_values) if hasattr(sv, "base_values") else np.array([0]))
        except Exception:
            return None

    def summary_plot(self, X: np.ndarray, feature_names: list[str] | None = None, output_path: str = "shap_summary.png") -> str:
        try:
            import shap, matplotlib.pyplot as plt
            sv = self.explainer(X)
            shap.summary_plot(sv, X, feature_names=feature_names, show=False)
            import matplotlib.pyplot as plt
            plt.tight_layout()
            plt.savefig(output_path, dpi=120, bbox_inches="tight")
            plt.close()
            return output_path
        except Exception as e:
            return f"plot failed: {e}"
