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

"""Statistical tests: McNemar, paired t-test, Wilcoxon, bootstrap CIs, effect size."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai_engineer.utils.errors import AIEngineerError


@dataclass
class StatisticalTestResult:
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float
    ci_lower: float
    ci_upper: float
    conclusion: str


class StatisticalTester:
    def mcnemar(self, y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, alpha: float = 0.05) -> StatisticalTestResult:
        """Compare two classifiers."""
        a_correct = pred_a == y_true
        b_correct = pred_b == y_true
        b = int(((~a_correct) & b_correct).sum())
        c = int((a_correct & (~b_correct)).sum())
        if b + c == 0:
            return StatisticalTestResult("mcnemar", 0.0, 1.0, False, 0.0, 0.0, 0.0, "Both models identical")
        from statsmodels.stats.contingency_tables import mcnemar
        chi2, p = mcnemar([[0, b], [c, 0]], exact=False, correction=True)
        return StatisticalTestResult("mcnemar", float(chi2), float(p), float(p) < alpha, float((b - c) / (b + c)), 0.0, 0.0, f"p={p:.4f}")

    def paired_ttest(self, scores_a: np.ndarray, scores_b: np.ndarray, alpha: float = 0.05) -> StatisticalTestResult:
        from scipy.stats import ttest_rel
        t, p = ttest_rel(scores_a, scores_b)
        d = float((scores_a - scores_b).mean() / max((scores_a - scores_b).std(), 1e-8))
        return StatisticalTestResult("paired_ttest", float(t), float(p), float(p) < alpha, d, 0.0, 0.0, f"p={p:.4f}, d={d:.2f}")

    def wilcoxon(self, scores_a: np.ndarray, scores_b: np.ndarray, alpha: float = 0.05) -> StatisticalTestResult:
        from scipy.stats import wilcoxon
        try:
            w, p = wilcoxon(scores_a, scores_b)
        except ValueError:
            return StatisticalTestResult("wilcoxon", 0.0, 1.0, False, 0.0, 0.0, 0.0, "No variance")
        return StatisticalTestResult("wilcoxon", float(w), float(p), float(p) < alpha, 0.0, 0.0, 0.0, f"p={p:.4f}")

    def bootstrap_ci(self, y_true: np.ndarray, y_pred: np.ndarray, metric_fn, n_bootstraps: int = 1000, alpha: float = 0.05) -> StatisticalTestResult:
        scores = []
        rng = np.random.default_rng(42)
        n = len(y_true)
        for _ in range(n_bootstraps):
            idx = rng.integers(0, n, n)
            try:
                scores.append(metric_fn(y_true[idx], y_pred[idx]))
            except Exception:
                continue
        if not scores:
            return StatisticalTestResult("bootstrap", 0.0, 1.0, False, 0.0, 0.0, 0.0, "All bootstraps failed")
        scores = np.array(scores)
        lo, hi = np.percentile(scores, [alpha / 2 * 100, (1 - alpha / 2) * 100])
        return StatisticalTestResult(
            "bootstrap",
            float(scores.mean()),
            0.0,
            False,
            float(scores.std()),
            float(lo),
            float(hi),
            f"mean={scores.mean():.4f}, 95% CI=[{lo:.4f}, {hi:.4f}]",
        )

    def cohens_d(self, a: np.ndarray, b: np.ndarray) -> float:
        return float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2))
