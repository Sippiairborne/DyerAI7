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

"""A/B testing framework for model variants."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ABTestConfig:
    name: str
    variants: dict[str, str]  # variant_name -> model_url
    traffic_split: dict[str, float]
    primary_metric: str = "conversion"
    min_samples: int = 1000
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    sticky_bucketing: bool = True


@dataclass
class ABTestResult:
    variant: str
    n: int
    metric_mean: float
    metric_std: float
    ci_lower: float
    ci_upper: float
    is_control: bool


class ABTestFramework:
    def __init__(self) -> None:
        self.tests: dict[str, ABTestConfig] = {}
        self.assignments: dict[str, dict[str, str]] = {}  # test -> {user_id -> variant}
        self.observations: dict[str, dict[str, list[float]]] = {}  # test -> variant -> [metric]

    def create_test(self, config: ABTestConfig) -> None:
        self.tests[config.name] = config
        self.observations[config.name] = {v: [] for v in config.variants}

    def assign(self, test_name: str, user_id: str) -> str:
        cfg = self.tests[test_name]
        if cfg.sticky_bucketing and test_name in self.assignments and user_id in self.assignments[test_name]:
            return self.assignments[test_name][user_id]
        rng = np.random.default_rng(int(hashlib.md5(f"{test_name}:{user_id}".encode()).hexdigest()[:8], 16))
        variants = list(cfg.traffic_split.keys())
        probs = np.array(list(cfg.traffic_split.values()))
        probs = probs / probs.sum()
        v = variants[rng.choice(len(variants), p=probs)]
        self.assignments.setdefault(test_name, {})[user_id] = v
        return v

    def record(self, test_name: str, variant: str, metric: float) -> None:
        self.observations[test_name][variant].append(metric)

    def analyze(self, test_name: str) -> dict[str, Any]:
        cfg = self.tests[test_name]
        results: dict[str, ABTestResult] = {}
        for i, (v, vals) in enumerate(self.observations[test_name].items()):
            arr = np.array(vals)
            if len(arr) == 0:
                continue
            lo, hi = np.percentile(arr, [2.5, 97.5])
            results[v] = ABTestResult(
                variant=v, n=len(arr),
                metric_mean=float(arr.mean()), metric_std=float(arr.std()),
                ci_lower=float(lo), ci_upper=float(hi),
                is_control=(i == 0),
            )
        # Statistical comparison
        control_name = next((v for v, r in results.items() if r.is_control), None)
        comparisons: dict[str, dict[str, float]] = {}
        if control_name and len(results) > 1:
            ctrl_vals = np.array(self.observations[test_name][control_name])
            for v, r in results.items():
                if v == control_name or len(self.observations[test_name][v]) < 10:
                    continue
                t_vals = np.array(self.observations[test_name][v])
                try:
                    from scipy.stats import ttest_ind
                    t, p = ttest_ind(t_vals, ctrl_vals, equal_var=False)
                    d = (t_vals.mean() - ctrl_vals.mean()) / np.sqrt((t_vals.var() + ctrl_vals.var()) / 2)
                    comparisons[v] = {"p_value": float(p), "cohens_d": float(d), "uplift": float((t_vals.mean() - ctrl_vals.mean()) / max(ctrl_vals.mean(), 1e-9))}
                except Exception:
                    pass
        return {
            "results": {k: r.__dict__ for k, r in results.items()},
            "comparisons": comparisons,
            "winner": max(results.items(), key=lambda x: x[1].metric_mean)[0] if results else None,
            "ready": min(len(v) for v in self.observations[test_name].values()) >= cfg.min_samples,
        }
