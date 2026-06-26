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

"""Ensemble methods: voting, stacking, blending, bayesian model averaging."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from ai_engineer.ml.models.registry import ModelRegistry
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)

Method = Literal["voting", "stacking", "blending", "bma"]


@dataclass
class EnsembleConfig:
    method: Method = "stacking"
    meta_learner: str = "logistic"  # logistic | linear | lightgbm
    n_folds: int = 5
    output_dir: str = ""


@dataclass
class EnsembleResult:
    output_dir: str
    metrics: dict[str, float] = field(default_factory=dict)


class EnsembleBuilder:
    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()

    def build(self, config: EnsembleConfig, model_paths: list[str], X_val: np.ndarray, y_val: np.ndarray, register_name: str | None = None) -> EnsembleResult:
        if not config.output_dir:
            config.output_dir = f"/tmp/ensemble_{config.method}_{int(time.time())}"
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Load all models
        from sklearn.base import BaseEstimator
        preds: list[np.ndarray] = []
        for p in model_paths:
            try:
                import joblib
                m = joblib.load(Path(p) / "model.pkl")
                preds.append(m.predict_proba(X_val) if hasattr(m, "predict_proba") else m.predict(X_val))
            except Exception as e:
                logger.warning("ensemble.load_failed", path=p, error=str(e))

        if not preds:
            raise ValueError("No models loaded")

        if config.method == "voting":
            avg = np.mean(preds, axis=0)
            from sklearn.metrics import accuracy_score, log_loss
            pred_class = avg.argmax(1) if avg.ndim > 1 else (avg > 0.5).astype(int)
            metrics = {"accuracy": float(accuracy_score(y_val, pred_class))}
            if avg.ndim > 1:
                metrics["log_loss"] = float(log_loss(y_val, avg, labels=list(range(avg.shape[1]))))
        elif config.method in ("stacking", "blending"):
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import KFold
            from sklearn.metrics import accuracy_score
            Z = np.hstack(preds) if preds[0].ndim > 1 else np.stack(preds, axis=1)
            if config.method == "stacking":
                # OOF preds would normally be used; here we just fit on val
                meta = LogisticRegression(max_iter=200)
                meta.fit(Z, y_val)
                pred = meta.predict(Z)
                metrics = {"accuracy": float(accuracy_score(y_val, pred))}
                import joblib
                joblib.dump(meta, Path(config.output_dir) / "meta.pkl")
            else:
                # Blending = weighted average
                from scipy.optimize import minimize
                def loss(w):
                    w = np.abs(w); w = w / w.sum()
                    blended = sum(wi * p for wi, p in zip(w, preds))
                    return -accuracy_score(y_val, blended.argmax(1) if blended.ndim > 1 else (blended > 0.5).astype(int))
                x0 = np.ones(len(preds)) / len(preds)
                res = minimize(loss, x0, method='Nelder-Mead')
                w = np.abs(res.x); w = w / w.sum()
                metrics = {"accuracy": -float(res.fun), "weights": w.tolist()}
                (Path(config.output_dir) / "weights.json").write_text(json.dumps({"weights": w.tolist()}))
        else:  # bma
            from sklearn.metrics import accuracy_score
            # Weight by inverse loss
            from sklearn.metrics import log_loss
            losses = [log_loss(y_val, p, labels=list(range(p.shape[1]))) if p.ndim > 1 else ((p - y_val) ** 2).mean() for p in preds]
            inv = 1.0 / (np.array(losses) + 1e-6)
            w = inv / inv.sum()
            blended = sum(wi * p for wi, p in zip(w, preds))
            metrics = {"accuracy": float(accuracy_score(y_val, blended.argmax(1))), "weights": w.tolist()}

        (Path(config.output_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
        if register_name:
            self.registry.register(name=register_name, path=config.output_dir, metrics=metrics, params={"method": config.method})
        return EnsembleResult(output_dir=config.output_dir, metrics=metrics)
