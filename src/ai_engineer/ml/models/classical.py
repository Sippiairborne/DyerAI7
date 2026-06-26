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

"""Classical ML trainers: sklearn, XGBoost, LightGBM, CatBoost."""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from ai_engineer.ml.models.registry import ModelRegistry
from ai_engineer.utils.errors import AIEngineerError
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)

Library = Literal["sklearn", "xgboost", "lightgbm", "catboost"]
Task = Literal["classification", "regression"]


@dataclass
class ClassicalTrainingResult:
    model_path: str
    metrics: dict[str, float]
    best_params: dict[str, Any]
    training_time_s: float
    feature_importance: dict[str, float] = field(default_factory=dict)
    cv_scores: list[float] = field(default_factory=list)


class ClassicalTrainer:
    """Unified trainer for classical ML models with CV, tuning, and registration."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        library: Library = "lightgbm",
        task: Task = "classification",
        params: dict[str, Any] | None = None,
        n_splits: int = 5,
        register_name: str | None = None,
        output_dir: str | None = None,
    ) -> ClassicalTrainingResult:
        params = params or {}
        start = time.time()
        # CV
        from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

        if library == "lightgbm":
            import lightgbm as lgb
            default = {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31, "verbosity": -1, "random_state": 42}
            default.update(params)
            model = lgb.LGBMClassifier(**default) if task == "classification" else lgb.LGBMRegressor(**default)
            scoring = "accuracy" if task == "classification" else "r2"
        elif library == "xgboost":
            import xgboost as xgb
            default = {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6, "verbosity": 0, "random_state": 42, "tree_method": "hist"}
            default.update(params)
            model = xgb.XGBClassifier(**default) if task == "classification" else xgb.XGBRegressor(**default)
            scoring = "accuracy" if task == "classification" else "r2"
        elif library == "catboost":
            from catboost import CatBoostClassifier, CatBoostRegressor
            default = {"iterations": 500, "learning_rate": 0.05, "depth": 6, "random_seed": 42, "verbose": False}
            default.update(params)
            model = CatBoostClassifier(**default) if task == "classification" else CatBoostRegressor(**default)
            scoring = "accuracy" if task == "classification" else "r2"
        elif library == "sklearn":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
            default = {"n_estimators": 300, "max_depth": 12, "random_state": 42, "n_jobs": -1}
            default.update(params)
            if task == "classification":
                model = RandomForestClassifier(**default) if "boosting" not in str(params.get("algo", "")) else GradientBoostingClassifier(**default)
            else:
                model = RandomForestRegressor(**default)
            scoring = "accuracy" if task == "classification" else "r2"
        else:
            raise AIEngineerError(f"Unknown library: {library}")

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) if task == "classification" else KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=kf, scoring=scoring, n_jobs=-1)

        model.fit(X, y)
        # Feature importance
        fi = {}
        if hasattr(model, "feature_importances_"):
            fi = {c: float(i) for c, i in zip(X.columns, model.feature_importances_)}
        elif hasattr(model, "get_feature_importance"):
            fi = {c: float(i) for c, i in zip(X.columns, model.get_feature_importance())}

        # Persist
        output_dir = output_dir or f"/tmp/classical_{int(time.time())}"
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        model_file = out_path / "model.pkl"
        with model_file.open("wb") as f:
            pickle.dump(model, f)
        (out_path / "params.json").write_text(json.dumps(default, indent=2, default=str))
        (out_path / "metrics.json").write_text(json.dumps({
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "cv_scores": cv_scores.tolist(),
            "library": library,
            "task": task,
        }, indent=2))

        metrics = {
            f"cv_{scoring}_mean": float(cv_scores.mean()),
            f"cv_{scoring}_std": float(cv_scores.std()),
        }

        if register_name:
            rm = self.registry.register(
                name=register_name,
                path=out_path,
                metrics=metrics,
                params=default,
                tags={"library": library, "task": task},
                description=f"{library} {task} model",
            )
            model_file = Path(rm.path) / "model.pkl"

        return ClassicalTrainingResult(
            model_path=str(model_file),
            metrics=metrics,
            best_params=default,
            training_time_s=time.time() - start,
            feature_importance=fi,
            cv_scores=cv_scores.tolist(),
        )
