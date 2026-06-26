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

"""Feature selection methods."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)

from ai_engineer.utils.errors import AIEngineerError
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SelectionResult:
    selected: list[str]
    scores: dict[str, float]
    method: str


class FeatureSelector:
    """Filter, wrapper, embedded feature selection."""

    def select(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "mutual_info",  # mutual_info | kbest | rfe | importance
        k: int = 10,
        task: str = "classification",  # classification | regression
    ) -> SelectionResult:
        if method == "mutual_info":
            fn = mutual_info_classif if task == "classification" else mutual_info_regression
            mi = fn(X.fillna(0), y)
            scores = {c: float(s) for c, s in zip(X.columns, mi)}
            selected = sorted(scores, key=scores.get, reverse=True)[:k]
        elif method == "kbest":
            fn = f_classif if task == "classification" else f_regression
            skb = SelectKBest(fn, k=min(k, X.shape[1])).fit(X.fillna(0), y)
            scores = {c: float(s) for c, s in zip(X.columns, skb.scores_)}
            selected = list(X.columns[skb.get_support()])
        elif method == "rfe":
            est = RandomForestClassifier(n_estimators=50, random_state=42) if task == "classification" else RandomForestRegressor(n_estimators=50, random_state=42)
            rfe = RFE(est, n_features_to_select=min(k, X.shape[1])).fit(X.fillna(0), y)
            scores = {c: float(r) for c, r in zip(X.columns, rfe.ranking_)}
            selected = list(X.columns[rfe.support_])
        elif method == "importance":
            est = RandomForestClassifier(n_estimators=100, random_state=42) if task == "classification" else RandomForestRegressor(n_estimators=100, random_state=42)
            est.fit(X.fillna(0), y)
            scores = {c: float(i) for c, i in zip(X.columns, est.feature_importances_)}
            selected = sorted(scores, key=scores.get, reverse=True)[:k]
        else:
            raise AIEngineerError(f"Unknown method: {method}")
        return SelectionResult(selected=selected, scores=scores, method=method)
