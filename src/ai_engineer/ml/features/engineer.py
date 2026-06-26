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

"""Automated feature engineering — generates interaction, polynomial, datetime, target encoding features."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ai_engineer.utils.errors import AIEngineerError
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureEngineeringReport:
    original_features: list[str]
    new_features: list[str]
    dropped: list[str] = field(default_factory=list)
    n_original: int = 0
    n_new: int = 0
    actions: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        return "\n".join([
            "# Feature Engineering Report",
            f"- Original: {self.n_original}",
            f"- New: {self.n_new}",
            f"- Dropped: {self.dropped}",
            "## Actions",
            *[f"- {a}" for a in self.actions],
        ])


class FeatureEngineer:
    """Generate features from raw data automatically."""

    def engineer(
        self,
        df: pd.DataFrame,
        target: str | None = None,
        *,
        datetime_features: bool = True,
        interaction_features: bool = True,
        polynomial_degree: int = 2,
        target_encode: bool = True,
        text_features: list[str] | None = None,
        max_features: int = 500,
    ) -> tuple[pd.DataFrame, FeatureEngineeringReport]:
        out = df.copy()
        report = FeatureEngineeringReport(
            original_features=list(df.columns),
            new_features=[],
            n_original=len(df.columns),
        )

        # Datetime
        if datetime_features:
            for c in out.select_dtypes(include=["datetime", "datetimetz"]).columns:
                out[f"{c}_year"] = out[c].dt.year
                out[f"{c}_month"] = out[c].dt.month
                out[f"{c}_day"] = out[c].dt.day
                out[f"{c}_dayofweek"] = out[c].dt.dayofweek
                out[f"{c}_hour"] = out[c].dt.hour
                out[f"{c}_is_weekend"] = (out[c].dt.dayofweek >= 5).astype(int)
                out[f"{c}_sin_month"] = np.sin(2 * np.pi * out[c].dt.month / 12)
                out[f"{c}_cos_month"] = np.cos(2 * np.pi * out[c].dt.month / 12)
                report.new_features += [f"{c}_year", f"{c}_month", f"{c}_day", f"{c}_dayofweek", f"{c}_hour", f"{c}_is_weekend", f"{c}_sin_month", f"{c}_cos_month"]
                report.actions.append(f"Extracted datetime features from {c}")

        # Numeric interactions and polynomials
        num_cols = out.select_dtypes(include="number").columns.tolist()
        if target and target in num_cols:
            num_cols.remove(target)
        if interaction_features and len(num_cols) >= 2:
            for i, a in enumerate(num_cols):
                for b in num_cols[i + 1:i + 3]:
                    out[f"{a}__x__{b}"] = out[a] * out[b]
                    out[f"{a}__div__{b}"] = out[a] / (out[b] + 1e-8)
                    report.new_features += [f"{a}__x__{b}", f"{a}__div__{b}"]
        if polynomial_degree >= 2 and num_cols:
            for a in num_cols[:5]:
                out[f"{a}__sq"] = out[a] ** 2
                report.new_features.append(f"{a}__sq")

        # Target encoding
        if target_encode and target and target in out.columns:
            cat_cols = out.select_dtypes(include=["object", "category"]).columns.tolist()
            if target in cat_cols:
                cat_cols.remove(target)
            for c in cat_cols:
                if out[c].nunique() > 50:
                    continue
                means = out.groupby(c)[target].mean()
                out[f"{c}__te"] = out[c].map(means)
                report.new_features.append(f"{c}__te")

        # Text features
        if text_features:
            for c in text_features:
                if c not in out.columns:
                    continue
                s = out[c].astype(str)
                out[f"{c}__len"] = s.str.len()
                out[f"{c}__n_words"] = s.str.split().str.len()
                out[f"{c}__n_unique"] = s.apply(lambda x: len(set(x.split())))
                out[f"{c}__has_num"] = s.str.contains(r"\d").astype(int)
                report.new_features += [f"{c}__len", f"{c}__n_words", f"{c}__n_unique", f"{c}__has_num"]

        # Drop constant / all-NaN
        for c in list(out.columns):
            if c in (target,) if target else False:
                continue
            if out[c].nunique(dropna=True) <= 1 or out[c].isna().all():
                out = out.drop(columns=[c])
                report.dropped.append(c)

        if len(out.columns) > max_features:
            keep = out.var(numeric_only=True).sort_values(ascending=False).head(max_features).index.tolist()
            if target and target in out.columns:
                keep = [target] + [c for c in keep if c != target]
            dropped = [c for c in out.columns if c not in keep]
            out = out[keep]
            report.dropped += dropped
            report.actions.append(f"Reduced features to top-{max_features} by variance")

        report.n_new = len([c for c in out.columns if c not in report.original_features])
        return out, report
