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

"""Feature scaling & normalization."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from ai_engineer.utils.errors import AIEngineerError

Method = Literal["standard", "minmax", "robust", "maxabs", "quantile", "power"]


@dataclass
class FittedScaler:
    method: Method
    scaler: Any
    columns: list[str]
    fitted: bool = False


class FeatureScaler:
    def __init__(self, method: Method = "standard") -> None:
        self.method = method
        self._state: FittedScaler | None = None

    def _make(self) -> Any:
        return {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "robust": RobustScaler,
            "maxabs": MaxAbsScaler,
            "quantile": lambda: QuantileTransformer(output_distribution="normal", random_state=42),
            "power": lambda: PowerTransformer(method="yeo-johnson"),
        }[self.method]()

    def fit_transform(self, df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        columns = columns or df.select_dtypes(include="number").columns.tolist()
        scaler = self._make()
        out = df.copy()
        out[columns] = scaler.fit_transform(df[columns].fillna(0))
        self._state = FittedScaler(method=self.method, scaler=scaler, columns=columns, fitted=True)
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._state or not self._state.fitted:
            raise AIEngineerError("Scaler not fitted")
        out = df.copy()
        out[self._state.columns] = self._state.scaler.transform(df[self._state.columns].fillna(0))
        return out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._state:
            raise AIEngineerError("Scaler not fitted")
        out = df.copy()
        out[self._state.columns] = self._state.scaler.inverse_transform(df[self._state.columns])
        return out

    def save(self, path: str | Path) -> None:
        if not self._state:
            raise AIEngineerError("Scaler not fitted")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as f:
            pickle.dump(self._state, f)

    def load(self, path: str | Path) -> "FeatureScaler":
        with Path(path).open("rb") as f:
            self._state = pickle.load(f)
        self.method = self._state.method
        return self
