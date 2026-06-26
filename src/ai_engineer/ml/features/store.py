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

"""Feature store — backed by Postgres + Redis for online/offline."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ai_engineer.config import get_settings
from ai_engineer.utils.errors import AIEngineerError
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureDefinition:
    name: str
    version: int
    description: str
    schema: dict[str, str]
    owner: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class FeatureStore:
    """Simple in-process feature store with optional Redis backend for online lookup."""

    def __init__(self) -> None:
        self._offline: dict[str, pd.DataFrame] = {}
        self._online: dict[str, dict[str, Any]] = {}
        self._defs: dict[str, FeatureDefinition] = {}
        self._redis = None
        try:
            import redis as redis_lib
            self._redis = redis_lib.from_url(get_settings().redis_url)
        except Exception:
            pass

    def register(self, definition: FeatureDefinition) -> None:
        key = f"{definition.name}:v{definition.version}"
        self._defs[key] = definition
        if self._redis:
            self._redis.set(f"feature:def:{key}", json.dumps(definition.__dict__))

    def write_offline(self, name: str, version: int, df: pd.DataFrame) -> None:
        key = f"{name}:v{version}"
        self._offline[key] = df
        logger.info("feature_store.write_offline", key=key, rows=len(df))

    def read_offline(self, name: str, version: int) -> pd.DataFrame:
        return self._offline[f"{name}:v{version}"]

    def write_online(self, name: str, version: int, entity_id: str, features: dict[str, Any]) -> None:
        key = f"{name}:v{version}:{entity_id}"
        self._online[key] = features
        if self._redis:
            self._redis.set(key, json.dumps(features), ex=86400 * 30)

    def read_online(self, name: str, version: int, entity_id: str) -> dict[str, Any] | None:
        key = f"{name}:v{version}:{entity_id}"
        if self._redis:
            v = self._redis.get(key)
            if v:
                return json.loads(v)
        return self._online.get(key)

    def list_features(self) -> list[dict[str, Any]]:
        return [
            {"name": d.name, "version": d.version, "owner": d.owner, "tags": d.tags}
            for d in self._defs.values()
        ]
