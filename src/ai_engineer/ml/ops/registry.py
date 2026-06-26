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

"""Higher-level model registry operations (search, filter, lineage-aware)."""
from __future__ import annotations

from typing import Any

from ai_engineer.ml.models.registry import ModelRegistry, RegisteredModel


class RegistryOps:
    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()

    def search(self, query: str, tag_filter: dict[str, str] | None = None) -> list[RegisteredModel]:
        results: list[RegisteredModel] = []
        for name in self.registry._index:
            for v in self.registry._index[name].values():
                if query.lower() in v.name.lower() or query.lower() in v.description.lower():
                    if tag_filter and not all(v.tags.get(k) == val for k, val in tag_filter.items()):
                        continue
                    results.append(v)
        return results

    def best_model(self, name: str, metric: str, mode: str = "max") -> RegisteredModel | None:
        versions = self.registry.list_versions(name)
        if not versions:
            return None
        if mode == "max":
            return max((v for v in versions if metric in v.metrics), key=lambda v: v.metrics[metric], default=None)
        return min((v for v in versions if metric in v.metrics), key=lambda v: v.metrics[metric], default=None)

    def compare(self, name: str, versions: list[str]) -> list[dict[str, Any]]:
        out = []
        for v in versions:
            rm = self.registry.get(name, v)
            out.append({"version": rm.version, "stage": rm.stage, "metrics": rm.metrics, "params": rm.params, "created_at": rm.created_at})
        return out

    def archive_old_versions(self, name: str, keep: int = 5) -> int:
        versions = sorted(self.registry.list_versions(name), key=lambda v: v.created_at, reverse=True)
        n = 0
        for v in versions[keep:]:
            if v.stage not in ("production", "staging"):
                self.registry.delete(name, v.version)
                n += 1
        return n
