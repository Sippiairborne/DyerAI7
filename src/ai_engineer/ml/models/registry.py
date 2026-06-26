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

"""Model registry with versioning, stage transitions, and metadata."""
from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ai_engineer.config import get_settings
from ai_engineer.utils.errors import AIEngineerError
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)

Stage = Literal["none", "staging", "production", "archived"]


@dataclass
class RegisteredModel:
    name: str
    version: str
    stage: Stage
    path: str
    run_id: str
    metrics: dict[str, float]
    params: dict[str, Any]
    tags: dict[str, str]
    description: str
    created_at: str
    updated_at: str
    lineage: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """Local file-based model registry (drop-in for MLflow Model Registry)."""

    def __init__(self, base_dir: str | None = None) -> None:
        s = get_settings()
        self.base = Path(base_dir or s.models_dir) / "_registry"
        self.base.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, dict[str, RegisteredModel]] = {}
        self._load_index()

    def _load_index(self) -> None:
        idx_path = self.base / "_index.json"
        if idx_path.exists():
            self._index = json.loads(idx_path.read_text())

    def _save_index(self) -> None:
        (self.base / "_index.json").write_text(json.dumps(self._index, indent=2, default=str))

    def register(
        self,
        name: str,
        path: str | Path,
        run_id: str = "",
        metrics: dict[str, float] | None = None,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        description: str = "",
        lineage: dict[str, Any] | None = None,
    ) -> RegisteredModel:
        version = f"v{int(time.time())}_{uuid.uuid4().hex[:6]}"
        ver_dir = self.base / name / version
        ver_dir.mkdir(parents=True, exist_ok=True)
        if Path(path).is_dir():
            for item in Path(path).iterdir():
                if item.is_file():
                    shutil.copy2(item, ver_dir / item.name)
        elif Path(path).is_file():
            shutil.copy2(path, ver_dir / Path(path).name)
        now = datetime.now(timezone.utc).isoformat()
        rm = RegisteredModel(
            name=name,
            version=version,
            stage="none",
            path=str(ver_dir),
            run_id=run_id,
            metrics=metrics or {},
            params=params or {},
            tags=tags or {},
            description=description,
            created_at=now,
            updated_at=now,
            lineage=lineage or {},
        )
        self._index.setdefault(name, {})[version] = rm
        self._save_index()
        (ver_dir / "metadata.json").write_text(json.dumps(asdict(rm), indent=2, default=str))
        logger.info("registry.registered", name=name, version=version)
        return rm

    def get(self, name: str, version: str | None = None, stage: Stage | None = None) -> RegisteredModel:
        versions = self._index.get(name)
        if not versions:
            raise AIEngineerError(f"Model not found: {name}")
        if stage:
            for v in versions.values():
                if v.stage == stage:
                    return v
            raise AIEngineerError(f"No model in stage {stage} for {name}")
        if version:
            return versions[version]
        # Return latest
        return max(versions.values(), key=lambda r: r.created_at)

    def transition(self, name: str, version: str, stage: Stage) -> RegisteredModel:
        rm = self.get(name, version)
        rm.stage = stage
        rm.updated_at = datetime.now(timezone.utc).isoformat()
        if stage == "production":
            # Demote other production versions
            for v in self._index[name].values():
                if v.version != version and v.stage == "production":
                    v.stage = "archived"
        self._save_index()
        (Path(rm.path) / "metadata.json").write_text(json.dumps(asdict(rm), indent=2, default=str))
        logger.info("registry.transition", name=name, version=version, stage=stage)
        return rm

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "latest_version": max(vs.values(), key=lambda r: r.created_at).version,
                "versions": len(vs),
                "stages": {s: sum(1 for v in vs.values() if v.stage == s) for s in ("none", "staging", "production", "archived")},
            }
            for name, vs in self._index.items()
        ]

    def list_versions(self, name: str) -> list[RegisteredModel]:
        return list(self._index.get(name, {}).values())

    def delete(self, name: str, version: str) -> None:
        if name in self._index and version in self._index[name]:
            rm = self._index[name].pop(version)
            shutil.rmtree(rm.path, ignore_errors=True)
            self._save_index()

    def load(self, name: str, version: str | None = None, stage: Stage | None = None) -> Path:
        rm = self.get(name, version=version, stage=stage)
        return Path(rm.path)
