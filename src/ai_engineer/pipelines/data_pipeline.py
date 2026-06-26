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

"""Data pipeline — acquire, clean, validate, version."""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ai_engineer.config import get_settings
from ai_engineer.utils.errors import PipelineError
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetVersion:
    name: str
    version: str
    path: str
    hash: str
    num_samples: int
    created_at: str
    metadata: dict[str, Any]


class DataPipeline:
    """Acquire, validate, and version a dataset."""

    def __init__(self) -> None:
        s = get_settings()
        self.base = Path(s.datasets_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    def acquire(self, source: str, name: str, **kwargs: Any) -> Path:
        if source == "hf":
            from datasets import load_dataset

            ds = load_dataset(name, **kwargs)
            path = self.base / name.replace("/", "__")
            path.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(path))
            return path
        if source == "url":
            import httpx

            dest = self.base / Path(name).name
            with httpx.stream("GET", name, follow_redirects=True, timeout=600) as r:
                r.raise_for_status()
                with dest.open("wb") as f:
                    for chunk in r.iter_bytes(1024 * 1024):
                        f.write(chunk)
            return dest
        raise PipelineError(f"Unknown source: {source}")

    def validate(self, path: Path) -> dict[str, Any]:
        if path.is_dir():
            from datasets import load_from_disk

            ds = load_from_disk(str(path))
            return {"rows": len(ds), "columns": ds.column_names}
        if path.suffix == ".csv":
            df = pd.read_csv(path, nrows=10_000)
            return {"rows": len(df), "columns": list(df.columns), "dtypes": {c: str(t) for c, t in df.dtypes.items()}}
        return {"path": str(path), "size_bytes": path.stat().st_size}

    def version(self, name: str, path: Path, metadata: dict[str, Any] | None = None) -> DatasetVersion:
        metadata = metadata or {}
        metadata.update(self.validate(path))
        h = hashlib.sha256()
        if path.is_file():
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
        else:
            for p in sorted(path.rglob("*")):
                if p.is_file():
                    h.update(p.read_bytes())
        version = h.hexdigest()[:12]
        ver_dir = self.base / "_versions" / name / version
        ver_dir.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            if ver_dir.joinpath("data").exists():
                shutil.rmtree(ver_dir / "data")
            shutil.copytree(path, ver_dir / "data")
        else:
            shutil.copy2(path, ver_dir / path.name)
        meta_path = ver_dir / "metadata.json"
        info = DatasetVersion(
            name=name,
            version=version,
            path=str(ver_dir),
            hash=h.hexdigest(),
            num_samples=metadata.get("rows", 0),
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata,
        )
        meta_path.write_text(json.dumps(asdict(info), indent=2, default=str))
        logger.info("data.versioned", name=name, version=version)
        return info
