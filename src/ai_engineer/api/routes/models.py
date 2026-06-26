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

"""Model routes."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from ai_engineer.api.schemas import ModelInfo
from ai_engineer.config import get_settings

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", response_model=list[ModelInfo])
async def list_models() -> list[ModelInfo]:
    s = get_settings()
    base = Path(s.models_dir)
    out: list[ModelInfo] = []
    for p in base.iterdir():
        if p.is_dir():
            try:
                size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            except Exception:
                size = 0
            out.append(ModelInfo(name=p.name, path=str(p), size_bytes=size))
    return out
