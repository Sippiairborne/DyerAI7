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

"""System status and health."""
from __future__ import annotations

from fastapi import APIRouter

from ai_engineer import __version__
from ai_engineer.api.deps import get_state
from ai_engineer.api.schemas import HealthResponse
from ai_engineer.config import get_settings

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    state = get_state()
    s = get_settings()
    return HealthResponse(
        status="ok" if state.orchestrator else "degraded",
        llm=s.llm_base_url,
        vector_store=s.qdrant_url,
        graph_store=s.neo4j_uri,
        redis=s.redis_url,
        version=__version__,
    )


@router.get("/config")
async def config() -> dict:
    s = get_settings()
    return s.model_dump()
