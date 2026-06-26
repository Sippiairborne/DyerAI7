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

"""FastAPI server entry point."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai_engineer import __version__
from ai_engineer.api.deps import get_state
from ai_engineer.api.routes import (
    datasets,
    deployment,
    evaluation,
    models,
    projects,
    system,
    tasks,
    training,
)
from ai_engineer.api.websocket import router as ws_router
from ai_engineer.config import get_settings
from ai_engineer.core.llm import LLMClient
from ai_engineer.core.memory import MemorySystem
from ai_engineer.core.orchestrator import Orchestrator
from ai_engineer.tools import file_ops, git_ops, shell  # noqa: F401  (register tools)
from ai_engineer.tools.dataset_loader import _registry as dl_registry
from ai_engineer.tools.experiment_tracker import _registry as et_registry
from ai_engineer.tools.file_ops import _registry as fo_registry
from ai_engineer.tools.git_ops import _registry as go_registry
from ai_engineer.tools.inference import _registry as inf_registry
from ai_engineer.tools.registry import ToolRegistry
from ai_engineer.tools.sandbox import Sandbox
from ai_engineer.tools.search import _registry as search_registry
from ai_engineer.tools.shell import _registry as shell_registry
from ai_engineer.tools.training import _registry as train_registry
from ai_engineer.utils.logging import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = get_state()
    settings = get_settings()
    log.info("startup.begin", version=__version__)

    state.llm = LLMClient()
    state.memory = MemorySystem(state.llm)
    await state.memory.init()
    state.sandbox = Sandbox()
    state.tools = ToolRegistry()
    for reg in [fo_registry, go_registry, shell_registry, dl_registry, train_registry, inf_registry, search_registry, et_registry]:
        for spec in reg.list():
            state.tools.register(spec)
    state.orchestrator = Orchestrator(state.llm, state.memory, state.tools, state.sandbox)

    log.info("startup.complete", tools=len(state.tools.list()))
    try:
        yield
    finally:
        if state.llm:
            await state.llm.close()
        if state.memory:
            await state.memory.close()
        log.info("shutdown.complete")


app = FastAPI(
    title="AI Engineer",
    description="Autonomous AI Engineering System",
    version=__version__,
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router)
app.include_router(tasks.router)
app.include_router(projects.router)
app.include_router(datasets.router)
app.include_router(models.router)
app.include_router(training.router)
app.include_router(evaluation.router)
app.include_router(deployment.router)
app.include_router(ws_router)
