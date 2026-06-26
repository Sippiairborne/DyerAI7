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

"""Application configuration loaded from environment."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # LLM
    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key: str = "sk-local"
    llm_model: str = "your-custom-model"
    llm_fast_model: str = "your-custom-model-fast"
    llm_code_model: str = "your-custom-model-code"
    llm_timeout: int = 120
    llm_max_retries: int = 3

    # Database
    database_url: str = "postgresql+asyncpg://aieng:aieng@localhost:5432/ai_engineer"
    redis_url: str = "redis://localhost:6379/0"

    # Vector
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "ai_engineer_memory"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Graph
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j_password"

    # Sandbox
    e2b_api_key: str = ""
    modal_token_id: str = ""
    modal_token_secret: str = ""
    docker_host: str = "unix:///var/run/docker.sock"
    sandbox_timeout: int = 600
    sandbox_gpu: str = "A100"
    sandbox_memory_gb: int = 64
    sandbox_disk_gb: int = 500

    # Experiment tracking
    wandb_api_key: str = ""
    wandb_project: str = "ai-engineer"
    mlflow_tracking_uri: str = "http://localhost:5000"

    # HuggingFace
    hf_token: str = ""
    hf_home: str = "/data/hf"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_workers: int = 4
    api_secret_key: str = "change-me"
    api_cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8501"])

    # UI
    ui_port: int = 8501
    ui_api_url: str = "http://localhost:8080"

    # Storage
    workspace_dir: str = "/data/workspaces"
    artifacts_dir: str = "/data/artifacts"
    datasets_dir: str = "/data/datasets"
    models_dir: str = "/data/models"
    logs_dir: str = "/data/logs"

    # Self improvement
    enable_self_improvement: bool = True
    fine_tune_threshold: int = 50
    fine_tune_base_model: str = "your-custom-model"

    sandbox_backend: Literal["e2b", "modal", "docker", "local"] = "docker"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    for d in (s.workspace_dir, s.artifacts_dir, s.datasets_dir, s.models_dir, s.logs_dir, s.hf_home):
        Path(d).mkdir(parents=True, exist_ok=True)
    return s
