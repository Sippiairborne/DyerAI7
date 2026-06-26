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

"""Pydantic schemas for the API."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Status = Literal["pending", "running", "done", "failed"]


class TaskRequest(BaseModel):
    goal: str
    max_replans: int = 3
    max_task_retries: int = 2
    stream: bool = False


class TaskResponse(BaseModel):
    id: str
    goal: str
    status: Status
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class ProjectCreate(BaseModel):
    name: str
    description: str = ""


class ProjectOut(BaseModel):
    id: str
    name: str
    description: str
    created_at: datetime


class DatasetInfo(BaseModel):
    name: str
    path: str
    size_bytes: int
    num_samples: int | None = None


class ModelInfo(BaseModel):
    name: str
    path: str
    size_bytes: int
    base_model: str | None = None
    created_at: datetime


class TrainingRunRequest(BaseModel):
    model_name: str
    dataset_path: str
    output_dir: str
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5
    use_lora: bool = True


class TrainingRunOut(BaseModel):
    id: str
    status: Status
    metrics: dict[str, float] = Field(default_factory=dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class EvalRequest(BaseModel):
    model_path: str
    dataset_path: str
    metrics: list[str] = Field(default_factory=lambda: ["accuracy"])


class HealthResponse(BaseModel):
    status: str
    llm: str
    vector_store: str
    graph_store: str
    redis: str
    version: str
