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

"""Project routes."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/projects", tags=["projects"])


class Project(BaseModel):
    id: str
    name: str
    description: str
    created_at: str


# In-memory store (replace with DB in production)
_projects: dict[str, Project] = {}


@router.get("", response_model=list[Project])
async def list_projects() -> list[Project]:
    return list(_projects.values())


@router.post("", response_model=Project)
async def create_project(name: str, description: str = "") -> Project:
    p = Project(
        id=str(uuid.uuid4()),
        name=name,
        description=description,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _projects[p.id] = p
    return p


@router.get("/{project_id}", response_model=Project)
async def get_project(project_id: str) -> Project:
    p = _projects.get(project_id)
    if not p:
        raise HTTPException(404, "Not found")
    return p


@router.delete("/{project_id}")
async def delete_project(project_id: str) -> dict:
    if project_id not in _projects:
        raise HTTPException(404, "Not found")
    del _projects[project_id]
    return {"deleted": project_id}
