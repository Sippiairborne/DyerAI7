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

"""Training routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ai_engineer.api.deps import get_state
from ai_engineer.api.schemas import TrainingRunRequest, TrainingRunOut

router = APIRouter(prefix="/api/training", tags=["training"])


@router.post("/start", response_model=TrainingRunOut)
async def start_training(req: TrainingRunRequest) -> TrainingRunOut:
    state = get_state()
    if not state.sandbox:
        raise HTTPException(503, "Sandbox not initialized")
    script = f"""
import json
from ai_engineer.tools.training import start_finetune
print(start_finetune(
    model_name='{req.model_name}',
    dataset_path='{req.dataset_path}',
    output_dir='{req.output_dir}',
    num_epochs={req.num_epochs},
    batch_size={req.batch_size},
    learning_rate={req.learning_rate},
    use_lora={str(req.use_lora).lower()},
))
"""
    result = await state.sandbox.execute(script, timeout=300)
    if result.exit_code != 0:
        raise HTTPException(500, f"Failed to start: {result.stderr}")
    return TrainingRunOut(id=req.output_dir, status="running")
