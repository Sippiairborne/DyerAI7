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

"""Evaluation routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ai_engineer.api.deps import get_state
from ai_engineer.api.schemas import EvalRequest

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


@router.post("/run")
async def run_evaluation(req: EvalRequest) -> dict:
    state = get_state()
    if not state.sandbox:
        raise HTTPException(503, "Sandbox not initialized")
    code = f"""
import json
from ai_engineer.tools.evaluation import run_eval_suite
result = run_eval_suite(
    model_path='{req.model_path}',
    dataset_path='{req.dataset_path}',
    metrics={req.metrics!r},
)
print(json.dumps(result))
"""
    result = await state.sandbox.execute(code, timeout=3600)
    if result.exit_code != 0:
        raise HTTPException(500, result.stderr)
    return {"output": result.stdout}
