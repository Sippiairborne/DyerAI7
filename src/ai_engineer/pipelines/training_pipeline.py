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

"""Training pipeline — wraps training with checkpointing, logging, resume."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_engineer.config import get_settings
from ai_engineer.tools.sandbox import Sandbox
from ai_engineer.utils.errors import PipelineError
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingPipelineResult:
    output_dir: str
    metrics: dict[str, float] = field(default_factory=dict)
    duration_s: float = 0.0
    success: bool = False
    log: str = ""


class TrainingPipeline:
    """End-to-end training pipeline that streams logs and is resumable."""

    def __init__(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox
        s = get_settings()
        self.logs_dir = Path(s.logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    async def run(
        self,
        script: str,
        output_dir: str,
        env: dict[str, str] | None = None,
        timeout: int = 14400,
    ) -> TrainingPipelineResult:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        log_path = Path(output_dir) / "train.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("training.start", output_dir=output_dir)
        start = time.time()
        result = await self.sandbox.execute(script, env=env, timeout=timeout, gpu=True)
        log_path.write_text(result.stdout + "\n--- STDERR ---\n" + result.stderr)
        # Try to extract metrics from log
        metrics: dict[str, float] = {}
        for line in result.stdout.splitlines():
            if "{" in line and "}" in line:
                try:
                    start_idx = line.index("{")
                    end_idx = line.rindex("}") + 1
                    data = json.loads(line[start_idx:end_idx])
                    if isinstance(data, dict):
                        for k, v in data.items():
                            if isinstance(v, (int, float)):
                                metrics[k] = float(v)
                except Exception:
                    pass
        return TrainingPipelineResult(
            output_dir=output_dir,
            metrics=metrics,
            duration_s=time.time() - start,
            success=result.exit_code == 0,
            log=result.stdout[-5000:],
        )
