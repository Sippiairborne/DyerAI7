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

"""Deployment pipeline."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ai_engineer.tools.sandbox import Sandbox
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DeploymentInfo:
    model_path: str
    command: str
    port: int
    container_id: str | None = None


class DeploymentPipeline:
    def __init__(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox

    def build_serving(self, model_path: str, port: int = 8000) -> DeploymentInfo:
        cmd = (
            f"docker run --gpus all -d -p {port}:8000 "
            f"-v {model_path}:/model vllm/vllm-openai:latest "
            f"--model /model --port 8000"
        )
        try:
            cid = subprocess.check_output(cmd, shell=True, text=True).strip()
        except subprocess.CalledProcessError:
            cid = None
        return DeploymentInfo(model_path=model_path, command=cmd, port=port, container_id=cid)
