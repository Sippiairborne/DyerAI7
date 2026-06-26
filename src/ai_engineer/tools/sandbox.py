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

"""Code sandbox — runs Python in an isolated environment with GPU."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import docker
from docker.errors import ContainerError, ImageNotFound

from ai_engineer.config import get_settings
from ai_engineer.utils.errors import SandboxError
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)

Backend = Literal["docker", "e2b", "local"]


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_s: float
    artifacts: dict[str, str]  # path -> file content or url


class Sandbox:
    """Executes code in an isolated environment.

    Backends:
      - docker: local docker with GPU passthrough (default)
      - e2b: managed E2B interpreter
      - local: just runs in-process (NO isolation — only for trusted dev)
    """

    def __init__(self, backend: Backend | None = None) -> None:
        s = get_settings()
        self.backend: Backend = backend or s.sandbox_backend
        self.timeout = s.sandbox_timeout
        self.memory_gb = s.sandbox_memory_gb
        self.disk_gb = s.sandbox_disk_gb
        self.gpu = s.sandbox_gpu
        self.workspaces = Path(s.workspace_dir)
        self.workspaces.mkdir(parents=True, exist_ok=True)
        self._docker = None
        self._e2b = None

    def _get_docker(self) -> docker.DockerClient:
        if self._docker is None:
            self._docker = docker.from_env()
        return self._docker

    def _get_e2b(self) -> Any:
        if self._e2b is None:
            from e2b_code_interpreter import Sandbox as E2BSandbox

            self._e2b = E2BSandbox
        return self._e2b

    async def execute(
        self,
        code: str,
        *,
        files: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
        gpu: bool = True,
    ) -> ExecutionResult:
        timeout = timeout or self.timeout
        if self.backend == "docker":
            return await self._run_docker(code, files, env, timeout, gpu)
        if self.backend == "e2b":
            return await self._run_e2b(code, files, env, timeout)
        return await self._run_local(code, files, env, timeout)

    async def _run_docker(
        self,
        code: str,
        files: dict[str, str] | None,
        env: dict[str, str] | None,
        timeout: int,
        gpu: bool,
    ) -> ExecutionResult:
        client = self._get_docker()
        run_id = uuid.uuid4().hex[:12]
        workspace = self.workspaces / run_id
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "main.py").write_text(code)
        if files:
            for rel, content in files.items():
                p = workspace / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content)

        image = "ai-engineer-runtime:latest"
        device_requests = []
        if gpu:
            device_requests.append(
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            )

        env_full = {**os.environ, **(env or {})}
        env_full.setdefault("HF_HOME", "/data/hf")
        env_full.setdefault("PYTHONUNBUFFERED", "1")

        start = time.time()
        try:
            container = client.containers.run(
                image=image,
                command=["python", "/workspace/main.py"],
                volumes={str(workspace): {"bind": "/workspace", "mode": "rw"}},
                environment=env_full,
                working_dir="/workspace",
                mem_limit=f"{self.memory_gb}g",
                shm_size="16g",
                device_requests=device_requests,
                detach=True,
                network_mode="bridge",
            )
        except ImageNotFound:
            raise SandboxError(
                f"Runtime image '{image}' not found. Build it with: docker build -t {image} ."
            )
        except Exception as e:
            raise SandboxError(f"Failed to start container: {e}") from e

        try:
            result = await asyncio.to_thread(container.wait, timeout=timeout)
            exit_code = result.get("StatusCode", -1)
            stdout = container.logs(stdout=True, stderr=False).decode(errors="replace")
            stderr = container.logs(stdout=False, stderr=True).decode(errors="replace")
        except Exception as e:
            container.kill()
            raise SandboxError(f"Container execution failed: {e}") from e
        finally:
            try:
                container.remove(force=True)
            except Exception:
                pass

        # Collect artifacts
        artifacts: dict[str, str] = {}
        for p in workspace.rglob("*"):
            if p.is_file() and p.name != "main.py":
                try:
                    rel = str(p.relative_to(workspace))
                    if p.stat().st_size < 10_000_000:  # 10MB cap
                        artifacts[rel] = p.read_text(errors="replace")
                except Exception:
                    pass

        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_s=time.time() - start,
            artifacts=artifacts,
        )

    async def _run_e2b(
        self,
        code: str,
        files: dict[str, str] | None,
        env: dict[str, str] | None,
        timeout: int,
    ) -> ExecutionResult:
        E2BSandbox = self._get_e2b()
        start = time.time()
        s = E2BSandbox()
        try:
            if files:
                for name, content in files.items():
                    s.files.write(name, content)
            execution = s.run_code(code, timeout=timeout)
            return ExecutionResult(
                stdout="\n".join(str(l.text) for l in execution.logs.stdout),
                stderr="\n".join(str(l.text) for l in execution.logs.stderr),
                exit_code=0 if execution.error is None else 1,
                duration_s=time.time() - start,
                artifacts={},
            )
        except Exception as e:
            raise SandboxError(f"E2B execution failed: {e}") from e
        finally:
            s.kill()

    async def _run_local(
        self,
        code: str,
        files: dict[str, str] | None,
        env: dict[str, str] | None,
        timeout: int,
    ) -> ExecutionResult:
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            (tmpp / "main.py").write_text(code)
            if files:
                for rel, content in files.items():
                    p = tmpp / rel
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(content)
            start = time.time()
            env_full = {**os.environ, **(env or {})}
            proc = await asyncio.create_subprocess_exec(
                "python", str(tmpp / "main.py"),
                cwd=tmp,
                env=env_full,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError as e:
                proc.kill()
                raise SandboxError("Local execution timed out") from e
            return ExecutionResult(
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
                exit_code=proc.returncode or 0,
                duration_s=time.time() - start,
                artifacts={},
            )
