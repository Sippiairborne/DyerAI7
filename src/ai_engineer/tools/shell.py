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

"""Shell command execution."""
from __future__ import annotations

import asyncio

from ai_engineer.tools.registry import ToolRegistry, tool
from ai_engineer.utils.errors import ToolError

_registry = ToolRegistry()


@tool(
    name="shell",
    description="Run a shell command. Returns combined stdout/stderr and exit code. Use carefully.",
)
async def shell(command: str, timeout: int = 300) -> str:
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        rc = proc.returncode
    except asyncio.TimeoutError as e:
        raise ToolError(f"Shell command timed out: {command[:100]}") from e
    out = stdout.decode(errors="replace")
    err = stderr.decode(errors="replace")
    return f"EXIT_CODE: {rc}\nSTDOUT:\n{out}\nSTDERR:\n{err}"


@tool(
    name="pip_install",
    description="Install a Python package with pip.",
)
async def pip_install(packages: str) -> str:
    return await shell(f"pip install {packages}", timeout=600)
