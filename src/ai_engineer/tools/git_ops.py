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

"""Git operations."""
from __future__ import annotations

import subprocess
from pathlib import Path

import git
from git import Repo

from ai_engineer.tools.registry import ToolRegistry, tool
from ai_engineer.utils.errors import ToolError

_registry = ToolRegistry()


def _run(cwd: str, args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout + ("\n" + result.stderr if result.stderr else "")


@tool(name="git_status", description="Show git status of a repo.")
def git_status(path: str) -> str:
    return _run(path, ["status"])


@tool(name="git_diff", description="Show diff for a repo path.")
def git_diff(path: str, staged: bool = False) -> str:
    return _run(path, ["diff", "--staged"] if staged else ["diff"])


@tool(name="git_commit", description="Stage all and commit with a message.")
def git_commit(path: str, message: str) -> str:
    repo = Repo(path)
    repo.git.add(A=True)
    if not repo.index.diff("HEAD"):
        return "Nothing to commit"
    return _run(path, ["commit", "-m", message])


@tool(name="git_log", description="Show last N commits.")
def git_log(path: str, n: int = 10) -> str:
    return _run(path, ["log", f"-n{n}", "--oneline"])


@tool(name="git_clone", description="Clone a repository to a target directory.")
def git_clone(url: str, target: str) -> str:
    try:
        Repo.clone_from(url, target)
        return f"Cloned {url} → {target}"
    except git.GitCommandError as e:
        raise ToolError(f"Clone failed: {e}") from e
