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

"""File operations."""
from __future__ import annotations

from pathlib import Path

from ai_engineer.tools.registry import ToolRegistry, tool
from ai_engineer.utils.errors import ToolError

_registry = ToolRegistry()


@tool(
    name="read_file",
    description="Read the contents of a file. Returns the full text content.",
)
def read_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise ToolError(f"File not found: {path}")
    return p.read_text(encoding="utf-8", errors="replace")


@tool(
    name="write_file",
    description="Write text to a file, creating directories as needed.",
)
def write_file(path: str, content: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {path}"


@tool(
    name="edit_file",
    description="Edit a file by replacing an exact string. The match must be unique.",
)
def edit_file(path: str, old_text: str, new_text: str) -> str:
    p = Path(path)
    if not p.exists():
        raise ToolError(f"File not found: {path}")
    text = p.read_text(encoding="utf-8")
    if text.count(old_text) != 1:
        raise ToolError(f"Expected exactly 1 occurrence of old_text, found {text.count(old_text)}")
    p.write_text(text.replace(old_text, new_text), encoding="utf-8")
    return f"Edited {path}"


@tool(
    name="list_dir",
    description="List files in a directory (non-recursive).",
)
def list_dir(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise ToolError(f"Directory not found: {path}")
    return "\n".join(sorted(str(x.relative_to(p)) for x in p.iterdir()))


@tool(
    name="glob_files",
    description="List files matching a glob pattern, e.g. '**/*.py'.",
)
def glob_files(pattern: str, base: str = ".") -> str:
    return "\n".join(sorted(str(p) for p in Path(base).glob(pattern)))


@tool(
    name="delete_file",
    description="Delete a file or empty directory.",
)
def delete_file(path: str) -> str:
    p = Path(path)
    if p.is_dir():
        p.rmdir()
    else:
        p.unlink()
    return f"Deleted {path}"
