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

import pytest

from ai_engineer.tools.sandbox import Sandbox


@pytest.mark.asyncio
async def test_local_execution(sandbox: Sandbox) -> None:
    result = await sandbox.execute('print("hello")\nimport os; print(os.getcwd())')
    assert result.exit_code == 0
    assert "hello" in result.stdout


@pytest.mark.asyncio
async def test_local_with_files(sandbox: Sandbox) -> None:
    code = "print(open('data.txt').read())"
    result = await sandbox.execute(code, files={"data.txt": "abc123"})
    assert "abc123" in result.stdout


@pytest.mark.asyncio
async def test_local_error(sandbox: Sandbox) -> None:
    result = await sandbox.execute("raise ValueError('boom')")
    assert result.exit_code != 0
    assert "ValueError" in result.stderr
