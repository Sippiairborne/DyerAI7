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

"""Dataset loading from various sources."""
from __future__ import annotations

import os
from pathlib import Path

from ai_engineer.config import get_settings
from ai_engineer.tools.registry import ToolRegistry, tool
from ai_engineer.utils.errors import ToolError

_registry = ToolRegistry()


@tool(
    name="load_hf_dataset",
    description="Load a HuggingFace dataset to disk and return its local path.",
)
def load_hf_dataset(name: str, split: str = "train", cache_dir: str | None = None) -> str:
    from datasets import load_dataset

    settings = get_settings()
    cache = cache_dir or settings.datasets_dir
    Path(cache).mkdir(parents=True, exist_ok=True)
    if settings.hf_token:
        os.environ["HF_TOKEN"] = settings.hf_token
    ds = load_dataset(name, split=split, cache_dir=cache)
    local = Path(cache) / f"{name.replace('/', '__')}_{split}"
    local.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(local))
    return f"Loaded {len(ds)} samples to {local}"


@tool(
    name="download_file",
    description="Download a file via URL to a local path.",
)
def download_file(url: str, dest: str) -> str:
    import httpx

    settings = get_settings()
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, follow_redirects=True, timeout=600) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                f.write(chunk)
    return f"Downloaded {url} → {dest}"


@tool(
    name="dataset_stats",
    description="Compute basic statistics for a CSV or JSONL file (rows, columns, dtypes).",
)
def dataset_stats(path: str) -> str:
    import pandas as pd

    if path.endswith(".csv"):
        df = pd.read_csv(path, nrows=100_000)
    elif path.endswith(".jsonl") or path.endswith(".json"):
        df = pd.read_json(path, lines=path.endswith(".jsonl"))
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        raise ToolError(f"Unsupported format: {path}")
    return f"Shape: {df.shape}\nColumns: {list(df.columns)}\nDtypes:\n{df.dtypes}\nHead:\n{df.head(3).to_string()}"
