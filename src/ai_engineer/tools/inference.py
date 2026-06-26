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

"""Inference utilities — vLLM, transformers."""
from __future__ import annotations

from pathlib import Path

from ai_engineer.config import get_settings
from ai_engineer.tools.registry import ToolRegistry, tool
from ai_engineer.utils.errors import ToolError

_registry = ToolRegistry()


@tool(
    name="generate",
    description="Run text generation with a HF model. Returns the generated text.",
)
def generate(
    model_path: str,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


@tool(
    name="build_vllm_server_config",
    description="Generate a vLLM serving config and Docker command for a model.",
)
def build_vllm_server_config(model_path: str, port: int = 8000) -> str:
    return (
        f"docker run --gpus all -p {port}:8000 -v {model_path}:/model "
        f"vllm/vllm-openai:latest --model /model --served-model-name {Path(model_path).name} --port 8000"
    )
