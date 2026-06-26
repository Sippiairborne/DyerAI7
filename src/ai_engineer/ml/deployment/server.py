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

"""Server deployers: vLLM, TGI, Triton, BentoML, Ray Serve."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)

Framework = Literal["vllm", "tgi", "triton", "bentoml", "ray", "fastapi"]


@dataclass
class ServingConfig:
    framework: Framework = "vllm"
    model_path: str = ""
    port: int = 8000
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    tensor_parallel_size: int = 1
    quantization: str | None = None
    host: str = "0.0.0.0"


@dataclass
class ServingHandle:
    framework: str
    command: str
    container_id: str | None = None
    url: str | None = None
    pid: int | None = None


class ServingDeployer:
    def deploy(self, config: ServingConfig) -> ServingHandle:
        if config.framework == "vllm":
            return self._vllm(config)
        if config.framework == "tgi":
            return self._tgi(config)
        if config.framework == "triton":
            return self._triton(config)
        if config.framework == "bentoml":
            return self._bentoml(config)
        if config.framework == "ray":
            return self._ray(config)
        return self._fastapi(config)

    def _vllm(self, c: ServingConfig) -> ServingHandle:
        cmd = (
            f"docker run --gpus all -d -p {c.port}:8000 -v {c.model_path}:/model "
            f"vllm/vllm-openai:latest --model /model --port 8000 "
            f"--gpu-memory-utilization {c.gpu_memory_utilization} "
            f"--max-model-len {c.max_model_len} "
            f"--tensor-parallel-size {c.tensor_parallel_size}"
        )
        if c.quantization:
            cmd += f" --quantization {c.quantization}"
        try:
            cid = subprocess.check_output(cmd, shell=True, text=True).strip()
        except subprocess.CalledProcessError:
            cid = None
        return ServingHandle(framework="vllm", command=cmd, container_id=cid, url=f"http://localhost:{c.port}/v1")

    def _tgi(self, c: ServingConfig) -> ServingHandle:
        cmd = (
            f"docker run --gpus all -d -p {c.port}:80 -v {c.model_path}:/data "
            f"ghcr.io/huggingface/text-generation-inference:latest "
            f"--model-id /data --port 80 --max-input-length {c.max_model_len}"
        )
        try:
            cid = subprocess.check_output(cmd, shell=True, text=True).strip()
        except subprocess.CalledProcessError:
            cid = None
        return ServingHandle(framework="tgi", command=cmd, container_id=cid, url=f"http://localhost:{c.port}")

    def _triton(self, c: ServingConfig) -> ServingHandle:
        model_repo = Path(c.model_path) / "triton_repo"
        model_repo.mkdir(parents=True, exist_ok=True)
        cmd = (
            f"docker run --gpus all -d -p {c.port}:8000 -p {c.port + 1}:8001 -p {c.port + 2}:8002 "
            f"-v {model_repo}:/models nvcr.io/nvidia/tritonserver:latest tritonserver --model-repository=/models"
        )
        try:
            cid = subprocess.check_output(cmd, shell=True, text=True).strip()
        except subprocess.CalledProcessError:
            cid = None
        return ServingHandle(framework="triton", command=cmd, container_id=cid, url=f"http://localhost:{c.port}")

    def _bentoml(self, c: ServingConfig) -> ServingHandle:
        # Caller would have built a Bento
        try:
            proc = subprocess.Popen(["bentoml", "serve", c.model_path, "--port", str(c.port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return ServingHandle(framework="bentoml", command="bentoml serve", pid=proc.pid, url=f"http://localhost:{c.port}")
        except Exception as e:
            return ServingHandle(framework="bentoml", command=str(e))

    def _ray(self, c: ServingConfig) -> ServingHandle:
        from ai_engineer.ml.deployment._ray_serve import serve
        handle = serve(c.model_path, port=c.port)
        return ServingHandle(framework="ray", command="ray serve deploy", url=f"http://localhost:{c.port}")

    def _fastapi(self, c: ServingConfig) -> ServingHandle:
        # Generate a serving FastAPI app
        app_path = Path(c.model_path).parent / "serve.py"
        app_path.write_text(f"""
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()
tok = AutoTokenizer.from_pretrained('{c.model_path}')
model = AutoModelForCausalLM.from_pretrained('{c.model_path}', torch_dtype=torch.bfloat16, device_map='auto')

class Req(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7

@app.post('/generate')
def generate(req: Req):
    inp = tok(req.prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=req.max_new_tokens, temperature=req.temperature, do_sample=req.temperature > 0)
    return {{'text': tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)}}

@app.get('/health')
def health():
    return {{'status': 'ok'}}
""")
        try:
            proc = subprocess.Popen(["uvicorn", "serve:app", "--host", c.host, "--port", str(c.port)])
            return ServingHandle(framework="fastapi", command="uvicorn", pid=proc.pid, url=f"http://localhost:{c.port}")
        except Exception as e:
            return ServingHandle(framework="fastapi", command=str(e))
