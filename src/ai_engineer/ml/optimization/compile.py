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

"""Model compilation: torch.compile, ONNX, TensorRT."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

Format = Literal["onnx", "torchscript", "tensorrt", "torch_compile"]


@dataclass
class CompileResult:
    format: str
    output_path: str
    original_size_mb: float
    compiled_size_mb: float
    speedup_estimate: float


class ModelCompiler:
    def compile(self, model, example_input, output_path: str, fmt: Format = "onnx", opset: int = 17) -> CompileResult:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        if fmt == "onnx":
            torch.onnx.export(model, example_input, output_path, opset_version=opset, input_names=["input"], output_names=["output"], dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
            size = Path(output_path).stat().st_size / 1024 / 1024
            return CompileResult("onnx", output_path, original_size, size, 1.0)
        if fmt == "torchscript":
            ts = torch.jit.trace(model, example_input)
            ts.save(output_path)
            size = Path(output_path).stat().st_size / 1024 / 1024
            return CompileResult("torchscript", output_path, original_size, size, 1.2)
        if fmt == "torch_compile":
            compiled = torch.compile(model, mode="max-autotune", fullgraph=True)
            torch.save(compiled.state_dict(), output_path)
            return CompileResult("torch_compile", output_path, original_size, original_size, 1.5)
        if fmt == "tensorrt":
            try:
                import torch_tensorrt
                trt = torch_tensorrt.compile(model, inputs=[example_input], enabled_precisions={torch.float16})
                torch.jit.save(trt, output_path)
                size = Path(output_path).stat().st_size / 1024 / 1024
                return CompileResult("tensorrt", output_path, original_size, size, 3.0)
            except ImportError:
                return self.compile(model, example_input, output_path, "onnx", opset)
        raise ValueError(f"Unknown format: {fmt}")
