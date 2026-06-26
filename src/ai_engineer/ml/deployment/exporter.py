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

"""Model export to ONNX, TorchScript, safetensors, CoreML, TFLite."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

Format = Literal["onnx", "torchscript", "safetensors", "coreml", "tflite"]


@dataclass
class ExportResult:
    format: str
    output_path: str
    size_mb: float


class ModelExporter:
    def export(self, model, output_path: str, fmt: Format = "safetensors", example_input: torch.Tensor | None = None) -> ExportResult:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if fmt == "safetensors":
            from safetensors.torch import save_file
            save_file(model.state_dict(), output_path)
        elif fmt == "onnx":
            torch.onnx.export(model, example_input, output_path, input_names=["input"], output_names=["output"], dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
        elif fmt == "torchscript":
            ts = torch.jit.trace(model, example_input)
            ts.save(output_path)
        elif fmt == "coreml":
            import coremltools as ct
            cm = ct.convert(model, inputs=[ct.TensorType(name="input", shape=example_input.shape)])
            cm.save(output_path)
        elif fmt == "tflite":
            import tensorflow as tf
            onnx_path = output_path.replace(".tflite", ".onnx")
            torch.onnx.export(model, example_input, onnx_path, opset_version=13)
            converter = tf.lite.TFLiteConverter.from_onnx(onnx_path)
            tflite_model = converter.convert()
            Path(output_path).write_bytes(tflite_model)
        return ExportResult(format=fmt, output_path=output_path, size_mb=Path(output_path).stat().st_size / 1024 / 1024)
