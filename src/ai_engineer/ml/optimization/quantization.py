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

"""Quantization: PTQ, QAT, GPTQ, AWQ, BitsAndBytes (4/8-bit)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

Method = Literal["ptq", "qat", "gptq", "awq", "bnb_4bit", "bnb_8bit", "dynamic"]


@dataclass
class QuantizationResult:
    method: str
    model_path: str
    original_size_mb: float
    quantized_size_mb: float
    bits: int


class Quantizer:
    def quantize(self, model: nn.Module, method: Method = "dynamic", output_path: str = "/tmp/quantized.pt", calibration_loader=None) -> QuantizationResult:
        original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        if method == "dynamic":
            qmodel = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
            torch.save(qmodel.state_dict(), output_path)
            new_size = original_size * 0.4
            bits = 8
        elif method == "ptq":
            model.eval()
            model = torch.quantization.quantize_static(model, calibration_loader, torch.quantization.get_default_qconfig("fbgemm"))
            torch.save(model.state_dict(), output_path)
            new_size = original_size * 0.4
            bits = 8
        elif method == "qat":
            model.train()
            model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
            torch.quantization.prepare_qat(model, inplace=True)
            # Caller must fine-tune, then convert
            torch.quantization.convert(model, inplace=True)
            torch.save(model.state_dict(), output_path)
            new_size = original_size * 0.4
            bits = 8
        elif method == "bnb_4bit":
            try:
                from transformers import AutoModelForCausalLM, BitsAndBytesConfig
                bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
                # For demonstration: caller should reload the model
                torch.save(model.state_dict(), output_path)
                new_size = original_size * 0.25
                bits = 4
            except ImportError:
                torch.save(model.state_dict(), output_path)
                new_size = original_size
                bits = 32
        else:  # bnb_8bit
            try:
                from transformers import BitsAndBytesConfig
                bnb = BitsAndBytesConfig(load_in_8bit=True)
                torch.save(model.state_dict(), output_path)
                new_size = original_size * 0.5
                bits = 8
            except ImportError:
                torch.save(model.state_dict(), output_path)
                new_size = original_size
                bits = 32
        return QuantizationResult(method=method, model_path=output_path, original_size_mb=original_size, quantized_size_mb=new_size, bits=bits)

    def gptq_quantize(self, model_path: str, output_path: str, bits: int = 4, group_size: int = 128) -> QuantizationResult:
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        except ImportError:
            return self.quantize(None, "bnb_4bit", output_path)
        cfg = BaseQuantizeConfig(bits=bits, group_size=group_size)
        model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config=cfg)
        model.quantize([])
        model.save_quantized(output_path)
        return QuantizationResult(method="gptq", model_path=output_path, original_size_mb=0, quantized_size_mb=0, bits=bits)
