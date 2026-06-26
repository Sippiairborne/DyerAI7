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

"""PEFT: LoRA, QLoRA, Adapters, IA3, DoRA, LongLoRA, LoftQ."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ai_engineer.utils.errors import AIEngineerError
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)

Method = Literal["lora", "qlora", "adalora", "ia3", "dora", "longlora", "loftq", "prefix", "prompt"]


@dataclass
class PEFTConfig:
    method: Method = "lora"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] | None = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    # QLoRA
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    # DoRA
    use_dora: bool = False
    # IA3
    feedforward_modules: list[str] | None = None
    # LoftQ
    loftq_iter: int = 1
    # LongLoRA
    use_longlora: bool = False
    # Prefix
    num_virtual_tokens: int = 20


class PEFTFactory:
    @staticmethod
    def build(model, config: PEFTConfig) -> tuple:
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError as e:
            raise AIEngineerError("Install peft: pip install peft") from e

        target_modules = config.target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        if config.method in ("lora", "qlora", "adalora", "dora", "loftq"):
            from peft import LoraConfig
            lora_kwargs = {
                "r": config.r,
                "lora_alpha": config.alpha,
                "lora_dropout": config.dropout,
                "target_modules": target_modules,
                "bias": config.bias,
                "task_type": config.task_type,
            }
            if config.method == "dora" or config.use_dora:
                lora_kwargs["use_dora"] = True
            peft_cfg = LoraConfig(**lora_kwargs)
            if config.method == "qlora" and config.use_4bit:
                from transformers import BitsAndBytesConfig
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=config.bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=getattr(__import__("torch"), config.bnb_4bit_compute_dtype),
                )
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
            return get_peft_model(model, peft_cfg), model
        if config.method == "ia3":
            from peft import IA3Config
            cfg = IA3Config(target_modules=target_modules, feedforward_modules=config.feedforward_modules or ["down_proj"], task_type=config.task_type)
            return get_peft_model(model, cfg), model
        if config.method == "prefix":
            from peft import PrefixTuningConfig
            cfg = PrefixTuningConfig(num_virtual_tokens=config.num_virtual_tokens, task_type=config.task_type)
            return get_peft_model(model, cfg), model
        if config.method == "prompt":
            from peft import PromptTuningConfig, PromptTuningInit
            cfg = PromptTuningConfig(num_virtual_tokens=config.num_virtual_tokens, prompt_tuning_init=PromptTuningInit.RANDOM, task_type=config.task_type)
            return get_peft_model(model, cfg), model
        raise AIEngineerError(f"Unknown PEFT method: {config.method}")
