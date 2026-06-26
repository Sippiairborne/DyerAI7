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

"""LLM fine-tuning: SFT, DPO, ORPO, KTO, PPO."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ai_engineer.ml.models.registry import ModelRegistry
from ai_engineer.utils.errors import AIEngineerError
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)

Method = Literal["sft", "dpo", "orpo", "kto", "ppo"]
Backend = Literal["transformers", "unsloth", "axolotl", "trl"]


@dataclass
class LLMTrainingConfig:
    method: Method = "sft"
    backend: Backend = "unsloth"
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    dataset_path: str = ""
    output_dir: str = ""
    num_epochs: int = 3
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler: str = "cosine"
    use_lora: bool = True
    use_qlora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    bf16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    max_steps: int = -1
    seed: int = 42
    beta: float = 0.1  # for DPO/ORPO/KTO
    max_prompt_length: int = 1024
    # Generation
    temperature: float = 0.7
    top_p: float = 0.9
    # Reward model (for PPO)
    reward_model_path: str | None = None


@dataclass
class LLMTrainingResult:
    output_dir: str
    final_loss: float
    best_metric: float
    training_time_s: float
    metrics: dict[str, float] = field(default_factory=dict)
    registered_version: str | None = None


class LLMTrainer:
    """Comprehensive LLM fine-tuning with multiple methods and backends."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()

    def train(self, config: LLMTrainingConfig, register_name: str | None = None) -> LLMTrainingResult:
        if not config.output_dir:
            config.output_dir = f"/tmp/llm_{config.method}_{int(time.time())}"
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        start = time.time()

        if config.backend == "unsloth":
            script = self._build_unsloth_script(config)
        elif config.backend == "axolotl":
            script = self._build_axolotl_config(config)
        else:
            script = self._build_trl_script(config)

        script_path = Path(config.output_dir) / "train.py"
        script_path.write_text(script)
        cfg_path = Path(config.output_dir) / "config.json"
        cfg_path.write_text(json.dumps(config.__dict__, indent=2, default=str))

        # Training itself runs in the sandbox by the orchestrator.
        # We return a recipe that the orchestrator can execute.
        return LLMTrainingResult(
            output_dir=config.output_dir,
            final_loss=0.0,
            best_metric=0.0,
            training_time_s=0.0,
            metrics={"script_path": str(script_path)},
        )

    def _build_unsloth_script(self, c: LLMTrainingConfig) -> str:
        if c.method == "sft":
            return f"""
import os
os.environ['HF_HOME'] = '/data/hf'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='{c.model_name}',
    max_seq_length={c.max_seq_length},
    dtype=None,
    load_in_4bit={c.use_qlora},
)
model = FastLanguageModel.get_peft_model(
    model,
    r={c.lora_r},
    target_modules={c.lora_target_modules!r},
    lora_alpha={c.lora_alpha},
    lora_dropout={c.lora_dropout},
    use_gradient_checkpointing='unsloth',
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset('json', data_files='{c.dataset_path}', split='train')

def fmt(ex):
    if 'messages' in ex:
        text = tokenizer.apply_chat_template(ex['messages'], tokenize=False)
    elif 'prompt' in ex and 'response' in ex:
        text = f"### INSTRUCT\\n{{ex['prompt']}}\\n\\n### RESPONSE\\n{{ex['response']}}"
    else:
        text = ex.get('text', str(ex))
    return {{'text': text}}

dataset = dataset.map(fmt, remove_columns=dataset.column_names)

args = SFTConfig(
    output_dir='{c.output_dir}',
    num_train_epochs={c.num_epochs},
    per_device_train_batch_size={c.per_device_batch_size},
    gradient_accumulation_steps={c.gradient_accumulation_steps},
    learning_rate={c.learning_rate},
    max_seq_length={c.max_seq_length},
    warmup_ratio={c.warmup_ratio},
    lr_scheduler_type='{c.lr_scheduler}',
    weight_decay={c.weight_decay},
    bf16={c.bf16},
    logging_steps={c.logging_steps},
    save_steps={c.save_steps},
    max_steps={c.max_steps},
    seed={c.seed},
    report_to='none',
)
trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer)
trainer.train()
trainer.save_model('{c.output_dir}')
tokenizer.save_pretrained('{c.output_dir}')
print('LLM_SFT_COMPLETE')
"""
        if c.method == "dpo":
            return f"""
import os
os.environ['HF_HOME'] = '/data/hf'
from unsloth import FastLanguageModel, PatchDPOTrainer
PatchDPOTrainer()
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='{c.model_name}',
    max_seq_length={c.max_seq_length},
    load_in_4bit={c.use_qlora},
)
model = FastLanguageModel.get_peft_model(
    model, r={c.lora_r}, lora_alpha={c.lora_alpha}, target_modules={c.lora_target_modules!r},
    lora_dropout={c.lora_dropout}, use_gradient_checkpointing='unsloth',
)

dataset = load_dataset('json', data_files='{c.dataset_path}', split='train')
args = DPOConfig(
    output_dir='{c.output_dir}',
    num_train_epochs={c.num_epochs},
    per_device_train_batch_size={c.per_device_batch_size},
    gradient_accumulation_steps={c.gradient_accumulation_steps},
    learning_rate={c.learning_rate},
    max_length={c.max_seq_length},
    max_prompt_length={c.max_prompt_length},
    beta={c.beta},
    bf16={c.bf16},
    logging_steps={c.logging_steps},
    save_steps={c.save_steps},
    seed={c.seed},
    report_to='none',
)
trainer = DPOTrainer(model=model, ref_model=None, args=args, train_dataset=dataset, tokenizer=tokenizer)
trainer.train()
trainer.save_model('{c.output_dir}')
print('LLM_DPO_COMPLETE')
"""
        if c.method == "orpo":
            return f"""
import os
os.environ['HF_HOME'] = '/data/hf'
from unsloth import FastLanguageModel, PatchDPOTrainer
PatchDPOTrainer()
from datasets import load_dataset
from trl import ORPOTrainer, ORPOConfig
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='{c.model_name}', max_seq_length={c.max_seq_length}, load_in_4bit={c.use_qlora},
)
model = FastLanguageModel.get_peft_model(model, r={c.lora_r}, lora_alpha={c.lora_alpha}, target_modules={c.lora_target_modules!r})
dataset = load_dataset('json', data_files='{c.dataset_path}', split='train')
args = ORPOConfig(
    output_dir='{c.output_dir}', num_train_epochs={c.num_epochs},
    per_device_train_batch_size={c.per_device_batch_size},
    gradient_accumulation_steps={c.gradient_accumulation_steps},
    learning_rate={c.learning_rate}, max_length={c.max_seq_length},
    max_prompt_length={c.max_prompt_length}, beta={c.beta},
    bf16={c.bf16}, report_to='none', seed={c.seed},
)
trainer = ORPOTrainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer)
trainer.train()
trainer.save_model('{c.output_dir}')
print('LLM_ORPO_COMPLETE')
"""
        if c.method == "kto":
            return f"""
import os
os.environ['HF_HOME'] = '/data/hf'
from unsloth import FastLanguageModel, PatchDPOTrainer
PatchDPOTrainer()
from datasets import load_dataset
from trl import KTOTrainer, KTOConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='{c.model_name}', max_seq_length={c.max_seq_length}, load_in_4bit={c.use_qlora},
)
model = FastLanguageModel.get_peft_model(model, r={c.lora_r}, lora_alpha={c.lora_alpha}, target_modules={c.lora_target_modules!r})
dataset = load_dataset('json', data_files='{c.dataset_path}', split='train')
args = KTOConfig(
    output_dir='{c.output_dir}', num_train_epochs={c.num_epochs},
    per_device_train_batch_size={c.per_device_batch_size}, learning_rate={c.learning_rate},
    max_length={c.max_seq_length}, beta={c.beta}, bf16={c.bf16}, report_to='none', seed={c.seed},
)
trainer = KTOTrainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer)
trainer.train()
trainer.save_model('{c.output_dir}')
print('LLM_KTO_COMPLETE')
"""
        # ppo
        return f"""
import os
os.environ['HF_HOME'] = '/data/hf'
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='{c.model_name}', max_seq_length={c.max_seq_length}, load_in_4bit={c.use_qlora},
)
model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

ref_model, _ = FastLanguageModel.from_pretrained(
    model_name='{c.model_name}', max_seq_length={c.max_seq_length}, load_in_4bit={c.use_qlora},
)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ref_model)

dataset = load_dataset('json', data_files='{c.dataset_path}', split='train')
args = PPOConfig(
    output_dir='{c.output_dir}', num_train_epochs={c.num_epochs},
    per_device_train_batch_size={c.per_device_batch_size}, learning_rate={c.learning_rate},
    bf16={c.bf16}, report_to='none', seed={c.seed},
)
trainer = PPOTrainer(args=args, model=model, ref_model=ref_model, tokenizer=tokenizer)
trainer.train()
trainer.save_model('{c.output_dir}')
print('LLM_PPO_COMPLETE')
"""

    def _build_trl_script(self, c: LLMTrainingConfig) -> str:
        # Fallback to TRL directly
        return self._build_unsloth_script(c).replace("from unsloth import", "from transformers import").replace(
            "FastLanguageModel.from_pretrained", "AutoModelForCausalLM.from_pretrained"
        )

    def _build_axolotl_config(self, c: LLMTrainingConfig) -> str:
        return f"""
# Axolotl YAML config (auto-generated)
base_model: {c.model_name}
model_type: LlamaForCausalLM
tokenizer_type: AutoTokenizer
load_in_4bit: {str(c.use_qlora).lower()}
adapter: qlora
lora_r: {c.lora_r}
lora_alpha: {c.lora_alpha}
lora_dropout: {c.lora_dropout}
lora_target_modules: {' '.join(c.lora_target_modules)}
datasets:
  - path: {c.dataset_path}
    type: alpaca
output_dir: {c.output_dir}
sequence_len: {c.max_seq_length}
sample_packing: true
pad_to_sequence_len: true
bf16: {str(c.bf16).lower()}
learning_rate: {c.learning_rate}
num_train_epochs: {c.num_epochs}
per_device_train_batch_size: {c.per_device_batch_size}
gradient_accumulation_steps: {c.gradient_accumulation_steps}
warmup_ratio: {c.warmup_ratio}
lr_scheduler_type: {c.lr_scheduler}
weight_decay: {c.weight_decay}
gradient_checkpointing: {str(c.gradient_checkpointing).lower()}
seed: {c.seed}
"""
