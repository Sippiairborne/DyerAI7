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

"""Training utilities (Unsloth, HF Trainer)."""
from __future__ import annotations

import json
from pathlib import Path

from ai_engineer.config import get_settings
from ai_engineer.tools.registry import ToolRegistry, tool
from ai_engineer.utils.errors import ToolError

_registry = ToolRegistry()


@tool(
    name="start_finetune",
    description="Launch a HuggingFace fine-tuning job. Returns the run path on disk. Use this tool from inside the sandbox.",
)
def start_finetune(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_seq_length: int = 2048,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    use_4bit: bool = True,
) -> str:
    """Build a training script and write it to disk. The orchestrator executes it in the sandbox."""
    settings = get_settings()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    script_path = Path(output_dir) / "train.py"
    script = f"""
import os
os.environ['HF_HOME'] = '{settings.hf_home}'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_name = '{model_name}'
dataset_path = '{dataset_path}'
output_dir = '{output_dir}'
max_seq_length = {max_seq_length}

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    load_in_4bit={use_4bit},
)
model.config.use_cache = False

dataset = load_from_disk(dataset_path)

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, max_length=max_seq_length, padding='max_length')

dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
dataset = dataset.with_format('torch')

if {use_lora}:
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r={lora_r},
        lora_alpha={lora_alpha},
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs={num_epochs},
    per_device_train_batch_size={batch_size},
    gradient_accumulation_steps=4,
    learning_rate={learning_rate},
    bf16=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=3,
    warmup_ratio=0.03,
    lr_scheduler_type='cosine',
    report_to='none',
)

trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer)
trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print('TRAINING_COMPLETE')
"""
    script_path.write_text(script)
    cfg_path = Path(output_dir) / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "dataset_path": dataset_path,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "use_lora": use_lora,
            },
            indent=2,
        )
    )
    return f"Training script written to {script_path}. Execute with shell tool."
