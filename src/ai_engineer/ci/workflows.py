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

"""Continuous improvement workflows — triggered by event hooks."""
from __future__ import annotations

import asyncio
from typing import Any

from ai_engineer.config import get_settings
from ai_engineer.core.llm import LLMClient
from ai_engineer.core.memory import MemorySystem
from ai_engineer.tools.sandbox import Sandbox
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


class CIWorkflow:
    """Runs periodic improvements: pattern extraction, fine-tune, skill induction."""

    def __init__(self, memory: MemorySystem, llm: LLMClient, sandbox: Sandbox) -> None:
        self.memory = memory
        self.llm = llm
        self.sandbox = sandbox
        self.settings = get_settings()

    async def extract_patterns(self, n: int = 20) -> int:
        """Analyze recent successful trajectories and extract reusable skills."""
        trajs = await self.memory.trajectories.successful(limit=n)
        added = 0
        for t in trajs:
            for step in t.steps:
                if not step.observation or "Traceback" in step.observation:
                    continue
                # Heuristic: if a step succeeded with code output, consider it a candidate skill
                if "def " in step.observation or "class " in step.observation:
                    try:
                        await self.memory.learn_skill(
                            name=f"pattern_{t.id}_{step.agent}",
                            description=step.action,
                            template=step.observation[:4000],
                            tags=[step.agent],
                        )
                        added += 1
                    except Exception as e:
                        logger.warning("ci.skill_extract_failed", error=str(e))
        logger.info("ci.patterns_extracted", count=added)
        return added

    async def maybe_finetune(self) -> dict[str, Any] | None:
        if not self.settings.enable_self_improvement:
            return None
        trajs = await self.memory.trajectories.successful(limit=self.settings.fine_tune_threshold)
        if len(trajs) < self.settings.fine_tune_threshold:
            return None
        from pathlib import Path

        out = Path(self.settings.artifacts_dir) / "training_data.jsonl"
        count = await self.memory.trajectories.export_training_data(out)
        # Run a LoRA fine-tune in the sandbox
        script = f"""
import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

base = '{self.settings.fine_tune_base_model}'
data_path = '{out}'
out_dir = '{self.settings.models_dir}/finetuned_self'

tokenizer = AutoTokenizer.from_pretrained(base)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map='auto', load_in_4bit=True)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=['q_proj','k_proj','v_proj','o_proj'], task_type='CAUSAL_LM'))

ds = load_dataset('json', data_files=str(data_path), split='train')

def fmt(ex):
    text = f\"### INSTRUCTION\\n{{ex['prompt']}}\\n\\n### RESPONSE\\n{{ex['response']}}\\n### END\"
    return tokenizer(text, truncation=True, max_length=2048, padding='max_length')

ds = ds.map(fmt, batched=False)
ds = ds.with_format('torch')

args = TrainingArguments(
    output_dir=out_dir,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    bf16=True,
    logging_steps=5,
    save_steps=200,
    report_to='none',
)
trainer = Trainer(model=model, args=args, train_dataset=ds, tokenizer=tokenizer)
trainer.train()
trainer.save_model(out_dir)
print('CI_TRAINING_DONE')
"""
        result = await self.sandbox.execute(script, timeout=14400, gpu=True)
        return {"success": result.exit_code == 0, "examples": count, "output_dir": f"{self.settings.models_dir}/finetuned_self"}

    async def run_cycle(self) -> dict[str, Any]:
        patterns = await self.extract_patterns()
        ft = await self.maybe_finetune()
        return {"patterns_extracted": patterns, "finetune": ft}
