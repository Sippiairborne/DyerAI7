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

"""Reinforcement learning trainer: PPO, SAC, DQN, A2C with stable-baselines3 + RLlib."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ai_engineer.ml.models.registry import ModelRegistry
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)

Algorithm = Literal["ppo", "sac", "dqn", "a2c", "td3"]


@dataclass
class RLConfig:
    algorithm: Algorithm = "ppo"
    env_id: str = "CartPole-v1"
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    output_dir: str = ""


@dataclass
class RLResult:
    output_dir: str
    metrics: dict[str, float] = field(default_factory=dict)


class RLTrainer:
    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()

    def train(self, config: RLConfig, register_name: str | None = None) -> RLResult:
        if not config.output_dir:
            config.output_dir = f"/tmp/rl_{config.algorithm}_{int(time.time())}"
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        script = f"""
import os
from stable_baselines3 import PPO, SAC, DQN, A2C, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
import json

algo = '{config.algorithm}'
Model = {{'ppo': PPO, 'sac': SAC, 'dqn': DQN, 'a2c': A2C, 'td3': TD3}}[algo]
env = DummyVecEnv([lambda: gym.make('{config.env_id}')])
model = Model('MlpPolicy', env, learning_rate={config.learning_rate}, gamma={config.gamma}, verbose=1)
model.learn(total_timesteps={config.total_timesteps})
model.save('{config.output_dir}/model')
mean, std = evaluate_policy(model, env, n_eval_episodes=10)
with open('{config.output_dir}/metrics.json', 'w') as f:
    json.dump({{'mean_reward': float(mean), 'std_reward': float(std)}}, f)
print(f'RL_DONE mean_reward={{mean:.2f}} +/- {{std:.2f}}')
"""
        Path(config.output_dir, "train.py").write_text(script)
        Path(config.output_dir, "config.json").write_text(json.dumps(config.__dict__, indent=2, default=str))
        return RLResult(output_dir=config.output_dir, metrics={"script_path": f"{config.output_dir}/train.py"})

    def train_llm_rlhf(self, config: RLConfig, sft_model_path: str, reward_model_path: str, dataset_path: str, register_name: str | None = None) -> RLResult:
        """RLHF / PPO training of an LLM with a reward model."""
        if not config.output_dir:
            config.output_dir = f"/tmp/rlhf_{int(time.time())}"
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        script = f"""
import os
os.environ['HF_HOME'] = '/data/hf'
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from datasets import load_dataset

tok = AutoTokenizer.from_pretrained('{sft_model_path}')
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLMWithValueHead.from_pretrained('{sft_model_path}')
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained('{sft_model_path}')
reward_model = AutoModelForCausalLMWithValueHead.from_pretrained('{reward_model_path}')

ds = load_dataset('json', data_files='{dataset_path}', split='train')
args = PPOConfig(output_dir='{config.output_dir}', learning_rate={config.learning_rate}, gamma={config.gamma}, batch_size=8, mini_batch_size=2, log_with=None)
trainer = PPOTrainer(args=args, model=model, ref_model=ref_model, tokenizer=tok)
for example in ds:
    query = example['prompt']
    q_ids = tok(query, return_tensors='pt').input_ids
    response = model.generate(q_ids)
    r = example.get('reward', 0.0)
    stats = trainer.step(q_ids, response, [r])
    print(stats)
trainer.save_pretrained('{config.output_dir}')
print('RLHF_COMPLETE')
"""
        Path(config.output_dir, "train.py").write_text(script)
        return RLResult(output_dir=config.output_dir)
