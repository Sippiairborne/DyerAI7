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

"""Distributed training launchers: DDP, FSDP, DeepSpeed."""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ai_engineer.utils.errors import AIEngineerError
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)

Strategy = Literal["ddp", "fsdp", "deepspeed"]


@dataclass
class DistributedConfig:
    strategy: Strategy = "ddp"
    n_gpus: int = 1
    n_nodes: int = 1
    node_rank: int = 0
    master_addr: str = "localhost"
    master_port: int = 29500
    mixed_precision: str = "bf16"
    fsdp_sharding: str = "full_shard"  # full_shard | shard_grad_op | no_shard
    fsdp_cpu_offload: bool = False
    fsdp_backward_prefetch: str = "backward_pre"
    deepspeed_config: dict | None = None
    output_dir: str = ""


class DistributedLauncher:
    """Launch distributed training jobs."""

    def build_command(self, config: DistributedConfig, train_script: str, script_args: list[str] | None = None) -> list[str]:
        script_args = script_args or []
        env = os.environ.copy()
        env["MASTER_ADDR"] = config.master_addr
        env["MASTER_PORT"] = str(config.master_port)
        env["WORLD_SIZE"] = str(config.n_gpus * config.n_nodes)
        env["RANK"] = str(config.node_rank * config.n_gpus)
        env["LOCAL_RANK"] = "0"  # torchrun handles this

        if config.strategy == "deepspeed":
            return self._deepspeed_cmd(config, train_script, script_args, env)
        if config.strategy == "fsdp":
            return self._torchrun_cmd(config, train_script, script_args, env, fsdp=True)
        return self._torchrun_cmd(config, train_script, script_args, env)

    def _torchrun_cmd(self, c: DistributedConfig, script: str, args: list[str], env: dict, fsdp: bool = False) -> list[str]:
        cmd = [
            "torchrun",
            f"--nproc_per_node={c.n_gpus}",
            f"--nnodes={c.n_nodes}",
            f"--node_rank={c.node_rank}",
            f"--master_addr={c.master_addr}",
            f"--master_port={c.master_port}",
        ]
        if fsdp:
            cmd += ["--standalone"]
        cmd += [script, *args]
        if fsdp:
            cmd += [f"--strategy=fsdp", f"--sharding={c.fsdp_sharding}", f"--cpu_offload={str(c.fsdp_cpu_offload).lower()}"]
        return cmd

    def _deepspeed_cmd(self, c: DistributedConfig, script: str, args: list[str], env: dict) -> list[str]:
        ds_config = c.deepspeed_config or {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "bf16": {"enabled": c.mixed_precision == "bf16"},
            "fp16": {"enabled": c.mixed_precision == "fp16"},
            "zero_optimization": {"stage": 2, "allgather_partitions": True, "allgather_bucket_size": 5e8, "reduce_scatter": True, "reduce_bucket_size": 5e8, "contiguous_gradients": True, "round_robin_gradients": True},
            "optimizer": {"type": "AdamW", "params": {"lr": "auto", "weight_decay": "auto"}},
        }
        ds_path = Path(c.output_dir or "/tmp/deepspeed") / "ds_config.json"
        ds_path.parent.mkdir(parents=True, exist_ok=True)
        ds_path.write_text(json.dumps(ds_config, indent=2))
        return [
            "deepspeed", f"--num_gpus={c.n_gpus}", f"--num_nodes={c.n_nodes}",
            f"--master_addr={c.master_addr}", f"--master_port={c.master_port}",
            f"--hostfile=/dev/null",
            script, f"--deepspeed={ds_path}", *args,
        ]

    def generate_launcher_script(self, config: DistributedConfig, train_script: str, output_path: str) -> str:
        cmd = self.build_command(config, train_script)
        script = "#!/usr/bin/env bash\nset -euo pipefail\n" + " ".join(cmd) + "\n"
        Path(output_path).write_text(script)
        os.chmod(output_path, 0o755)
        return script

    def write_fsdp_entrypoint(self, train_script: str, output_dir: str) -> str:
        """Generate an FSDP-compatible entrypoint wrapping the user's training script."""
        ep = f"""
import os, sys, torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, BackwardPrefetch, ShardingStrategy, CPUOffload

def setup():
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.distributed.init_process_group('nccl')
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank

def wrap_model(model, args):
    mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    strategy = {{
        'full_shard': ShardingStrategy.FULL_SHARD,
        'shard_grad_op': ShardingStrategy.SHARD_GRAD_OP,
        'no_shard': ShardingStrategy.NO_SHARD,
    }}[args.sharding]
    return FSDP(model, sharding_strategy=strategy, mixed_precision=mp,
        cpu_offload=CPUOffload(offload_params=args.cpu_offload),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=int(os.environ.get('LOCAL_RANK', 0)))

sys.path.insert(0, os.path.dirname('{train_script}'))
exec(open('{train_script}').read(), {{'__name__': '__main__', 'setup': setup, 'wrap_model': wrap_model}})
"""
        ep_path = Path(output_dir) / "fsdp_entrypoint.py"
        ep_path.write_text(ep)
        return str(ep_path)
