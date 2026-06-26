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

"""Optimizer factory."""
from __future__ import annotations

import torch


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    if name == "adagrad":
        return torch.optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    if name == "lion":
        try:
            from lion_pytorch import Lion
            return Lion(params, lr=lr, weight_decay=weight_decay)
        except ImportError:
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adafactor":
        try:
            from transformers.optimization import Adafactor
            return Adafactor(params, lr=lr, weight_decay=weight_decay, scale_parameter=False, relative_step=False)
        except ImportError:
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sophia":
        try:
            from sophia import SophiaG
            return SophiaG(params, lr=lr, weight_decay=weight_decay)
        except ImportError:
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "lamb":
        try:
            from torch_optimizer import Lamb
            return Lamb(params, lr=lr, weight_decay=weight_decay)
        except ImportError:
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "ranger":
        try:
            from torch_optimizer import Ranger
            return Ranger(params, lr=lr, weight_decay=weight_decay)
        except ImportError:
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")
