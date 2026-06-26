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

"""Graph Neural Network trainer (PyTorch Geometric / DGL)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ai_engineer.ml.models.registry import ModelRegistry
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)

Architecture = Literal["gcn", "gat", "graphsage", "gin", "rgcn", "han"]


@dataclass
class GNNConfig:
    architecture: Architecture = "gcn"
    num_layers: int = 3
    hidden_dim: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-2
    output_dir: str = ""
    task: str = "node_classification"


@dataclass
class GNNResult:
    output_dir: str
    metrics: dict[str, float] = field(default_factory=dict)


class GNNTrainer:
    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()

    def train(self, config: GNNConfig, dataset_path: str, register_name: str | None = None) -> GNNResult:
        if not config.output_dir:
            config.output_dir = f"/tmp/gnn_{config.architecture}_{int(time.time())}"
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.output_dir, "train.py").write_text(f"""
import os
os.environ['HF_HOME'] = '/data/hf'
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
import pandas as pd, numpy as np, pickle

with open('{dataset_path}', 'rb') as f:
    data = pickle.load(f)

if isinstance(data, dict):
    edge_index = torch.tensor(data['edge_index'], dtype=torch.long)
    x = torch.tensor(data['x'], dtype=torch.float32)
    y = torch.tensor(data['y'], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)

arch = '{config.architecture}'
class GNN(nn.Module):
    def __init__(self, in_dim, hid, out, n_layers={config.num_layers}):
        super().__init__()
        if arch == 'gcn':
            self.layers = nn.ModuleList([GCNConv(in_dim if i==0 else hid, hid) for i in range(n_layers)])
        elif arch == 'gat':
            self.layers = nn.ModuleList([GATConv(in_dim if i==0 else hid, hid, heads=1) for i in range(n_layers)])
        elif arch == 'graphsage':
            self.layers = nn.ModuleList([SAGEConv(in_dim if i==0 else hid, hid) for i in range(n_layers)])
        elif arch == 'gin':
            self.layers = nn.ModuleList([GINConv(nn.Sequential(nn.Linear(in_dim if i==0 else hid, hid), nn.ReLU(), nn.Linear(hid, hid))) for i in range(n_layers)])
        self.head = nn.Linear(hid, out)
    def forward(self, x, ei):
        for l in self.layers: x = F.relu(l(x, ei))
        return self.head(x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GNN(data.x.shape[1], {config.hidden_dim}, int(data.y.max())+1).to(device)
opt = torch.optim.Adam(model.parameters(), lr={config.learning_rate}, weight_decay=5e-4)
data = data.to(device)

for epoch in range({config.num_epochs}):
    model.train()
    opt.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask]) if hasattr(data, 'train_mask') else F.cross_entropy(out, data.y)
    loss.backward(); opt.step()
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(1)
            if hasattr(data, 'test_mask'):
                acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
                print(f'epoch {{epoch}} loss {{loss.item():.4f}} test_acc {{acc:.4f}}')
            else:
                print(f'epoch {{epoch}} loss {{loss.item():.4f}}')
torch.save(model.state_dict(), '{config.output_dir}/gnn.pt')
print('GNN_COMPLETE')
""")
        Path(config.output_dir, "config.json").write_text(json.dumps(config.__dict__, indent=2, default=str))
        return GNNResult(output_dir=config.output_dir)
