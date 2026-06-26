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

"""Tabular deep learning: TabNet, FT-Transformer, SAINT, NODE, TabTransformer."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ai_engineer.ml.models.registry import ModelRegistry
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)

Architecture = Literal["tabnet", "ft_transformer", "saint", "tab_transformer", "node"]


@dataclass
class TabularDLConfig:
    architecture: Architecture = "tabnet"
    num_epochs: int = 50
    batch_size: int = 1024
    learning_rate: float = 2e-2
    output_dir: str = ""
    cat_cols: list[str] = field(default_factory=list)
    num_cols: list[str] = field(default_factory=list)
    target: str = ""
    task: str = "classification"
    num_classes: int = 1


@dataclass
class TabularDLResult:
    output_dir: str
    metrics: dict[str, float] = field(default_factory=dict)


class TabularDLTrainer:
    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()

    def train(self, config: TabularDLConfig, dataset_path: str, register_name: str | None = None) -> TabularDLResult:
        if not config.output_dir:
            config.output_dir = f"/tmp/tabdl_{config.architecture}_{int(time.time())}"
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        if config.architecture == "tabnet":
            script = self._tabnet_script(config, dataset_path)
        elif config.architecture == "ft_transformer":
            script = self._ftt_script(config, dataset_path)
        elif config.architecture == "saint":
            script = self._saint_script(config, dataset_path)
        else:
            script = self._tab_transformer_script(config, dataset_path)
        Path(config.output_dir, "train.py").write_text(script)
        Path(config.output_dir, "config.json").write_text(json.dumps(config.__dict__, indent=2, default=str))
        return TabularDLResult(output_dir=config.output_dir, metrics={"script_path": f"{config.output_dir}/train.py"})

    def _tabnet_script(self, c: TabularDLConfig, dp: str) -> str:
        return f"""
import os
os.environ['HF_HOME'] = '/data/hf'
import pandas as pd
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from sklearn.preprocessing import StandardScaler

df = pd.read_parquet('{dp}') if '{dp}'.endswith('.parquet') else pd.read_csv('{dp}')
cat = {c.cat_cols!r}
num = {c.num_cols!r}
target = '{c.target}'
y = df[target].values
X_num = df[num].fillna(0).values.astype(np.float32)
X_cat = pd.get_dummies(df[cat].astype(str), dummy_na=True).values.astype(np.float32) if cat else np.zeros((len(df), 0), dtype=np.float32)
X = np.hstack([X_num, X_cat])
sc = StandardScaler(); X[:, :len(num)] = sc.fit_transform(X[:, :len(num)])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
Model = TabNetClassifier if '{c.task}' == 'classification' else TabNetRegressor
model = Model(
    n_d=32, n_a=32, n_steps=3, gamma=1.3, n_independent=2, n_shared=2,
    cat_idxs=list(range(len(num), len(num) + X_cat.shape[1])),
    cat_dims=[2]*X_cat.shape[1], cat_emb_dim=4,
    optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr={c.learning_rate}),
    scheduler_params={{'step_size': 20, 'gamma': 0.9}}, scheduler_fn=torch.optim.lr_scheduler.StepLR,
    device_name=device, verbose=10,
)
model.fit(X, y, max_epochs={c.num_epochs}, patience=15, batch_size={c.batch_size}, virtual_batch_size=128, num_workers=4)
model.save_model('{c.output_dir}/tabnet')
print('TABNET_COMPLETE')
"""

    def _ftt_script(self, c: TabularDLConfig, dp: str) -> str:
        return f"""
import os
os.environ['HF_HOME'] = '/data/hf'
import pandas as pd, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from rtdl import FTTransformer

df = pd.read_parquet('{dp}') if '{dp}'.endswith('.parquet') else pd.read_csv('{dp}')
cat = {c.cat_cols!r}
num = {c.num_cols!r}
target = '{c.target}'
y = df[target].values.astype(np.int64 if '{c.task}' == 'classification' else np.float32)
X_num = df[num].fillna(0).values.astype(np.float32)
cat_card = [int(df[c].astype(str).nunique()) + 1 for c in cat]
X_cat = np.stack([df[c].astype('category').cat.codes.values + 1 for c in cat], axis=1).astype(np.int64) if cat else np.zeros((len(df), 0), dtype=np.int64)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = FTTransformer(
    n_cont_features=len(num), cat_cardinalities=cat_card, d_out={c.num_classes},
    n_blocks=3, attention_dropout=0.2, ffn_dropout=0.2, residual_dropout=0.0,
).to(device)
opt = torch.optim.AdamW(model.parameters(), lr={c.learning_rate}, weight_decay=1e-5)
crit = nn.CrossEntropyLoss() if '{c.task}' == 'classification' else nn.MSELoss()

dl = DataLoader(TensorDataset(torch.tensor(X_num), torch.tensor(X_cat), torch.tensor(y)), batch_size={c.batch_size}, shuffle=True)
for epoch in range({c.num_epochs}):
    losses = []
    for xn, xc, yt in dl:
        xn, xc, yt = xn.to(device), xc.to(device), yt.to(device)
        opt.zero_grad()
        out = model(xn, xc)
        loss = crit(out, yt)
        loss.backward(); opt.step()
        losses.append(loss.item())
    print(f'epoch {{epoch}} loss {{sum(losses)/max(len(losses),1):.4f}}')
torch.save(model.state_dict(), '{c.output_dir}/ftt.pt')
print('FTT_COMPLETE')
"""

    def _saint_script(self, c: TabularDLConfig, dp: str) -> str:
        return self._ftt_script(c, dp).replace("FTTransformer", "SAINT").replace("rtdl", "rtdl_redux")

    def _tab_transformer_script(self, c: TabularDLConfig, dp: str) -> str:
        return f"""
import os, json
os.environ['HF_HOME'] = '/data/hf'
import pandas as pd, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

df = pd.read_parquet('{dp}') if '{dp}'.endswith('.parquet') else pd.read_csv('{dp}')
cat = {c.cat_cols!r}
num = {c.num_cols!r}
target = '{c.target}'
y = df[target].values.astype(np.int64 if '{c.task}' == 'classification' else np.float32)
X_num = df[num].fillna(0).values.astype(np.float32)
X_cat = np.stack([df[c].astype('category').cat.codes.values + 1 for c in cat], axis=1).astype(np.int64) if cat else np.zeros((len(df), 0), dtype=np.int64)
cat_card = [int(df[c].astype(str).nunique()) + 1 for c in cat]

class TabTransformer(nn.Module):
    def __init__(self, n_num, cat_card, d=32, n_heads=4, n_layers=3, n_classes=1):
        super().__init__()
        self.num_emb = nn.Linear(1, d) if n_num else None
        self.cat_emb = nn.ModuleList([nn.Embedding(c, d) for c in cat_card])
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d, n_classes)
    def forward(self, xn, xc):
        embs = []
        if self.num_emb is not None and xn.shape[1] > 0:
            embs.append(self.num_emb(xn.unsqueeze(-1)).mean(dim=1))
        for i, emb in enumerate(self.cat_emb):
            embs.append(emb(xc[:, i]))
        if not embs: raise ValueError('no features')
        x = torch.stack(embs, dim=1)
        x = self.encoder(x)
        return self.head(x.mean(dim=1))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TabTransformer(len(num), cat_card, n_classes={c.num_classes}).to(device)
opt = torch.optim.AdamW(model.parameters(), lr={c.learning_rate})
crit = nn.CrossEntropyLoss() if '{c.task}' == 'classification' else nn.MSELoss()
dl = DataLoader(TensorDataset(torch.tensor(X_num), torch.tensor(X_cat), torch.tensor(y)), batch_size={c.batch_size}, shuffle=True)
for epoch in range({c.num_epochs}):
    losses = []
    for xn, xc, yt in dl:
        xn, xc, yt = xn.to(device), xc.to(device), yt.to(device)
        opt.zero_grad()
        out = model(xn, xc)
        loss = crit(out, yt)
        loss.backward(); opt.step()
        losses.append(loss.item())
    print(f'epoch {{epoch}} loss {{sum(losses)/max(len(losses),1):.4f}}')
torch.save(model.state_dict(), '{c.output_dir}/tab_transformer.pt')
print('TAB_TRANSFORMER_COMPLETE')
"""
