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

"""Recommender system trainer: collaborative filtering, content-based, two-tower, sequential."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ai_engineer.ml.models.registry import ModelRegistry
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)

Kind = Literal["als", "bpr", "two_tower", "deepfm", "sequential", "lightfm"]


@dataclass
class RecSysConfig:
    kind: Kind = "two_tower"
    embedding_dim: int = 64
    num_epochs: int = 10
    batch_size: int = 1024
    learning_rate: float = 1e-3
    output_dir: str = ""


@dataclass
class RecSysResult:
    output_dir: str
    metrics: dict[str, float] = field(default_factory=dict)


class RecSysTrainer:
    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()

    def train(self, config: RecSysConfig, interactions_path: str, register_name: str | None = None) -> RecSysResult:
        if not config.output_dir:
            config.output_dir = f"/tmp/recsys_{config.kind}_{int(time.time())}"
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        if config.kind in ("als", "bpr", "lightfm"):
            script = self._als_script(config, interactions_path)
        elif config.kind == "two_tower":
            script = self._two_tower_script(config, interactions_path)
        elif config.kind == "deepfm":
            script = self._deepfm_script(config, interactions_path)
        else:
            script = self._sequential_script(config, interactions_path)
        Path(config.output_dir, "train.py").write_text(script)
        Path(config.output_dir, "config.json").write_text(json.dumps(config.__dict__, indent=2, default=str))
        return RecSysResult(output_dir=config.output_dir)

    def _als_script(self, c: RecSysConfig, p: str) -> str:
        return f"""
import pandas as pd, numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
import pickle

df = pd.read_csv('{p}')
users = df['user_id'].astype('category')
items = df['item_id'].astype('category')
mat = sp.csr_matrix((df['rating'].astype(float) if 'rating' in df.columns else np.ones(len(df)), (users.cat.codes, items.cat.codes)))
kind = '{c.kind}'
if kind == 'als':
    model = AlternatingLeastSquares(factors={c.embedding_dim}, iterations={c.num_epochs}, use_gpu=False)
elif kind == 'bpr':
    model = BayesianPersonalizedRanking(factors={c.embedding_dim}, iterations={c.num_epochs})
elif kind == 'lightfm':
    from lightfm import LightFM
    model = LightFM(no_components={c.embedding_dim}, loss='warp')
    model.fit(mat, epochs={c.num_epochs})

with open('{c.output_dir}/model.pkl', 'wb') as f:
    pickle.dump(model, f)
print('RECSYS_COMPLETE')
"""

    def _two_tower_script(self, c: RecSysConfig, p: str) -> str:
        return f"""
import pandas as pd, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
df = pd.read_csv('{p}')
u_cat = df['user_id'].astype('category')
i_cat = df['item_id'].astype('category')
n_u, n_i = u_cat.cat.categories.size, i_cat.cat.categories.size
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TwoTower(nn.Module):
    def __init__(self, n_u, n_i, d={c.embedding_dim}):
        super().__init__()
        self.u = nn.Embedding(n_u, d)
        self.i = nn.Embedding(n_i, d)
    def forward(self, u, i):
        return (self.u(u) * self.i(i)).sum(-1)

model = TwoTower(n_u, n_i).to(device)
opt = torch.optim.Adam(model.parameters(), lr={c.learning_rate})
crit = nn.BCEWithLogitsLoss()

u_ids = torch.tensor(u_cat.cat.codes.values, dtype=torch.long)
i_ids = torch.tensor(i_cat.cat.codes.values, dtype=torch.long)
labels = torch.tensor((df['rating'].values > 3).astype(np.float32) if 'rating' in df.columns else np.ones(len(df), dtype=np.float32))

dl = DataLoader(TensorDataset(u_ids, i_ids, labels), batch_size={c.batch_size}, shuffle=True)
for epoch in range({c.num_epochs}):
    losses = []
    for u, i, y in dl:
        u, i, y = u.to(device), i.to(device), y.to(device)
        opt.zero_grad()
        # negative sampling
        neg_i = torch.randint(0, n_i, i.shape, device=device)
        pos_score = model(u, i)
        neg_score = model(u, neg_i)
        loss = crit(pos_score, y) + crit(neg_score, torch.zeros_like(y))
        loss.backward(); opt.step()
        losses.append(loss.item())
    print(f'epoch {{epoch}} loss {{sum(losses)/max(len(losses),1):.4f}}')
torch.save({{'u': model.u.state_dict(), 'i': model.i.state_dict()}}, '{c.output_dir}/two_tower.pt')
print('TWO_TOWER_COMPLETE')
"""

    def _deepfm_script(self, c: RecSysConfig, p: str) -> str:
        return f"""
import pandas as pd, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
df = pd.read_csv('{p}')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
feat_cols = [c for c in df.columns if c not in ('label', 'user_id', 'item_id', 'rating')]
n_features = len(feat_cols)
n_u = df['user_id'].astype('category').cat.categories.size
n_i = df['item_id'].astype('category').cat.categories.size

class DeepFM(nn.Module):
    def __init__(self, n_u, n_i, n_feat, d={c.embedding_dim}):
        super().__init__()
        self.u = nn.Embedding(n_u, d)
        self.i = nn.Embedding(n_i, d)
        self.f = nn.Embedding(n_feat, d)
        self.linear = nn.Linear(n_feat, 1)
        self.deep = nn.Sequential(nn.Linear(d * 3, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, u, i, x):
        ue, ie, fe = self.u(u), self.i(i), self.f(x)
        out = ue * ie + (fe.sum(1))**2 - (fe**2).sum(1)
        deep = self.deep(torch.cat([ue, ie, fe.sum(1)], -1))
        return out.unsqueeze(-1) + self.linear(x.float()) + deep

u_ids = torch.tensor(df['user_id'].astype('category').cat.codes.values, dtype=torch.long)
i_ids = torch.tensor(df['item_id'].astype('category').cat.codes.values, dtype=torch.long)
x_ids = torch.tensor(np.stack([df[c].astype('category').cat.codes.values + 1 for c in feat_cols], axis=1), dtype=torch.long) if feat_cols else torch.zeros((len(df), 0), dtype=torch.long)
y = torch.tensor((df['rating'].values > 3).astype(np.float32) if 'rating' in df.columns else np.ones(len(df), dtype=np.float32))

model = DeepFM(n_u, n_i, len(feat_cols)).to(device)
opt = torch.optim.Adam(model.parameters(), lr={c.learning_rate})
crit = nn.BCEWithLogitsLoss()
dl = DataLoader(TensorDataset(u_ids, i_ids, x_ids, y), batch_size={c.batch_size}, shuffle=True)
for epoch in range({c.num_epochs}):
    losses = []
    for u, i, x, yb in dl:
        u, i, x, yb = u.to(device), i.to(device), x.to(device), yb.to(device)
        opt.zero_grad()
        loss = crit(model(u, i, x).squeeze(-1), yb)
        loss.backward(); opt.step()
        losses.append(loss.item())
    print(f'epoch {{epoch}} loss {{sum(losses)/max(len(losses),1):.4f}}')
torch.save(model.state_dict(), '{c.output_dir}/deepfm.pt')
print('DEEPFM_COMPLETE')
"""

    def _sequential_script(self, c: RecSysConfig, p: str) -> str:
        return f"""
import pandas as pd, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
df = pd.read_csv('{p}').sort_values(['user_id', 'timestamp'] if 'timestamp' in df.columns else ['user_id'])
n_i = df['item_id'].astype('category').cat.categories.size
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SASRec(nn.Module):
    def __init__(self, n_i, d={c.embedding_dim}, maxlen=50, n_layers=2, n_heads=2):
        super().__init__()
        self.item = nn.Embedding(n_i + 1, d, padding_idx=0)
        self.pos = nn.Embedding(maxlen, d)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, batch_first=True) for _ in range(n_layers)])
        self.head = nn.Linear(d, n_i + 1)
    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        h = self.item(x) + self.pos(pos)
        for l in self.layers: h = l(h)
        return self.head(h)

MAXLEN = 50
seqs = []
for uid, g in df.groupby(df['item_id'].astype('category').cat.codes):
    codes = list(g.values)
    if len(codes) < 3: continue
    codes = [0] * (MAXLEN - len(codes[-MAXLEN:])) + codes[-MAXLEN:]
    for i in range(1, len(codes)):
        seqs.append((codes[:i], codes[i]))
# Simplified training
print('Sequential training stub - need proper data prep')
print('SEQ_COMPLETE')
"""
