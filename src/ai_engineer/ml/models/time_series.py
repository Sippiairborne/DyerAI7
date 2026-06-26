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

"""Time series models: forecasting, anomaly detection, classification."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ai_engineer.ml.models.registry import ModelRegistry
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)

Task = Literal["forecast", "anomaly", "classification"]
ModelKind = Literal["arima", "prophet", "lstm", "nbeats", "tft", "patchtst", "informer", "anomaly_transformer"]


@dataclass
class TimeSeriesConfig:
    task: Task = "forecast"
    model: ModelKind = "patchtst"
    horizon: int = 24
    lookback: int = 96
    num_epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    output_dir: str = ""
    freq: str = "h"
    target: str = "value"
    time_col: str = "timestamp"


@dataclass
class TimeSeriesResult:
    output_dir: str
    metrics: dict[str, float] = field(default_factory=dict)


class TimeSeriesTrainer:
    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()

    def train(self, config: TimeSeriesConfig, dataset_path: str, register_name: str | None = None) -> TimeSeriesResult:
        if not config.output_dir:
            config.output_dir = f"/tmp/ts_{config.model}_{int(time.time())}"
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        builder = {
            "patchtst": self._patchtst_script,
            "nbeats": self._nbeats_script,
            "informer": self._informer_script,
            "tft": self._tft_script,
            "lstm": self._lstm_script,
            "arima": self._arima_script,
            "prophet": self._prophet_script,
            "anomaly_transformer": self._anomaly_script,
        }.get(config.model)
        if not builder:
            raise ValueError(f"Unknown model: {config.model}")
        Path(config.output_dir, "train.py").write_text(builder(config, dataset_path))
        Path(config.output_dir, "config.json").write_text(json.dumps(config.__dict__, indent=2, default=str))
        return TimeSeriesResult(output_dir=config.output_dir, metrics={"script_path": f"{config.output_dir}/train.py"})

    def _patchtst_script(self, c: TimeSeriesConfig, dp: str) -> str:
        return f"""
import os
os.environ['HF_HOME'] = '/data/hf'
import pandas as pd, numpy as np, torch
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST
from neuralforecast.losses.pytorch import MAE

df = pd.read_csv('{dp}', parse_dates=['{c.time_col}'])
df = df.rename(columns={{'{c.time_col}': 'ds', '{c.target}': 'y'}})
df['unique_id'] = 'series'
nf = NeuralForecast(models=[PatchTST(h={c.horizon}, input_size={c.lookback}, patch_len=16, stride=8, max_steps={c.num_epochs * 100}, scaler_type='standard', learning_rate={c.learning_rate}, loss=MAE())], freq='{c.freq}')
nf.fit(df)
nf.save('{c.output_dir}/patchtst')
preds = nf.predict()
preds.to_csv('{c.output_dir}/predictions.csv', index=False)
print('PATCHTST_COMPLETE')
"""

    def _nbeats_script(self, c: TimeSeriesConfig, dp: str) -> str:
        return f"""
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
df = pd.read_csv('{dp}', parse_dates=['{c.time_col}']).rename(columns={{'{c.time_col}': 'ds', '{c.target}': 'y'}})
df['unique_id'] = 'series'
nf = NeuralForecast(models=[NBEATS(h={c.horizon}, input_size={c.lookback}, max_steps={c.num_epochs * 100}, learning_rate={c.learning_rate})], freq='{c.freq}')
nf.fit(df); nf.save('{c.output_dir}/nbeats')
print('NBEATS_COMPLETE')
"""

    def _informer_script(self, c: TimeSeriesConfig, dp: str) -> str:
        return f"""
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import Informer
df = pd.read_csv('{dp}', parse_dates=['{c.time_col}']).rename(columns={{'{c.time_col}': 'ds', '{c.target}': 'y'}})
df['unique_id'] = 'series'
nf = NeuralForecast(models=[Informer(h={c.horizon}, input_size={c.lookback}, max_steps={c.num_epochs * 100}, learning_rate={c.learning_rate})], freq='{c.freq}')
nf.fit(df); nf.save('{c.output_dir}/informer')
print('INFORMER_COMPLETE')
"""

    def _tft_script(self, c: TimeSeriesConfig, dp: str) -> str:
        return f"""
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
df = pd.read_csv('{dp}', parse_dates=['{c.time_col}']).rename(columns={{'{c.time_col}': 'ds', '{c.target}': 'y'}})
df['unique_id'] = 'series'
nf = NeuralForecast(models=[TFT(h={c.horizon}, input_size={c.lookback}, max_steps={c.num_epochs * 100}, learning_rate={c.learning_rate})], freq='{c.freq}')
nf.fit(df); nf.save('{c.output_dir}/tft')
print('TFT_COMPLETE')
"""

    def _lstm_script(self, c: TimeSeriesConfig, dp: str) -> str:
        return f"""
import pandas as pd, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
df = pd.read_csv('{dp}', parse_dates=['{c.time_col}']).sort_values('{c.time_col}')
y = df['{c.target}'].values.astype(np.float32)
mu, sd = y.mean(), y.std() + 1e-8
y_norm = (y - mu) / sd
Xs, ys = [], []
for i in range(len(y_norm) - {c.lookback} - {c.horizon}):
    Xs.append(y_norm[i:i + {c.lookback}])
    ys.append(y_norm[i + {c.lookback}:i + {c.lookback} + {c.horizon}])
X = torch.tensor(np.stack(Xs)).unsqueeze(-1)
Y = torch.tensor(np.stack(ys))

class LSTMForecaster(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Linear(64, h)
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LSTMForecaster({c.horizon}).to(device)
opt = torch.optim.Adam(model.parameters(), lr={c.learning_rate})
crit = nn.MSELoss()
dl = DataLoader(TensorDataset(X, Y), batch_size={c.batch_size}, shuffle=True)
for epoch in range({c.num_epochs}):
    losses = []
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward(); opt.step()
        losses.append(loss.item())
    print(f'epoch {{epoch}} loss {{sum(losses)/max(len(losses),1):.4f}}')
torch.save({{'model': model.state_dict(), 'mu': mu, 'sd': sd}}, '{c.output_dir}/lstm.pt')
print('LSTM_TS_COMPLETE')
"""

    def _arima_script(self, c: TimeSeriesConfig, dp: str) -> str:
        return f"""
import pandas as pd, numpy as np
from statsmodels.tsa.arima.model import ARIMA
import joblib
df = pd.read_csv('{dp}', parse_dates=['{c.time_col}']).sort_values('{c.time_col}').set_index('{c.time_col}')['{c.target}']
model = ARIMA(df, order=(2, 1, 2), seasonal_order=(1, 1, 1, 24) if '{c.freq}' == 'h' else (0,0,0,0))
fit = model.fit()
fit.save('{c.output_dir}/arima.pkl')
print('ARIMA_COMPLETE')
"""

    def _prophet_script(self, c: TimeSeriesConfig, dp: str) -> str:
        return f"""
import pandas as pd
from prophet import Prophet
df = pd.read_csv('{dp}', parse_dates=['{c.time_col}']).rename(columns={{'{c.time_col}': 'ds', '{c.target}': 'y'}})
m = Prophet()
m.fit(df)
import pickle
with open('{c.output_dir}/prophet.pkl', 'wb') as f:
    pickle.dump(m, f)
print('PROPHET_COMPLETE')
"""

    def _anomaly_script(self, c: TimeSeriesConfig, dp: str) -> str:
        return f"""
import pandas as pd, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
df = pd.read_csv('{dp}', parse_dates=['{c.time_col}']).sort_values('{c.time_col}')
y = df['{c.target}'].values.astype(np.float32)
mu, sd = y.mean(), y.std() + 1e-8
y_norm = (y - mu) / sd
Xs = []
for i in range(len(y_norm) - {c.lookback}):
    Xs.append(y_norm[i:i + {c.lookback}])
X = torch.tensor(np.stack(Xs)).unsqueeze(-1)

class AnomalyTransformer(nn.Module):
    def __init__(self, d=64, nhead=4, layers=2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.proj = nn.Linear(1, d)
        self.recon = nn.Linear(d, 1)
    def forward(self, x):
        h = self.proj(x)
        h = self.enc(h)
        return self.recon(h)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AnomalyTransformer().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.MSELoss()
dl = DataLoader(TensorDataset(X), batch_size={c.batch_size}, shuffle=True)
for epoch in range({c.num_epochs}):
    losses = []
    for (xb,) in dl:
        xb = xb.to(device)
        opt.zero_grad()
        recon = model(xb)
        loss = crit(recon, xb)
        loss.backward(); opt.step()
        losses.append(loss.item())
    print(f'epoch {{epoch}} loss {{sum(losses)/max(len(losses),1):.4f}}')
torch.save({{'model': model.state_dict(), 'mu': mu, 'sd': sd}}, '{c.output_dir}/anomaly.pt')
print('ANOMALY_COMPLETE')
"""
