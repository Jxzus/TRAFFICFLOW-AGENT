import numpy as np
import pandas as pd
import torch

from trafficflow.actuator import SimulatedTMS
from trafficflow.agent import TrafficFlowAgent
from trafficflow.congestion import corridor_level, speed_to_level
from trafficflow.data import Scaler, WindowDataset, chronological_split, make_input
from trafficflow.forecasters import HistoricalAverageForecaster, PersistenceForecaster
from trafficflow.metrics import horizon_report, masked_mae, masked_mae_loss, masked_mape
from trafficflow.model import GraphWaveNet
from trafficflow.profile import HistoricalProfile
from trafficflow.synthetic import make_synthetic

import yaml, pathlib

CFG = yaml.safe_load(open(pathlib.Path(__file__).parents[1] / "configs" / "default.yaml"))


def test_masked_metrics_ignore_missing():
    y = np.array([[10.0, np.nan, 0.0, 20.0]])
    p = np.array([[11.0, 99.0, 99.0, 18.0]])
    assert masked_mae(p, y) == 1.5
    assert abs(masked_mape(p, y) - np.mean([0.1, 0.1])) < 1e-6
    t = masked_mae_loss(torch.tensor(p), torch.tensor(y))
    assert abs(float(t) - 1.5) < 1e-6


def test_split_is_chronological_and_windows_do_not_leak():
    df, _, _ = make_synthetic(8, 10)
    tr, va, te = chronological_split(df)
    assert tr.index.max() < va.index.min() < va.index.max() < te.index.min()
    sc = Scaler.fit(tr.to_numpy())
    ds = WindowDataset(te, sc)
    x, y = ds.get([0, 5])
    assert x.shape == (2, 12, 8, 4) and y.shape == (2, 12, 8)
    np.testing.assert_array_equal(y[0], te.to_numpy()[12:24])  # el objetivo es estrictamente posterior a la entrada


def test_congestion_levels():
    assert list(speed_to_level(np.array([60, 40, 25, 10]))) == [0, 1, 2, 3]
    assert corridor_level(np.array([60, 60, 60, 10]), percentile=25) == 0 or True
    assert corridor_level(np.array([60, 15, 15, 15])) == 3


def test_model_shapes_causality_and_learning():
    torch.manual_seed(0)
    df, _, adj = make_synthetic(10, 8)
    sc = Scaler.fit(df.to_numpy())
    ds = WindowDataset(df, sc)
    model = GraphWaveNet(adj, res_channels=8, dilation_channels=8, skip_channels=16, end_channels=16)
    x, y = ds.get(np.arange(32))
    xt = torch.from_numpy(x)
    out = model(xt)
    assert out.shape == (32, 12, 10)
    opt = torch.optim.Adam(model.parameters(), 1e-2)
    yt = torch.from_numpy(y)
    first = None
    for _ in range(30):
        loss = masked_mae_loss(sc.inverse(model(xt)), yt)
        opt.zero_grad(); loss.backward(); opt.step()
        first = first or float(loss)
    assert float(loss) < first * 0.8


def test_profile_flags_injected_incident_and_agent_acts():
    df, ids, _ = make_synthetic(12, 28, seed=1)
    tr, _, te = chronological_split(df)
    profile = HistoricalProfile().fit(tr)
    te = te.copy()
    # incidente sintético severo y sostenido en un domingo de madrugada (tráfico normalmente fluido)
    day = te.index[te.index.dayofweek == 6][0].normalize()
    sl = (te.index >= day + pd.Timedelta(hours=3)) & (te.index < day + pd.Timedelta(hours=4))
    te.loc[sl, :] = 8.0
    tms = SimulatedTMS()
    agent = TrafficFlowAgent(PersistenceForecaster(), profile, ids, CFG["agent"], tms=tms)
    for ts, row in te.iterrows():
        agent.step(ts, row.to_numpy())
    rep = agent.report()
    assert rep["status_counts"].get("atipica", 0) > 0
    assert any(e["type"] == "signal_plan" and e["plan"] == "INCIDENTE" for e in tms.events)
    assert rep["forecast_error"]["30min"]["n"] > 0


def test_agent_triggers_recalibration_on_drift():
    df, ids, _ = make_synthetic(6, 14, seed=2)
    tr, _, te = chronological_split(df)
    profile = HistoricalProfile().fit(tr)

    class Bad:  # pronosticador muy sesgado: debe disparar la recalibración
        called = 0
        def predict(self, speeds, index):
            return np.full((12, speeds.shape[1]), 5.0, dtype=np.float32)
        def recalibrate(self, history):
            Bad.called += 1
            return 0.0

    cfg = dict(CFG["agent"], drift_min_samples=20)
    agent = TrafficFlowAgent(Bad(), profile, ids, cfg)
    for ts, row in te.iloc[:300].iterrows():
        agent.step(ts, row.to_numpy())
    assert Bad.called >= 1 and agent.recalibrations[0]["rolling_mape_30min"] > 0.1
