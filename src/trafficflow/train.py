"""Entrenamiento y evaluación de Graph WaveNet en METR-LA (comparado con líneas base)."""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import yaml

from .data import (
    Scaler,
    WindowDataset,
    chronological_split,
    load_adjacency,
    load_speed_frame,
)
from .forecasters import HistoricalAverageForecaster, PersistenceForecaster
from .metrics import horizon_report, masked_mae, masked_mae_loss
from .model import GraphWaveNet
from .profile import HistoricalProfile
from .synthetic import make_synthetic


def load_everything(cfg):
    d = cfg["data"]
    if d.get("synthetic"):
        df, _, adj = make_synthetic(n_sensors=d.get("synthetic_sensors", 20), days=d.get("synthetic_days", 28))
    else:
        df = load_speed_frame(d["h5_path"])
        _, adj = load_adjacency(d["adj_path"])
    return df, adj


@torch.no_grad()
def predict_dataset(model, ds, scaler, device, batch_size=128):
    model.eval()
    preds, ys = [], []
    for b in range(0, len(ds), batch_size):
        ids = np.arange(b, min(b + batch_size, len(ds)))
        x, y = ds.get(ids)
        p = model(torch.from_numpy(x).to(device)).cpu().numpy()
        preds.append(scaler.inverse(p))
        ys.append(y)
    return np.concatenate(preds), np.concatenate(ys)


def baseline_predictions(ds, forecaster):
    out = []
    for i in range(len(ds)):
        out.append(forecaster.predict(ds.raw_inputs([i])[0], ds.index[i : i + ds.in_len]))
    return np.stack(out)


def run(cfg) -> dict:
    t, m, dcfg = cfg["train"], cfg["model"], cfg["data"]
    torch.manual_seed(t["seed"])
    np.random.seed(t["seed"])
    device = "cuda" if (t["device"] == "auto" and torch.cuda.is_available()) else ("cpu" if t["device"] == "auto" else t["device"])
    df, adj = load_everything(cfg)
    tr, va, te = chronological_split(df, dcfg["split"])
    scaler = Scaler.fit(tr.to_numpy())
    in_len, out_len = dcfg["in_len"], dcfg["out_len"]
    ds_tr, ds_va, ds_te = (WindowDataset(x, scaler, in_len, out_len) for x in (tr, va, te))

    model = GraphWaveNet(adj, out_len=out_len, **m).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(t["epochs"]):
        model.train()
        t0, losses = time.time(), []
        order = np.random.permutation(len(ds_tr))
        for b in range(0, len(order) - t["batch_size"] + 1, t["batch_size"]):
            x, y = ds_tr.get(order[b : b + t["batch_size"]])
            x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            loss = masked_mae_loss(scaler.inverse(model(x)), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            opt.step()
            losses.append(float(loss))
        pv, yv = predict_dataset(model, ds_va, scaler, device)
        val = masked_mae(pv, yv)
        sched.step(val)
        print(f"época {epoch + 1:02d} | train MAE {np.mean(losses):.3f} | val MAE {val:.3f} | {time.time() - t0:.0f}s", flush=True)
        if val < best - 1e-4:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= t["patience"]:
                print("parada temprana")
                break
    model.load_state_dict(best_state)

    pt, yt = predict_dataset(model, ds_te, scaler, device)
    profile = HistoricalProfile().fit(tr)
    results = {"GraphWaveNet": horizon_report(pt, yt)}
    results["HistoricalAverage"] = horizon_report(baseline_predictions(ds_te, HistoricalAverageForecaster(profile, out_len)), yt)
    results["Persistence"] = horizon_report(baseline_predictions(ds_te, PersistenceForecaster(out_len)), yt)

    os.makedirs(t["artifacts_dir"], exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "model_cfg": m, "out_len": out_len, "scaler": {"mean": scaler.mean, "std": scaler.std}},
        os.path.join(t["artifacts_dir"], "model.pt"),
    )
    with open(os.path.join(t["artifacts_dir"], "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    mape30 = results["GraphWaveNet"]["30min"]["MAPE"]
    print(json.dumps(results, indent=2))
    print(f"\nMAPE a 30 min (prueba): {mape30:.2%} | meta < {cfg['agent']['mape_target']:.0%}: {'CUMPLE' if mape30 < cfg['agent']['mape_target'] else 'NO CUMPLE'}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--synthetic", action="store_true", help="usar datos sintéticos")
    ap.add_argument("--epochs", type=int)
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    if a.synthetic:
        cfg["data"]["synthetic"] = True
    if a.epochs:
        cfg["train"]["epochs"] = a.epochs
    run(cfg)


if __name__ == "__main__":
    main()
