"""TCN v2: multi-lead (MIMO, 72 outputs) history model for the FDL BenchLab solar-wind task.

Inputs  (all built by windows.build_sample, shared by training and inference)
  recent   : last 168 h of V, Bx, By, Bz, log N  + observed-masks   (ends at t0 - latency)
  27 d     : V... in the window around t0 + h - 648  (h = 1..72, +-48 h margin)      [lead-aligned recurrence]
  54 d     : same around t0 + h - 1296
  scalars  : sin/cos day-of-year (Earth's heliographic latitude), age of last valid V, valid fraction, 7 d / 27 d mean V
Output   : V(t0+1 .. t0+72)  in one forward pass.
  pred = persistence + MLP(features) + sum_p gate_p[h] * (recurrence_p[h] - persistence)     (gates start at 0)
Loss     : weighted MSE on valid targets; weight = (1 + gamma*N*c(v)) [TIMSE tail term] x per-lead 1/RMSE_h [metric gradient]
Validation: S_final (the tournament metric) over all 72 leads on a recent block, issue times every 6 h.

Recipe
  1) python tcn_v2.py smoke                                           # synthetic end-to-end check (needs torch)
  2) python tcn_v2.py train --hf data/solar_wind_historical_HuggFaceDataset.parquet \
         --extra "data/swpc_ace_historical/swpc_ace_swepam_hourly.parquet:V=speed_km_s" \
         --train-end 2022-12-31 --val-start 2023-01-01 --val-end 2024-12-31 --out runs/v2a
  3) python tcn_v2.py train ... --no-val --epochs <best epoch printed in 2> --train-end 2024-12-31 --out runs/v2_final
  4) python tcn_v2.py predict --run runs/v2_final --hf ... --extra ... --t0-start 2025-01-01 --t0-end 2026-08-31 \
         --score-ace data/swpc_ace_historical/swpc_ace_swepam_hourly.parquet --out-parquet runs/v2_final/net_predictions_v2.parquet
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, get_worker_info

from windows import (FEATS, N_SCAL, Aug, Grid, Scaler, WCfg, build_sample, make_tail_lut, score_matrix,
                     tail_weight, valid_t0)


# ------------------------------------------------------------------------------------------------ data
def _stamp(e):
    if "timestamp" in e.columns:
        return pd.to_datetime(e["timestamp"], utc=True).dt.tz_localize(None).dt.floor("h")
    if {"YEAR", "DOY", "HOUR"} <= set(e.columns):
        return pd.to_datetime(e["YEAR"].astype(int).astype(str) + " " + e["DOY"].astype(int).astype(str) + " "
                              + e["HOUR"].astype(int).astype(str), format="%Y %j %H")
    raise SystemExit(f"cannot find a time column in {list(e.columns)[:12]}")


def load_table(hf, extras=()):
    """Hourly table with columns FEATS (naive UTC, NaN = missing).  `extras`: 'path:V=col,N=col' overrides/adds
    columns where the extra file has data (use it for 2025+ OMNI and for the ACE feed that you will also get live)."""
    d = pd.read_parquet(hf)
    t = d.set_index(_stamp(d))[FEATS].astype(float)
    t = t.groupby(level=0).mean().sort_index()
    for spec in extras:
        path, _, mp = spec.partition(":")
        e = pd.read_parquet(path)
        mapping = dict(kv.split("=") for kv in mp.split(",") if kv)
        miss = [v for v in mapping.values() if v not in e.columns]
        if miss:
            raise SystemExit(f"{path}: columns {miss} not found; available: {list(e.columns)}")
        x = pd.DataFrame({k: pd.to_numeric(e[v], errors="coerce").to_numpy() for k, v in mapping.items()},
                         index=_stamp(e))
        for k in x.columns:                                         # OMNI fill values -> NaN
            x.loc[x[k] >= (9999 if k == "V" else 999), k] = np.nan
        x = x.groupby(level=0).mean().sort_index()
        t = x.combine_first(t)[FEATS]
    return t.sort_index()


def coverage(t):
    c = t.notna().groupby(t.index.year).mean().round(2)
    print("fraction of hours observed per year:\n" + c.to_string())


# ----------------------------------------------------------------------------------------------- torch
class WinDS(Dataset):
    def __init__(self, g, idx, cfg, train, aug=None, lut=None, gamma=0.25, seed=0):
        self.g, self.idx, self.cfg, self.train, self.aug, self.lut, self.gamma = g, idx, cfg, train, aug, lut, gamma
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        s = build_sample(self.g, int(self.idx[k]), self.cfg,
                         self.rng if self.train else None, self.aug if self.train else None)
        s["w"] = (tail_weight(s["yv"], s["ym"], self.lut, self.gamma) if self.lut is not None
                  else np.ones(self.cfg.horizon, np.float32))
        return {k_: torch.from_numpy(np.ascontiguousarray(v)) for k_, v in s.items()}


def _worker_init(_):
    info = get_worker_info()
    info.dataset.rng = np.random.default_rng((torch.initial_seed() + info.id) % (2 ** 32))


class Enc(nn.Module):
    """(B, L, C) -> (B, d).  Conv stack with 3 poolings, then flatten (keeps the position inside the window)."""

    def __init__(self, c_in, L, d, ch=(32, 48, 64), k=5, p=0.1):
        super().__init__()
        layers, c, n = [], c_in, L
        for o in ch:
            layers += [nn.Conv1d(c, o, k, padding=k // 2), nn.GroupNorm(4, o), nn.GELU(),
                       nn.Conv1d(o, o, k, padding=k // 2), nn.GELU(), nn.MaxPool1d(2)]
            c, n = o, n // 2
        self.net = nn.Sequential(*layers)
        self.proj = nn.Sequential(nn.Flatten(), nn.Dropout(p), nn.Linear(ch[-1] * n, d), nn.GELU())

    def forward(self, x):
        return self.proj(self.net(x.transpose(1, 2)))


class MimoNet(nn.Module):
    def __init__(self, cfg, d=128, hid=512, p=0.2):
        super().__init__()
        c_in = 2 * len(FEATS)
        self.er = Enc(c_in, cfg.recent, d, p=p / 2)
        self.ep = nn.ModuleList([Enc(c_in, cfg.seg, d, p=p / 2) for _ in cfg.periods])
        self.sc = nn.Sequential(nn.Linear(N_SCAL, 32), nn.GELU())
        self.head = nn.Sequential(nn.Linear(d * (1 + len(cfg.periods)) + 32, hid), nn.GELU(), nn.Dropout(p),
                                  nn.Linear(hid, 256), nn.GELU(), nn.Linear(256, cfg.horizon))
        nn.init.zeros_(self.head[-1].bias)
        with torch.no_grad():
            self.head[-1].weight.mul_(0.1)                           # start close to persistence
        self.gate = nn.Parameter(torch.zeros(len(cfg.periods), cfg.horizon))

    def forward(self, b):
        f = [self.er(b["xr"])] + [e(b["xrec"][:, i]) for i, e in enumerate(self.ep)] + [self.sc(b["scal"])]
        delta = self.head(torch.cat(f, 1))
        base = b["base"].view(-1, 1)
        skip = ((b["fill"] - base.unsqueeze(1)) * self.gate).sum(1)  # (B,P,H) - (B,1,1)
        return base + delta + skip


def to_dev(b, dev):
    return {k: v.to(dev) for k, v in b.items()}


@torch.no_grad()
def predict_idx(model, g, idx, cfg, dev, bs=512):
    """Predictions in km/s for issue indices `idx` -> (n, H)."""
    model.eval()
    dl = DataLoader(WinDS(g, idx, cfg, train=False), batch_size=bs, shuffle=False)
    out = [model(to_dev(b, dev)).cpu().numpy() for b in dl]
    z = np.concatenate(out) if out else np.zeros((0, cfg.horizon))
    return z * g.scaler.std[0] + g.scaler.mean[0]


def obs_matrix(g, idx, cfg):
    return np.stack([g.v[i + 1:i + 1 + cfg.horizon] for i in idx])


def evaluate(model, g, idx, cfg, dev):
    p = predict_idx(model, g, idx, cfg, dev)
    tab, s = score_matrix(obs_matrix(g, idx, cfg), p)
    return s, tab, p


# --------------------------------------------------------------------------------------------- training
def train_one(seed, g, cfg, tr_idx, va_idx, lut, a, out_path, dev):
    torch.manual_seed(seed)
    np.random.seed(seed)
    H = cfg.horizon
    arch = dict(d=a.width, hid=4 * a.width, p=a.dropout)
    model = MimoNet(cfg, **arch).to(dev)
    print(f"  seed {seed}: {sum(p.numel() for p in model.parameters()):,} parameters, "
          f"{len(tr_idx):,} train issue times, {0 if va_idx is None else len(va_idx):,} val")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    n_ep = min(a.samples_per_epoch, len(tr_idx) * 4)
    steps = (n_ep // a.bs) * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
    ds = WinDS(g, tr_idx, cfg, train=True, aug=Aug(), lut=lut, gamma=a.tail_gamma, seed=seed)
    dl = DataLoader(ds, batch_size=a.bs, sampler=RandomSampler(ds, replacement=True, num_samples=n_ep),
                    num_workers=a.workers, drop_last=True, worker_init_fn=_worker_init if a.workers else None)
    ema = torch.ones(H, device=dev)                                   # running per-lead MSE (scaled units)
    best, best_ep, bad = np.inf, -1, 0

    for ep in range(a.epochs):
        model.train()
        t_ep, tot, nb = time.time(), 0.0, 0
        for b in dl:
            b = to_dev(b, dev)
            pred = model(b)
            e2 = (pred - b["y"]) ** 2
            lw = (ema + 1e-3).rsqrt()
            lw = lw / lw.mean()                                       # d S_tau / d MSE_tau ~ 1/RMSE_tau
            w = b["w"] * b["ym"] * lw.unsqueeze(0)
            loss = (w * e2).sum() / w.sum().clamp_min(1e-6)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                m = b["ym"]
                ema = 0.99 * ema + 0.01 * torch.where(m.sum(0) > 0, (e2 * m).sum(0) / m.sum(0).clamp_min(1), ema)
            tot += loss.item()
            nb += 1
        msg = f"    ep {ep + 1:2d}/{a.epochs} loss {tot / max(nb, 1):.4f}"
        if va_idx is not None:
            s, tab, _ = evaluate(model, g, va_idx, cfg, dev)
            msg += (f" | val S_final {s:.2f}  RMSE {tab.RMSE.mean():.1f}  TRMSE {tab.TRMSE.mean():.1f}"
                    f"  S[1-24] {tab.S.loc[1:24].mean():.1f}  S[49-72] {tab.S.loc[49:72].mean():.1f}")
            if s < best - 1e-3:
                best, best_ep, bad = s, ep + 1, 0
                save(model, cfg, g.scaler, out_path, dict(epoch=ep + 1, val_S=s, seed=seed, arch=arch))
            else:
                bad += 1
        print(msg + f" ({time.time() - t_ep:.0f}s)")
        if va_idx is not None and bad >= a.patience:
            print("    early stop")
            break
    if va_idx is None:
        save(model, cfg, g.scaler, out_path, dict(epoch=a.epochs, seed=seed, arch=arch))
        return None, a.epochs
    print(f"  seed {seed}: best val S_final {best:.2f} at epoch {best_ep}")
    return best, best_ep


def save(model, cfg, scaler, path, meta):
    torch.save(dict(state=model.state_dict(), cfg=cfg.to_dict(), scaler=scaler.to_dict(), meta=meta), path)


def load_run(run_dir, dev):
    models = []
    for p in sorted(Path(run_dir).glob("tcn_v2_seed*.pt")):
        ck = torch.load(p, map_location=dev, weights_only=False)
        cfg = WCfg.from_dict(ck["cfg"])
        m = MimoNet(cfg, **ck["meta"].get("arch", {})).to(dev)
        m.load_state_dict(ck["state"])
        m.eval()
        models.append(m)
    if not models:
        raise SystemExit(f"no tcn_v2_seed*.pt in {run_dir}")
    return models, cfg, Scaler.from_dict(ck["scaler"])


def get_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------------------------- commands
def cmd_train(a):
    dev = get_device(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = WCfg(latency=a.latency, drop=tuple(x for x in a.drop.split(",") if x))
    t = load_table(a.hf, a.extra)
    print(f"table {t.index.min()} .. {t.index.max()}  ({len(t):,} h)")
    coverage(t)
    train_end = pd.Timestamp(a.train_end)
    scaler = Scaler.fit(t.loc[:train_end])
    g = Grid(t, scaler, cfg)
    tr = valid_t0(g, cfg, end=train_end, step=1)
    va = None
    if not a.no_val:
        va = valid_t0(g, cfg, start=a.val_start, end=a.val_end, step=6)
        # purge: training targets must not overlap the validation period
        tr = tr[g.t[tr + cfg.horizon] < pd.Timestamp(a.val_start)]
    ref_start = train_end - pd.DateOffset(years=a.tail_ref_years)
    lut = make_tail_lut(t.loc[ref_start:train_end, "V"].to_numpy())
    print(f"tail weights from V in {ref_start.date()}..{train_end.date()}  (N={len(lut[0]):,})")
    res = []
    for k in range(a.seeds):
        res.append(train_one(a.seed0 + k, g, cfg, tr, va, lut, a, out / f"tcn_v2_seed{a.seed0 + k}.pt", dev))
    json.dump(dict(args=vars(a), results=res), open(out / "train_log.json", "w"), indent=1, default=str)
    if va is not None:
        eps = [r[1] for r in res]
        print(f"\nbest epoch per seed {eps} of --epochs {a.epochs}.  NOTE: OneCycle's schedule depends on --epochs, so do not "
              f"refit with the best epoch number; if the best epoch is early, validate a shorter schedule (e.g. --epochs 3) "
              f"and only refit with --no-val if its LAST epoch is also the best.")
        models, _, _ = load_run(out, dev)
        for k, m in enumerate(models):                                  # does the net use the 27 d / 54 d recurrence?
            gt = m.gate.detach().cpu().numpy()
            print(f"  seed {k} recurrence gates (mean over leads 1-24 / 25-48 / 49-72): " + "  ".join(
                f"{p}d: {gt[i, :24].mean():+.2f}/{gt[i, 24:48].mean():+.2f}/{gt[i, 48:].mean():+.2f}"
                for i, p in enumerate((27, 54))))
        P = np.mean([predict_idx(m, g, va, cfg, dev) for m in models], axis=0)
        np.savez(out / "val_preds.npz", t0=g.t[va].astype(str), pred=P, obs=obs_matrix(g, va, cfg))
        tab, s = score_matrix(obs_matrix(g, va, cfg), P)
        print(f"ensemble of {len(models)} seeds: val S_final {s:.2f}  RMSE {tab.RMSE.mean():.1f}  "
              f"TRMSE {tab.TRMSE.mean():.1f}")


def cmd_predict(a):
    dev = get_device(a.device)
    models, cfg, scaler = load_run(a.run, dev)
    t = load_table(a.hf, a.extra)
    if a.live_t0:                                                     # one live issue time
        t0 = pd.Timestamp(a.live_t0)
        g = Grid(t, scaler, cfg, t_end=t0 + pd.Timedelta(hours=cfg.horizon))
        idx = np.array([g.index_of(t0)])
    else:
        g = Grid(t, scaler, cfg, t_end=pd.Timestamp(a.t0_end) + pd.Timedelta(hours=cfg.horizon))
        idx = np.array([g.index_of(x) for x in pd.date_range(a.t0_start, a.t0_end, freq=f"{a.step}h")
                        if x in g.t])
    P = np.mean([predict_idx(m, g, idx, cfg, dev) for m in models], axis=0)
    cols = [f"h{k:02d}" for k in range(1, cfg.horizon + 1)]
    df = pd.DataFrame(P, index=pd.DatetimeIndex(g.t[idx], name="t0").tz_localize("UTC"), columns=cols)
    Path(a.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out_parquet)
    print(f"wrote {a.out_parquet}: {df.shape}  (index = issue time t0; set features.Cfg.tcn_index_offset_h = 0)")
    if a.score_ace and not a.live_t0:
        ace = pd.read_parquet(a.score_ace)
        s = pd.Series(ace["speed_km_s"].to_numpy(float), index=_stamp(ace)).groupby(level=0).mean()
        s = s.reindex(pd.date_range(s.index.min(), s.index.max() + pd.Timedelta(hours=cfg.horizon), freq="h"))
        pos = s.index.get_indexer(g.t[idx])
        ok = pos >= 0                                                 # issue times inside the ACE file
        pos, P = pos[ok], P[ok]
        obs = np.stack([s.to_numpy()[p + 1:p + 1 + cfg.horizon] for p in pos])
        last = s.ffill().to_numpy()[pos]                              # persistence: latest valid ACE at/before t0
        idx = idx[ok]
        tab, S = score_matrix(obs, P)
        tab_p, S_p = score_matrix(obs, np.repeat(last[:, None], cfg.horizon, 1))
        print(f"\nvs ACE ({len(idx)} issue times): S_final {S:.2f}  RMSE {tab.RMSE.mean():.1f}  "
              f"TRMSE {tab.TRMSE.mean():.1f}   [persistence S_final {S_p:.2f}]")
        print(f"  S[1-24] {tab.S.loc[1:24].mean():.1f}  S[25-48] {tab.S.loc[25:48].mean():.1f}  "
              f"S[49-72] {tab.S.loc[49:72].mean():.1f}")
        tab.to_csv(Path(a.out_parquet).with_suffix(".per_lead.csv"))


def cmd_smoke(a):
    """Synthetic end-to-end check: 4 years of fake solar wind with a 27-day recurrence."""
    import tempfile
    rng = np.random.default_rng(0)
    n = 24 * 365 * 4
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    ph = 2 * np.pi * np.arange(n) / (27 * 24)
    V = 420 + 120 * np.maximum(np.sin(ph) + 0.5 * np.sin(2.3 * ph + 1), 0) + rng.normal(0, 20, n)
    df = pd.DataFrame(dict(V=V, BX_GSE=rng.normal(0, 3, n), BY_GSM=rng.normal(0, 3, n), BZ_GSM=rng.normal(0, 3, n),
                           N=rng.gamma(2, 3, n)), index=idx)
    df = df.mask(rng.random(df.shape) < 0.1)
    tmp = Path(tempfile.mkdtemp())
    d2 = df.copy()
    d2.insert(0, "timestamp", idx.tz_localize("UTC"))
    d2.reset_index(drop=True).to_parquet(tmp / "hf.parquet")
    ns = argparse.Namespace(hf=str(tmp / "hf.parquet"), extra=[], out=str(tmp / "run"), latency=3,
                            train_end="2022-12-31", val_start="2023-01-01", val_end="2023-12-31", no_val=False,
                            seeds=1, seed0=0, epochs=3, samples_per_epoch=3000, bs=128, lr=2e-3, wd=1e-2,
                            tail_gamma=0.25, tail_ref_years=2, patience=3, workers=0, device=a.device,
                            drop="BX_GSE", width=32, dropout=0.3)
    cmd_train(ns)
    pa = argparse.Namespace(run=ns.out, hf=ns.hf, extra=[], t0_start="2023-06-01", t0_end="2023-08-31", step=6,
                            live_t0=None, out_parquet=str(tmp / "pred.parquet"), score_ace=None, device=a.device)
    cmd_predict(pa)
    p = pd.read_parquet(tmp / "pred.parquet")
    assert p.shape[1] == 72 and np.isfinite(p.to_numpy()).all()
    # live-style call: newest observation is 3 h old
    t0 = df.index[-1] + pd.Timedelta(hours=3)
    pl = argparse.Namespace(run=ns.out, hf=ns.hf, extra=[], t0_start=None, t0_end=None, step=6,
                            live_t0=str(t0), out_parquet=str(tmp / "live.parquet"), score_ace=None, device=a.device)
    cmd_predict(pl)
    assert pd.read_parquet(tmp / "live.parquet").shape == (1, 72)
    print("\nsmoke test passed")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "predict", "smoke"):
        sp = sub.add_parser(name)
        sp.add_argument("--device", default="auto")
        if name == "smoke":
            continue
        sp.add_argument("--hf", required=True)
        sp.add_argument("--extra", action="append", default=[],
                        help="'path:V=col,N=col'  (repeatable; later ones override earlier)")
        if name == "train":
            sp.add_argument("--out", required=True)
            sp.add_argument("--train-end", default="2022-12-31")
            sp.add_argument("--val-start", default="2023-01-01")
            sp.add_argument("--val-end", default="2024-12-31")
            sp.add_argument("--no-val", action="store_true")
            sp.add_argument("--latency", type=int, default=3)
            sp.add_argument("--seeds", type=int, default=5)
            sp.add_argument("--seed0", type=int, default=0)
            sp.add_argument("--epochs", type=int, default=30)
            sp.add_argument("--samples-per-epoch", type=int, default=40000)
            sp.add_argument("--bs", type=int, default=256)
            sp.add_argument("--lr", type=float, default=2e-3)
            sp.add_argument("--wd", type=float, default=1e-2)
            sp.add_argument("--tail-gamma", type=float, default=0.25)
            sp.add_argument("--tail-ref-years", type=int, default=3)
            sp.add_argument("--patience", type=int, default=6)
            sp.add_argument("--drop", default="", help="comma list of features to disable, e.g. BX_GSE,BY_GSM,BZ_GSM")
            sp.add_argument("--width", type=int, default=128, help="embedding width d (head hidden = 4*d)")
            sp.add_argument("--dropout", type=float, default=0.2)
            sp.add_argument("--workers", type=int, default=0)
        else:
            sp.add_argument("--run", required=True)
            sp.add_argument("--out-parquet", required=True)
            sp.add_argument("--t0-start")
            sp.add_argument("--t0-end")
            sp.add_argument("--step", type=int, default=6)
            sp.add_argument("--live-t0", help="single live issue time, e.g. '2026-11-12 06:00'")
            sp.add_argument("--score-ace")
    a = ap.parse_args()
    {"train": cmd_train, "predict": cmd_predict, "smoke": cmd_smoke}[a.cmd](a)


if __name__ == "__main__":
    main()
