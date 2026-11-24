"""Store Timestral weights in fp16 (about half the size) and measure the effect on the speed feature.
python shrink_timestral.py --speed data/swpc_ace_historical/swpc_ace_swepam_hourly.parquet --col speed_km_s --out timestral_fp16
Writes <out>/model.safetensors (fp16) + config.json, then compares fp32 vs fp16 forecasts for a few t0.
"""
import argparse, glob, os, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
from safetensors.torch import load_file, save_file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import timestral_speed as T
from timestral import TimestralForecaster

ap = argparse.ArgumentParser(); ap.add_argument("--speed", required=True); ap.add_argument("--col", default="speed_km_s")
ap.add_argument("--out", default="timestral_fp16"); ap.add_argument("--latency", type=int, default=0)
a = ap.parse_args()
snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--mistralai--Timestral-123M/snapshots/*/"))[0]
sd = load_file(os.path.join(snap, "model.safetensors"))
out = Path(a.out); out.mkdir(exist_ok=True)
sd16 = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}
save_file(sd16, str(out / "model.safetensors"), metadata={"format": "pt"})
shutil.copy(os.path.join(snap, "config.json"), out / "config.json")
print("sizes MiB: fp32 %.0f -> fp16 %.0f" % (os.path.getsize(os.path.join(snap, "model.safetensors")) / 2**20, os.path.getsize(out / "model.safetensors") / 2**20))
f32 = TimestralForecaster.from_pretrained("mistralai/Timestral-123M", device="cpu")
f16 = TimestralForecaster.from_pretrained(str(out), device="cpu")
s = T.load_speed(a.speed, a.col)
diffs = []
for t0 in pd.date_range("2025-03-01", periods=8, freq="2D"):
    r32, r16 = T.speed_features(f32, s, t0, a.latency), T.speed_features(f16, s, t0, a.latency)
    diffs.append((np.abs(r32.timestral_med - r16.timestral_med).mean(), np.abs(r32.timestral_spread - r16.timestral_spread).mean()))
d = np.array(diffs)
print("fp32 vs fp16 mean |diff| over 8 cycles: median %.3f km/s, spread %.3f km/s  (max %.3f / %.3f)" % (d[:, 0].mean(), d[:, 1].mean(), d[:, 0].max(), d[:, 1].max()))
