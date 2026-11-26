"""Offline tests: data cut-offs (leak guards), ENLIL admission rules, fallbacks. No network."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from swp import model

T0 = pd.Timestamp("2026-11-10 06:00")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
BLEND = json.load(open(os.path.join(model.HERE, "final_blend.json")))


def monthly_text(start, hours, speed=450.0, bad_every=0):
    lines = []
    for i in range(hours):
        t = start + pd.Timedelta(hours=i)
        status = 9 if (bad_every and i % bad_every == 0) else 0
        v = -9999.9 if status else speed + (i % 24)
        lines.append(f"{t:%Y %m %d %H%M} 61000 0 {status} 5.0 {v} 100000".replace(" 61000 0 ", " 61000 3600 "))
    return "\n".join(lines)


def fake_http(start, hours):
    txt = monthly_text(start, hours)
    return lambda url, timeout=30, retries=2: txt.encode()


def test_parse_monthly_skips_bad_status():
    rows = model.parse_monthly(monthly_text(pd.Timestamp("2026-11-01"), 48, bad_every=5))
    assert 0 < len(rows) < 48
    assert all(0 < v < 3000 for v in rows.values())


def test_ace_never_uses_labels_after_cutoff(monkeypatch):
    # the file contains data up to t0 + 40 h: the future must be invisible
    monkeypatch.setattr(model, "http_get", fake_http(pd.Timestamp("2026-10-01"), 24 * 45))
    obs, _ = model.fetch_ace(T0)
    assert obs.index.max() <= T0 - pd.Timedelta(hours=model.LAG_H)
    assert obs.dropna().index.max() <= T0 - pd.Timedelta(hours=model.LAG_H)


def test_enlil_run_too_new_or_too_old_is_rejected():
    ser = pd.Series(500.0, index=pd.date_range("2026-11-01", periods=300, freq="h"))
    new_run = T0 - pd.Timedelta(hours=11)          # started < 12 h before t0
    old_run = T0 - pd.Timedelta(hours=73)          # older than 72 h
    ok_run = T0 - pd.Timedelta(hours=30)
    assert not np.isfinite(model.enlil_for(T0, (new_run, ser.set_axis(pd.date_range(new_run, periods=300, freq="h"))))[0]).any()
    assert not np.isfinite(model.enlil_for(T0, (old_run, ser.set_axis(pd.date_range(old_run, periods=300, freq="h"))))[0]).any()
    vals, run = model.enlil_for(T0, (ok_run, ser.set_axis(pd.date_range(ok_run, periods=300, freq="h"))))
    assert run == ok_run and np.isfinite(vals[:40]).all()
    assert np.isnan(vals[-1])                       # lead beyond 72 h from the run start is not used


def test_core_outputs_72_finite_values():
    idx = pd.date_range(T0 - pd.Timedelta(days=36), T0 - pd.Timedelta(hours=3), freq="h")
    obs = pd.Series(450 + 60 * np.sin(np.arange(len(idx)) / 40), index=idx)
    pred, info = model.core(T0, obs, None, None, BLEND)
    assert pred.shape == (72,) and np.isfinite(pred).all()
    assert info["n_enlil"] == 0 and not info["strip_used"]


def test_ace_raises_when_every_source_fails(monkeypatch):
    def boom(url, timeout=30, retries=2):
        raise OSError("no network")
    monkeypatch.setattr(model, "http_get", boom)
    with pytest.raises(RuntimeError, match="no network"):
        model.fetch_ace(T0)


def test_ace_survives_one_failed_month(monkeypatch):
    good = fake_http(pd.Timestamp("2026-10-01"), 24 * 45)

    def flaky(url, timeout=30, retries=2):
        if "202610" in url:
            raise OSError("boom")
        return good(url)
    monkeypatch.setattr(model, "http_get", flaky)
    obs, src = model.fetch_ace(T0)
    assert obs is not None and "monthly:202611" in src and "monthly:202610" not in src

@pytest.mark.parametrize("obs", [None, pd.Series(dtype=float)])
def test_core_without_obs_does_not_crash(obs):
    pred, info = model.core(T0, obs, None, None, BLEND)
    assert pred.shape == (72,) and np.isfinite(pred).all()

def test_finalize_fallback_levels():
    idx = pd.date_range(T0 - pd.Timedelta(days=36), T0 - pd.Timedelta(hours=3), freq="h")
    obs = pd.Series(450.0, index=idx)
    _, info = model.finalize(T0, obs, ["x"], None, np.zeros(72))
    assert info["level"] == "full"
    _, info = model.finalize(T0, obs, ["x"], None, None)
    assert info["level"] == "no_strip"
    pred, info = model.finalize(T0, None, [], None, None)
    assert info["level"] == "no_ace" and np.allclose(pred, model.CLIM)

def test_strip_adjustment_is_additive_and_scaled():
    idx = pd.date_range(T0 - pd.Timedelta(days=36), T0 - pd.Timedelta(hours=3), freq="h")
    obs = pd.Series(450.0, index=idx)
    base, _ = model.core(T0, obs, None, None, BLEND)
    adj = np.full(72, 40.0)
    out, info = model.core(T0, obs, None, adj, BLEND)
    assert np.allclose(out - base, model.STRIP_LAMBDA * 40.0) and info["strip_used"]
