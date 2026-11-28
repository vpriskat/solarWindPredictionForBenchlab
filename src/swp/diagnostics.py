"""Diagnostics for one forecast cycle: pure functions returning JSON-serialisable values."""
import hashlib, os, platform
import numpy as np
import pandas as pd

SCHEMA_VERSION = 1


def _ts(x):
    return None if x is None else pd.Timestamp(x).strftime("%Y-%m-%dT%H:%M:%SZ")


def _h(a, b):
    return round((pd.Timestamp(a) - pd.Timestamp(b)) / pd.Timedelta(hours=1), 2)


def longest_run(mask):
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def ace_diag(t0, obs, window_h=72):
    if obs is None or obs.dropna().empty:
        return {"ace_n_rows": 0 if obs is None else int(len(obs)), "ace_n_valid": 0}
    valid = obs.dropna()
    last, cut = valid.index.max(), obs.index.max()
    win = obs.iloc[-window_h:]
    return {
        "ace_first_label": _ts(obs.index.min()), "ace_last_label": _ts(last),
        "ace_last_lag_h": _h(t0, last), "ace_trailing_nan_h": _h(cut, last),
        "ace_n_rows": int(len(obs)), "ace_n_valid": int(len(valid)),
        "ace_missing_72h": int(win.isna().sum()), "ace_longest_gap_h": int(longest_run(win.isna().to_numpy())),
        "ace_last_value": round(float(valid.iloc[-1]), 1), "ace_mean_24h": round(float(valid.iloc[-24:].mean()), 1),
    }


def enlil_diag(t0, enlil):
    if enlil is None:
        return {"enlil_n_points": 0}
    run, ser = enlil
    return {
        "enlil_run": _ts(run), "enlil_run_age_h": _h(t0, run),
        "enlil_first_valid": _ts(ser.index.min()), "enlil_last_valid": _ts(ser.index.max()),
        "enlil_series_end_lead_h": _h(ser.index.max(), t0), "enlil_n_points": int(len(ser)),
    }


def strip_diag(adj):
    if adj is None:
        return {}
    a = np.asarray(adj, float)
    return {"strip_adj_mean": round(float(np.nanmean(a)), 2), "strip_adj_min": round(float(np.nanmin(a)), 2),
            "strip_adj_max": round(float(np.nanmax(a)), 2)}


def forecast_diag(pred, lo=180.0, hi=1200.0):
    steps = np.abs(np.diff(pred))
    return {
        "fc_mean": round(float(np.mean(pred)), 1), "fc_min": round(float(np.min(pred)), 1), "fc_max": round(float(np.max(pred)), 1),
        "fc_n_clipped": int(np.sum((pred <= lo) | (pred >= hi))),
        "fc_max_step_kms": round(float(steps.max()), 1), "fc_max_step_lead": int(steps.argmax()) + 2,
    }


def _sha(path):
    try:
        return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]
    except OSError:
        return None


def provenance(artifacts_dir):
    import numpy, pandas, scipy
    return {
        "schema_version": SCHEMA_VERSION,
        "git_sha": os.environ.get("SWP_GIT_SHA", "unknown"), "run_url": os.environ.get("SWP_RUN_URL"),
        "python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__,
        "blend_sha": _sha(os.path.join(artifacts_dir, "final_blend.json")),
        "strip_sha": _sha(os.path.join(artifacts_dir, "strip_model.npz")),
    }


def describe_cycle(t0, obs, enlil, adj, pred):
    return {**ace_diag(t0, obs), **enlil_diag(t0, enlil), **strip_diag(adj), **forecast_diag(pred)}