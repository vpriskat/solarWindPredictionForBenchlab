"""Common benchmark on top of the OFFICIAL Benchlab scorer (benchlab-starter-kit/scoring_demo/swp_scoring).

Truth, persistence fill, missing-cycle handling and the metric are the platform's own code, so the numbers are what the
leaderboard would compute for the same cycles.  What this script adds: several models side by side, overlapping windows,
paired block-bootstrap confidence intervals, and an optional "without this event" table.

  git clone https://github.com/Trillium-Technologies/benchlab-starter-kit        # once (the scorer lives in scoring_demo/)
  python bench_score.py --kit benchlab-starter-kit/scoring_demo \
      --pred mine=results/mine.json --pred partner=partner_forecasts/ \
      --live --cache-dir data/ace_monthly --exclude 2026-01-14:2026-01-22 --csv results/bench.csv

Ground truth: NOAA SWPC monthly ACE SWEPAM files (status flag 0 only, -9999.9 = missing), the same files the platform uses.
  --live             download the months a window needs from NASA/SWPC (saved in --cache-dir, so re-runs work offline)
  --archive-dir DIR  use already downloaded YYYYMM_ace_swepam_1h.txt files (the kit ships Jun-Sep 2026 in scoring_demo/data/ace)

Rules copied from the platform (do not change them if you want comparable numbers)
  * the grid is EVERY scheduled cycle (00/06/12/18 UTC) in the window, the same for every model;
  * a cycle a model has no forecast for, and any null / NaN / out-of-range value, is filled with persistence = the latest valid
    observation at or before t0 (looking back at most 72 h).  Coverage is printed, because filling is not free;
  * S_tau = sqrt(RMSE_tau * TRMSE_tau) per lead, S_final = mean over the 72 leads (lower is better).
  --common-only keeps just the cycles every model really forecast (use it when a backtest file has holes).

Forecast files (one file, or a directory of files that are merged), any of
  * the kit format      {"name": ..., "forecasts": {"20260701T060000Z": {"forecast_speed_kms": [72 values]}, ...}}
  * container outputs   {"t0": "2026-11-11T06:00:00Z", "forecast_speed_kms": [...]}   (one per file, or a list of them)
  * a wide table (.parquet / .csv): a t0 column (or index) + 72 columns h01..h72 (or 1..72)
  * a long table (.parquet / .csv): t0, lead_time, and a value column (pred / forecast / speed / value)
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON = 72
TIME_KEYS = ("t0", "issue_time", "issued_at", "forecast_time", "init_time", "timestamp", "time", "datetime", "date")
VALUE_KEYS = ("forecast_speed_kms", "forecast", "prediction", "predictions", "pred", "values", "speeds", "speed", "speed_km_s", "value")
LEAD_KEYS = ("lead_time", "lead", "horizon", "step", "hour", "h")


# ------------------------------------------------------------------------------------------------ official scorer
def import_scorer(kit):
    cands = [kit] if kit else []
    cands += ["benchlab-starter-kit/scoring_demo", "../benchlab-starter-kit/scoring_demo", "scoring_demo", "../scoring_demo"]
    for c in cands:
        p = Path(c).expanduser()
        if (p / "swp_scoring").is_dir():
            sys.path.insert(0, str(p.resolve()))
            return p.resolve()
    raise SystemExit("cannot find the official scorer: clone https://github.com/Trillium-Technologies/benchlab-starter-kit and pass "
                     "--kit <clone>/scoring_demo")


# ------------------------------------------------------------------------------------------------ forecasts
def _utc(x):
    """Anything time-like -> aware UTC datetime."""
    s = str(x).strip()
    for fmt in ("%Y%m%dT%H%M%SZ",):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    t = pd.Timestamp(s)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.to_pydatetime()


def _pick(keys, cands):
    low = {str(k).lower(): k for k in keys}
    return next((low[c] for c in cands if c in low), None)


def _num_list(x):
    return isinstance(x, list) and len(x) > 0 and all(v is None or isinstance(v, (int, float)) for v in x)


def _from_table(d):
    tcol, lcol, vcol = _pick(d.columns, TIME_KEYS), _pick(d.columns, LEAD_KEYS), _pick(d.columns, VALUE_KEYS)
    out = {}
    if tcol is not None and lcol is not None and vcol is not None and len(d.columns) < 10:       # long
        for t, g in d.groupby(tcol):
            row = [None] * HORIZON
            for l, v in zip(g[lcol], g[vcol]):
                if 1 <= int(l) <= HORIZON:
                    row[int(l) - 1] = None if pd.isna(v) else float(v)
            out[_utc(t)] = row
        return out
    cols = [c for c in d.columns if c != tcol]                                                     # wide
    hmap = {}
    for c in cols:
        s = str(c).lower().lstrip("h").replace("lead_", "").replace("+", "")
        if s.isdigit() and 1 <= int(s) <= HORIZON:
            hmap[int(s)] = c
    if len(hmap) != HORIZON:
        num = [c for c in cols if pd.api.types.is_numeric_dtype(d[c])]
        if len(num) != HORIZON:
            raise SystemExit(f"cannot find {HORIZON} forecast columns (named h01..h72: {len(hmap)}, numeric: {len(num)}): {cols[:8]}")
        hmap = {i + 1: c for i, c in enumerate(num)}
    times = d[tcol] if tcol is not None else d.index
    for t, vals in zip(times, d[[hmap[h] for h in range(1, HORIZON + 1)]].to_numpy(float)):
        out[_utc(t)] = [None if np.isnan(v) else float(v) for v in vals]
    return out


def _from_json_obj(obj, path):
    """-> {t0: [72 values]}"""
    if isinstance(obj, dict) and isinstance(obj.get("forecasts"), dict):                    # kit format
        return {_utc(k): (b.get("forecast_speed_kms") if isinstance(b, dict) else None) for k, b in obj["forecasts"].items()}
    if isinstance(obj, dict) and isinstance(obj.get("forecasts"), list):
        return _from_json_obj(obj["forecasts"], path)
    if isinstance(obj, dict) and obj and all(_num_list(v) and len(v) == HORIZON for v in obj.values()):   # {t0: [72]}
        return {_utc(k): v for k, v in obj.items()}
    recs = obj if isinstance(obj, list) else [obj]
    out = {}
    for r in recs:
        if not isinstance(r, dict):
            raise SystemExit(f"{path}: unexpected JSON content")
        tk = _pick(r.keys(), TIME_KEYS)
        vk = _pick(r.keys(), VALUE_KEYS)
        if tk is None or vk is None:
            raise SystemExit(f"{path}: need an issue-time key and a value key, found {list(r.keys())[:10]} "
                             "(extend TIME_KEYS / VALUE_KEYS at the top of bench_score.py)")
        v = r[vk]
        if _num_list(v):
            out[_utc(r[tk])] = v
        elif isinstance(v, list) and v and isinstance(v[0], dict):                           # [{lead, value}, ...]
            row = [None] * HORIZON
            for e in v:
                lk, ek = _pick(e.keys(), LEAD_KEYS), _pick(e.keys(), VALUE_KEYS)
                if lk and ek and 1 <= int(e[lk]) <= HORIZON:
                    row[int(e[lk]) - 1] = e[ek]
            out[_utc(r[tk])] = row
        else:
            raise SystemExit(f"{path}: cannot read the 72 values under '{vk}'")
    return out


def load_model(path):
    """-> (name, {t0: [72 raw values or None]})"""
    p = Path(path)
    files = sorted(f for f in p.rglob("*") if f.suffix.lower() in (".json", ".parquet", ".pq", ".csv")) if p.is_dir() else [p]
    if not files:
        raise SystemExit(f"no forecast files in {path}")
    name, merged = p.stem if not p.is_dir() else p.name, {}
    for f in files:
        suf = f.suffix.lower()
        if suf == ".json":
            obj = json.loads(f.read_text())
            if isinstance(obj, dict) and obj.get("name") and len(files) == 1:
                name = obj["name"]
            merged.update(_from_json_obj(obj, f))
        else:
            merged.update(_from_table(pd.read_parquet(f) if suf in (".parquet", ".pq") else pd.read_csv(f)))
    return name, merged


# ------------------------------------------------------------------------------------------------ truth
def load_observations(contract, start, end, archive_dir, live, cache_dir, A):
    """{valid_time: speed} of VALID rows (status 0, not -9999.9) from the SWPC monthly files, cached when downloaded."""
    obs, lo, hi = {}, A.format_valid_time(start), A.format_valid_time(end)
    for yyyymm in A.months_between(start, end):
        name = Path(A.month_url(yyyymm)).name
        text = None
        for d in (cache_dir, archive_dir):
            if d and (Path(d) / name).exists():
                text = (Path(d) / name).read_text(encoding="ascii", errors="replace")
                break
        if text is None:
            if not live:
                raise SystemExit(f"missing {name} (needed for {yyyymm}); add --live to download it, or point --archive-dir at it")
            print(f"  downloading {name} ...")
            text = A.fetch_text(A.month_url(yyyymm))
            if cache_dir:
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                (Path(cache_dir) / name).write_text(text, encoding="ascii", errors="replace")
        for row in A.parse(text):
            if lo <= row.valid_time <= hi and row.is_valid(contract):
                obs[row.valid_time] = row.speed
    return obs


# ------------------------------------------------------------------------------------------------ scoring
def score_matrices(S, obs, pred, contract):
    return S.score(obs, pred, contract.percentile_intervals, contract.quantile_method)


def summarize(rep):
    s, r, t = np.array(rep.s_curve, float), np.array(rep.rmse_curve, float), np.array(rep.trmse_curve, float)
    return dict(S_final=rep.s_final, RMSE=rep.mean_rmse, TRMSE=rep.mean_trmse, TIMSE=float(np.nanmean(t ** 2)), bias=rep.mean_bias,
                S_1_24=float(np.nanmean(s[:24])), S_25_48=float(np.nanmean(s[24:48])), S_49_72=rep.s_49_72)


def block_bootstrap(S, contract, cycles, obs, preds, ref, n_boot, seed, block_days=5):
    """Paired: S_final(model) - S_final(ref) on the same resampled blocks of days.  -> {model: (mean, lo, hi)}"""
    days = np.array([c.date().toordinal() for c in cycles])
    block = (days - days.min()) // block_days
    groups = [np.flatnonzero(block == b) for b in np.unique(block)]
    rng = np.random.default_rng(seed)
    names = list(preds)
    diffs = {n: [] for n in names if n != ref}
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        sref = score_matrices(S, obs[idx], preds[ref][idx], contract).s_final
        for n in diffs:
            diffs[n].append(score_matrices(S, obs[idx], preds[n][idx], contract).s_final - sref)
    return {n: (float(np.mean(v)), *np.percentile(v, [2.5, 97.5])) for n, v in diffs.items()}


def parse_windows(specs):
    out = []
    for s in specs:
        name, a, b = s.split(":")
        out.append((name, _utc(a), _utc(b) + pd.Timedelta(hours=23, minutes=59).to_pytimedelta()))
    return out


DEFAULT_WINDOWS = ["Jan-Apr:2026-01-01:2026-04-30", "Mar-Jun:2026-03-01:2026-06-30", "May-Aug:2026-05-01:2026-08-31",
                   "Jan-Aug:2026-01-01:2026-08-31"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", action="append", default=[], help="MODEL FOLDER with one subfolder per window (jan-apr/, mar-jun/, may-aug/, jan-aug/), "
                    "each holding that run's per-cycle forecast JSONs; repeat per model (name = folder name or manifest.json 'name')")
    ap.add_argument("--pred", action="append", default=[], help="[name=]path (file or directory) used for ALL windows; repeat per model")
    ap.add_argument("--kit", default=None, help="path to benchlab-starter-kit/scoring_demo")
    ap.add_argument("--live", action="store_true", help="download missing ACE months from NASA/SWPC")
    ap.add_argument("--archive-dir", default=None, help="folder with YYYYMM_ace_swepam_1h.txt files")
    ap.add_argument("--cache-dir", default=None, help="folder where downloaded months are kept (recommended with --live)")
    ap.add_argument("--windows", nargs="*", default=DEFAULT_WINDOWS, help="name:start:end (cycle dates, inclusive)")
    ap.add_argument("--ref", default=None, help="reference model for the confidence intervals (default: the first --pred)")
    ap.add_argument("--common-only", action="store_true", help="score only the cycles every model forecast")
    ap.add_argument("--exclude", default=None, help="start:end cycle dates, adds a second table without them (e.g. 2026-01-14:2026-01-22)")
    ap.add_argument("--no-baselines", action="store_true", help="do not add the three official baselines as rows")
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    kit = import_scorer(a.kit)
    from swp_scoring.core import evaluate as E, grid as G, metric as S, types as T
    from swp_scoring.sources import ace_archive as A

    class _A:                                                    # tiny namespace so helper signatures stay short
        format_valid_time = staticmethod(T.format_valid_time)
        months_between, month_url, fetch_text, parse = (staticmethod(f) for f in (A.months_between, A.month_url, A.fetch_text, A.parse))

    contract = T.load_competition_contract()
    archive = a.archive_dir or str(kit / "data" / "ace")

    wins = parse_windows(a.windows)
    if not a.pred and not a.runs:
        raise SystemExit("give --runs <model folder> (one subfolder per window) or --pred <file>")
    models_w = {label: {} for label, _, _ in wins}              # window -> {model: raw forecasts}
    for spec in a.pred:                                          # one file / folder, cut to each window by the grid
        name, sep, path = spec.partition("=")
        nm, raw = load_model(path)
        nm = name if sep else nm
        for label in models_w:
            models_w[label][nm] = raw
        ts = sorted(raw)
        print(f"  {nm}: {len(raw):,} forecasts, {ts[0]:%Y-%m-%d %H:%M} .. {ts[-1]:%Y-%m-%d %H:%M}")
    for run in a.runs:                                           # <model>/<window>/*.json
        root = Path(run).expanduser()
        if not root.is_dir():
            raise SystemExit(f"--runs {run}: not a folder")
        nm = root.name
        if (root / "manifest.json").exists():
            nm = json.loads((root / "manifest.json").read_text()).get("name", nm)
        subs = {d.name.lower(): d for d in root.iterdir() if d.is_dir()}
        seen = {}
        for label, _, _ in wins:
            d = subs.get(label.lower())
            if d is None:
                print(f"  {nm}: NO folder for window {label!r} (expected {root / label.lower()}) -> not scored there")
                continue
            _, raw = load_model(d)
            models_w[label][nm] = raw
            for t, v in raw.items():                             # same cycle in two overlapping runs: report differences
                if t in seen and not np.allclose(np.array(seen[t], float), np.array(v, float), equal_nan=True):
                    seen.setdefault("_diff", set()).add(t)
                seen[t] = v
            ts = sorted(raw)
            print(f"  {nm} [{label}]: {len(raw):,} forecasts, {ts[0]:%Y-%m-%d %H:%M} .. {ts[-1]:%Y-%m-%d %H:%M}")
        if seen.get("_diff"):
            print(f"  note: {nm}: {len(seen['_diff'])} cycles differ between overlapping runs (fine if each run was retrained/re-run; "
                  f"each window is scored only on its own folder)")
    names = list(dict.fromkeys(n for m in models_w.values() for n in m))
    ref = a.ref or names[0]
    if ref not in names:
        raise SystemExit(f"--ref {ref!r} is not one of {names}")
    parsed_w = {label: {n: {t: G.parse_forecast({contract.forecast_field: v}, contract) for t, v in raw.items()} for n, raw in m.items()}
                for label, m in models_w.items()}

    obs_lo, obs_hi = (min(E.observation_window(s, e, contract)[0] for _, s, e in wins),
                      max(E.observation_window(s, e, contract)[1] for _, s, e in wins))
    observations = load_observations(contract, obs_lo, obs_hi, archive, a.live, a.cache_dir, _A)
    print(f"  ground truth: {len(observations):,} valid ACE hours (status 0) {obs_lo:%Y-%m-%d} .. {obs_hi:%Y-%m-%d}\n")

    excl = None
    if a.exclude:
        x0, x1 = a.exclude.split(":")
        excl = (_utc(x0), _utc(x1) + pd.Timedelta(hours=23, minutes=59).to_pytimedelta())

    all_rows = []
    for label, start, end in wins:
        parsed = parsed_w[label]
        if not parsed:
            print(f"[{label}] no model has a run for this window\n")
            continue
        ref_w = ref if ref in parsed else next(iter(parsed))
        grid = E.build_grid(observations, start, end, contract)
        cycles = list(grid.cycles)
        keep = np.isfinite(grid.persistence)
        if a.common_only:
            keep &= np.array([all(any(np.isfinite(f.get(c, np.full(1, np.nan)))) for f in parsed.values()) for c in cycles])
        if not keep.any():
            print(f"[{label}] no scorable cycles\n")
            continue
        for tag, mask in [(label, keep)] + ([(f"{label} (without {a.exclude})", keep & np.array([not (excl[0] <= c <= excl[1]) for c in cycles]))] if excl else []):
            if tag != label and mask.sum() == keep.sum():
                continue
            cyc = [c for c, k in zip(cycles, mask) if k]
            obs = grid.obs[mask]
            pers = grid.persistence[mask]
            preds, cov = {}, {}
            for n, fc in parsed.items():
                preds[n] = G.prediction_matrix(cyc, fc, pers, HORIZON)
                sub = [c for c in cyc if c in fc]
                raw = np.array([fc[c] if c in fc else np.full(HORIZON, np.nan) for c in cyc], float)
                scor = np.isfinite(obs)                          # only cells that have a truth value are scored
                filled = float((~np.isfinite(raw) & scor).sum() / max(scor.sum(), 1))
                cov[n] = (len(sub), filled)
            base = {}
            if not a.no_baselines:
                g2 = E.EvaluationGrid(cyc, grid.observations, obs, pers)
                base = {f"official:{k}": v for k, v in E.baseline_predictions(g2, contract).items()}
            rows = []
            for n, p in {**preds, **base}.items():
                m = summarize(score_matrices(S, obs, p, contract))
                rows.append(dict(window=tag, model=n, cycles=len(cyc), forecast_cycles=cov.get(n, (len(cyc), 0))[0],
                                 filled_pct=100 * cov.get(n, (0, 0))[1], **m))
            ci = block_bootstrap(S, contract, cyc, obs, preds, ref_w, a.boot, a.seed) if (a.boot and len(preds) > 1 and len(cyc) >= 20) else {}
            for r in rows:
                if r["model"] in ci:
                    r["dS_vs_ref"], r["ci_lo"], r["ci_hi"] = ci[r["model"]]
            t = pd.DataFrame(rows)
            all_rows.append(t)
            show = t.copy()
            if "dS_vs_ref" in show:
                show["vs " + ref_w] = [("" if pd.isna(x) else f"{x:+.2f} [{lo:+.2f},{hi:+.2f}]{'*' if (lo > 0 or hi < 0) else ''}")
                                     for x, lo, hi in zip(show.dS_vs_ref, show.ci_lo, show.ci_hi)]
                show = show.drop(columns=["dS_vs_ref", "ci_lo", "ci_hi"])
            show = show.drop(columns=["window"]).sort_values("S_final")
            print(f"=== {tag}  ({start:%Y-%m-%d} .. {end:%Y-%m-%d}, {len(cyc)} cycles) ===")
            print(show.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
            print()
    print("S_final = mean over the 72 leads of sqrt(RMSE_tau*TRMSE_tau); TIMSE = TRMSE^2 (mean over leads); lower is better.  "
          "filled_pct = share of SCORED values (those with a truth value) replaced by persistence.  'vs ref' = S_final(model) - S_final(ref), paired 5-day "
          "block bootstrap 95% CI, * = CI excludes 0 (negative = better than the reference).")
    if a.csv and all_rows:
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        pd.concat(all_rows).to_csv(a.csv, index=False)
        print("wrote", a.csv)


if __name__ == "__main__":
    main()
