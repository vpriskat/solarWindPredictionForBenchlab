# Common benchmark: 2026 backtest, official scorer (scoring only)

Each developer produces the forecasts with their own code. `bench_score.py` only reads the results and computes the metrics.

## Folder contract (one folder per model)
    <model>/
      manifest.json            optional: {"name": "mine", "notes": "...", "lag_hours": 3}
      jan-apr/                 one run = one subfolder, named exactly like the window
        20260101T000000Z.json  one file per cycle, = the container output:
        20260101T060000Z.json    {"t0": "2026-01-01T00:00:00Z", "valid_from": "...", "forecast_speed_kms": [72 floats]}
        ...  (00/06/12/18 UTC, 2026-01-01 .. 2026-04-30 = 480 files)
      mar-jun/                 2026-03-01 .. 2026-06-30 (488 files)
      may-aug/                 2026-05-01 .. 2026-08-31 (492 files)
      jan-aug/                 optional, 972 files
A subfolder may instead hold a single kit-format file ({"name","forecasts":{stamp:{...}}}). Overlapping months may differ between
runs (e.g. each run retrained on data before its own start); a window is scored only on its own subfolder.

## Rules (identical for both)
1. Information cut-off t0 - 3 h for every feed (ACE, ENLIL, images, OMNI). 2. No training on the scored period.
3. A missing / null / NaN / out-of-range cycle is scored as persistence (platform rule); coverage is printed (filled_pct).
4. Numbers come only from bench_score.py (official scorer code).

## Run
    python bench_score.py --kit <kit>/scoring_demo --live --cache-dir data/ace_monthly \
        --runs results/mine --runs results/partner --ref mine --exclude 2026-01-14:2026-01-22 --csv results/bench.csv
Prints per window: S_final, RMSE, TRMSE, TIMSE, bias, S for leads 1-24/25-48/49-72, cycles, forecast_cycles, filled_pct, and the
paired block-bootstrap difference vs --ref. Official persistence / 27-day recurrence / climatology rows are added as yardsticks.
