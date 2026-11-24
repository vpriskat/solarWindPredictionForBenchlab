# Roadmap

- [x] **Phase 0: foundation.** Package layout, tests (leak guards), Docker, CI.
- [ ] **Phase 1: evaluation layer.** Truth fetcher, scoring CLI (RMSE + the competition's TIMSE-based score), replay harness that writes Parquet.
- [ ] **Phase 2: live pipeline.** Forecast every 6 h (GitHub Actions cron → Prefect flow), ingest truth, update the metrics table, feed-health checks (ACE lag, ENLIL age, AIA count).
- [ ] **Phase 3: website.** Live forecast vs truth, scoreboard vs persistence/recurrence, feed health, "method and negative results", architecture.
- [ ] **Phase 4: extras.** MLflow for the experiment ladder, Evidently drift reports, online recalibration experiment.
- [ ] **Phase 5: agent layer.** Weekly reviewer first; a constrained experiment proposer behind the fixed evaluation gate.

Minimum showcase = phases 0–3.
