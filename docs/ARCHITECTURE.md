# Architecture

**Model.** Per 6-lead group ridge blend of `[persistence, rec27_eff, enlil]` (fitted on 2025-01…2025-09) + 0.75 × coronal-hole adjustment (ridge on AIA-193 strips, fitted ≤ 2024). `core()` is a pure function of its inputs, so it is testable offline.

**Layers and fallbacks.** full → no strip → no ENLIL (replaced by persistence) → persistence → constant 440 km/s. The final clip is [180, 1200] km/s.

**Known limits.** The SWPC ENLIL feed holds only the latest run, which is often 2–3 days old, so ENLIL covers only the first leads. NOAA's ACE files lag 2–5 h and have invalid hours. See `research/` for the evidence behind each decision.
