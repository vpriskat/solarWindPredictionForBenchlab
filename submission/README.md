# Solar-wind speed forecast (t0+1…t0+72 h)

**Method.** Per-lead-group (6 h) ridge blend of three inputs: last valid ACE/SWEPAM speed, the 27-day recurrence value (25 h centred mean if missing) and the NOAA WSA-ENLIL speed at Earth; coefficients fitted offline on 2025-01…2025-09. A ridge correction (weight 0.75) uses coronal-hole area strips segmented from near-real-time SDO/AIA 193 Å images (00:00 UTC, lags 0–72 h), fitted on data ≤ 2024. Missing inputs fall back to persistence; output is clipped to [180, 1200] km/s.

**Data cut-off (enforced in `model.py`).**
1. ACE: NOAA hourly SWPC files have no publication time, so we admit only measurements labelled ≤ t0 − 3 h (constant `LAG_H`). Labels are read as the timestamp of the hourly record as published; status 0 only.
2. WSA-ENLIL: only a run whose start time ≤ t0 − 12 h and age ≤ 72 h; otherwise ignored.
3. AIA: images with observation time ≤ t0 − 1 h.
4. All data are fetched live during the run; the retrieval time (UTC) is logged and written to the `meta` field of the output.

**Domains:** services.swpc.noaa.gov, sohoftp.nascom.nasa.gov, jsoc1.stanford.edu. No training data or pre-computed forecasts are in the image.
