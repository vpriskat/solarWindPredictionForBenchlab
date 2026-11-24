# Experiment protocol (for trying new model ideas)

1. **No leaks.** Every input must be available at t0 with the live delays: ACE ≤ t0 − 3 h, ENLIL run ≤ t0 − 12 h, AIA ≤ t0 − 1 h. A feature that uses data up to t0 (as the old Timestral feature did) is invalid even if it scores well.
2. **Same ladder.** Compare against persistence, 27-day recurrence and the current model (`FINAL strip x0.75`) on the same windows (Jan–Apr, Mar–Jun, May–Aug 2026) with the official scorer.
3. **Same gate.** Paired 5-day block bootstrap. Keep a change only if the gain is above ~1.5 S **and** the CI excludes 0.
4. **Many tries need a correction.** If you test many ideas, hold back a forward period (live cycles) that you never tuned on.
5. **Log everything,** including failures. Negative results are part of the showcase.

Idea log: copy the table below into `notes/ideas.md`.

| idea | source (paper/link) | hypothesis | cost | result | gate passed? |
|---|---|---|---|---|---|
