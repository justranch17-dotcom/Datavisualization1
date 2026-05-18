# Next Chat Handoff

Continue the structural pattern learning project from here.

## What Was Done In This Chat

- Read the user's new manual feedback in `mach2_group_feedback.csv`.
- Added configurable thresholds to `automated_structural_feedback.py`.
- Added `run_structural_feedback_sweeps.py`.
- Ran three 80% coverage feedback/backtest sweeps:

```text
tight_g12_14_16
very_tight_g14_18_22
balanced_g10_14_18
```

- Appended audited automated feedback after each sweep.
- Retrained `structural_pattern_learner.py` after all sweeps.

## Current Feedback State

```text
Total training rows: 9,391
Good pattern rows: 748
Not useful rows: 8,643
```

The model is intentionally strict. It strongly rejects loose groups.

## Latest Model Validation

From the final `structural_pattern_learner.py` run:

```text
precision good-pattern class: 0.68
recall good-pattern class: 0.72
f1 good-pattern class: 0.70
accuracy: 0.95
```

## Best Sweep

The strictest structure quality came from:

```text
very_tight_g14_18_22
```

It had:

```text
good_avg_corr: 0.922453
good_avg_cohesion: 0.781633
good_test_avg_abs_return: 2.588363
bad_test_avg_abs_return: 1.402498
```

The best final model balance came after:

```text
balanced_g10_14_18
```

## Current Top Scored Candidates

Open:

```text
model_artifacts/structural_day_scores.csv
```

Recent top names included:

```text
COST 2025-06-05
BTC_USD 2024-09-04
NFLX 2025-07-22
XLI 2025-11-13
ETH_USD 2025-08-02
GC_F 2026-03-20
MU 2025-10-24
MA 2025-01-27
ORCL 2025-06-20
NVDA 2026-02-12
QQQ 2025-02-27
QQQ 2026-01-21
```

## Continue With This Command

Run another stricter sweep, ideally extending group counts:

```powershell
.\.venv\Scripts\python.exe .\automated_structural_feedback.py --coverage 0.80 --group-counts 18 22 26 --min-group-days 3 --min-corr 0.91 --max-rmse 0.95 --max-band 1.05 --min-same-direction 0.74 --min-median-abs-return 0.35 --run-name next_very_tight_g18_22_26 --append-feedback
```

Then retrain:

```powershell
.\.venv\Scripts\python.exe .\structural_pattern_learner.py --interval 5m
```

## Important Guidance

- Keep volume shape out of the core loop for now.
- Use volume only as a separate experiment after price-structure quality stabilizes.
- The user's preference is visual/structural tightness:

```text
days close to the signature line
similar to each other
similar to the group average
not loose noisy clusters
```

- Watch for false negatives. The model is strict and may reject some usable patterns.
