# Structural Pattern Learning System

This is the new direction for the project: keep Mach 2 as the human feedback
workbench, and use a separate learner to turn that feedback into a reusable
structural pattern model.

## Active Pieces

```text
Mach2AImarket.py
```

Use this to group similar days and mark groups as:

```text
Good pattern
Not useful
```

Those ratings are saved in:

```text
mach2_group_feedback.csv
```

```text
structural_pattern_learner.py
```

This builds day-level structural features across every downloaded symbol, trains
from `mach2_group_feedback.csv`, and scores all available days.

Outputs:

```text
model_artifacts/structural_day_features.csv
model_artifacts/feedback_training_rows.csv
model_artifacts/structural_feedback_model.joblib
model_artifacts/structural_day_scores.csv
```

```text
download_yfinance_universe.py
```

This downloads extra 5m forex and futures symbols into
`downloaded_historical_data`.

```text
automated_structural_feedback.py
```

This is the current automated feedback/backtest loop. It reads the day feature
universe, groups each symbol multiple ways, judges whether each group is tight
around its signature line, appends audited feedback, retrains the model, and
writes a comparison report.

```text
pattern_grouping_lab.py
```

This is the pattern-recognition inspection tool. It groups days into structural
families, scores how tight each family is, and writes reports without making
trade predictions or appending feedback.

## Main Commands

Retrain and rescore the local universe:

```powershell
.\.venv\Scripts\python.exe .\structural_pattern_learner.py --interval 5m
```

Run the automated feedback/backtest loop across 80% of symbols:

```powershell
.\.venv\Scripts\python.exe .\automated_structural_feedback.py --coverage 0.90 --group-counts 14 16 18 20 --min-group-days 3 --append-feedback
```

Run the pattern grouping lab without changing feedback:

```powershell
.\.venv\Scripts\python.exe .\pattern_grouping_lab.py --group-counts 12 14 16 18 20 --min-group-days 3
```

Download the default forex/futures expansion set:

```powershell
.\.venv\Scripts\python.exe .\download_yfinance_universe.py --period 60d --interval 5m
```

Run Mach 2:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\Mach2AImarket.py
```

## How The Learning Loop Improves

1. Run similar-day grouping in Mach 2.
2. Mark groups as `Good pattern` or `Not useful`.
3. Rerun `structural_pattern_learner.py`.
4. Open `model_artifacts/structural_day_scores.csv`.
5. Use high-scoring days as candidates for backtesting and deeper inspection.

## Current Model

The first model intentionally starts simple. It learns from these structural
features:

```text
avg_return
avg_range
avg_early_move
avg_shift_strength
avg_shift_count
```

This makes it easy to inspect and debug. The next upgrade should train on the
full normalized intraday shape columns (`shape_000`, `shape_001`, etc.) plus
cross-symbol regime features.

## Automated Feedback Rule

The automated feedback pass is based on the user's stated preference:

```text
Good groups should have days close to the signature line.
Good groups should have similar structure to the other days in the group.
Good groups should resemble the average day rather than being loose collections.
```

The runner checks:

```text
avg_shape_corr
signature_rmse
signature_band_width
same_direction_rate
median_abs_return
```

Volume shape is intentionally not used in the automated pass yet. We should test
volume only sparingly after the price-structure learner has a stable baseline.

## Pattern Grouping Refocus

The project direction is now explicitly pattern recognition first. Prediction
models are useful as measuring tools, but the core job is grouping days into
cleaner structural families.

Changes made for that direction:

```text
automated_structural_feedback.py defaults now use group counts 12, 14, 16, 18, 20
run_structural_feedback_sweeps.py now sweeps around 14-20 groups
pattern_grouping_lab.py creates non-predictive pattern-family reports
```

Latest grouping lab:

```text
total_groups: 6,470
20-group pass high-quality groups: 482
18-group pass high-quality groups: 390
16-group pass high-quality groups: 361
14-group pass high-quality groups: 273
12-group pass high-quality groups: 187
```

Interpretation: tighter group counts around 16-20 produce more coherent pattern
families than the older 6/8/10-style grouping. Keep using 16 as the center, but
let 18 and 20 compete when the universe has enough examples.

Latest grouping outputs:

```text
model_artifacts/pattern_grouping_candidates.csv
model_artifacts/pattern_grouping_members.csv
model_artifacts/pattern_grouping_lab_report.md
```

The dashboard now has a `Pattern Groups` tab for these files:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\model_dashboard.py --server.port 8502
```

The Pattern Groups tab supports:

```text
multi-ticker filtering
group-count filtering
pattern-type filtering
ticker quality comparison
signature-line comparison for selected pattern families
member-day line overlays for each selected family
representative date tables
```

Use it when checking whether a few tickers share clean structural families. The
member CSV is generated locally by `pattern_grouping_lab.py` and is intentionally
not committed because it is a larger derived artifact.

Latest audited grouping feedback pass:

```text
run_name: pattern_focus_g14_16_18_20
coverage: 90%
group_counts: 14, 16, 18, 20
automated feedback rows: 4,497
total feedback rows after append: 22,139
good pattern rows after append: 1,449
not useful rows after append: 20,690
```

Feedback model check after this pass:

```text
ROC AUC: 0.960
Average precision: 0.765
Good-pattern precision: 0.68
Good-pattern recall: 0.75
Good-pattern F1: 0.71
Accuracy: 0.96
```

This is a better shape-learning result than the previous feedback state because
the model is being taught from denser, more visually coherent pattern families.

## Feedback Quality Lesson

A strict de-duplication experiment collapsed the feedback file from 22,139 rows
to 12,050 rows because many distinct structural groups share rounded summary
signatures. Validation dropped sharply:

```text
average precision fell from 0.765 to 0.412
good-pattern F1 fell from 0.71 to 0.45
```

The file was restored to the richer 22,139-row feedback state. Do not de-dup
automated labels only by rounded signature, ticker, group name, and rating. The
rounded signature is too coarse to represent all distinct pattern families.

## Latest Automated Run

The latest automated run processed 84 of 105 symbols, which is 80% coverage.

It generated:

```text
model_artifacts/automated_structural_feedback.csv
model_artifacts/automated_feedback_backtest.csv
model_artifacts/learner_improvement_report.txt
```

Summary:

```text
Automated feedback rows: 1,529
Good pattern rows: 84
Not useful rows: 1,445
Before ROC AUC: 0.889
After ROC AUC: 0.929
Good group avg cohesion: 0.7220
Bad group avg cohesion: 0.6000
Good group test avg absolute return: 2.3962
Bad group test avg absolute return: 1.4538
```

## Latest Sweep Update

After adding more manual feedback, we ran three tighter 80% coverage sweeps with
higher similar-day group counts:

```text
tight_g12_14_16
very_tight_g14_18_22
balanced_g10_14_18
```

Final feedback/training state:

```text
Training rows: 9,391
Good pattern rows: 748
Not useful rows: 8,643
```

Final model validation:

```text
Good-pattern precision: 0.68
Good-pattern recall: 0.72
Good-pattern F1: 0.70
Overall accuracy: 0.95
```

The strongest tightness pass was:

```text
very_tight_g14_18_22
good_avg_corr: 0.922453
good_avg_cohesion: 0.781633
good_test_avg_abs_return: 2.588363
bad_test_avg_abs_return: 1.402498
```

Continuation notes are in:

```text
NEXT_CHAT_HANDOFF.md
```

## Second Continuation Update

The latest continuation added ensemble scoring and an early-chart predictor.

Current feedback state:

```text
Total feedback rows: 17,688
Good pattern rows: 1,188
Not useful rows: 16,500
```

Current summary model:

```text
ROC AUC: 0.962
Average precision: 0.723
Good-pattern precision: 0.69
Good-pattern recall: 0.69
Accuracy: 0.96
```

The rich shape model is not good as a standalone classifier, but its ranking is
useful. The current best candidate ranking is:

```text
structural_day_ensemble_scores.csv
```

Score ranking lift:

```text
Summary top 1% lift vs bottom half: 3.045x
Rich top 1% lift vs bottom half: 4.289x
Ensemble top 1% lift vs bottom half: 4.473x
```

The early predictor tests whether the first ~35% of a day can predict whether
the full day becomes a top-decile structural pattern:

```text
early_pattern_predictor.py
early_pattern_predictor_report.txt
```

Latest early walk-forward result:

```text
ROC AUC: 0.912
Average precision: 0.555
Top 1% early prediction target rate: 0.8161
Top 5% early prediction target rate: 0.6582
Bottom 50% target rate: 0.0039
```

## Directional Early Predictor

Added:

```text
early_directional_predictor.py
early_directional_predictor_report.txt
early_directional_predictions.csv
```

This model splits high-structure candidates into:

```text
high-structure up day
high-structure down day
```

Important design correction: the directional model avoids full-day leakage. It
does not use full-day `avg_range`, `avg_shift_strength`, or `avg_shift_count`.
It uses `avg_early_move`, engineered early-path features, and the first 35% of
the saved normalized shape columns.

Latest walk-forward result:

```text
Up model ROC AUC: 0.831
Up model average precision: 0.249
Down model ROC AUC: 0.841
Down model average precision: 0.227
Top 1% early-up target rate: 0.5057
Top 1% early-down target rate: 0.4023
```

This is useful as a ranking/filtering signal, not a standalone trade trigger.
The next step is a bar-based trade simulation that combines:

```text
early_pattern_probability
early_up_probability
early_down_probability
early_direction_edge
```

## Early Trade Proxy

Added:

```text
early_trade_simulator.py
early_trade_simulator_report.md
early_trade_signals.csv
```

This joins the general early-structure predictor with the directional predictor
and tests signed full-day return as a proxy outcome.

Latest discovery result:

```text
pattern_threshold: 0.75
edge_threshold: 0.40
direction_probability_threshold: 0.55
signals: 35
avg_trade_return_proxy: 2.3456
median_trade_return_proxy: 2.2984
hit_rate_proxy: 1.0000
long_signals: 27
short_signals: 8
```

Important caution: this is a threshold discovery pass on the same walk-forward
test rows. It is not validated trading performance. The result is strong enough
to justify a true bar-entry simulation using raw intraday OHLCV.

## Bar Entry Simulation

Added:

```text
bar_entry_trade_simulator.py
bar_entry_trade_simulator_report.md
bar_entry_trade_results.csv
```

This is the first raw-OHLCV follow-through check. It uses the selected early
trade signals and simulates fixed entries at 5m bar indexes:

```text
20
25
28
```

Latest result across the discovery-selected signals:

```text
trades: 105
avg_close_return_pct: 0.6040
median_close_return_pct: 0.7012
hit_rate_to_close: 0.7524
avg_mfe_pct: 1.1811
avg_mae_pct: -0.3489
target_hit_rate: 0.4190
stop_hit_rate: 0.0190
```

Best entry was bar 20:

```text
bar_20_avg_close_return_pct: 0.8645
bar_20_hit_rate_to_close: 0.8286
```

## Early Signal Holdout Validation

Added:

```text
early_signal_holdout_validator.py
early_signal_holdout_validator_report.md
early_holdout_trade_signals.csv
early_holdout_bar_entry_results.csv
```

This chooses thresholds on the earlier half of the existing walk-forward
prediction rows, then evaluates those thresholds on the later half.

Latest holdout result:

```text
calibration_rows: 3,476
holdout_rows: 5,179
split_date: 2026-02-07
selected_pattern_threshold: 0.70
selected_edge_threshold: 0.20
selected_direction_probability_threshold: 0.50
holdout_signals: 37
holdout_proxy_avg_trade_return: 2.2231
holdout_proxy_hit_rate: 1.0000
holdout_bar_trades: 111
holdout_bar_avg_close_return_pct: 0.5423
holdout_bar_hit_rate_to_close: 0.7658
holdout_bar_20_avg_close_return_pct: 0.7985
holdout_bar_20_hit_rate_to_close: 0.8378
```

Interpretation: the early signal survives a stricter threshold holdout, but the
realistic bar-entry return is much smaller than the full-day proxy. Bar 20 is
currently the most promising entry point.

## Losing Signal Audit

Added:

```text
losing_signal_audit.py
losing_signal_audit_report.md
continuation_filter_experiment.py
continuation_filter_experiment_report.md
```

The audit shows that failures are mostly high-confidence long continuation
failures, not low-confidence model errors. In the stricter holdout rows:

```text
bar20_all_avg_close_return_pct: 0.7985
bar20_all_hit_rate: 0.8378
bar20_short_only_avg_close_return_pct: 1.2007
bar20_short_only_hit_rate: 1.0000
bar20_long_only_avg_close_return_pct: 0.6283
bar20_long_only_hit_rate: 0.7692
```

Best exploratory mixed guardrail:

```text
bar20_long_ensemble_gte_090_edge_lte_075_or_short
rows: 21
avg_close_return_pct: 1.1451
hit_rate: 0.9524
target_hit_rate: 0.7619
stop_hit_rate: 0.0000
```

Interpretation: raising early probability alone is not the next fix. The next
feature should measure whether the early long move is exhausted or still
building pressure after entry.

## Bar-20 Entry Quality Model

Added:

```text
bar20_entry_quality_model.py
bar20_entry_quality_features.csv
bar20_entry_quality_predictions.csv
bar20_entry_quality_model.joblib
bar20_entry_quality_model_report.txt
```

This builds raw OHLCV features known by bar 20 for every current walk-forward
prediction row, then trains a second-stage model to predict whether the implied
long/short side has positive bar-20-to-close follow-through.

Raw bar-20 features include:

```text
session_open_to_entry_pct
bar20_early_range_pct
bar20_close_position
bar20_distance_from_high_pct
bar20_distance_from_low_pct
bar20_last5_return_pct
bar20_last5_range_pct
bar20_last5_body_pct
bar20_volume_late_early_ratio
bar20_last5_volume_share
```

Latest chronological test:

```text
feature_rows: 8,655
train_rows: 4,567
test_rows: 4,088
split_date: 2026-02-26
ROC AUC: 0.729
Average precision: 0.770
accuracy: 0.67
```

Ranked behavior:

```text
top_1pct_rows: 41
top_1pct_avg_signed_return: 1.1619
top_1pct_hit_rate: 0.9512
top_5pct_rows: 205
top_5pct_avg_signed_return: 0.9376
top_5pct_hit_rate: 0.9024
bottom_50pct_avg_signed_return: -0.1292
bottom_50pct_hit_rate: 0.4056
```

Interpretation: this is now the best bridge between structural pattern
recognition and tradable entry quality. The next production-style signal should
combine:

```text
early_pattern_probability
early_direction_edge
bar20_entry_quality_probability
```

## Entry Timing Sweep

Added:

```text
entry_bar_timing_sweep.py
entry_bar_timing_features.csv
entry_bar_timing_sweep_summary.csv
entry_bar_timing_predictions.csv
entry_bar_timing_models.joblib
entry_bar_timing_sweep_report.md
```

This trains the same raw entry-quality model separately for several possible
5m entry bars:

```text
8, 12, 15, 18, 20, 22, 25, 28, 32
```

Latest result: bar 8 is currently best in the broader timing sweep.

```text
bar_8_top5_avg_signed_return: 1.3989
bar_8_top5_hit_rate: 0.9756
bar_8_top10_avg_signed_return: 1.3474
bar_8_top10_hit_rate: 0.9658
bar_8_roc_auc: 0.7430
bar_8_average_precision: 0.8529
```

For comparison:

```text
bar_20_top5_avg_signed_return: 0.9887
bar_20_top5_hit_rate: 0.9122
bar_20_roc_auc: 0.7289
bar_20_average_precision: 0.7700
```

Interpretation: bar 20 was not random, but it is no longer the best default.
Earlier structure appears to become tradable around bar 8 in the current data.
This needs a stricter live-style validation before replacing bar 20 everywhere.

The visual dashboard now includes an Entry Timing tab:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\model_dashboard.py --server.port 8502
```

## Combined Live-Style Ranker

Added:

```text
combined_live_signal_ranker.py
combined_live_signal_rankings.csv
combined_live_signal_ranker_report.md
```

The ranker currently favors the timing sweep's best bar, bar 8, and combines:

```text
0.45 * entry_quality_rank
0.20 * early_pattern_rank
0.20 * abs(direction_edge)_rank
0.15 * ensemble_rank
```

Latest comparison:

```text
bar_8_top5_avg_signed_return: 1.4932
bar_8_top5_hit_rate: 0.9709
bar_20_top5_avg_signed_return: 1.0253
bar_20_top5_hit_rate: 0.9171
```

Default bar-8 combined score:

```text
top_1pct_avg_signed_return: 1.9410
top_1pct_hit_rate: 0.9756
top_5pct_avg_signed_return: 1.4932
top_5pct_hit_rate: 0.9709
top_10pct_avg_signed_return: 1.3584
top_10pct_hit_rate: 0.9535
```

Guarded candidates only:

```text
top_5pct_avg_signed_return: 1.7088
top_5pct_hit_rate: 0.9722
```

The dashboard now has a Combined Ranker tab.

Detailed handoff:

```text
NEXT_CHAT_HANDOFF_2.md
```

## Important Note

The model is only as smart as the feedback file. Right now there are enough rows
to prove the loop works, but not enough to trust it blindly. The next big gain
comes from rating many more structural groups across NVDA, ETH, indexes, forex,
and futures.
