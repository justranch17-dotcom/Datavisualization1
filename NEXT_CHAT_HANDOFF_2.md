# Next Chat Handoff 2

Continue from this file if a new chat starts.

## What Was Done In This Pass

Started from `NEXT_CHAT_HANDOFF.md` and continued the structural pattern learner.

Added:

```text
build_ensemble_scores.py
score_backtest_report.py
early_pattern_predictor.py
```

Updated:

```text
structural_pattern_learner.py
automated_structural_feedback.py
score_backtest_report.py
```

Ran:

```powershell
.\.venv\Scripts\python.exe .\automated_structural_feedback.py --coverage 0.80 --group-counts 18 22 26 --min-group-days 3 --min-corr 0.91 --max-rmse 0.95 --max-band 1.05 --min-same-direction 0.74 --min-median-abs-return 0.35 --run-name next_very_tight_g18_22_26 --append-feedback
```

Then ran an even stricter 90% coverage pass:

```powershell
.\.venv\Scripts\python.exe .\automated_structural_feedback.py --coverage 0.90 --group-counts 22 26 30 --min-group-days 3 --min-corr 0.92 --max-rmse 0.90 --max-band 1.00 --min-same-direction 0.76 --min-median-abs-return 0.40 --run-name ultra_tight_90pct_g22_26_30 --append-feedback
```

Then retrained and rebuilt scores:

```powershell
.\.venv\Scripts\python.exe .\structural_pattern_learner.py --interval 5m --model-mode both
.\.venv\Scripts\python.exe .\build_ensemble_scores.py
.\.venv\Scripts\python.exe .\score_backtest_report.py
.\.venv\Scripts\python.exe .\early_pattern_predictor.py
```

## Current Feedback State

```text
Total feedback rows: 17,688
Good pattern rows: 1,188
Not useful rows: 16,500
```

The model is now very strict. It is heavily trained to reject loose or noisy
groups.

## Summary Model

Current summary model validation:

```text
rows=17,688
positives=1,188
negatives=16,500
roc_auc=0.962
average_precision=0.723
good-pattern precision=0.69
good-pattern recall=0.69
good-pattern f1=0.69
accuracy=0.96
```

This remains the best classifier.

## Rich Shape Model

The rich shape model was added and trained on full normalized shape columns.

Important finding:

```text
As a classifier, the rich model is poor.
As a ranking signal, the rich model is useful.
```

Rich model validation:

```text
roc_auc=0.502
average_precision=0.166
```

Do not use rich model alone yet.

## Ensemble Score

Added:

```text
model_artifacts/structural_day_ensemble_scores.csv
```

The ensemble combines:

```text
60% summary model rank
40% rich shape model rank
```

This currently gives the best structural ranking behavior.

Score backtest:

```text
Summary top 1% lift vs bottom half: 3.045x
Rich top 1% lift vs bottom half: 4.289x
Ensemble top 1% lift vs bottom half: 4.473x
```

The ensemble is now the best candidate-ranking output.

Open:

```text
model_artifacts/score_backtest_report.md
model_artifacts/structural_day_ensemble_scores.csv
```

## Early Pattern Predictor

Added a walk-forward early-chart predictor:

```text
early_pattern_predictor.py
model_artifacts/early_pattern_predictor.joblib
model_artifacts/early_pattern_predictions.csv
model_artifacts/early_pattern_predictor_report.txt
```

It uses the first ~35% of the normalized intraday shape to predict whether the
full day will become a top-decile ensemble structural pattern.

Latest walk-forward test:

```text
train_rows=34,621
test_rows=8,655
roc_auc=0.912
average_precision=0.555
top_1pct_target_rate=0.8161
top_5pct_target_rate=0.6582
bottom_50pct_target_rate=0.0039
top_1pct_avg_abs_return=2.1177
top_5pct_avg_abs_return=1.9267
bottom_50pct_avg_abs_return=0.8776
```

This is the strongest evidence so far that the system can recognize likely
high-structure days early, not just label full days after the fact.

## Current Top Ensemble Candidates

Recent top ensemble candidates included:

```text
CVX 2025-04-03
CSCO 2025-11-12
DIA 2024-05-23
BA 2024-07-16
NFLX 2024-05-20
QCOM 2025-03-07
MCHI 2026-02-06
TXN 2025-10-10
TLT 2025-03-13
NKE 2024-08-02
BTC_USD 2024-09-09
MA 2025-01-27
LOW 2024-06-25
AAPL 2025-03-03
QQQ 2026-01-21
```

Top early predictions included:

```text
WFC 2025-12-11
WMT 2026-02-02
CVX 2026-01-02
NKE 2026-04-23
XOM 2026-01-08
V 2026-01-05
INDA 2026-04-23
DIS 2026-01-05
BA 2025-12-09
NQ_F 2026-05-08
NVDA 2026-03-02
```

## Next Best Improvements

1. Add a cleaner early-window feature builder from raw OHLCV instead of using the
   first 35% of the already normalized full-day shape.
2. Add direction-specific early predictors:

```text
high-structure up day
high-structure down day
```

3. Add a real trade-style backtest:

```text
At bar 20 or 25, if early_pattern_probability is high, estimate expected full-day direction/range.
```

4. Keep volume shape separate for now. Price structure is improving; volume can
   be tested as an optional second-channel experiment later.

## Continue Command

Good next command for another pass:

```powershell
.\.venv\Scripts\python.exe .\early_pattern_predictor.py
```

Then build a new script:

```text
early_directional_predictor.py
```

Goal:

```text
predict high-structure up vs high-structure down from early shape
```

## Third Continuation Update

This pass read:

```text
NEXT_CHAT_HANDOFF.md
NEXT_CHAT_HANDOFF_2.md
PATTERN_LEARNING_SYSTEM.md
MARKET_UNIVERSE.md
archive/phone_app_cleanup_2026-05-15/codex_tasks.md
archive/phone_app_cleanup_2026-05-15/streamlit_codex_phone_starter/streamlit_codex_phone_starter/app.py
archive/phone_app_cleanup_2026-05-15/streamlit_codex_phone_starter/streamlit_codex_phone_starter/scanner.py
```

Phone-app finding:

```text
The archived phone Streamlit app already has a phone-safe scanner adapter.
It reads mach2_group_feedback.csv and project status without importing
Mach2AImarket.py directly, which is the right direction for keeping the phone
app lightweight.
```

Added:

```text
early_directional_predictor.py
model_artifacts/early_directional_predictor.joblib
model_artifacts/early_directional_predictions.csv
model_artifacts/early_directional_predictor_report.txt
```

Ran:

```powershell
.\.venv\Scripts\python.exe .\early_directional_predictor.py
```

The directional model predicts two separate targets:

```text
target_high_structure_up = top-decile ensemble score and positive full-day return
target_high_structure_down = top-decile ensemble score and negative full-day return
```

Important improvement:

```text
The directional model avoids full-day leakage. It does not use avg_range,
avg_shift_strength, or avg_shift_count, because those are only fully known after
the day is over. It uses avg_early_move, early-path features, and the first 35%
of the saved normalized shape.
```

Latest directional walk-forward result:

```text
train_rows=34,621
test_rows=8,655
up_test_positives=474
down_test_positives=404
up_roc_auc=0.831
up_average_precision=0.249
down_roc_auc=0.841
down_average_precision=0.227
top_1pct_up_target_rate=0.5057
top_5pct_up_target_rate=0.3095
top_1pct_down_target_rate=0.4023
top_5pct_down_target_rate=0.2887
```

Interpretation:

```text
The directional signal is real but weaker than the general early structural
signal. Use it as a ranked bias/filter, not as a standalone trade trigger.
```

Best next script:

```text
early_trade_simulator.py
```

Goal:

```text
Join early_pattern_predictions.csv with early_directional_predictions.csv.
At high early_pattern_probability and strong early_direction_edge, simulate
simple long/short decisions and report average return, hit rate, drawdown proxy,
and behavior by ticker family.
```

Suggested command sequence for the next chat:

```powershell
.\.venv\Scripts\python.exe .\early_pattern_predictor.py
.\.venv\Scripts\python.exe .\early_directional_predictor.py
```

Then build:

```text
early_trade_simulator.py
```

## Fourth Continuation Update

Added:

```text
early_trade_simulator.py
model_artifacts/early_trade_simulator_report.md
model_artifacts/early_trade_signals.csv
```

Ran:

```powershell
.\.venv\Scripts\python.exe .\early_trade_simulator.py
```

What it does:

```text
Joins early_pattern_predictions.csv with early_directional_predictions.csv.
Filters for high early_pattern_probability plus strong early_direction_edge.
Uses signed full-day avg_return as a proxy outcome:
long = avg_return
short = -avg_return
```

Best discovery threshold set:

```text
pattern_threshold=0.75
edge_threshold=0.40
direction_probability_threshold=0.55
signals=35
avg_trade_return_proxy=2.345629
median_trade_return_proxy=2.298378
hit_rate_proxy=1.000000
long_signals=27
short_signals=8
```

Baseline comparison from the same test rows:

```text
all_test_rows=8,655
avg_abs_return=1.1316
early-move side avg return proxy=0.9133
early-move side hit rate proxy=0.7980
```

Important caution:

```text
This is a discovery pass, not validated trading performance. The threshold sweep
chooses the best thresholds on the same walk-forward test rows. The result is
strong enough to justify a real bar-entry simulator, but do not treat the 100%
proxy hit rate as durable yet.
```

Next best implementation:

```text
bar_entry_trade_simulator.py
```

Goal:

```text
For each signal row, open the matching raw ticker CSV, locate the session, enter
around bar 20/25 or after the first 35% window, and measure forward return,
max favorable excursion, max adverse excursion, stop/target behavior, and
time-to-target. This is the first point where the project can start judging
trade mechanics instead of only full-day structure.
```

## Fifth Continuation Update

Added:

```text
bar_entry_trade_simulator.py
model_artifacts/bar_entry_trade_results.csv
model_artifacts/bar_entry_trade_simulator_report.md
```

Ran:

```powershell
.\.venv\Scripts\python.exe .\bar_entry_trade_simulator.py
```

What it does:

```text
Uses early_trade_signals.csv.
Loads the matching raw 5m OHLCV session.
Tests fixed entry bars 20, 25, and 28.
Measures close return, MFE, MAE, target hits, stop hits, and bars to barrier.
```

Latest bar-entry result:

```text
trades=105
avg_close_return_pct=0.604019
median_close_return_pct=0.701231
hit_rate_to_close=0.752381
avg_mfe_pct=1.181103
avg_mae_pct=-0.348871
target_hit_rate=0.419048
stop_hit_rate=0.019048
```

Best entry bar:

```text
bar_20_trades=35
bar_20_avg_close_return_pct=0.864508
bar_20_median_close_return_pct=0.965529
bar_20_hit_rate_to_close=0.828571
```

Side finding:

```text
Shorts were stronger than longs in this sample:
short_avg_close_return_pct=1.050134
long_avg_close_return_pct=0.471837
```

Important interpretation:

```text
The full-day proxy was flattering the signal, but the signal still has real
follow-through after a fixed bar entry. Bar 20 currently looks better than
waiting to bar 25 or 28.
```

Added:

```text
early_signal_holdout_validator.py
model_artifacts/early_signal_holdout_validator_report.md
model_artifacts/early_holdout_trade_signals.csv
model_artifacts/early_holdout_bar_entry_results.csv
```

Ran:

```powershell
.\.venv\Scripts\python.exe .\early_signal_holdout_validator.py
```

What it does:

```text
Splits the existing walk-forward prediction rows chronologically.
Chooses thresholds on the earlier half.
Evaluates proxy and bar-entry behavior on the later half.
```

Latest holdout:

```text
calibration_rows=3,476
holdout_rows=5,179
split_date=2026-02-07
selected_pattern_threshold=0.70
selected_edge_threshold=0.20
selected_direction_probability_threshold=0.50
holdout_signals=37
holdout_proxy_avg_trade_return=2.223092
holdout_proxy_hit_rate=1.000000
holdout_bar_trades=111
holdout_bar_avg_close_return_pct=0.542335
holdout_bar_median_close_return_pct=0.525086
holdout_bar_hit_rate_to_close=0.765766
```

Holdout by entry:

```text
bar_20_avg_close_return_pct=0.798456
bar_20_hit_rate_to_close=0.837838
bar_25_avg_close_return_pct=0.489466
bar_25_hit_rate_to_close=0.756757
bar_28_avg_close_return_pct=0.339083
bar_28_hit_rate_to_close=0.702703
```

Next best improvements:

```text
1. Build a real walk-forward retraining script that trains early models only on
   dates before each test period, instead of using current saved test predictions.
2. Add a bar-20 live feature builder from raw OHLCV so the early models no
   longer rely on saved normalized full-day shape columns.
3. Add a losing-signal audit to identify why XOM 2026-03-24, NKE 2026-02-10,
   and similar high-confidence long signals failed after entry.
4. Test a short-only and bar-20-only variant, because shorts and earlier entries
   are currently stronger.
```

## Sixth Continuation Update

Added:

```text
losing_signal_audit.py
model_artifacts/losing_signal_audit_report.md
continuation_filter_experiment.py
model_artifacts/continuation_filter_experiment_report.md
```

Ran:

```powershell
.\.venv\Scripts\python.exe .\losing_signal_audit.py
.\.venv\Scripts\python.exe .\continuation_filter_experiment.py
```

Losing audit finding:

```text
The bad rows are mostly high-confidence longs, not weak model scores.
Losers actually had higher average early_up_probability and early_direction_edge
than winners. This points to long-side exhaustion/fade risk.
```

Holdout variant results:

```text
bar20_all:
rows=37
avg_close_return_pct=0.798456
hit_rate=0.837838

bar20_short_only:
rows=11
avg_close_return_pct=1.200736
hit_rate=1.000000
target_hit_rate=0.909091
stop_hit_rate=0.000000

bar20_long_only:
rows=26
avg_close_return_pct=0.628261
hit_rate=0.769231
target_hit_rate=0.538462
stop_hit_rate=0.038462
```

Best exploratory mixed guardrail:

```text
bar20_long_ensemble_gte_090_edge_lte_075_or_short
rows=21
avg_close_return_pct=1.145137
median_close_return_pct=1.017088
hit_rate=0.952381
target_hit_rate=0.761905
stop_hit_rate=0.000000
```

Important interpretation:

```text
The next filter should not simply raise early_pattern_probability. It should
detect exhaustion in high-confidence long signals. Shorts are currently cleaner
than longs. Bar 20 remains better than waiting.
```

Best next implementation:

```text
raw_bar20_feature_builder.py
```

Goal:

```text
For every scored test day, open raw 5m OHLCV and compute features known by bar
20: early range, distance from early high/low, last-5-bar slope, pullback from
early extreme, VWAP/proxy position if volume exists, candle compression, and
post-open continuation ratio. Then train an entry-quality model on bar-20
outcomes, especially separating long exhaustion from short continuation.
```

## Seventh Continuation Update

Added:

```text
bar20_entry_quality_model.py
model_artifacts/bar20_entry_quality_features.csv
model_artifacts/bar20_entry_quality_predictions.csv
model_artifacts/bar20_entry_quality_model.joblib
model_artifacts/bar20_entry_quality_model_report.txt
```

Ran:

```powershell
.\.venv\Scripts\python.exe .\bar20_entry_quality_model.py
```

What it does:

```text
Loads the current early prediction rows.
Loads raw 5m OHLCV by ticker.
Builds features known at bar 20 only.
Creates an implied side from early_direction_edge:
  edge >= 0 -> long
  edge < 0 -> short
Trains a second-stage model to predict positive bar-20-to-close signed return.
```

Raw bar-20 features:

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
feature_rows=8,655
train_rows=4,567
test_rows=4,088
split_date=2026-02-26
roc_auc=0.729
average_precision=0.770
accuracy=0.67
```

Ranked test behavior:

```text
top_1pct_rows=41
top_1pct_avg_signed_return=1.161920
top_1pct_hit_rate=0.951220
top_1pct_strong_hit_rate=0.878049

top_5pct_rows=205
top_5pct_avg_signed_return=0.937569
top_5pct_hit_rate=0.902439
top_5pct_strong_hit_rate=0.760976

bottom_50pct_rows=2,044
bottom_50pct_avg_signed_return=-0.129203
bottom_50pct_hit_rate=0.405577
```

Important interpretation:

```text
This is now the strongest bridge from structural recognition to entry quality.
The old early model says "this day can become structurally useful."
The directional model says "which side is favored."
The bar20 entry-quality model says "does the entry still have follow-through
left from here?"
```

Best next implementation:

```text
combined_live_signal_ranker.py
```

Goal:

```text
Join early_pattern_predictions.csv, early_directional_predictions.csv, and
bar20_entry_quality_predictions.csv. Rank candidates by a combined live score,
for example:

0.40 * early_pattern_rank
0.25 * abs_direction_edge_rank
0.35 * bar20_entry_quality_rank

Then report top candidates, side, expected bar20-to-close behavior, and whether
they pass the exploratory guardrails:
bar20 entry, short-cleanliness, long ensemble >= 0.90, long edge <= 0.75.
```

## Eighth Continuation Update

Added:

```text
model_dashboard.py
```

Started dashboard:

```text
http://localhost:8502
```

Run later with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\model_dashboard.py --server.port 8502
```

Dashboard views:

```text
Overview: quality deciles, probability vs outcome, side/asset behavior.
Entry Timing: entry-bar timing sweep charts.
Ranked Rows: top model predictions and cumulative behavior.
Ticker Chart: raw 5m candlestick chart with entry marker.
Model Report: text reports.
```

Added:

```text
entry_bar_timing_sweep.py
model_artifacts/entry_bar_timing_features.csv
model_artifacts/entry_bar_timing_sweep_summary.csv
model_artifacts/entry_bar_timing_predictions.csv
model_artifacts/entry_bar_timing_models.joblib
model_artifacts/entry_bar_timing_sweep_report.md
```

Ran:

```powershell
.\.venv\Scripts\python.exe .\entry_bar_timing_sweep.py
```

What it does:

```text
Builds raw OHLCV entry features for bars:
8, 12, 15, 18, 20, 22, 25, 28, 32

Trains a separate entry-quality model for each bar.
Compares top ranked test behavior for each entry time.
```

Timing sweep result:

```text
Best entry bar: 8
bar_8_roc_auc=0.743011
bar_8_average_precision=0.852897
bar_8_top5_avg_signed_return=1.398912
bar_8_top5_hit_rate=0.975610
bar_8_top10_avg_signed_return=1.347434
bar_8_top10_hit_rate=0.965770
```

Comparison to bar 20:

```text
bar_20_roc_auc=0.728873
bar_20_average_precision=0.769989
bar_20_top5_avg_signed_return=0.988709
bar_20_top5_hit_rate=0.912195
bar_20_top10_avg_signed_return=0.854821
bar_20_top10_hit_rate=0.880196
```

Important interpretation:

```text
Bar 20 was a sensible first checkpoint, but the broader timing sweep says the
signal becomes useful earlier. Bar 8 is currently the strongest tested entry.
Do not blindly replace all bar-20 logic yet; first build a stricter live-style
validation for bar 8 and compare transaction/slippage assumptions.
```

Best next implementation:

```text
combined_live_signal_ranker.py
```

Updated goal:

```text
Use entry_bar_timing_predictions.csv, favor bar 8 for now, and rank live-style
candidates by entry_quality_probability plus structural context. Include a
toggle/report comparing bar 8 vs bar 20 candidates and charts.
```

## Ninth Continuation Update

Added:

```text
combined_live_signal_ranker.py
model_artifacts/combined_live_signal_rankings.csv
model_artifacts/combined_live_signal_ranker_report.md
```

Ran:

```powershell
.\.venv\Scripts\python.exe .\combined_live_signal_ranker.py
```

What it does:

```text
Uses entry_bar_timing_predictions.csv.
Defaults to entry bar 8 because timing sweep found bar 8 strongest.
Creates one combined live-style score:

0.45 * entry_quality_rank
0.20 * early_pattern_rank
0.20 * abs(direction_edge)_rank
0.15 * ensemble_rank
```

Entry bar comparison from combined ranker:

```text
bar_8_top5_avg_signed_return=1.493168
bar_8_top5_hit_rate=0.970874
bar_8_top10_avg_signed_return=1.358381
bar_8_top10_hit_rate=0.953545

bar_20_top5_avg_signed_return=1.025273
bar_20_top5_hit_rate=0.917073
bar_20_top10_avg_signed_return=0.866984
bar_20_top10_hit_rate=0.894866
```

Default bar-8 combined score summary:

```text
top_1pct_rows=41
top_1pct_avg_signed_return=1.940992
top_1pct_hit_rate=0.975610

top_5pct_rows=206
top_5pct_avg_signed_return=1.493168
top_5pct_hit_rate=0.970874

top_10pct_rows=409
top_10pct_avg_signed_return=1.358381
top_10pct_hit_rate=0.953545
```

Guarded candidates:

```text
top_5pct_rows=108
top_5pct_avg_signed_return=1.708830
top_5pct_hit_rate=0.972222
top_5pct_strong_hit_rate=0.953704
```

Updated:

```text
model_dashboard.py
```

Dashboard now includes:

```text
Combined Ranker tab
Entry Timing tab
Overview tab
Ranked Rows tab
Ticker Chart tab
Model Report tab
```

Current best file to inspect:

```text
model_artifacts/combined_live_signal_rankings.csv
```

Current best dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\model_dashboard.py --server.port 8502
```

Best next improvement:

```text
live_replay_scanner.py
```

Goal:

```text
For historical days, reveal bars one at a time up to bar 8, compute only the
features known by then, score the candidate, and then plot the rest of the day.
This will make the model feel like it is acting live instead of scoring from a
saved prediction table.
```

## Tenth Continuation Update

Added:

```text
live_replay_scanner.py
model_artifacts/live_replay_scanner_results.csv
model_artifacts/live_replay_scanner_model.joblib
model_artifacts/live_replay_scanner_report.md
```

Ran:

```powershell
.\.venv\Scripts\python.exe .\live_replay_scanner.py
```

What it does:

```text
Loads the existing entry-bar timing features.
Filters to entry_bar=8.
Trains a raw-only replay model using only OHLCV features known by bar 8:
session open to entry, early range, close position, distance from high/low,
last-5-bar return/range/body, and volume shape.

It also creates a separate research_context_score that blends:
60% raw replay rank
15% early pattern rank
15% abs(direction edge) rank
10% ensemble rank
```

Important live-style finding:

```text
Raw bar-8 OHLCV by itself is weak.
The strong bar-8 ranking behavior mainly comes from structural context plus
entry quality, not from raw bar-8 candles alone.
```

Latest raw-only replay model:

```text
feature_rows=8,655
entry_bar=8
train_rows=4,567
test_rows=4,088
split_date=2026-02-26
roc_auc=0.512885
average_precision=0.678201
```

Raw-only ranking:

```text
top_1pct_rows=41
top_1pct_avg_signed_return=0.424846
top_1pct_hit_rate=0.658537

top_5pct_rows=205
top_5pct_avg_signed_return=0.475460
top_5pct_hit_rate=0.687805

bottom_50pct_rows=2,044
bottom_50pct_avg_signed_return=0.427037
bottom_50pct_hit_rate=0.661448
```

Research-context ranking:

```text
top_1pct_rows=41
top_1pct_avg_signed_return=1.500231
top_1pct_hit_rate=0.951220

top_5pct_rows=205
top_5pct_avg_signed_return=1.017824
top_5pct_hit_rate=0.887805

top_10pct_rows=409
top_10pct_avg_signed_return=0.880116
top_10pct_hit_rate=0.838631
```

Guarded raw-only replay:

```text
top_5pct_rows=108
top_5pct_avg_signed_return=0.703593
top_5pct_hit_rate=0.787037
top_5pct_short_rate=0.879630
```

Side finding:

```text
The raw-only replay score still prefers the same broad theme:
shorts are cleaner than longs.

Top 5% raw replay stock shorts:
rows=40
avg_signed_return=0.720341
hit_rate=0.850000

Top 5% raw replay stock longs:
rows=69
avg_signed_return=0.514660
hit_rate=0.608696
```

Updated:

```text
model_dashboard.py
```

Dashboard now includes:

```text
Live Replay tab
Replay score toggle:
  research_context_score
  raw_replay_probability
Guarded-only replay filter
Replay scatter and ranked rows
Live replay report appended to Model Report tab
```

Verified:

```powershell
.\.venv\Scripts\python.exe -m py_compile .\live_replay_scanner.py .\model_dashboard.py
```

Current interpretation:

```text
Do not replace the combined live ranker with raw-only bar-8 replay.
Raw bar-8 features alone are not enough yet. The system needs either:

1. A true live early-structure model built directly from raw bars available by
   bar 8/12/20, or
2. A staged scanner that waits until enough bars exist to recreate the early
   structural context without full-day leakage.
```

Best next implementation:

```text
raw_early_structure_predictor.py
```

Goal:

```text
Train an early-structure predictor from raw OHLCV prefixes at several checkpoints
(bar 8, 12, 20, 28), targeting the same top-decile ensemble structural label.
Then feed that live-safe early-structure probability into the entry timing model
instead of using saved first-35%-of-normalized-day predictions.
```

Suggested command sequence:

```powershell
.\.venv\Scripts\python.exe .\live_replay_scanner.py
.\.venv\Scripts\python.exe -m streamlit run .\model_dashboard.py --server.port 8502
```

## Eleventh Continuation Update

Added:

```text
raw_early_structure_predictor.py
model_artifacts/raw_early_structure_features.csv
model_artifacts/raw_early_structure_predictions.csv
model_artifacts/raw_early_structure_summary.csv
model_artifacts/raw_early_structure_models.joblib
model_artifacts/raw_early_structure_predictor_report.md
```

Ran:

```powershell
.\.venv\Scripts\python.exe .\raw_early_structure_predictor.py
```

Note:

```text
The first run hit a 3-minute timeout while building the full raw feature universe.
The script was updated to cache/reuse raw_early_structure_features.csv and use a
lighter 350-tree model. The second run completed in about 5 minutes.
```

What it does:

```text
Builds live-safe raw OHLCV prefix features for every scored structural day at:
bar 8, bar 12, bar 20, bar 28.

Target:
top-decile useful_pattern_ensemble_score

This replaces the older normalized first-35%-of-day structural context with a
raw prefix model that can be computed from live bars.
```

Latest result:

```text
feature_rows=173,104
train_rows=138,483
test_rows=34,621
target_threshold_top_decile_ensemble_score=0.862420
checkpoint_bars=[8, 12, 20, 28]
```

Checkpoint summary:

```text
bar_28:
roc_auc=0.679941
average_precision=0.214593
top_1pct_target_rate=0.436782
top_5pct_target_rate=0.297921
top_5pct_avg_abs_return=1.654490
bottom_50pct_target_rate=0.057532

bar_20:
roc_auc=0.650599
average_precision=0.176865
top_1pct_target_rate=0.321839
top_5pct_target_rate=0.240185
top_5pct_avg_abs_return=1.391559
bottom_50pct_target_rate=0.067006

bar_12:
roc_auc=0.609549
average_precision=0.135448
top_5pct_target_rate=0.152425

bar_8:
roc_auc=0.599109
average_precision=0.136299
top_5pct_target_rate=0.138568
```

Important interpretation:

```text
Raw early structure is real, but it matures later than the bar-8 entry-quality
signal. Bar 28 is the best raw structural checkpoint, and bar 20 is second.
Bar 8 raw structure is still too weak to replace the old structural context.
```

Updated:

```text
model_dashboard.py
```

Dashboard now includes:

```text
Raw Structure tab
Raw checkpoint summary chart
Target-rate and AUC/AP charts by checkpoint
Raw early-structure ranked rows
Raw early-structure report appended to Model Report tab
```

Verified:

```powershell
.\.venv\Scripts\python.exe -m py_compile .\raw_early_structure_predictor.py .\model_dashboard.py
```

New best implementation:

```text
staged_live_signal_ranker.py
```

Goal:

```text
Combine:
1. bar-8 entry-quality ranking for very early continuation entries
2. bar-20/bar-28 raw early-structure probabilities for confirmation
3. directional edge from the existing directional model until a raw directional
   prefix replacement is built

Compare staged variants:
early_aggressive_bar8
confirmed_bar20
confirmed_bar28

Report trade-style behavior, hit rate, average signed return, and whether the
bar-20/bar-28 confirmation filters reduce the bar-8 false positives.
```

Suggested next commands:

```powershell
.\.venv\Scripts\python.exe .\raw_early_structure_predictor.py
.\.venv\Scripts\python.exe .\live_replay_scanner.py
```

Then build:

```text
staged_live_signal_ranker.py
```

## Twelfth Continuation Update

Added:

```text
staged_live_signal_ranker.py
model_artifacts/staged_live_signal_rankings.csv
model_artifacts/staged_live_signal_summary.csv
model_artifacts/staged_live_signal_ranker_report.md
```

Ran:

```powershell
.\.venv\Scripts\python.exe .\staged_live_signal_ranker.py
```

What it does:

```text
Compares three staged workflows:

early_aggressive_bar8:
  entry_bar=8
  raw_structure_bar=8

confirmed_bar20:
  entry_bar=20
  raw_structure_bar=20

confirmed_bar28:
  entry_bar=28
  raw_structure_bar=28

staged_live_score =
  0.45 * entry_quality_rank
  0.30 * raw_structure_rank
  0.25 * abs(direction_edge)_rank
```

Key staged result:

```text
Bar 8 still has the strongest trade-style follow-through.
Bar 20 has the cleanest high-structure confirmation.
Bar 28 has even higher structure confirmation but gives up too much entry edge.
```

Top 5% by staged_live_score:

```text
early_aggressive_bar8:
rows=205
avg_signed_return=1.115891
median_signed_return=1.073627
hit_rate=0.931707
strong_hit_rate=0.780488
target_high_structure_rate=0.468293

confirmed_bar20:
rows=205
avg_signed_return=0.801771
median_signed_return=0.810469
hit_rate=0.863415
strong_hit_rate=0.648780
target_high_structure_rate=0.639024

confirmed_bar28:
rows=205
avg_signed_return=0.598301
median_signed_return=0.538356
hit_rate=0.878049
strong_hit_rate=0.541463
target_high_structure_rate=0.707317
```

Important comparison:

```text
Entry-quality-only ranking remains stronger for pure return capture:

bar8 entry_quality top 5%:
avg_signed_return=1.398912
hit_rate=0.975610
strong_hit_rate=0.912195

bar20 entry_quality top 5%:
avg_signed_return=0.988709
hit_rate=0.912195

bar28 entry_quality top 5%:
avg_signed_return=0.722873
hit_rate=0.868293
```

False-positive audit:

```text
early_aggressive_bar8 top 5% had 205 rows and 14 losers.

bar20 confirmation:
losers_confirmed_rate=0.714286
winners_confirmed_rate=0.649215

bar28 confirmation:
losers_confirmed_rate=0.500000
winners_confirmed_rate=0.743455
```

Interpretation:

```text
Raw structure confirmation does not cleanly remove the bar-8 losers yet.
It improves structural purity, but it can also filter out profitable early
continuations or confirm some losing long setups. Do not use raw structure as a
simple hard filter until a better false-positive audit is built.
```

Updated:

```text
model_dashboard.py
```

Dashboard now includes:

```text
Staged Ranker tab
Top 5% staged variant comparison
Variant bar charts for avg signed return and high-structure rate
Staged ranked rows
Staged report appended to Model Report tab
```

Verified:

```powershell
.\.venv\Scripts\python.exe -m py_compile .\staged_live_signal_ranker.py .\model_dashboard.py
```

Best next implementation:

```text
bar8_false_positive_audit.py
```

Goal:

```text
Focus only on early_aggressive_bar8 top-ranked candidates.
Compare winners vs losers using:
raw bar-8 features
bar-20 and bar-28 raw structure probabilities
side
asset class
distance from high/low
close position
last-5-bar move
early_direction_edge
guardrails

Find filters that reduce the 14 top-5% losers without sacrificing too many of
the 191 winners.
```

Current most important files:

```text
model_artifacts/staged_live_signal_ranker_report.md
model_artifacts/staged_live_signal_rankings.csv
model_artifacts/raw_early_structure_predictor_report.md
model_artifacts/live_replay_scanner_report.md
```

## Thirteenth Continuation Update

The user noticed the project was drifting too much toward prediction and asked
to refocus on the original staple: grouping days into better pattern families.

Direction change:

```text
Prediction stays useful as a scorecard.
The core work is pattern recognition: grouping days by similar intraday shape.
Automatic grouping should use more groups, centered around roughly 16.
```

Updated:

```text
automated_structural_feedback.py
run_structural_feedback_sweeps.py
PATTERN_LEARNING_SYSTEM.md
model_dashboard.py
```

Added:

```text
pattern_grouping_lab.py
model_artifacts/pattern_grouping_candidates.csv
model_artifacts/pattern_grouping_lab_report.md
```

What changed:

```text
automated_structural_feedback.py default group counts:
  before: 6, 8, 10
  after: 12, 14, 16, 18, 20

run_structural_feedback_sweeps.py configs now center on:
  14, 16, 18
  16, 18, 20
  12, 16, 20

min_group_days default lowered to 3 so tighter groups can survive.
```

Ran the new non-predictive grouping lab:

```powershell
.\.venv\Scripts\python.exe .\pattern_grouping_lab.py --group-counts 12 14 16 18 20 --min-group-days 3
```

Grouping lab summary:

```text
total_groups: 6,470

group_count=20 groups=1,515 high_quality_groups=482 avg_tightness=0.623370
group_count=18 groups=1,395 high_quality_groups=390 avg_tightness=0.609312
group_count=16 groups=1,336 high_quality_groups=361 avg_tightness=0.602935
group_count=14 groups=1,188 high_quality_groups=273 avg_tightness=0.582577
group_count=12 groups=1,036 high_quality_groups=187 avg_tightness=0.561317
```

Interpretation:

```text
The denser 16-20 group range creates more tight pattern families.
16 is a good center point.
20 currently finds the most high-quality groups.
The old loose grouping style should no longer be the default.
```

Ran a stricter audited grouping-feedback pass and appended feedback:

```powershell
.\.venv\Scripts\python.exe .\automated_structural_feedback.py --coverage 0.90 --group-counts 14 16 18 20 --min-group-days 3 --min-corr 0.90 --max-rmse 0.95 --max-band 1.05 --min-same-direction 0.72 --min-median-abs-return 0.30 --run-name pattern_focus_g14_16_18_20 --append-feedback
```

Result:

```text
processed_symbols: 94 of 105
automated_feedback_rows: 4,497
mach2_group_feedback.csv rows after append: 22,139
good pattern rows: 1,449
not useful rows: 20,690
```

Model check before/after the appended grouping feedback:

```text
before:
  rows=17,688 positives=1,188 negatives=16,500
  roc_auc=0.952 average_precision=0.714

after:
  rows=22,139 positives=1,449 negatives=20,690
  roc_auc=0.960 average_precision=0.765
  good-pattern precision=0.68
  good-pattern recall=0.75
  good-pattern f1=0.71
  accuracy=0.96
```

Backtest grouping check:

```text
good_groups=266
bad_groups=4,231
good_test_avg_abs_return=2.4033
bad_test_avg_abs_return=1.3737
good_avg_cohesion=0.7897
bad_avg_cohesion=0.6844
```

Refreshed ensemble ranking and score report:

```powershell
.\.venv\Scripts\python.exe .\build_ensemble_scores.py
.\.venv\Scripts\python.exe .\score_backtest_report.py
```

Updated ensemble score behavior:

```text
structural_day_ensemble_scores.csv rows scored: 43,276
top 1% avg abs return: 2.3500
top 5% avg abs return: 2.1271
bottom 50% avg abs return: 0.5646
top 1% lift vs bottom half: 4.162x
top 5% lift vs bottom half: 3.767x
```

Top refreshed ensemble candidates:

```text
CSCO 2025-11-12
CVX 2025-04-03
LOW 2024-06-25
BA 2024-07-16
NKE 2025-12-31
TXN 2025-10-10
NFLX 2024-05-20
EWZ 2025-08-27
LLY 2026-02-27
DIA 2024-05-23
```

Verified:

```powershell
.\.venv\Scripts\python.exe -m py_compile .\model_dashboard.py .\pattern_grouping_lab.py .\automated_structural_feedback.py .\run_structural_feedback_sweeps.py
```

Dashboard update:

```text
Added a Pattern Groups tab to model_dashboard.py.
It reads model_artifacts/pattern_grouping_candidates.csv.
It shows:
  group-count comparison
  tightness/correlation scatter
  ticker/group-count/pattern-type filters
  representative dates for each pattern family
```

Dashboard is running here:

```text
http://localhost:8502
```

GitHub note:

```text
The user asked to make a GitHub repository and push after stopping.
Git is not available in this shell:
  git : The term 'git' is not recognized...

Do not claim this was pushed. Install/configure Git or GitHub CLI first, then
create the repo and push the current project state.
```

Best next implementation:

```text
Add a Pattern Groups tab to model_dashboard.py.
Use model_artifacts/pattern_grouping_candidates.csv.
Show:
  group count comparison
  top tight pattern groups
  ticker filter
  direction/pattern type filter
  representative dates for each group
```

Best next command:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\model_dashboard.py --server.port 8502
```

## Fourteenth Continuation Update

The user asked to improve the dashboard so a few tickers' pattern groups can be
checked visually, then continue improving feedback and learners.

Updated:

```text
pattern_grouping_lab.py
model_dashboard.py
PATTERN_LEARNING_SYSTEM.md
```

Pattern grouping lab now writes a second local artifact:

```text
model_artifacts/pattern_grouping_members.csv
```

This file records each day inside each pattern family:

```text
ticker
group_count_setting
pattern_group
group_key
date
signature_distance
avg_return
avg_range
avg_early_move
avg_shift_strength
avg_shift_count
```

Ran:

```powershell
.\.venv\Scripts\python.exe .\pattern_grouping_lab.py --group-counts 12 14 16 18 20 --min-group-days 3
```

Grouping result stayed stable:

```text
total_groups: 6,470
20-group high-quality groups: 482
18-group high-quality groups: 390
16-group high-quality groups: 361
14-group high-quality groups: 273
12-group high-quality groups: 187
```

Dashboard Pattern Groups tab now supports:

```text
multi-ticker filtering
group-count filtering
pattern-type filtering
minimum tightness and minimum-days filters
ticker pattern-quality bar chart
pattern family table
multi-family signature-line comparison
member-day line overlays
member date table sorted by signature distance
```

The dashboard is still running:

```text
http://localhost:8502
```

Feedback/learner note:

```text
An attempted stricter de-duplication of mach2_group_feedback.csv was too
aggressive. It collapsed distinct pattern families that share rounded summary
signatures and reduced the file from 22,139 rows to 12,050 rows.

Validation fell:
  average_precision: 0.765 -> 0.412
  good-pattern f1: 0.71 -> 0.45

The feedback file was restored to the 22,139-row state from git.
Do not de-dup only by rounded signature/ticker/group/rating.
```

Then retrained the learner from the restored feedback:

```powershell
.\.venv\Scripts\python.exe .\structural_pattern_learner.py --interval 5m --model-mode both
.\.venv\Scripts\python.exe .\build_ensemble_scores.py
.\.venv\Scripts\python.exe .\score_backtest_report.py
```

Restored learner state:

```text
feedback_training_rows: 22,139
summary validation accuracy: 0.96
good-pattern precision: 0.68
good-pattern recall: 0.73
good-pattern f1: 0.70
```

Updated score report:

```text
structural_day_ensemble_scores.csv rows scored: 43,276
top 1% avg abs return: 2.3243
top 5% avg abs return: 2.0670
bottom 50% avg abs return: 0.5732
top 1% lift vs bottom half: 4.055x
top 5% lift vs bottom half: 3.606x
```

Current top ensemble candidates:

```text
HD 2025-11-25
CSCO 2025-11-12
LOW 2024-06-25
AVGO 2024-08-13
JPM 2026-03-03
V 2025-04-22
MA 2025-01-27
TGT 2024-07-11
BA 2024-07-16
MCHI 2026-01-12
```

Verified:

```powershell
.\.venv\Scripts\python.exe -m py_compile .\model_dashboard.py .\pattern_grouping_lab.py .\automated_structural_feedback.py
```

Next best implementation:

```text
Add a "pattern family review queue" that picks the top uncertain/tight groups
for human rating in Mach2AImarket.py, rather than adding more automated
feedback blindly.
```
