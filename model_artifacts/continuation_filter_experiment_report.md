# Continuation Filter Experiment

This is an exploratory holdout-only filter experiment. It tests simple bar-20 guardrails suggested by the losing-signal audit. Treat it as a clue generator, not as final strategy validation.

## Variant Results

```text
                                           subset  rows  avg_close_return_pct  median_close_return_pct  hit_rate  avg_mfe_pct  avg_mae_pct  target_hit_rate  stop_hit_rate
                                 bar20_short_only    11              1.200736                 1.211729  1.000000     1.731516    -0.144793         0.909091       0.000000
bar20_long_ensemble_gte_090_edge_lte_075_or_short    21              1.145137                 1.017088  0.952381     1.588085    -0.155839         0.761905       0.000000
bar20_long_ensemble_gte_085_edge_lte_075_or_short    25              1.085992                 0.996528  0.960000     1.552626    -0.156674         0.760000       0.000000
             bar20_long_ensemble_gte_090_or_short    28              1.018009                 1.006808  0.928571     1.480155    -0.153157         0.714286       0.000000
             bar20_long_ensemble_gte_085_or_short    32              0.987693                 0.917840  0.937500     1.465944    -0.154144         0.718750       0.000000
                           bar20_ensemble_gte_090    25              0.980819                 0.996528  0.920000     1.410045    -0.160969         0.680000       0.000000
                           bar20_ensemble_gte_085    31              0.973190                 0.839152  0.935484     1.432970    -0.158683         0.709677       0.000000
                           bar20_ensemble_gte_080    33              0.964710                 0.839152  0.939394     1.443046    -0.152879         0.696970       0.000000
                 bar20_long_edge_lte_075_or_short    28              0.943599                 0.834190  0.892857     1.463411    -0.193565         0.714286       0.000000
                 bar20_long_edge_lte_070_or_short    25              0.929222                 0.839152  0.920000     1.405967    -0.192338         0.680000       0.000000
                                        bar20_all    37              0.798456                 0.816620  0.837838     1.365151    -0.237334         0.648649       0.027027
                                  bar20_long_only    26              0.628261                 0.688621  0.769231     1.210150    -0.276486         0.538462       0.038462
```

## Best Variant Worst Rows

```text
ticker       date asset_class trade_side  entry_bar  close_return_pct  mfe_pct   mae_pct barrier_result  early_pattern_probability  early_direction_edge  useful_pattern_ensemble_score
   CVX 2026-04-27       stock      short         20          0.463262 1.061194 -0.199310         target                   0.738166             -0.687425                       0.864022
   NKE 2026-04-23       stock      short         20          0.466978 1.623304 -0.177896         target                   0.859668             -0.526973                       0.904293
  AMZN 2026-03-27       stock      short         20          0.770007 0.969908 -0.392937        neither                   0.756801             -0.639865                       0.988566
   IWM 2026-03-20         etf      short         20          0.996528 1.846028 -0.281805         target                   0.732729             -0.534331                       0.900513
  AAPL 2026-04-21       stock      short         20          1.017088 1.303434 -0.003719         target                   0.758415             -0.659125                       0.921069
   NKE 2026-03-18       stock      short         20          1.211729 1.470724 -0.064564         target                   0.759900             -0.534064                       0.938936
   SLV 2026-03-13         etf      short         20          1.437288 2.488136 -0.013424         target                   0.772423             -0.740746                       0.826578
  SBUX 2026-03-20       stock      short         20          1.448195 1.799595 -0.234267         target                   0.805147             -0.672615                       0.931560
   EWJ 2026-03-20         etf      short         20          1.539954 1.964351 -0.157633         target                   0.795011             -0.654484                       0.979273
  AAPL 2026-03-13       stock      short         20          1.773845 1.876106 -0.015733         target                   0.794826             -0.566292                       0.937587
   CVX 2026-04-09       stock      short         20          2.083226 2.643897 -0.051438         target                   0.743267             -0.579854                       0.896331
```

## Practical Read

```text
The current best exploratory variants favor bar-20 entries and either shorts or longs with stronger ensemble confirmation.
Raising early_pattern_probability alone is not the right next filter; the failure mode is long-side exhaustion.
The next model feature should describe post-entry continuation pressure from raw bars, not just the early normalized shape.
```