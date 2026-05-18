# Bar Entry Trade Simulator

This is the first raw-OHLCV check after the early signal files. It uses the already-selected early trade signals and enters at fixed 5m bar indexes inside the regular session.

Important: this still inherits the discovery-selected signal threshold from `early_trade_simulator.py`. It is more realistic than the full-day proxy, but it is not yet an out-of-sample strategy validation.

## Settings

```text
signals=35
entry_bars=[20, 25, 28]
session=09:30-16:00
target_pct=max(0.25, 0.50 * early_range_pct)
stop_pct=max(0.25, 0.50 * early_range_pct)
```

## Overall

```text
trades                     105.000000
avg_close_return_pct         0.604019
median_close_return_pct      0.701231
hit_rate_to_close            0.752381
avg_mfe_pct                  1.181103
avg_mae_pct                 -0.348871
target_hit_rate              0.419048
stop_hit_rate                0.019048
avg_barrier_bars            27.565217
```

## By Entry Bar

```text
 entry_bar  trades  avg_close_return_pct  median_close_return_pct  hit_rate_to_close  avg_mfe_pct  avg_mae_pct  target_hit_rate  stop_hit_rate  avg_barrier_bars
        20    35.0              0.864508                 0.965529           0.828571     1.453224    -0.240251         0.600000       0.028571         25.818182
        25    35.0              0.567373                 0.648102           0.742857     1.141867    -0.340161         0.371429       0.028571         30.785714
        28    35.0              0.380176                 0.429520           0.685714     0.948217    -0.466201         0.285714       0.000000         26.900000
```

## By Side

```text
trade_side  trades  avg_close_return_pct  median_close_return_pct  hit_rate_to_close  avg_mfe_pct  avg_mae_pct  target_hit_rate  stop_hit_rate  avg_barrier_bars
      long    81.0              0.471837                 0.561617           0.691358     1.102592    -0.381491         0.370370       0.024691          28.46875
     short    24.0              1.050134                 1.154919           0.958333     1.446075    -0.238777         0.583333       0.000000          25.50000
```

## By Entry Bar And Side

```text
 entry_bar trade_side  trades  avg_close_return_pct  median_close_return_pct  hit_rate_to_close  avg_mfe_pct  avg_mae_pct  target_hit_rate  stop_hit_rate  avg_barrier_bars
        20       long    27.0              0.745304                 0.839152           0.777778     1.391480    -0.272950         0.555556       0.037037         27.937500
        20      short     8.0              1.266823                 1.442742           1.000000     1.661610    -0.129891         0.750000       0.000000         20.166667
        25       long    27.0              0.428039                 0.553854           0.703704     1.055446    -0.378195         0.333333       0.037037         31.800000
        25      short     8.0              1.037627                 1.171620           0.875000     1.433539    -0.211795         0.500000       0.000000         28.250000
        28       long    27.0              0.242169                 0.386237           0.592593     0.860851    -0.493329         0.222222       0.000000         24.333333
        28      short     8.0              0.845951                 0.933406           1.000000     1.243077    -0.374645         0.500000       0.000000         30.750000
```

## By Asset Class

```text
asset_class  trades  avg_close_return_pct  median_close_return_pct  hit_rate_to_close  avg_mfe_pct  avg_mae_pct  target_hit_rate  stop_hit_rate  avg_barrier_bars
     crypto     3.0             -0.851129                -0.916805           0.000000     0.345089    -1.073978         0.000000       0.000000               NaN
        etf    12.0              0.802148                 0.771922           1.000000     1.254762    -0.231136         0.750000       0.000000         26.222222
      stock    90.0              0.626107                 0.733975           0.744444     1.199148    -0.340399         0.388889       0.022222         27.891892
```

## Best Bar Entries

```text
ticker       date asset_class trade_side  entry_bar                entry_time  close_return_pct  mfe_pct   mae_pct barrier_result  barrier_bars  early_pattern_probability  early_direction_edge
  SBUX 2026-03-19       stock       long         20 2026-03-19 11:10:00-04:00          2.821888 3.277897 -0.160944         target           4.0                   0.762231              0.720629
   XOM 2026-01-08       stock       long         20 2026-01-08 11:10:00-05:00          1.974448 2.559316 -0.033184         target          16.0                   0.853636              0.590500
    BA 2025-12-09       stock      short         20 2025-12-09 11:10:00-05:00          1.909892 2.047013 -0.173849         target          17.0                   0.838269             -0.575840
  SBUX 2026-03-19       stock       long         25 2026-03-19 11:35:00-04:00          1.827648 2.279248 -0.255021         target          41.0                   0.762231              0.720629
    BA 2025-12-09       stock      short         25 2025-12-09 11:35:00-05:00          1.808912 1.946174 -0.073533         target          13.0                   0.838269             -0.575840
    BA 2025-12-09       stock      short         28 2025-12-09 11:50:00-05:00          1.799284 1.936559  0.044124         target          10.0                   0.838269             -0.575840
  AAPL 2026-03-13       stock      short         20 2026-03-13 11:10:00-04:00          1.773845 1.876106 -0.015733         target           6.0                   0.794826             -0.566292
   OXY 2026-04-21       stock       long         25 2026-04-21 11:35:00-04:00          1.733610 2.004696 -0.090302         target          20.0                   0.779964              0.591682
   WMT 2026-02-02       stock       long         20 2026-02-02 11:10:00-05:00          1.730217 1.845018 -0.053301         target          19.0                   0.873956              0.836882
   OXY 2026-04-21       stock       long         20 2026-04-21 11:10:00-04:00          1.650997 1.921862 -0.297753         target          25.0                   0.779964              0.591682
   EWJ 2026-03-20         etf      short         20 2026-03-20 11:10:00-04:00          1.539954 1.964351 -0.157633         target          38.0                   0.795011             -0.654484
  MSFT 2026-04-13       stock       long         20 2026-04-13 11:10:00-04:00          1.537852 1.611838 -0.007927         target          52.0                   0.793406              0.596105
  SBUX 2026-03-19       stock       long         28 2026-03-19 11:50:00-04:00          1.525585 1.975845 -0.550906         target          39.0                   0.762231              0.720629
   XOM 2026-01-08       stock       long         28 2026-01-08 11:50:00-05:00          1.515464 2.097700 -0.099104         target          16.0                   0.853636              0.590500
   OXY 2026-04-21       stock       long         28 2026-04-21 11:50:00-04:00          1.495315 1.765766 -0.324324         target          18.0                   0.779964              0.591682
   NKE 2025-12-31       stock       long         20 2025-12-31 11:10:00-05:00          1.486507 2.122786 -0.008749         target           2.0                   0.823188              0.682709
   TGT 2026-02-02       stock       long         20 2026-02-02 11:10:00-05:00          1.479207 2.251372  0.027910         target          21.0                   0.781646              0.582112
  SBUX 2026-03-20       stock      short         20 2026-03-20 11:10:00-04:00          1.448195 1.799595 -0.234267         target          42.0                   0.805147             -0.672615
   SLV 2026-03-13         etf      short         20 2026-03-13 11:10:00-04:00          1.437288 2.488136 -0.013424         target          10.0                   0.772423             -0.740746
   XOM 2026-01-08       stock       long         25 2026-01-08 11:35:00-05:00          1.419058 2.000741 -0.193977         target          21.0                   0.853636              0.590500
  AMZN 2026-01-06       stock       long         20 2026-01-06 11:10:00-05:00          1.387721 2.262405 -0.088310         target          14.0                   0.754948              0.620907
   WMT 2026-02-02       stock       long         25 2026-02-02 11:35:00-05:00          1.381057 1.495465 -0.008172         target          50.0                   0.873956              0.836882
  AAPL 2026-03-13       stock      short         25 2026-03-13 11:35:00-04:00          1.378194 1.480867 -0.019745         target          16.0                   0.794826             -0.566292
  SBUX 2026-03-20       stock      short         28 2026-03-20 11:50:00-04:00          1.374680 1.726343 -0.309037         target          34.0                   0.805147             -0.672615
   EWJ 2026-03-20         etf      short         25 2026-03-20 11:35:00-04:00          1.366535 1.791679 -0.115396         target          33.0                   0.795011             -0.654484
```

## Worst Bar Entries

```text
 ticker       date asset_class trade_side  entry_bar                entry_time  close_return_pct  mfe_pct   mae_pct barrier_result  barrier_bars  early_pattern_probability  early_direction_edge
    XOM 2026-03-24       stock       long         28 2026-03-24 11:50:00-04:00         -1.109185 0.089643 -1.685293        neither           NaN                   0.834788              0.789785
    NKE 2026-02-10       stock       long         25 2026-02-10 11:35:00-05:00         -1.103373 0.046952 -1.533766           stop          52.0                   0.807673              0.791728
    NKE 2026-02-10       stock       long         20 2026-02-10 11:10:00-05:00         -1.041422 0.524626 -1.472085           stop          56.0                   0.807673              0.791728
    NKE 2026-02-10       stock       long         28 2026-02-10 11:50:00-05:00         -1.041422 0.007830 -1.472085        neither           NaN                   0.807673              0.791728
BTC_USD 2026-01-12      crypto       long         28 2026-01-12 11:50:00-05:00         -1.016432 0.177793 -1.238909        neither           NaN                   0.780614              0.640461
    XOM 2026-03-24       stock       long         25 2026-03-24 11:35:00-04:00         -1.014536 0.185440 -1.591195        neither           NaN                   0.834788              0.789785
BTC_USD 2026-01-12      crypto       long         25 2026-01-12 11:35:00-05:00         -0.916805 0.278621 -1.139506        neither           NaN                   0.780614              0.640461
   TSLA 2025-12-09       stock       long         28 2025-12-09 11:50:00-05:00         -0.848533 0.752767 -0.986615        neither           NaN                   0.821663              0.858159
BTC_USD 2026-01-12      crypto       long         20 2026-01-12 11:10:00-05:00         -0.620151 0.578855 -0.843519        neither           NaN                   0.780614              0.640461
   TSLA 2025-12-09       stock       long         25 2025-12-09 11:35:00-05:00         -0.589496 1.015988 -0.727939        neither           NaN                   0.821663              0.858159
      V 2026-01-05       stock       long         20 2026-01-05 11:10:00-05:00         -0.564908 0.484809 -0.593013        neither           NaN                   0.851473              0.711964
      V 2026-01-05       stock       long         28 2026-01-05 11:50:00-05:00         -0.534158 0.515884 -0.562272        neither           NaN                   0.851473              0.711964
     HD 2026-03-10       stock       long         28 2026-03-10 11:50:00-04:00         -0.531945 0.718543 -0.587646        neither           NaN                   0.793898              0.788745
      V 2026-01-05       stock       long         25 2026-01-05 11:35:00-05:00         -0.525768 0.524362 -0.553884        neither           NaN                   0.851473              0.711964
   TSLA 2025-12-09       stock       long         20 2025-12-09 11:10:00-05:00         -0.508408 1.098385 -0.646964        neither           NaN                   0.821663              0.858159
    DIS 2026-01-05       stock       long         28 2026-01-05 11:50:00-05:00         -0.458135 1.252236 -0.536673        neither           NaN                   0.841382              0.604846
    TXN 2026-04-16       stock       long         25 2026-04-16 11:35:00-04:00         -0.426520 0.011165 -0.765950        neither           NaN                   0.793557              0.674137
    DIS 2026-01-05       stock       long         25 2026-01-05 11:35:00-05:00         -0.301534 1.411528 -0.380195         target          22.0                   0.841382              0.604846
    XOM 2026-03-24       stock       long         20 2026-03-24 11:10:00-04:00         -0.292842 0.915883 -0.873705        neither           NaN                   0.834788              0.789785
     HD 2026-03-10       stock       long         25 2026-03-10 11:35:00-04:00         -0.283389 0.970223 -0.339229        neither           NaN                   0.793898              0.788745
    TXN 2026-04-16       stock       long         28 2026-04-16 11:50:00-04:00         -0.228229 0.107402 -0.568334        neither           NaN                   0.793557              0.674137
   NVDA 2026-03-02       stock       long         28 2026-03-02 11:50:00-05:00         -0.180559 0.410542 -0.582837        neither           NaN                   0.812136              0.662690
    NKE 2025-12-31       stock       long         28 2025-12-31 11:50:00-05:00         -0.109598 0.516674 -0.454047        neither           NaN                   0.823188              0.682709
     HD 2026-03-10       stock       long         20 2026-03-10 11:10:00-04:00         -0.086723 1.169362 -0.142673         target          26.0                   0.793898              0.788745
    WFC 2025-12-11       stock       long         28 2025-12-11 11:50:00-05:00         -0.053972 0.839054 -0.496546        neither           NaN                   0.884846              0.780656
```