# Early Signal Holdout Validator

This report chooses thresholds on the earlier half of the existing walk-forward prediction rows, then evaluates those thresholds on the later half. It is still not a full retrain-by-period validation, but it is stricter than selecting thresholds on the same rows being reported.

## Split

```text
calibration_rows=3476
holdout_rows=5179
split_date=2026-02-07
```

## Selected Thresholds

```text
pattern_threshold                   0.700000
edge_threshold                      0.200000
direction_probability_threshold     0.500000
signals                            27.000000
avg_trade_return_proxy              2.372824
median_trade_return_proxy           2.432175
hit_rate_proxy                      1.000000
selection_score                     2.856112
```

## Calibration Proxy Summary

```text
signals                      27.000000
avg_trade_return_proxy        2.372824
median_trade_return_proxy     2.432175
hit_rate_proxy                1.000000
avg_abs_return                2.372824
long_signals                 21.000000
short_signals                 6.000000
```

## Holdout Proxy Summary

```text
signals                      37.000000
avg_trade_return_proxy        2.223092
median_trade_return_proxy     2.277680
hit_rate_proxy                1.000000
avg_abs_return                2.223092
long_signals                 26.000000
short_signals                11.000000
```

## Holdout Bar Entry Summary

```text
trades                     111.000000
avg_close_return_pct         0.542335
median_close_return_pct      0.525086
hit_rate_to_close            0.765766
avg_mfe_pct                  1.096447
avg_mae_pct                 -0.346156
target_hit_rate              0.459459
stop_hit_rate                0.018018
avg_barrier_bars            29.037736
```

## Holdout Bar By Entry

```text
 entry_bar  trades  avg_close_return_pct  median_close_return_pct  hit_rate_to_close  avg_mfe_pct  avg_mae_pct  target_hit_rate  stop_hit_rate  avg_barrier_bars
        20    37.0              0.798456                 0.816620           0.837838     1.365151    -0.237334         0.648649       0.027027            24.760
        25    37.0              0.489466                 0.518756           0.756757     1.043290    -0.344402         0.405405       0.027027            33.125
        28    37.0              0.339083                 0.386237           0.702703     0.880900    -0.456733         0.324324       0.000000            32.500
```

## Holdout Bar By Side

```text
trade_side  trades  avg_close_return_pct  median_close_return_pct  hit_rate_to_close  avg_mfe_pct  avg_mae_pct  target_hit_rate  stop_hit_rate  avg_barrier_bars
      long    78.0              0.371836                 0.407204           0.679487     0.935144    -0.386403         0.333333       0.025641              30.5
     short    33.0              0.945333                 0.996528           0.969697     1.477707    -0.251027         0.757576       0.000000              27.4
```

## Holdout Bar By Asset

```text
asset_class  trades  avg_close_return_pct  median_close_return_pct  hit_rate_to_close  avg_mfe_pct  avg_mae_pct  target_hit_rate  stop_hit_rate  avg_barrier_bars
        etf    18.0              0.538945                 0.236222           0.777778     1.069561    -0.291288         0.500000       0.000000         31.111111
    futures     3.0              0.575068                 0.518756           1.000000     0.680419    -0.096220         1.000000       0.000000         53.000000
      stock    90.0              0.541922                 0.581426           0.755556     1.115692    -0.365461         0.433333       0.022222         26.829268
```

## Top Holdout Signals

```text
ticker       date asset_class trade_side  trade_return_proxy  avg_return  early_pattern_probability  early_up_probability  early_down_probability  early_direction_edge
   SLV 2026-03-13         etf      short            4.090249   -4.090249                   0.772423              0.000000                0.740746             -0.740746
  SBUX 2026-03-19       stock       long            4.072546    4.072546                   0.762231              0.773496                0.052867              0.720629
   TGT 2026-04-17       stock       long            3.305051    3.305051                   0.791564              0.519669                0.000000              0.519669
   OXY 2026-04-21       stock       long            3.291281    3.291281                   0.779964              0.594514                0.002832              0.591682
  MSFT 2026-04-13       stock       long            3.209605    3.209605                   0.793406              0.596105                0.000000              0.596105
  SBUX 2026-03-20       stock      short            3.180249   -3.180249                   0.805147              0.000000                0.672615             -0.672615
  CSCO 2026-03-24       stock       long            3.121019    3.121019                   0.782848              0.766124                0.001513              0.764611
  UBER 2026-04-15       stock       long            2.985320    2.985320                   0.776676              0.787228                0.000000              0.787228
   EWJ 2026-03-20         etf      short            2.975266   -2.975266                   0.795011              0.000000                0.654484             -0.654484
  NFLX 2026-02-20       stock       long            2.935984    2.935984                   0.774549              0.682921                0.000000              0.682921
   UNH 2026-04-28       stock       long            2.887441    2.887441                   0.700802              0.728900                0.001490              0.727410
   CVX 2026-04-09       stock      short            2.842852   -2.842852                   0.743267              0.004110                0.583964             -0.579854
   TGT 2026-04-15       stock       long            2.469754    2.469754                   0.728399              0.694406                0.022365              0.672041
   BAC 2026-03-03       stock       long            2.460529    2.460529                   0.780610              0.539567                0.005380              0.534188
  GOOG 2026-04-14       stock       long            2.458654    2.458654                   0.705030              0.825257                0.000000              0.825257
   NKE 2026-03-18       stock      short            2.448114   -2.448114                   0.759900              0.009726                0.543790             -0.534064
 GOOGL 2026-04-14       stock       long            2.423263    2.423263                   0.725672              0.800487                0.006777              0.793710
  AAPL 2026-03-13       stock      short            2.298378   -2.298378                   0.794826              0.004000                0.570292             -0.566292
   CVX 2026-03-12       stock       long            2.277680    2.277680                   0.729919              0.775518                0.000000              0.775518
  AMZN 2026-03-27       stock      short            2.224123   -2.224123                   0.756801              0.000000                0.639865             -0.639865
    KO 2026-04-08       stock       long            2.185141    2.185141                   0.801712              0.556894                0.000000              0.556894
   XLU 2026-02-13         etf       long            2.120180    2.120180                   0.741255              0.852864                0.000000              0.852864
  NVDA 2026-03-02       stock       long            1.865505    1.865505                   0.812136              0.667150                0.004460              0.662690
  AAPL 2026-04-21       stock      short            1.773226   -1.773226                   0.758415              0.004307                0.663432             -0.659125
   TXN 2026-04-16       stock       long            1.708446    1.708446                   0.793557              0.676852                0.002715              0.674137
  META 2026-02-20       stock       long            1.689289    1.689289                   0.774072              0.518252                0.000000              0.518252
   CVX 2026-04-27       stock      short            1.450667   -1.450667                   0.738166              0.002579                0.690004             -0.687425
  CSCO 2026-04-24       stock       long            1.424339    1.424339                   0.737477              0.636984                0.004483              0.632500
   NKE 2026-04-23       stock      short            1.409692   -1.409692                   0.859668              0.002290                0.529263             -0.526973
    HD 2026-03-10       stock       long            1.398254    1.398254                   0.793898              0.789815                0.001070              0.788745
```

## Worst Holdout Signals

```text
ticker       date asset_class trade_side  trade_return_proxy  avg_return  early_pattern_probability  early_up_probability  early_down_probability  early_direction_edge
  UBER 2026-04-15       stock       long            2.985320    2.985320                   0.776676              0.787228                0.000000              0.787228
   EWJ 2026-03-20         etf      short            2.975266   -2.975266                   0.795011              0.000000                0.654484             -0.654484
  NFLX 2026-02-20       stock       long            2.935984    2.935984                   0.774549              0.682921                0.000000              0.682921
   UNH 2026-04-28       stock       long            2.887441    2.887441                   0.700802              0.728900                0.001490              0.727410
   CVX 2026-04-09       stock      short            2.842852   -2.842852                   0.743267              0.004110                0.583964             -0.579854
   TGT 2026-04-15       stock       long            2.469754    2.469754                   0.728399              0.694406                0.022365              0.672041
   BAC 2026-03-03       stock       long            2.460529    2.460529                   0.780610              0.539567                0.005380              0.534188
  GOOG 2026-04-14       stock       long            2.458654    2.458654                   0.705030              0.825257                0.000000              0.825257
   NKE 2026-03-18       stock      short            2.448114   -2.448114                   0.759900              0.009726                0.543790             -0.534064
 GOOGL 2026-04-14       stock       long            2.423263    2.423263                   0.725672              0.800487                0.006777              0.793710
  AAPL 2026-03-13       stock      short            2.298378   -2.298378                   0.794826              0.004000                0.570292             -0.566292
   CVX 2026-03-12       stock       long            2.277680    2.277680                   0.729919              0.775518                0.000000              0.775518
  AMZN 2026-03-27       stock      short            2.224123   -2.224123                   0.756801              0.000000                0.639865             -0.639865
    KO 2026-04-08       stock       long            2.185141    2.185141                   0.801712              0.556894                0.000000              0.556894
   XLU 2026-02-13         etf       long            2.120180    2.120180                   0.741255              0.852864                0.000000              0.852864
  NVDA 2026-03-02       stock       long            1.865505    1.865505                   0.812136              0.667150                0.004460              0.662690
  AAPL 2026-04-21       stock      short            1.773226   -1.773226                   0.758415              0.004307                0.663432             -0.659125
   TXN 2026-04-16       stock       long            1.708446    1.708446                   0.793557              0.676852                0.002715              0.674137
  META 2026-02-20       stock       long            1.689289    1.689289                   0.774072              0.518252                0.000000              0.518252
   CVX 2026-04-27       stock      short            1.450667   -1.450667                   0.738166              0.002579                0.690004             -0.687425
  CSCO 2026-04-24       stock       long            1.424339    1.424339                   0.737477              0.636984                0.004483              0.632500
   NKE 2026-04-23       stock      short            1.409692   -1.409692                   0.859668              0.002290                0.529263             -0.526973
    HD 2026-03-10       stock       long            1.398254    1.398254                   0.793898              0.789815                0.001070              0.788745
  NQ_F 2026-05-08     futures       long            1.356904    1.356904                   0.814939              0.525261                0.001513              0.523748
   IWM 2026-03-20         etf      short            1.355091   -1.355091                   0.732729              0.036693                0.571025             -0.534331
   LOW 2026-03-10       stock       long            1.209141    1.209141                   0.708212              0.710233                0.001070              0.709164
   XOM 2026-03-24       stock       long            1.133113    1.133113                   0.834788              0.789785                0.000000              0.789785
   EWJ 2026-02-09         etf       long            1.035724    1.035724                   0.781499              0.673565                0.008376              0.665189
   NKE 2026-02-10       stock       long            0.845835    0.845835                   0.807673              0.793207                0.001479              0.791728
   QQQ 2026-05-01         etf       long            0.336505    0.336505                   0.746112              0.562281                0.001415              0.560866
```

## Worst Holdout Bar Entries

```text
ticker       date asset_class trade_side  entry_bar  close_return_pct  mfe_pct   mae_pct barrier_result  early_pattern_probability  early_direction_edge
   XOM 2026-03-24       stock       long         28         -1.109185 0.089643 -1.685293        neither                   0.834788              0.789785
   NKE 2026-02-10       stock       long         25         -1.103373 0.046952 -1.533766           stop                   0.807673              0.791728
   NKE 2026-02-10       stock       long         20         -1.041422 0.524626 -1.472085           stop                   0.807673              0.791728
   NKE 2026-02-10       stock       long         28         -1.041422 0.007830 -1.472085        neither                   0.807673              0.791728
   XOM 2026-03-24       stock       long         25         -1.014536 0.185440 -1.591195        neither                   0.834788              0.789785
  META 2026-02-20       stock       long         28         -0.821445 0.012102 -1.161823        neither                   0.774072              0.518252
   LOW 2026-03-10       stock       long         28         -0.743112 0.640342 -0.834025        neither                   0.708212              0.709164
   LOW 2026-03-10       stock       long         25         -0.680299 0.704030 -0.771269        neither                   0.708212              0.709164
  META 2026-02-20       stock       long         20         -0.648603 0.369764 -0.989574        neither                   0.774072              0.518252
    HD 2026-03-10       stock       long         28         -0.531945 0.718543 -0.587646        neither                   0.793898              0.788745
  META 2026-02-20       stock       long         25         -0.482710 0.537357 -0.824251        neither                   0.774072              0.518252
   TXN 2026-04-16       stock       long         25         -0.426520 0.011165 -0.765950        neither                   0.793557              0.674137
   CVX 2026-03-12       stock       long         28         -0.323608 0.561258 -0.485412        neither                   0.729919              0.775518
   LOW 2026-03-10       stock       long         20         -0.309699 1.079796 -0.401008         target                   0.708212              0.709164
   XOM 2026-03-24       stock       long         20         -0.292842 0.915883 -0.873705        neither                   0.834788              0.789785
    HD 2026-03-10       stock       long         25         -0.283389 0.970223 -0.339229        neither                   0.793898              0.788745
   TXN 2026-04-16       stock       long         28         -0.228229 0.107402 -0.568334        neither                   0.793557              0.674137
   QQQ 2026-05-01         etf       long         25         -0.213974 0.096992 -0.329461        neither                   0.746112              0.560866
   QQQ 2026-05-01         etf       long         28         -0.196979 0.069609 -0.312485        neither                   0.746112              0.560866
  NVDA 2026-03-02       stock       long         28         -0.180559 0.410542 -0.582837        neither                   0.812136              0.662690
  CSCO 2026-04-24       stock       long         28         -0.123429 0.684470 -0.437612        neither                   0.737477              0.632500
    HD 2026-03-10       stock       long         20         -0.086723 1.169362 -0.142673         target                   0.793898              0.788745
   XLU 2026-02-13         etf       long         25         -0.085985 0.171969 -0.386930        neither                   0.741255              0.852864
   CVX 2026-03-12       stock       long         25         -0.038031 0.849370 -0.200299        neither                   0.729919              0.775518
   CVX 2026-04-27       stock      short         28         -0.037897 0.563045 -0.703806        neither                   0.738166             -0.687425
   QQQ 2026-05-01         etf       long         20         -0.029671 0.281870 -0.145371        neither                   0.746112              0.560866
   XLU 2026-02-13         etf       long         28          0.053816 0.312130 -0.247551        neither                   0.741255              0.852864
   NKE 2026-04-23       stock      short         25          0.055822 1.216925 -0.279111         target                   0.859668             -0.526973
   NKE 2026-04-23       stock      short         28          0.066979 1.227953 -0.267917         target                   0.859668             -0.526973
 GOOGL 2026-04-14       stock       long         28          0.081183 0.213482 -0.351795        neither                   0.725672              0.793710
```