# Losing Signal Audit

This audit studies the stricter holdout bar-entry rows when available. It is meant to teach the next filter what failure looks like, not to delete signals blindly.

## Scope

```text
combined_rows=216
audit_rows=111
winners=85
losers=26
```

## Variant Summary

```text
      subset  rows  avg_close_return_pct  median_close_return_pct  hit_rate  avg_mfe_pct  avg_mae_pct  target_hit_rate  stop_hit_rate
         all   111              0.542335                 0.525086  0.765766     1.096447    -0.346156         0.459459       0.018018
entry_bar_20    37              0.798456                 0.816620  0.837838     1.365151    -0.237334         0.648649       0.027027
entry_bar_25    37              0.489466                 0.518756  0.756757     1.043290    -0.344402         0.405405       0.027027
entry_bar_28    37              0.339083                 0.386237  0.702703     0.880900    -0.456733         0.324324       0.000000
   long_only    78              0.371836                 0.407204  0.679487     0.935144    -0.386403         0.333333       0.025641
  short_only    33              0.945333                 0.996528  0.969697     1.477707    -0.251027         0.757576       0.000000
  bar20_long    26              0.628261                 0.688621  0.769231     1.210150    -0.276486         0.538462       0.038462
 bar20_short    11              1.200736                 1.211729  1.000000     1.731516    -0.144793         0.909091       0.000000
   no_crypto   111              0.542335                 0.525086  0.765766     1.096447    -0.346156         0.459459       0.018018
  stock_only    90              0.541922                 0.581426  0.755556     1.115692    -0.365461         0.433333       0.022222
    etf_only    18              0.538945                 0.236222  0.777778     1.069561    -0.291288         0.500000       0.000000
```

## Winner Vs Loser Feature Contrast

```text
                      feature  winner_avg  loser_avg  loser_minus_winner
                      mfe_pct    1.293247   0.453061           -0.840186
                      mae_pct   -0.221473  -0.753774           -0.532301
       early_down_probability    0.237863   0.027672           -0.210191
useful_pattern_ensemble_score    0.929802   0.808476           -0.121326
    early_pattern_probability    0.769598   0.771933            0.002335
         early_up_probability    0.425272   0.675194            0.249922
              early_range_pct    2.387679   2.724918            0.337239
         early_direction_edge    0.187408   0.647521            0.460113
```

## Repeated Losing Signals

```text
ticker       date trade_side  losing_entries  worst_close_return_pct  avg_close_return_pct  max_mfe_pct  min_mae_pct  early_pattern_probability  early_direction_edge
   XOM 2026-03-24       long               3               -1.109185             -0.805521     0.915883    -1.685293                   0.834788              0.789785
   NKE 2026-02-10       long               3               -1.103373             -1.062072     0.524626    -1.533766                   0.807673              0.791728
  META 2026-02-20       long               3               -0.821445             -0.650920     0.537357    -1.161823                   0.774072              0.518252
   LOW 2026-03-10       long               3               -0.743112             -0.577703     1.079796    -0.834025                   0.708212              0.709164
    HD 2026-03-10       long               3               -0.531945             -0.300685     1.169362    -0.587646                   0.793898              0.788745
   QQQ 2026-05-01       long               3               -0.213974             -0.146874     0.281870    -0.329461                   0.746112              0.560866
   TXN 2026-04-16       long               2               -0.426520             -0.327374     0.107402    -0.765950                   0.793557              0.674137
   CVX 2026-03-12       long               2               -0.323608             -0.180820     0.849370    -0.485412                   0.729919              0.775518
  NVDA 2026-03-02       long               1               -0.180559             -0.180559     0.410542    -0.582837                   0.812136              0.662690
  CSCO 2026-04-24       long               1               -0.123429             -0.123429     0.684470    -0.437612                   0.737477              0.632500
   XLU 2026-02-13       long               1               -0.085985             -0.085985     0.171969    -0.386930                   0.741255              0.852864
   CVX 2026-04-27      short               1               -0.037897             -0.037897     0.563045    -0.703806                   0.738166             -0.687425
```

## High-Confidence Bar-20 Failures

```text
ticker       date asset_class trade_side  entry_bar  close_return_pct  mfe_pct   mae_pct barrier_result  early_pattern_probability  early_direction_edge  useful_pattern_ensemble_score
   NKE 2026-02-10       stock       long         20         -1.041422 0.524626 -1.472085           stop                   0.807673              0.791728                       0.710999
  META 2026-02-20       stock       long         20         -0.648603 0.369764 -0.989574        neither                   0.774072              0.518252                       0.725927
   XOM 2026-03-24       stock       long         20         -0.292842 0.915883 -0.873705        neither                   0.834788              0.789785                       0.643109
    HD 2026-03-10       stock       long         20         -0.086723 1.169362 -0.142673         target                   0.793898              0.788745                       0.949589
```

## Practical Read

```text
1. Bar 20 remains the strongest general entry variant.
2. Shorts remain cleaner than longs in the current holdout.
3. High-confidence long failures are not low-confidence model errors; they are mostly continuation failures after strong early moves.
4. The next filter should focus on exhaustion/follow-through after entry, not simply raising early_pattern_probability.
```