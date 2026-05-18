# Market Universe For Structural Learning

The current local dataset already has broad stock, ETF, mega-cap tech, crypto,
and sector coverage. The next learning system should keep expanding into symbols
that provide different structural regimes rather than only more similar equities.

## Current Priority Symbols

Already local:

```text
NVDA
ETH_USD
BTC_USD
QQQ
SPY
AAPL
MSFT
TSLA
AMD
META
AMZN
GOOGL
IWM
DIA
TLT
GLD
SLV
```

## Add Next With YFinance

Forex:

```text
EURUSD=X
GBPUSD=X
USDJPY=X
USDCHF=X
USDCAD=X
AUDUSD=X
NZDUSD=X
EURJPY=X
GBPJPY=X
```

Futures:

```text
ES=F
NQ=F
YM=F
RTY=F
GC=F
SI=F
CL=F
NG=F
ZB=F
ZN=F
6E=F
6J=F
```

## Learning Direction

The first model should learn from human feedback on structural groups:

```text
Mach2AImarket.py feedback -> mach2_group_feedback.csv
downloaded 5m days -> structural features
structural_pattern_learner.py -> scored candidate days
```

The model should improve as we add more feedback. The key behavior is not
"predict price perfectly"; it is "surface days/groups that resemble structures
the user repeatedly marks as meaningful."
