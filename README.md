# Fear & Greed Weighted DCA Backtester

This project backtests a Bitcoin DCA strategy that adjusts position size based on the Crypto Fear & Greed Index. It compares a standard flat DCA to a weighted DCA approach where you invest more during fear and less during greed.

🔍 Features

Uses CoinMarketCap Fear & Greed Index (historical API with pagination)

Pulls BTC daily prices from Binance

Supports daily, weekly, or monthly DCA frequency

Adjustable sentiment weights

Option to normalize weights so total invested matches flat DCA

Generates:

Portfolio value chart

Total capital deployed chart

Average cost basis chart

Prints ROI, BTC accumulated, and final portfolio value for both strategies

````
BASE_USD = 10.0
DCA_FREQUENCY = "daily"      # "daily", "weekly", "monthly"
NORMALIZE_WEIGHTS = True
```

```
WEIGHT_MAP = {
    "extreme_fear": 2.0,
    "fear": 1.5,
    "neutral": 1.0,
    "greed": 0.75,
    "extreme_greed": 0.5,
}
```
