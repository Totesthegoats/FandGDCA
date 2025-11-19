import os
import requests
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# -----------------------------
# 1. PARAMETERS
# -----------------------------

# Base DCA amount per contribution
BASE_USD = 10.0

# How many days of history to fetch (used for BTC prices & approximate F&G length)
DAYS_BACK = 730  # ~2 years

# DCA frequency: "daily", "weekly", or "monthly"
DCA_FREQUENCY = "daily"  # change to "weekly" or "monthly" as you like

# Whether to normalize weights so that the average weight over the period = 1
# (Makes total capital deployed similar to flat DCA; isolates *timing* effect)
NORMALIZE_WEIGHTS = False

# Fear & Greed bucket -> weight mapping
WEIGHT_MAP = {
    "extreme_fear": 2.0,
    "fear": 1.5,
    "neutral": 1.0,
    "greed": 0.9,
    "extreme_greed": 0.75,
}

CMC_API_KEY = "163c8f6f3b81405fa0308d4834fe9ebc"

# -----------------------------
# 2. GET FEAR & GREED FROM CMC
# -----------------------------

def get_fng_data():
    """
    Fetches Fear & Greed Index from CoinMarketCap using pagination (limit=100 per request).

    Handles both:
      - Unix timestamp seconds (e.g. "1736294400")
      - ISO8601 strings (e.g. "2024-09-02T12:00:00.000Z")

    Returns:
        DataFrame with columns:
            date (datetime64[ns] normalized to midnight),
            value (int),
            bucket (str in {extreme_fear, fear, neutral, greed, extreme_greed})
    """
    if not CMC_API_KEY:
        raise ValueError("CMC_API_KEY not set in environment variables")

    url = "https://pro-api.coinmarketcap.com/v3/fear-and-greed/historical"

    headers = {
        "X-CMC_PRO_API_KEY": CMC_API_KEY,
        "Accept": "application/json",
    }

    all_rows = []
    limit = 100
    start = 1

    # Paginate until we have at least DAYS_BACK rows or run out of data
    while len(all_rows) < DAYS_BACK:
        params = {"start": start, "limit": limit}
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        raw = resp.json()

        if "data" not in raw or not isinstance(raw["data"], list):
            break

        rows = raw["data"]
        if not rows:
            break

        all_rows.extend(rows)
        start += limit

    if not all_rows:
        raise ValueError("No data returned from CMC Fear & Greed endpoint")

    df = pd.DataFrame(all_rows)

    # ---- robust timestamp parsing ----
    ts_raw = df["timestamp"].astype(str)

    # If all values are digits, treat as Unix seconds
    if ts_raw.str.fullmatch(r"\d+").all():
        ts = pd.to_datetime(ts_raw.astype("int64"), unit="s", errors="coerce")
    else:
        # Otherwise assume ISO 8601 strings
        ts = pd.to_datetime(ts_raw, errors="coerce")

    df["date"] = ts.dt.normalize()

    df["value"] = df["value"].astype(int)

    # Map classification strings to our buckets
    def to_bucket(c: str) -> str:
        c = c.lower()
        if "extreme fear" in c:
            return "extreme_fear"
        if c == "fear":
            return "fear"
        if "neutral" in c:
            return "neutral"
        if "extreme greed" in c:
            return "extreme_greed"
        if c == "greed":
            return "greed"
        return "neutral"

    df["bucket"] = df["value_classification"].apply(to_bucket)

    df = df[["date", "value", "bucket"]].dropna().drop_duplicates("date")
    df = df.sort_values("date").reset_index(drop=True)
    return df

# -----------------------------
# 3. GET BTC PRICE FROM BINANCE
# -----------------------------

def get_btc_prices(days_back=DAYS_BACK):
    """
    Fetches BTCUSDT daily candles from Binance.

    Returns:
        DataFrame with columns:
            date (datetime64[ns] normalized to midnight),
            price (float)
    """
    url = (
        "https://api.binance.com/api/v3/klines"
        f"?symbol=BTCUSDT&interval=1d&limit={days_back}"
    )

    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbav", "tbqv", "ignore"
    ]

    df = pd.DataFrame(data, columns=cols)

    ts = pd.to_datetime(df["open_time"], unit="ms", errors="coerce")
    df["date"] = ts.dt.normalize()
    df["price"] = df["close"].astype(float)

    df = df.groupby("date", as_index=False)["price"].mean()
    return df[["date", "price"]]

# -----------------------------
# 4. WEIGHTS + BACKTEST SETUP
# -----------------------------

def build_weight_series(fng_df: pd.DataFrame) -> pd.DataFrame:
    df = fng_df.copy()
    df["weight_raw"] = df["bucket"].map(WEIGHT_MAP)

    if NORMALIZE_WEIGHTS:
        avg_weight = df["weight_raw"].mean()
        df["weight"] = df["weight_raw"] / avg_weight
    else:
        df["weight"] = df["weight_raw"]

    return df[["date", "value", "bucket", "weight"]]

# -----------------------------
# 5. BACKTEST
# -----------------------------

def backtest_dca(df: pd.DataFrame, base_usd: float = BASE_USD):
    df = df.sort_values("date").reset_index(drop=True).copy()

    # Flat DCA
    df["invest_flat"] = base_usd
    df["btc_flat"] = df["invest_flat"] / df["price"]
    df["cum_btc_flat"] = df["btc_flat"].cumsum()
    df["cum_invest_flat"] = df["invest_flat"].cumsum()

    # Weighted DCA
    df["invest_weighted"] = base_usd * df["weight"]
    df["btc_weighted"] = df["invest_weighted"] / df["price"]
    df["cum_btc_weighted"] = df["btc_weighted"].cumsum()
    df["cum_invest_weighted"] = df["invest_weighted"].cumsum()

    final_price = df["price"].iloc[-1]

    flat_total = df["cum_invest_flat"].iloc[-1]
    flat_btc = df["cum_btc_flat"].iloc[-1]
    flat_value = flat_btc * final_price

    w_total = df["cum_invest_weighted"].iloc[-1]
    w_btc = df["cum_btc_weighted"].iloc[-1]
    w_value = w_btc * final_price

    summary = {
        "start": df["date"].iloc[0],
        "end": df["date"].iloc[-1],
        "final_price": final_price,
        "flat": {
            "total": flat_total,
            "btc": flat_btc,
            "value": flat_value,
            "roi": flat_value / flat_total - 1,
        },
        "weighted": {
            "total": w_total,
            "btc": w_btc,
            "value": w_value,
            "roi": w_value / w_total - 1,
        },
    }

    return df, summary

# -----------------------------
# 6. PLOTTING
# -----------------------------

def plot_equity_curves(df: pd.DataFrame):
    df = df.copy()
    df["value_flat"] = df["cum_btc_flat"] * df["price"]
    df["value_weighted"] = df["cum_btc_weighted"] * df["price"]

    # Portfolio value
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["value_flat"], label="Flat DCA")
    plt.plot(df["date"], df["value_weighted"], label="Weighted DCA")
    plt.title("Portfolio Value")
    plt.ylabel("USD")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Capital deployed
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["cum_invest_flat"], label="Flat invested")
    plt.plot(df["date"], df["cum_invest_weighted"], label="Weighted invested")
    plt.title("Total Capital Deployed")
    plt.ylabel("USD")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Average cost basis
    df["avg_flat"] = df["cum_invest_flat"] / df["cum_btc_flat"]
    df["avg_weighted"] = df["cum_invest_weighted"] / df["cum_btc_weighted"]
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["avg_flat"], label="Flat avg cost")
    plt.plot(df["date"], df["avg_weighted"], label="Weighted avg cost")
    plt.title("Average Cost Basis")
    plt.ylabel("USD per BTC")
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------
# 7. MAIN
# -----------------------------

if __name__ == "__main__":
    print("Fetching Fear & Greed (CMC)...")
    fng_raw = get_fng_data()

    print("Fetching BTC prices (Binance)...")
    btc_prices = get_btc_prices(DAYS_BACK)

    # Dates already normalized, but make sure:
    fng_raw["date"] = pd.to_datetime(fng_raw["date"]).dt.normalize()
    btc_prices["date"] = pd.to_datetime(btc_prices["date"]).dt.normalize()

    # Filter FNG to BTC date range
    min_d = btc_prices["date"].min()
    max_d = btc_prices["date"].max()
    fng_raw = fng_raw[(fng_raw["date"] >= min_d) & (fng_raw["date"] <= max_d)]

    # Build weights + merge on normalized date
    fng_weights = build_weight_series(fng_raw)
    df = pd.merge(btc_prices, fng_weights, on="date", how="inner")

    if df.empty:
        raise ValueError(
            "Merged dataframe is empty – likely date alignment issue.\n"
            f"BTC dates range: {btc_prices['date'].min()} → {btc_prices['date'].max()}\n"
            f"FNG dates range: {fng_raw['date'].min()} → {fng_raw['date'].max()}"
        )

    # Apply DCA frequency
    if DCA_FREQUENCY == "weekly":
        df = df.set_index("date").resample("W-MON").last().dropna().reset_index()
    elif DCA_FREQUENCY == "monthly":
        df = df.set_index("date").resample("M").last().dropna().reset_index()
    # else: daily → leave as-is

    print(f"Using {len(df)} {DCA_FREQUENCY} points from {df['date'].min()} → {df['date'].max()}")

    backtest_df, summary = backtest_dca(df, BASE_USD)

    print("\n--- RESULTS ---")
    print(f"Period: {summary['start']} → {summary['end']}")
    print(f"Final BTC price: {summary['final_price']:.2f}\n")

    print("Flat DCA:")
    print(f"  Total invested: ${summary['flat']['total']:.2f}")
    print(f"  BTC held:       {summary['flat']['btc']:.6f}")
    print(f"  Final value:    ${summary['flat']['value']:.2f}")
    print(f"  ROI:            {summary['flat']['roi']*100:.2f}%\n")

    print("Weighted DCA:")
    print(f"  Total invested: ${summary['weighted']['total']:.2f}")
    print(f"  BTC held:       {summary['weighted']['btc']:.6f}")
    print(f"  Final value:    ${summary['weighted']['value']:.2f}")
    print(f"  ROI:            {summary['weighted']['roi']*100:.2f}%\n")

    flat_mult = summary["flat"]["value"] / summary["flat"]["total"]
    w_mult = summary["weighted"]["value"] / summary["weighted"]["total"]
    print("Value per $1 invested:")
    print(f"  Flat DCA:     {flat_mult:.3f}x")
    print(f"  Weighted DCA: {w_mult:.3f}x")

    plot_equity_curves(backtest_df)