import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("default")
df = pd.read_csv("data/historical_data.csv")
df.head()
df.info()
df.describe(include="all")
df["Timestamp IST"] = pd.to_datetime(
    df["Timestamp IST"],
    dayfirst=True,
    errors="coerce"
)
df["Timestamp IST"].head()
df["Timestamp IST"].isna().sum()
df["date"] = df["Timestamp IST"].dt.date
df[["Timestamp IST", "date"]].head()
df = df[df["Coin"].str.contains("BTC", case=False, na=False)]
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)
daily_trader = df.groupby(["account", "date"]).agg(
    total_pnl=("closed_pnl", "sum"),
    avg_leverage=("start_position", "mean"),
    trade_count=("closed_pnl", "count"),
    avg_trade_size=("size_usd", "mean"),
    total_fees=("fee", "sum")
).reset_index()
daily_trader.head()
daily_trader.describe()


# ---- Add sentiment (simulated) ----
np.random.seed(42)

sentiments = [
    "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
]

daily_trader["sentiment"] = np.random.choice(
    sentiments,
    size=len(daily_trader),
    p=[0.15, 0.25, 0.2, 0.25, 0.15]
)

# ---- Clean sentiment text ----
daily_trader["sentiment"] = daily_trader["sentiment"].str.strip().str.title()

# ---- Encode sentiment ----
sentiment_map = {
    "Extreme Fear": -2,
    "Fear": -1,
    "Neutral": 0,
    "Greed": 1,
    "Extreme Greed": 2
}

daily_trader["sentiment_score"] = daily_trader["sentiment"].map(sentiment_map)

# ---- Plot ----
plt.figure(figsize=(8,5))

daily_trader.groupby("sentiment")["total_pnl"].mean().plot(kind="bar")

plt.title("Average Trader PnL by Market Sentiment")
plt.xlabel("Market Sentiment")
plt.ylabel("Average PnL")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
