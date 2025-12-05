import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from binance.client import Client


def download_btc_data(output_path: Path | str | None = None, years: int = 3) -> pd.DataFrame:
    """
    Download historical BTC/USDT 1h candles from Binance and save to CSV.

    Args:
        output_path: Where to save the CSV. Defaults to ../data/btc_1h.csv.
        years: Number of years of history to download (backward from now).

    Returns:
        DataFrame of cleaned candle data with columns:
        timestamp, open, high, low, close, volume
    """
    output_path = Path(output_path) if output_path else Path(__file__).resolve().parents[1] / "data" / "btc_1h.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = Client()  # Public endpoints do not require keys
    end_time = int(datetime.utcnow().timestamp() * 1000)
    start_time = int((datetime.utcnow() - timedelta(days=365 * years)).timestamp() * 1000)

    klines = client.get_historical_klines(
        "BTCUSDT",
        Client.KLINE_INTERVAL_1HOUR,
        start_time,
        end_time,
    )

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]

    df = pd.DataFrame(klines, columns=columns)
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop(columns=["open_time"])
    df = df.rename(
        columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
    )
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp").reset_index(drop=True)

    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    default_path = Path(__file__).resolve().parents[1] / "data" / "btc_1h.csv"
    df_downloaded = download_btc_data(default_path)
    print(f"Downloaded {len(df_downloaded)} rows to {default_path}")
