from __future__ import annotations

import numpy as np
import pandas as pd


def _sharpe(series: pd.Series) -> float:
    returns = series.pct_change().dropna()
    if returns.empty or returns.std() == 0:
        return 0.0
    # Annualize for 1h bars
    annual_factor = 8760
    return float((returns.mean() * annual_factor) / (returns.std() * np.sqrt(annual_factor)))


def _slope(series: pd.Series) -> float:
    series = series.dropna()
    if len(series) < 10:
        return 0.0
    x = np.arange(len(series))
    slope = np.polyfit(x, series.values, 1)[0]
    return float(slope)


def compute_market_stats(df: pd.DataFrame, equity_series=None, window: int = 500) -> dict:
    """
    Compute snapshot of market/regime stats from the last `window` bars.
    """
    recent = df.tail(window).copy()
    if recent.empty:
        return {}

    close = recent["close"]
    returns = close.pct_change().dropna()

    vol20 = returns.tail(20).std() if len(returns) >= 20 else returns.std()
    vol100 = returns.tail(100).std() if len(returns) >= 100 else returns.std()

    long_ma = close.rolling(window=100, min_periods=20).mean()
    trend_ma_slope = _slope(long_ma.tail(100))

    running_max = close.cummax()
    drawdowns = close / running_max - 1
    recent_drawdown = float(drawdowns.tail(window).min()) if not drawdowns.empty else 0.0

    recent_return = float(close.iloc[-1] / close.iloc[0] - 1) if len(close) > 1 else 0.0

    stats = {
        "volatility_20": float(vol20) if pd.notna(vol20) else 0.0,
        "volatility_100": float(vol100) if pd.notna(vol100) else 0.0,
        "trend_ma_slope": trend_ma_slope,
        "recent_drawdown": recent_drawdown,
        "recent_return": recent_return,
    }

    if equity_series is not None:
        if isinstance(equity_series, dict):
            for name, series in equity_series.items():
                eq_recent = pd.Series(series).tail(window)
                stats[f"{name.lower()}_sharpe_recent"] = _sharpe(eq_recent)
        else:
            eq_recent = pd.Series(equity_series).tail(window)
            stats["ma_sharpe_recent"] = _sharpe(eq_recent)

    # Trade frequency placeholder: derive from equity drawdowns as proxy if available
    stats["trade_frequency_recent"] = 0.0

    return stats
