from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def compute_performance(
    equity: pd.Series,
    trades: Iterable[float] | None = None,
    freq_hours: int = 1,
) -> dict:
    trades = list(trades or [])
    equity = equity.dropna()

    final_equity = equity.iloc[-1]
    start_equity = equity.iloc[0]
    total_return = final_equity / start_equity - 1

    returns = equity.pct_change().dropna()
    periods = len(returns)
    annual_factor = 8760 / freq_hours  # hours per year for 1h bars

    if periods > 0:
        per_period = (final_equity / start_equity) ** (1 / periods) - 1
        annualized_return = (1 + per_period) ** annual_factor - 1
        annualized_volatility = returns.std() * sqrt(annual_factor)
    else:
        annualized_return = 0.0
        annualized_volatility = 0.0

    sharpe_ratio = (
        annualized_return / annualized_volatility if annualized_volatility > 0 else 0.0
    )

    running_max = equity.cummax()
    drawdowns = equity / running_max - 1
    max_drawdown = abs(drawdowns.min()) if not drawdowns.empty else 0.0

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t < 0]
    number_of_trades = len(trades)
    win_rate = len(wins) / number_of_trades if number_of_trades else 0.0
    average_win = float(np.mean(wins)) if wins else 0.0
    average_loss = float(np.mean(losses)) if losses else 0.0

    return {
        "final_equity": float(final_equity),
        "total_return": float(total_return),
        "max_drawdown": float(max_drawdown),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "number_of_trades": int(number_of_trades),
        "win_rate": float(win_rate),
        "average_win": average_win,
        "average_loss": average_loss,
    }


def save_metrics(metrics: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
