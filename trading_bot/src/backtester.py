from __future__ import annotations

from typing import Iterable

import pandas as pd


class Backtester:
    """
    Simple long-only backtester.
    Signals: 1 to be long, 0 to be flat. Trades executed at bar close.
    """

    def __init__(self, df: pd.DataFrame, fee: float = 0.0004) -> None:
        self.df = df.reset_index(drop=True)
        self.fee = fee
        self.trades: list[float] = []

    def run(self, signals: Iterable[int]) -> pd.Series:
        signals = list(signals)
        if len(signals) != len(self.df):
            raise ValueError("Signals length must match dataframe length.")

        self.trades = []
        equity = [1.0]
        position = 0
        position_size = 0.0
        entry_equity = None

        for i in range(len(self.df)):
            raw_signal = signals[i]
            if isinstance(raw_signal, dict):
                signal = int(raw_signal.get("signal", 0))
                size = float(raw_signal.get("size", 1.0))
            else:
                signal = int(raw_signal)
                size = 1.0

            size = min(max(size, 0.0), 2.0)

            # Entry/exit with fee applied at the moment of change
            if position == 0 and signal == 1:
                position = 1
                position_size = size
                equity[-1] -= equity[-1] * self.fee * position_size
                entry_equity = equity[-1]
            elif position == 1 and signal == 0:
                equity[-1] -= equity[-1] * self.fee * position_size
                if entry_equity is not None:
                    trade_return = (equity[-1] - entry_equity) / entry_equity
                    self.trades.append(trade_return)
                position = 0
                position_size = 0.0
                entry_equity = None
            elif position == 1 and signal == 1 and abs(size - position_size) > 1e-6:
                # Adjust size with fee on the changed portion
                equity[-1] -= equity[-1] * self.fee * abs(size - position_size)
                position_size = size

            # Apply P&L to next bar
            if i < len(self.df) - 1:
                price_ret = self.df["close"].iloc[i + 1] / self.df["close"].iloc[i] - 1
                next_equity = equity[-1] * (1 + price_ret * position_size) if position == 1 else equity[-1]
                equity.append(next_equity)

        # Close at the end if still in position
        if position == 1 and position_size > 0:
            equity[-1] -= equity[-1] * self.fee * position_size
            if entry_equity is not None:
                trade_return = (equity[-1] - entry_equity) / entry_equity
                self.trades.append(trade_return)

        index = self.df["timestamp"] if "timestamp" in self.df.columns else pd.RangeIndex(len(equity))
        return pd.Series(equity, index=index, name="equity")
