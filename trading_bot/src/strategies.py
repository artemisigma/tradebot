import numpy as np
import pandas as pd


def moving_average_strategy(df: pd.DataFrame, short: int = 20, long: int = 50) -> list[int]:
    """
    Moving average crossover: long when short MA is above long MA.
    Returns list of 0/1 signals aligned with df rows.
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must include 'close' column.")
    ma_short = df["close"].rolling(window=short).mean()
    ma_long = df["close"].rolling(window=long).mean()
    signals = (ma_short > ma_long).astype(int)
    signals.iloc[: max(short, long)] = 0  # avoid lookahead on warmup
    return signals.tolist()


def buy_and_hold_strategy(df: pd.DataFrame) -> list[int]:
    """
    Buy & hold: long from the first bar onward.
    """
    return [1] * len(df)


def ml_strategy(model, X: pd.DataFrame, threshold: float = 0.6, with_size: bool = True):
    """
    Use model predict_proba to generate signals.
    Optionally return dicts with position sizing.
    """
    probs = model.predict_proba(X)[:, 1]
    base_signals = (probs > threshold).astype(int)

    if not with_size:
        return base_signals.tolist()

    sizes = np.clip((probs - 0.5) * 4, 0, 1)
    signals = [{"signal": int(sig), "size": float(size)} for sig, size in zip(base_signals, sizes)]
    return signals
