from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature set:
        - return_1: 1 bar percent change
        - return_5: 5 bar percent change
        - volatility_20: rolling std of returns over 20 bars
        - rsi_14: simple RSI over 14 bars
    """
    close = df["close"]
    returns = close.pct_change()

    features = pd.DataFrame(index=df.index)
    features["return_1"] = returns
    features["return_5"] = close.pct_change(5)
    features["volatility_20"] = returns.rolling(window=20).std()
    features["rsi_14"] = _simple_rsi(close, window=14)

    return features


def _simple_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_target(df: pd.DataFrame) -> pd.Series:
    """
    Binary target: 1 if next close > current close else 0.
    """
    target = (df["close"].shift(-1) > df["close"]).astype(int)
    return target


def train_model(df: pd.DataFrame):
    """
    Train RandomForest on time-ordered split (80/20).
    Returns model and test predictions.
    """
    features = build_features(df)
    target = build_target(df)

    dataset = features.join(target.rename("target")).dropna()
    feature_cols = ["return_1", "return_5", "volatility_20", "rsi_14"]

    split_idx = int(len(dataset) * 0.8)
    train_data = dataset.iloc[:split_idx]
    test_data = dataset.iloc[split_idx:]

    X_train, y_train = train_data[feature_cols], train_data["target"]
    X_test, y_test = test_data[feature_cols], test_data["target"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]

    return model, X_test, y_test, test_pred, test_proba
