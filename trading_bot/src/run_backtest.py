from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    # When run as module: python -m src.run_backtest
    from .backtester import Backtester
    from .ml_model import train_model
    from .market_stats import compute_market_stats
    from .performance import compute_performance, save_metrics
    from .regime import detect_regime
    from .strategies import (
        buy_and_hold_strategy,
        ml_strategy,
        moving_average_strategy,
    )
except ImportError:
    # When executed directly: python src/run_backtest.py
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from backtester import Backtester
    from ml_model import train_model
    from market_stats import compute_market_stats
    from performance import compute_performance, save_metrics
    from regime import detect_regime
    from strategies import (
        buy_and_hold_strategy,
        ml_strategy,
        moving_average_strategy,
    )


def load_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found. Run data_loader.download_btc_data first.")
    df = pd.read_csv(data_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def run_strategy(name: str, df: pd.DataFrame, signals):
    bt = Backtester(df)
    equity = bt.run(signals)
    metrics = compute_performance(equity, getattr(bt, "trades", []))
    return equity, metrics


def align_equity(equity: pd.Series, df: pd.DataFrame) -> pd.Series:
    if "timestamp" in df.columns:
        return equity.reindex(df["timestamp"])
    return equity.reset_index(drop=True).reindex(range(len(df)))


def build_hybrid_signals(df: pd.DataFrame, signals_map: dict, equities_map: dict, window: int = 500):
    hybrid = []
    for i in range(len(df)):
        if i < window:
            regime_info = {"preferred_strategy": "buy_hold", "position_multiplier": 1.0}
        else:
            equity_slice = {k: v.iloc[: i + 1] for k, v in equities_map.items()}
            stats = compute_market_stats(df.iloc[: i + 1], equity_series=equity_slice, window=window)
            regime_info = detect_regime(stats)

        pref = regime_info.get("preferred_strategy", "buy_hold")
        multiplier = regime_info.get("position_multiplier", 1.0)

        chosen_signal = signals_map.get(pref, [0] * len(df))[i]
        if isinstance(chosen_signal, dict):
            base_signal = int(chosen_signal.get("signal", 0))
            base_size = float(chosen_signal.get("size", 1.0))
        else:
            base_signal = int(chosen_signal)
            base_size = 1.0

        size = max(0.0, min(2.0, base_size * multiplier))
        hybrid.append({"signal": base_signal, "size": size})
    return hybrid


def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "btc_1h.csv"
    notebooks_dir = root / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)

    equities = {}
    metrics_all = {}

    # Buy & Hold
    bh_signals = buy_and_hold_strategy(df)
    equities["Buy & Hold"], metrics_all["Buy & Hold"] = run_strategy("Buy & Hold", df, bh_signals)

    # Moving average strategy
    ma_signals = moving_average_strategy(df)
    equities["MA"], metrics_all["MA"] = run_strategy("MA", df, ma_signals)

    # ML strategy
    model, X_test, y_test, test_pred, test_proba = train_model(df)
    df_test = df.loc[X_test.index].copy()
    ml_signals_partial = ml_strategy(model, X_test, threshold=0.6, with_size=True)

    # Expand ML signals to full length
    ml_signals_full = [0] * len(df)
    for sig, idx in zip(ml_signals_partial, X_test.index):
        ml_signals_full[idx] = sig

    equities["ML"], metrics_all["ML"] = run_strategy("ML", df, ml_signals_full)

    # Hybrid strategy using regime detection
    aligned_equities = {
        "MA": align_equity(equities["MA"], df),
        "ML": align_equity(equities["ML"], df),
        "Buy & Hold": align_equity(equities["Buy & Hold"], df),
    }
    signals_map = {
        "buy_hold": bh_signals,
        "ma": ma_signals,
        "ml": ml_signals_full,
    }
    hybrid_signals = build_hybrid_signals(df, signals_map, aligned_equities)
    equities["Hybrid"], metrics_all["Hybrid"] = run_strategy("Hybrid", df, hybrid_signals)

    # Plot equity curves
    plt.figure(figsize=(10, 6))
    for name, equity in equities.items():
        equity.plot(label=name)

    plt.title("Equity Curve Comparison (1h BTCUSDT)")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()

    plot_path = notebooks_dir / "equity_curves.png"
    plt.savefig(plot_path)
    print(f"Equity curves saved to {plot_path}")

    metrics_path = notebooks_dir / "metrics.json"
    save_metrics(metrics_all, metrics_path)
    print(f"Metrics saved to {metrics_path}")

    print("\nStrategy Comparison:")
    print("--------------------------------------------")
    for name in ["Buy & Hold", "MA", "ML", "Hybrid"]:
        m = metrics_all.get(name, {})
        print(
            f"{name:12s}: final {m.get('final_equity', 0):.3f}, "
            f"Sharpe {m.get('sharpe_ratio', 0):.2f}, DD {m.get('max_drawdown', 0):.2f}"
        )

    print(f"ML test accuracy: {(y_test == test_pred).mean():.3f}")
    print(f"ML average prediction prob: {test_proba.mean():.3f}")


if __name__ == "__main__":
    main()
