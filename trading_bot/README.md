# BTC/USDT 1h Trading Bot (Backtest-Ready)

End-to-end scaffold for downloading Binance BTC/USDT data, running baseline and ML strategies, and plotting equity curves on the 1-hour timeframe.

## Setup
1. Python 3.10+ recommended.
2. Create env and install deps:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Data Download
Fetch 3 years of 1h BTC/USDT candles and save to `data/btc_1h.csv`:
```bash
python -m src.data_loader
```
Public endpoints are used; no API keys required.

## Run Backtests
Execute both strategies (moving average and ML) and plot equity curves:
```bash
python -m src.run_backtest
```
Equity plot is saved to `notebooks/equity_curve.png`. Summary metrics print to console.

## Project Layout
```
trading_bot/
  config/config.yaml     # Tunable parameters
  data/                  # CSV output from downloader
  notebooks/             # Figures/results
  src/
    data_loader.py       # Binance downloader + CSV writer
    backtester.py        # Simple long-only engine with fees
    strategies.py        # Moving average + ML signal generators
    ml_model.py          # Feature engineering and RF training
    regime.py            # Regime detection placeholder (GPT hook)
    run_backtest.py      # Orchestrates strategies and plotting
```

## Extending the Bot
- Adjust parameters in `config/config.yaml` or edit function args directly.
- Swap in new indicators within `strategies.py`.
- Enhance feature set or model in `ml_model.py` (add cross-validation, alternative learners, etc.).
- Replace `regime.detect_regime` with GPT-driven logic to modulate sizing (`position_multiplier`) based on stats you compute (e.g., volatility, trend strength).

## Notes
- Backtester is long-only and operates on close-to-close moves with fees applied on entries/exits.
- Signals align one bar ahead; warmup periods are zeroed to avoid lookahead bias.
