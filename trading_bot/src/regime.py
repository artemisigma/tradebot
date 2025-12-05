def detect_regime(stats: dict) -> dict:
    """
    Rule-based regime detection returning regime, preferred strategy, and position multiplier.
    """
    vol20 = stats.get("volatility_20", 0)
    vol100 = stats.get("volatility_100", 0)
    slope = stats.get("trend_ma_slope", 0)
    drawdown = stats.get("recent_drawdown", 0)
    recent_return = stats.get("recent_return", 0)

    regime = "flat"
    preferred_strategy = "buy_hold"
    position_multiplier = 1.0

    # High volatility or large recent drawdown -> defensive
    if vol20 > vol100 * 1.5 or drawdown < -0.1:
        regime = "high_vol"
        preferred_strategy = "ml"
        position_multiplier = 0.6
    # Trending with lower short-term vol vs long-term vol
    elif slope > 0 and vol20 <= vol100:
        regime = "trend"
        preferred_strategy = "ma"
        position_multiplier = 1.2
    # Sideways / low slope and muted vol -> range preference
    elif abs(slope) < 1e-4 and vol20 < vol100 * 0.8:
        regime = "range"
        preferred_strategy = "ml"
        position_multiplier = 1.0
    else:
        regime = "flat"
        preferred_strategy = "buy_hold"
        position_multiplier = 1.0

    # Additional tilt based on recent returns
    if recent_return < -0.05:
        position_multiplier *= 0.8
    elif recent_return > 0.05 and regime == "trend":
        position_multiplier *= 1.1

    position_multiplier = max(0.0, min(2.0, position_multiplier))

    return {
        "regime": regime,
        "preferred_strategy": preferred_strategy,
        "position_multiplier": position_multiplier,
    }
