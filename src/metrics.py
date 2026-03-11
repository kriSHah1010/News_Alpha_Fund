import numpy as np


def calculate_performance_metrics(df):

    strategy_returns = df["strategy_return"]
    market_returns = df["return"]

    # Total return
    total_return_strategy = (1 + strategy_returns).prod() - 1
    total_return_market = (1 + market_returns).prod() - 1

    # Annualization (252 trading days)
    annual_return_strategy = (1 + total_return_strategy) ** (252 / len(df)) - 1
    annual_vol_strategy = strategy_returns.std() * np.sqrt(252)

    sharpe_ratio = annual_return_strategy / annual_vol_strategy if annual_vol_strategy != 0 else 0

    # Max Drawdown
    cumulative = (1 + strategy_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    win_rate = (strategy_returns > 0).mean()

    return {
        "Total Return (Strategy)": total_return_strategy,
        "Total Return (Market)": total_return_market,
        "Annual Return (Strategy)": annual_return_strategy,
        "Annual Volatility (Strategy)": annual_vol_strategy,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate
    }