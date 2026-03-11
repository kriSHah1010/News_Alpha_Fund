import numpy as np


def backtest(df, model, feature_cols):

    df = df.copy()

    df["proba"] = model.predict_proba(df[feature_cols])[:, 1]

    # Probability-weighted position
    df["position"] = 2 * (df["proba"] - 0.5)

    # Clip to avoid extreme leverage
    df["position"] = df["position"].clip(-1, 1)

    df["strategy_return"] = df["position"] * df["return"]

    df["cumulative_strategy"] = (1 + df["strategy_return"]).cumprod()
    df["cumulative_market"] = (1 + df["return"]).cumprod()

    return df