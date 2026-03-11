import pandas as pd
import numpy as np


def create_return_features(price_df):

    price_df = price_df.copy()

    # Sort by date (very important for time series)
    price_df.sort_values("Date", inplace=True)

    # Daily returns
    price_df["return"] = price_df["Close"].pct_change()

    # Lag features
    price_df["lag_1"] = price_df["return"].shift(1)
    price_df["lag_2"] = price_df["return"].shift(2)

    # Rolling volatility (5-day std)
    price_df["rolling_vol"] = price_df["return"].rolling(5).std()

    return price_df


def merge_features(sentiment_df, price_df):

    df = pd.merge(sentiment_df, price_df, on="Date", how="inner")

    # 3-day forward cumulative return
    df["future_3d_return"] = df["return"].shift(-1) + \
                         df["return"].shift(-2) + \
                         df["return"].shift(-3)

    df["target"] = (df["future_3d_return"] > 0).astype(int)

    df.dropna(inplace=True)

    return df