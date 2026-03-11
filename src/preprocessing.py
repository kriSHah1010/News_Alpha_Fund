import pandas as pd


def combine_daily_headlines(news_df):

    headline_cols = [col for col in news_df.columns if "Top" in col]

    news_df["combined_text"] = (
        news_df[headline_cols]
        .astype(str)
        .agg(" ".join, axis=1)
    )

    return news_df[["Date", "combined_text"]]