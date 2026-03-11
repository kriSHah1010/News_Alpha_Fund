import pandas as pd
import numpy as np

from src.data_loader import DataLoader
from src.preprocessing import combine_daily_headlines
from src.feature_engineering import create_return_features, merge_features
from src.model import XGBModel
from src.backtester import backtest
from src.metrics import calculate_performance_metrics
from sklearn.metrics import roc_auc_score


# ==============================
# 1️⃣ LOAD DATA
# ==============================

loader = DataLoader(
    news_path="data/Combined_News_DJIA.csv",
    reddit_path="data/RedditNews.csv",
    price_path="data/upload_DJIA_table.csv"
)

news = loader.load_news()
reddit = loader.load_reddit()
prices = loader.load_prices()

print("Data Loaded")


# ==============================
# 2️⃣ COMBINE DAILY HEADLINES
# ==============================

news = combine_daily_headlines(news)

print("Headlines Combined")


# ==============================
# 3️⃣ LOAD SAVED SENTIMENT
# ==============================

sentiment_df = pd.read_csv("data/sentiment_output.csv")
sentiment_df["Date"] = pd.to_datetime(sentiment_df["Date"])

print("Sentiment Loaded From File")


# ==============================
# 4️⃣ MERGE SENTIMENT
# ==============================


# ==============================
# 5️⃣ PRICE FEATURES
# ==============================

prices = create_return_features(prices)

final_df = merge_features(sentiment_df, prices)

print("Features Created")


# ==============================
# 6️⃣ MODEL TRAINING
# ==============================

feature_cols = [
    "sentiment_news",
    "sentiment_reddit",
    "lag_1",
    "lag_2",
    "rolling_vol"
]

X = final_df[feature_cols]
y = final_df["target"]

model = XGBModel()
trained_model = model.train(X, y)

model.save("models/xgb_model.pkl")

print("Model Trained")


# ==============================
# PROPER OUT-OF-SAMPLE ROC-AUC
# ==============================

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import numpy as np

tscv = TimeSeriesSplit(n_splits=5)

oos_probs = np.zeros(len(X))

for train_idx, test_idx in tscv.split(X):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model_fold = XGBModel().model
    model_fold.fit(X_train, y_train)

    oos_probs[test_idx] = model_fold.predict_proba(X_test)[:, 1]

roc = roc_auc_score(y, oos_probs)

print(f"\nOut-of-Sample ROC-AUC: {roc:.4f}")


# ==============================
# REGIME SEGMENTATION (OOS)
# ==============================

final_df["oos_prob"] = oos_probs

vol_threshold = final_df["rolling_vol"].median()

high_vol = final_df[final_df["rolling_vol"] > vol_threshold]
low_vol = final_df[final_df["rolling_vol"] <= vol_threshold]

roc_high = roc_auc_score(high_vol["target"], high_vol["oos_prob"])
roc_low = roc_auc_score(low_vol["target"], low_vol["oos_prob"])

print("\n===== REGIME ROC-AUC (OOS) =====")
print(f"High Volatility ROC-AUC: {roc_high:.4f}")
print(f"Low Volatility ROC-AUC: {roc_low:.4f}")

# ==============================
# FEATURE IMPORTANCE
# ==============================

print("\n===== FEATURE IMPORTANCE =====")

importances = trained_model.feature_importances_

for name, val in zip(feature_cols, importances):
    print(f"{name}: {val:.4f}")


# ==============================
# 7️⃣ BACKTEST
# ==============================

results = backtest(final_df, trained_model, feature_cols)

results.to_csv("data/final_results.csv", index=False)

print("Backtest Complete")

# ==============================
# PERFORMANCE METRICS
# ==============================

metrics = calculate_performance_metrics(results)

print("\n===== PERFORMANCE METRICS =====")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")