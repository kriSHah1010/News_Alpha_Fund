import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(page_title="News Alpha Fund", layout="wide")

# ==============================
# DARK QUANT THEME
# ==============================

st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #e6e6e6;
}
.stMetric {
    background-color: #111827;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# TITLE
# ==============================

st.title("🧠 News Alpha Fund")
st.markdown("""
### Sentiment-Driven Market Prediction Using FinBERT & Time-Series ML

This project evaluates whether financial news sentiment provides predictive power 
for DJIA market movements.  

It combines:
- FinBERT NLP sentiment analysis  
- Time-series cross-validation  
- XGBoost classification  
- Regime segmentation  
- Strategy backtesting  

""")

# ==============================
# LOAD DATA
# ==============================

results = pd.read_csv("data/final_results.csv")
results["Date"] = pd.to_datetime(results["Date"])

sentiment = pd.read_csv("data/sentiment_output.csv")
sentiment["Date"] = pd.to_datetime(sentiment["Date"])

# ==============================
# PERFORMANCE METRICS
# ==============================

strategy_returns = results["strategy_return"]
market_returns = results["return"]

total_return_strategy = (1 + strategy_returns).prod() - 1
annual_return = (1 + total_return_strategy) ** (252 / len(results)) - 1
annual_vol = strategy_returns.std() * np.sqrt(252)
sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else 0

cumulative = results["cumulative_strategy"]
peak = cumulative.cummax()
drawdown = (cumulative - peak) / peak
max_drawdown = drawdown.min()
win_rate = (strategy_returns > 0).mean()

# ==============================
# SIDEBAR NAVIGATION
# ==============================

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Model Performance", "Regime Analysis", "Live Headline Predictor"]
)

# ==============================
# PAGE 1 — OVERVIEW
# ==============================

if page == "Overview":

    st.subheader("📊 Strategy Performance Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return", f"{total_return_strategy:.2%}")
    col2.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    col3.metric("Max Drawdown", f"{max_drawdown:.2%}")
    col4.metric("Win Rate", f"{win_rate:.2%}")

    st.markdown("---")

    st.subheader("📈 Strategy vs Market")

    st.markdown("""
    This chart compares cumulative returns of:
    - The sentiment-driven strategy  
    - Buy-and-hold DJIA benchmark  
    """)

    fig = px.line(
        results,
        x="Date",
        y=["cumulative_strategy", "cumulative_market"],
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📉 Drawdown")

    st.markdown("""
    Drawdown measures peak-to-trough decline in strategy value.
    Lower drawdown indicates better risk control.
    """)

    fig_dd = px.line(
        x=results["Date"],
        y=drawdown,
        template="plotly_dark"
    )
    st.plotly_chart(fig_dd, use_container_width=True)

# ==============================
# PAGE 2 — MODEL PERFORMANCE
# ==============================

elif page == "Model Performance":

    st.subheader("🔍 Feature Importance")

    st.markdown("""
    Feature importance shows which inputs most influence model predictions.
    Sentiment features are evaluated against traditional lag-based signals.
    """)

    feature_cols = [
        "sentiment_news",
        "sentiment_reddit",
        "lag_1",
        "lag_2",
        "rolling_vol"
    ]

    importance_values = [0.2007, 0.1884, 0.1925, 0.1950, 0.2234]

    fig_imp = px.bar(
        x=feature_cols,
        y=importance_values,
        template="plotly_dark"
    )
    st.plotly_chart(fig_imp, use_container_width=True)

# ==============================
# PAGE 3 — REGIME ANALYSIS
# ==============================

elif page == "Regime Analysis":

    st.subheader("🌡 Volatility Regime Segmentation")

    vol_threshold = results["rolling_vol"].median()

    results["Regime"] = np.where(
        results["rolling_vol"] > vol_threshold,
        "High Volatility",
        "Low Volatility"
    )

    st.markdown("""
    We test whether sentiment predictive power changes during:
    - High volatility periods  
    - Low volatility periods  
    """)

    fig_vol = px.line(
        results,
        x="Date",
        y="rolling_vol",
        color="Regime",
        template="plotly_dark"
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    # OOS ROC
    feature_cols = [
        "sentiment_news",
        "sentiment_reddit",
        "lag_1",
        "lag_2",
        "rolling_vol"
    ]

    X = results[feature_cols]
    y = results["target"]

    tscv = TimeSeriesSplit(n_splits=5)
    oos_probs = np.zeros(len(X))

    for train_idx, test_idx in tscv.split(X):
        model_temp = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        )
        model_temp.fit(X.iloc[train_idx], y.iloc[train_idx])
        oos_probs[test_idx] = model_temp.predict_proba(X.iloc[test_idx])[:, 1]

    results["oos_prob"] = oos_probs

    high_vol = results[results["Regime"] == "High Volatility"]
    low_vol = results[results["Regime"] == "Low Volatility"]

    roc_high = roc_auc_score(high_vol["target"], high_vol["oos_prob"])
    roc_low = roc_auc_score(low_vol["target"], low_vol["oos_prob"])

    col1, col2 = st.columns(2)
    col1.metric("High Vol ROC", f"{roc_high:.3f}")
    col2.metric("Low Vol ROC", f"{roc_low:.3f}")

# ==============================
# PAGE 4 — LIVE HEADLINE PREDICTOR
# ==============================

elif page == "Live Headline Predictor":

    st.subheader("📰 Enter a Financial Headline")

    headline = st.text_area("Type a news headline here:")

    if headline:

        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        model.eval()

        inputs = tokenizer(headline, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = F.softmax(outputs.logits, dim=1).numpy()[0]

        negative, neutral, positive = probs

        st.markdown("### Sentiment Probabilities")
        st.write(f"Positive: {positive:.3f}")
        st.write(f"Neutral: {neutral:.3f}")
        st.write(f"Negative: {negative:.3f}")

        sentiment_score = positive - negative

        st.markdown("### Sentiment Score")
        st.write(f"{sentiment_score:.3f}")

        st.markdown("""
        Sentiment score = Positive − Negative probability.
        """)