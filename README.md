![Python](https://img.shields.io/badge/Python-3.9-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)
![Transformers](https://img.shields.io/badge/HuggingFace-FinBERT-yellow)


# 🧠 News Alpha Fund  
### Sentiment-Driven Market Prediction Using FinBERT & Time-Series Machine Learning
 
---

## 📌 Project Overview

This project investigates whether financial news sentiment contains predictive information for DJIA market movements.

Using FinBERT-based NLP sentiment extraction combined with time-series machine learning, the study evaluates:

- Directional predictive power of news sentiment  
- Economic significance via backtesting  
- Regime-dependent behavior under high vs low volatility  
- Risk-adjusted performance metrics  

The objective is not curve-fitting profitability, but rigorous statistical and economic evaluation of sentiment-driven signals.

---

## 🏗 Architecture
Raw News + Reddit Data
↓
FinBERT Sentiment Extraction
↓
Feature Engineering
↓
TimeSeriesSplit Cross-Validation
↓
XGBoost Classification Model
↓
Out-of-Sample ROC Evaluation
↓
Strategy Backtest
↓
Regime Segmentation Analysis
↓
Interactive Streamlit Dashboard


---

## 📊 Dataset

- **Combined_News_DJIA** – 25 daily financial headlines
- **RedditNews** – 73,000+ Reddit financial news entries
- **DJIA Historical Prices** – Open, High, Low, Close, Volume

Time span: 1989 trading days

---

## 🧠 Methodology

### 1️⃣ Sentiment Extraction
- FinBERT (ProsusAI/finbert)
- Sentiment score = Positive − Negative probability
- Daily aggregation of news and Reddit sentiment

### 2️⃣ Feature Engineering
- Sentiment (News + Reddit)
- Lagged returns (1-day, 2-day)
- Rolling volatility (5-day)

### 3️⃣ Time-Series Validation
- 5-fold TimeSeriesSplit
- Strict out-of-sample evaluation
- No data leakage

### 4️⃣ Modeling
- XGBoost classifier
- Directional target (1-day & 3-day horizon tested)

### 5️⃣ Economic Evaluation
- Long-only & probability-weighted strategies
- Cumulative returns
- Sharpe ratio
- Maximum drawdown
- Win rate

### 6️⃣ Regime Segmentation
- Median rolling volatility split
- High-volatility vs Low-volatility evaluation
- Regime-specific ROC-AUC analysis

---

## 📈 Key Findings

- Daily sentiment shows weak unconditional predictive power (OOS ROC ≈ 0.49)
- Predictive power improves during high-volatility regimes (OOS ROC ≈ 0.52)
- No statistically significant alpha in calm markets
- Sentiment influence is regime-dependent
- Backtest confirms limited standalone trading profitability

### Interpretation:
Financial news sentiment appears to contain conditional information during stressed market regimes, but does not provide consistent alpha at daily horizons.

---

## 📊 Dashboard Features

Built using Streamlit:

- 📈 Strategy vs Market performance visualization  
- 📉 Drawdown analysis  
- 📊 Feature importance inspection  
- 🌡 Regime-dependent ROC evaluation  
- 🧠 Live FinBERT headline sentiment predictor  

Run locally:

```bash
streamlit run app/streamlit_app.py
```

## 🛠 Tech Stack

- Python
- Pandas / NumPy
- XGBoost
- HuggingFace Transformers (FinBERT)
- Scikit-Learn
- Plotly
- Streamlit

## 🎓 Academic Value

This project demonstrates:

- Proper time-series cross-validation
- Detection and correction of data leakage
- Regime-dependent modeling
- Statistical vs economic significance distinction
- End-to-end ML pipeline design
- Production-style dashboard deployment

## 🚀 Future Improvements

- Intraday data integration
- Transaction cost modeling
- Probabilistic calibration (Platt scaling)
- Transformer fine-tuning
- Ensemble sentiment models
- Multi-asset extension

## 👨‍💻 Author

- Krish Shah
- MSc Data Science

## ⚠️ Disclaimer

- This project is for academic and research purposes only.
- It does not constitute financial advice.

## 🔎 What This Project Does (Step-by-Step)

- Started with three datasets: daily financial headlines, Reddit financial news posts, and historical DJIA market data.
- Used FinBERT (a finance-specific NLP model) to convert raw news headlines into measurable sentiment scores (positive, neutral, negative).
- Aggregated daily sentiment from both traditional news and Reddit to create structured numerical features.
- Combined sentiment features with market-based indicators such as lagged returns and rolling volatility.
- Built a time-series machine learning pipeline using XGBoost to predict future market direction.
- Applied strict TimeSeriesSplit cross-validation to avoid data leakage and ensure realistic out-of-sample evaluation.
- Evaluated statistical performance using ROC-AUC rather than just accuracy.
- Translated predictions into a trading strategy and conducted full backtesting.
- Measured economic performance using Sharpe ratio, drawdown, win rate, and cumulative returns.
- Discovered that sentiment does not provide strong unconditional predictive power.
- Identified regime-dependent behavior: predictive signal improves during high-volatility periods.
- Built an interactive Streamlit dashboard to visualize strategy performance, risk metrics, regime segmentation, and live headline sentiment prediction.
- Delivered an end-to-end research pipeline from raw text data → NLP → ML → backtesting → deployment.
