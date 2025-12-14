# Cryptocurrency Price Prediction System

**Team Zephyrus** | OlympAI Hackathon 2025

Please run it 2-3 times if it doesn't work in the first time.

## What This Project Does

This is a machine learning web app that predicts whether cryptocurrency prices will go **UP or DOWN in the next 3 hours**. It analyzes market patterns using technical indicators and AI to help you understand crypto price movements.

## Features

- Predicts short-term price movements (3 hours ahead)
- Works with any cryptocurrency on Yahoo Finance (BTC, ETH, SOL, etc.)
- Shows prediction confidence percentage
- Displays 6 different analysis charts
- Simple, clean web interface

## How It Works

1. Downloads 2 years of hourly price data
2. Calculates 28 technical indicators (RSI, Bollinger Bands, ATR, etc.)
3. Trains an ensemble of machine learning models (XGBoost + Random Forest)
4. Uses Bitcoin data as market context
5. Makes predictions with confidence scores

## Project Structure

```
main.py              - Flask web server
model.py             - ML models and prediction logic
utils.py             - Helper functions for charts
requirements.txt     - Python dependencies
templates/
  - index.html      - Web interface
static/
  ─ script.js       - Frontend logic
  ─ styles.css      - Styling
```

## Quick Start

### Requirements

- Python 3.8+
- Internet connection

### Installation

```bash
pip install -r requirements.txt
```

### Run the App

```bash
python main.py
```

Then open your browser to: **http://127.0.0.1:5000**

## 📚 How to Use

1. Enter a crypto symbol (e.g., `BTC-USD`, `ETH-USD`)
2. Click **Predict**
3. Wait 15-30 seconds for analysis
4. View prediction, confidence, and 6 analysis charts

### Supported Cryptocurrencies

- `BTC-USD` - Bitcoin
- `ETH-USD` - Ethereum
- `SOL-USD` - Solana
- `ADA-USD` - Cardano
- `XRP-USD` - Ripple
- `DOGE-USD` - Dogecoin
- And many more from Yahoo Finance

## What You'll See

- **Prediction Result**: UP or DOWN
- **Confidence Score**: How sure the model is (0-100%)
- **Current Price**: Live market price
- **Model Accuracy**: How well it performs
- **6 Analysis Charts**:
  1. Price trends with moving averages
  2. Trading volume over time
  3. Return distribution
  4. Top 10 important features
  5. Confusion matrix
  6. ROC curve

## Technical Details

### Machine Learning Pipeline

1. **Data Collection** - 730 days of hourly data
2. **Feature Engineering** - 28 technical indicators
3. **Data Scaling** - Normalize features to 0-1
4. **Feature Stacking** - XGBoost predictions as extra feature
5. **Hyperparameter Tuning** - RandomizedSearchCV optimization
6. **Ensemble Model** - XGBoost + Random Forest voting
7. **Threshold Optimization** - Youden's J statistic
8. **Prediction** - Final result with confidence

### Technical Indicators

- Moving averages (7-day, 14-day, EMA)
- RSI (Relative Strength Index)
- ATR (Average True Range)
- Bollinger Bands
- OBV (On-Balance Volume)
- Momentum & Volatility
- Bitcoin correlation features

### Performance Metrics

- **Cross-validation ROC-AUC**: ~0.6
- **Test Accuracy**: 60-70% (varies by crypto)
- **Time-Series CV**: Prevents data leakage
- **Class Imbalance Handling**: Weighted training

## Troubleshooting

| Problem                        | Solution                                         |
| ------------------------------ | ------------------------------------------------ |
| "No data available for symbol" | Check format (needs `-USD`, like `ETH-USD`)      |
| "Module not found"             | Run `pip install -r requirements.txt`            |
| Import errors                  | Try `pip install --upgrade xgboost scikit-learn` |

## Important Disclaimers

This is a **school project for learning purposes only**:

- **NOT financial advice** - don't make real trades based on this
- Historical patterns may not repeat
- **No guarantees** - markets are unpredictable
- **Short-term only** - 3-hour predictions, not long-term
- **Varies by market** - accuracy changes with conditions

## What We Learned

This project demonstrates:

- Machine learning for time-series data
- Feature engineering for financial indicators
- Ensemble models and hyperparameter tuning
- Building web applications with Flask
- Data visualization and analysis

## Team

- **Arvin Das** - Machine Learning & Data Science, Web Development
- **Kanish Kannan Srinivasan** - Frontend Development
- **Muhammed Riswan Navas** - Financial Analysis & Features

---

## THANK YOU!!!
