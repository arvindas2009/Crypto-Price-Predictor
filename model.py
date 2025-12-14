# Team Zephyrus - Bright Riders School, Abu Dhabi
import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from scipy.stats import randint, uniform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from utils import create_plot_base64
import logging

logger = logging.getLogger(__name__)

# Feature list for model training
FEATURE_COLUMNS = [
    'Volume','ret','vol','ratio','mom','RSI','HL','HPC','LPC','TR','ATR',
    'ret_1','ret_2','ret_3','ret_4','mom_21','vol_21',
    'BB_width','BB_pos','OBV','OBV_mom','ema21','ret_5',
    'skew_21','kurt_21','mom_vol',
    'Year','Month','Day','BTC_ret_1','BTC_vol','vol_diff','corr_7d'
]

def engineer_features(df):
    # Datetime and returns
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df['ret'] = df['Close'].pct_change()
    df['vol'] = df['ret'].rolling(7).std()

    # df['avg_price'] = df[['Open', 'High', 'Low', 'Close']].mean(axis=1)
    
    
    # df['hl_range'] = df['High'] - df['Low']
    # df['oc_change'] = df['Close'] - df['Open']
    # df['range_pct'] = df['hl_range'] / df['Close']

    # Moving averages and ratio
    df['s7'] = df['Close'].rolling(7).mean()
    df['s14'] = df['Close'].rolling(14).mean()
    df['ratio'] = df['s7'] / df['s14']

    # Momentum indicators
    df['mom'] = df['Close'] - df['Close'].shift(7)
    df['mom_21'] = df['Close'] - df['Close'].shift(21)

    # True Range and ATR
    df['HL'] = df['High'] - df['Low']
    df['HPC'] = (df['High'] - df['Close'].shift(1)).abs()
    df['LPC'] = (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = df[['HL','HPC','LPC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()

    # Lagged returns
    df['ret_1'] = df['ret'].shift(1)
    df['ret_2'] = df['ret'].shift(2)
    df['ret_3'] = df['ret'].shift(3)
    df['ret_4'] = df['ret'].shift(4)
    df['vol_21'] = df['ret'].rolling(21).std()

    # Bollinger Bands
    w = 20
    df['BB_std'] = df['Close'].rolling(w).std()
    df['BB_mean'] = df['Close'].rolling(w).mean()
    df['BB_upper'] = df['BB_mean'] + 2*df['BB_std']
    df['BB_lower'] = df['BB_mean'] - 2*df['BB_std']
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_mean']
    df['BB_pos'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

    # On balance Volume
    df['volume_flow'] = df['Volume'] * df['Close'].diff().apply(
        lambda x: 1 if x>0 else (-1 if x<0 else 0)
    )
    df['OBV'] = df['volume_flow'].cumsum()
    df['OBV_mom'] = df['OBV'] - df['OBV'].shift(10)

    # additional features
    df['ema21'] = df['Close'].ewm(span=21).mean()
    df['ret_5'] = df['Close'].pct_change(5)
    df['skew_21'] = df['ret'].rolling(21).skew()
    df['kurt_21'] = df['ret'].rolling(21).kurt()
    df['mom_vol'] = df['mom'] * df['vol']

    # BTC correlation features
    df['vol_diff'] = df['vol'] - df['BTC_vol']
    df['corr_7d'] = df['ret'].rolling(7).corr(df['BTC_ret_1'])

    # Regression target (1h ahead)
    future_3h = df['Close'].shift(-3) - df['Close']
    future_1h = df['Close'].shift(-1) - df['Close']
    df['Target_Reg'] = future_1h / df['Close']

    # RSI calculation
    win = 14
    delta = df['Close'].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(win).mean()
    avg_loss = losses.rolling(win).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Date features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    # Binary target: 3h price movement vs ATR threshold
    thr = df['ATR'] * 0.5
    df['Target'] = (future_3h > thr).astype(float)

    # Drop intermediate columns
    df = df.drop(columns=['BB_std','BB_mean','BB_upper','BB_lower','volume_flow'], errors='ignore')
    return df


def scale_after_split(X_train, X_test, last_row):
    # scaling from 0 to 1
    scalers = {}
    Xtr = X_train.copy()
    Xte = X_test.copy()
    L = last_row.copy()
    for c in X_train.columns:
        s = MinMaxScaler()
        s.fit(X_train[[c]])
        scalers[c] = s
        Xtr[c] = s.transform(X_train[[c]]).ravel()
        Xte[c] = s.transform(X_test[[c]]).ravel()
        L[c] = s.transform(last_row[[c]]).ravel()
    return Xtr, Xte, L, scalers


def generate_plots(df, cm, fpr, tpr, roc_auc, importance, feature_names):
    # Graphs
    fig, ax = plt.subplots(3,2, figsize=(18,14))
    ax = ax.ravel()

    # Price with moving averages
    ax[0].plot(df.index, df['Close'], label='Close')
    ax[0].plot(df.index, df['s7'], label='7d MA')
    ax[0].plot(df.index, df['s14'], label='14d MA')
    ax[0].set_title('Price Trends')
    ax[0].legend()

    # Trading volume
    ax[1].bar(df.index, df['Volume'], alpha=0.6)
    ax[1].set_title('Volume')

    # Return distribution
    sns.histplot(df['ret'], bins=30, kde=True, ax=ax[2])
    ax[2].set_title('Return Distribution')

    # 10 important features
    idx = importance.argsort()[-10:]
    # print(idx)
    # print(feature_names[idx[1:]])
    top_feats = feature_names[idx]
    top_imps = importance[idx]
    mask = top_feats != 'reg_pred'
    top_feats = top_feats[mask]
    top_imps = top_imps[mask]
    ax[3].barh(top_feats, top_imps)
    ax[3].set_title('10 Imp Features')

    # Confusion matrix heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[4])
    ax[4].set_title('Confusion Matrix')

    # ROC curve
    ax[5].plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    ax[5].plot([0,1],[0,1],'k--', label='Random')
    ax[5].legend()
    ax[5].set_title('ROC Curve')

    plt.tight_layout()
    img = create_plot_base64(fig)
    plt.close(fig)
    return img


def analyze_crypto(symbol):
    # Main analysis
    try:
        # Download Bitcoin data as context for other cryptos, as it often influences the market
        print("Downloading BTC")
        btc = yf.download("BTC-USD", interval="1h", period="730d", progress=False)
        if btc.empty:
            return None
        if isinstance(btc.columns, pd.MultiIndex):
            btc.columns = [c[0] for c in btc.columns]
        btc = btc.reset_index()
        btc.rename(columns={btc.columns[0]:'Date'}, inplace=True)
        btc['Date'] = pd.to_datetime(btc['Date'], utc=True)
        btc['BTC_ret_1'] = btc['Close'].pct_change().shift(1)
        btc['BTC_vol'] = btc['BTC_ret_1'].rolling(7).std().shift(1)
        btc_ctx = btc[['Date','BTC_ret_1','BTC_vol']]

        # Download target data for the given symbol
        print(f"Downloading {symbol}")
        c = yf.download(symbol, interval="1h", period="730d", progress=False)
        if c.empty:
            return None
        if isinstance(c.columns, pd.MultiIndex):
            c.columns = [x[0] for x in c.columns]
        c = c.reset_index()
        c.rename(columns={c.columns[0]:'Date'}, inplace=True)
        c['Date'] = pd.to_datetime(c['Date'], utc=True)

        # Merge with BTC context
        c = pd.merge(c, btc_ctx, on='Date', how='left')

        # Engineer features
        print("Engineering features.....")
        feats = engineer_features(c.copy())
        clean = feats.dropna().reset_index(drop=True)

        # Training and testing split
        X = clean[FEATURE_COLUMNS].copy()
        y = clean['Target'].astype(int)
        cut = int(len(X)*0.7)
        X_train = X.iloc[:cut].copy()
        X_test = X.iloc[cut:].copy()
        y_train = y.iloc[:cut].copy()
        y_test = y.iloc[cut:].copy()

        # Get last row for live prediction
        raw_last = feats[FEATURE_COLUMNS].iloc[-1:].copy()
        if raw_last.isna().any().any():
            logger.warning("Missing features in last row.")
            return None

        # Train regression model for stacking feature
        print("Target Reg model train")
        y_reg = clean['Target_Reg']
        y_tr_reg = y_reg.iloc[:cut]
        reg = XGBRegressor(tree_method='hist', n_estimators=100, max_depth=4, random_state=42)
        reg.fit(X_train, y_tr_reg)
        X_train['reg_pred'] = reg.predict(X_train)
        X_test['reg_pred'] = reg.predict(X_test)
        raw_last['reg_pred'] = reg.predict(raw_last)

        # Scale features without leakage
        print("Scaling features...")
        X_train, X_test, raw_last, _ = scale_after_split(X_train, X_test, raw_last)

        # Account for class imbalance
        pos = (y_train==1).sum()
        neg = (y_train==0).sum()
        s_w = max(1.0, neg/(pos+1e-6)) if pos>0 else 1

        # Hyperparameter optimization via random search
        print("Optimizing hyperparameters")
        tscv = TimeSeriesSplit(n_splits=3)
        xgb = XGBClassifier(eval_metric='logloss', tree_method='hist', random_state=42, scale_pos_weight=s_w)
        params = {
            'n_estimators': randint(100,400),
            'learning_rate': uniform(0.01,0.15),
            'max_depth': randint(3,7),
            'subsample': uniform(0.7,0.3),
            'colsample_bytree': uniform(0.7,0.3)
        }
        search = RandomizedSearchCV(xgb, params, cv=tscv, scoring='roc_auc', n_iter=20, random_state=42, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_params_
        best.update({'eval_metric':'logloss','random_state':42,'tree_method':'hist'})

        # Train final ensemble
        print("Training XGBoost RF")
        xgb_final = XGBClassifier(**best)
        rf = RandomForestClassifier(n_estimators=350, max_depth=8, random_state=42)
        ens = VotingClassifier([('xgb', xgb_final), ('rf', rf)], voting='soft')
        ens.fit(X_train, y_train)

        # Evaluate on test set
        print("Evaluating model")
        preds = ens.predict(X_test)
        probs = ens.predict_proba(X_test)[:,1]
        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        fpr, tpr, th = roc_curve(y_test, probs)
        test_auc = auc(fpr, tpr)
        best_cv = search.best_score_

        print(f"Actual ROC Score is {test_auc}")
        print(f"Train ROC Score is {best_cv}")

        # Find optimal threshold using Youden's J statistic
        j = tpr - fpr
        optimal_idx = j.argmax()
        threshold = th[optimal_idx]
        if threshold > 0.95:
            threshold = 0.5

        # Make next prediction
        print("Predicting next movement....")
        pred_probs = ens.predict_proba(raw_last)[0]
        prob_up = pred_probs[1]
        final_pred = 1 if prob_up >= threshold else 0
        confidence = pred_probs[final_pred]

        # Get current price
        latest = yf.download(symbol, period='1d', progress=False)
        price = float(latest['Close'].iloc[-1]) if not latest.empty else None

        # Determine currency code and symbol from ticker (e.g., BTC-USD -> USD -> $)
        try:
            currency_code = symbol.split('-')[-1].upper()
        except Exception:
            currency_code = ''
 
        currency_map = {
            'USD': '$', 'INR': '₹', 'EUR': '€', 'GBP': '£', 'JPY': '¥',
            'CNY': '¥', 'KRW': '₩', 'AUD': 'A$', 'CAD': 'C$', 'SGD': 'S$',
            'CHF': 'CHF', 'NZD': 'NZ$'
        }
        currency_sym = currency_map.get(currency_code, '$') if currency_code else '$'
 
        # Average feature importances from both models
        fi_xgb = ens.named_estimators_['xgb'].feature_importances_
        fi_rf = ens.named_estimators_['rf'].feature_importances_
        fi = (fi_xgb + fi_rf) / 2
        fn = np.array(X_train.columns)

        # Generate diagnostic plots
        print("Generating plots...")
        plot_img = generate_plots(clean, cm, fpr, tpr, test_auc, fi, fn)

        # Return results
        return {
            'symbol': symbol,
            'prediction': 'UP' if final_pred==1 else 'DOWN',
            'confidence': round(confidence*100,1),
            'current_price': round(price,2) if price else None,
            'currency': currency_code,
            'currency_symbol': currency_sym,
            'accuracy': round(acc*100,1),
            'roc_auc': round(test_auc,2),
            'best_cv': round(best_cv, 2),
            'plot': plot_img
        }

    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return None