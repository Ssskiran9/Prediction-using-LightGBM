import os
import numpy as np
import pandas as pd
#from alpha_vantage.timeseries import TimeSeries
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
#from backtesting import Backtest, Strategy
import pandas_ta as ta
import joblib
import vectorbt as vbt
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio


def readstockdata():
    df = pd.read_csv(
        "C:\\Users\\saiki\\Desktop\\projects\\lightgbm_adani_day\\sorted_data.csv",
        parse_dates=True,
        index_col=0
    )
    df.dropna(inplace=True)
    for col in ["open", 'high', "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.reset_index()
    return df

def add_indicators(df):
    df['return'] = df['close'].pct_change()
    df.ta.rsi(length=14, append=True)
    df.ta.bbands(length=20, append=True)
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=200, append=True)  # ✅ EMA 200 for trend filter
    df.ta.ema(length=100, append=True) 
    df.ta.roc(length=10, append=True)

    macd_result = ta.macd(df['close'])
    df['MACD'] = macd_result['MACD_12_26_9']
    df['MACD_signal'] = macd_result['MACDs_12_26_9']

    df['ATR_4'] = ta.atr(df['high'], df['low'], df['close'], length=4)

    for lag in range(1, 6):
        df[f'return_lag_{lag}'] = df['return'].shift(lag)

    df['volatility_5'] = df['return'].rolling(window=5).std()
    df['volatility_10'] = df['return'].rolling(window=10).std()
    df['volatility_20'] = df['return'].rolling(window=20).std()
    

    # Assuming df has 'close' and is sorted by date
    df['std_5'] = df['close'].rolling(window=5).std()
    df['std_20'] = df['close'].rolling(window=20).std()

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])  # or whatever your time column is
    df.set_index('date', inplace=True)

    df.ta.obv(close='close', volume='volume', append=True)


    df.ta.vwap(                        # adds a column named 'VWAP'
        high='high',
        low='low',
        close='close',
        volume='volume',
        append=True
    )

    # Assuming 'df' has a 'close' column
    df['sma_42'] = df['close'].rolling(window=42).mean()
    df['sma_242'] = df['close'].rolling(window=242).mean()

    df['sma_signal'] = np.where(df['sma_42'] > df['sma_242'], 1, 0)

    print(f"all columns: {df.columns}")

    return df.dropna()

# Load and process data
df = readstockdata()
df = add_indicators(df)

# Add direction
df["returns"] = np.log(df["close"] / df["close"].shift(1))
# df["direction"] = np.where(df["returns"] > 0, 1, 0)
df['direction'] = (df['close'].shift(-1) > df['close']).astype(int)
df.dropna(inplace=True)
df['price_fluctuation'] = np.where(df['direction'] == 0, df['low'] - df['open'], df['high'] - df['open'])
df['fluc_percentage'] = df['price_fluctuation'] / df['open']

# Features for ML
features = [
     'open', 'high', 'low', 'close','volume',
    'RSI_14', 'BBL_20_2.0_2.0', 'BBM_20_2.0_2.0', 'BBU_20_2.0_2.0',
    'EMA_20', 'EMA_200',
    'ROC_10', 'return_lag_1', 'return_lag_2', 'return_lag_3',
    'return_lag_4', 'return_lag_5',
    'volatility_5', 'volatility_10', 'volatility_20',
    'MACD', 'MACD_signal', 'ATR_4','EMA_100', 'OBV', 'VWAP_D',
    'std_5', 'std_20'
]

# Save for review
df.to_csv("C:\\Users\\saiki\\Desktop\\projects\\lightgbm_adani_day\\adani_1day_features.csv", index=False)

# top_features = importance.head(20)['feature'].tolist()

# Split dataset
split = int(0.8 * len(df))



df_train = df.iloc[:split]
df_test = df.iloc[split:]

# #Convert index to datetime (if not already)
# df_test.index = pd.to_datetime(df_test.index)

# # Filter rows after June 2025
# df_test = df_test[df_test.index > '2025-06-01']

X_train, y_train = df_train[features], df_train['direction']
X_test, y_test = df_test[features], df_test['direction']

# Train model
model = LGBMClassifier(
    objective='binary',
    boosting_type='goss',
    metric='binary_logloss',
    random_state=42,
    num_leaves=30,
    min_data_in_leaf=50,
    max_depth=-1,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    n_estimators=2000,
    learning_rate=0.01
)



# Train model
model.fit(X_train, y_train)

# # Get feature importance (gain-based)
# importance = pd.DataFrame({
#     'feature': X_train.columns,
#     'importance': model.feature_importances_  # gain-based by default
# }).sort_values('importance', ascending=False)

# # Plot top N features
# plt.figure(figsize=(10, 6))
# sns.barplot(x='importance', y='feature', data=importance.head(20))
# plt.title('Top 20 Most Important Features (Gain-based)')
# plt.show()






train_preds = model.predict(X_train)

train_acc = accuracy_score(y_train, train_preds)
print(f"Training accuracy: {train_acc:.4f}")

print(f"all columns: {df.columns}")

# Predict
y_pred = model.predict(X_test)
probs = model.predict_proba(X_test)
buy_prob = probs[:, 1]
sell_prob = probs[:, 0]

print(classification_report(y_test, y_pred, target_names=['sell', 'buy']))

# Save model
model.booster_.save_model('C:\\Users\\saiki\\Desktop\\projects\\lightgbm_adani_day\\newadani_model.txt')
joblib.dump(model, "C:\\Users\\saiki\\Desktop\\projects\\lightgbm_adani_day\\newadani_model.pkl")

# ==========================
# ✅ VECTORBT BACKTEST
# ==========================
df_test = df_test.copy()
df_test['pred'] = y_pred

# 2) Define your long/short entry & exit logic
long_entries  = df_test['pred'] == 1
long_exits    = df_test['pred'] ==0


short_entries = df_test['pred'] == 0
short_exits   = df_test['pred'] == 1


# 3) Stick those into your df and backtest
df_test = df_test.copy()
df_test['long_entries']  = long_entries
df_test['long_exits']    =  long_exits
df_test['short_entries'] = short_entries
df_test['short_exits']   = short_exits





#ATR based stop loss and take profit
tp_array = df_test['ATR_4'] * 3
sl_array = df_test['ATR_4'] * 1

size = 50000/ df_test['close']  # ₹30000 per trade

# tps = np.linspace(0.005, 0.05, 46)  # from 0.5% to 5% in 0.1% steps

# ✅ VectorBT Portfolio
pf = vbt.Portfolio.from_signals(
    close=df_test['close'],
    open =df_test['open'],
    high=df_test['high'],
    low=df_test['low'],
    entries=df_test['long_entries'],
    #exits=long_exits, 
    short_entries=df_test['short_entries'],
    #short_exits=short_exits,
    freq='1day',
    size=size,

      # ₹50000 per trade
    # fixed_fees=20,

    init_cash=50_000,
    tp_stop = tp_array,
    sl_stop= sl_array,
    # sl_trail=True,
    
    
    
)

print(pf.stats())
# make sure your vectorbt call returns a figure
fig = pf.plot()    # or whatever returns a plotly Figure
fig.show(renderer = "browser")
pf.positions.records_readable.to_csv("C:\\Users\\saiki\\Desktop\\projects\\lightgbm_adani_day\\positions.csv", index=False)

# # How many entry signals you generated
# total_entries = int(entries.sum())
# # How many exit signals you generated
# total_exits = int(exits.sum())

# # How many trades actually ran
# executed_trades = len(pf.trades.records_readable)

# # How many signals got skipped (cash/position constraints)
# skipped_entries = total_entries - executed_trades

# print(f"Total entry signals:      {total_entries}")
# print(f"Total exit signals:       {total_exits}")
# print(f"Total executed trades:    {executed_trades}")
