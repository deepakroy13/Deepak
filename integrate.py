import pandas as pd
import numpy as np
import lightgbm as lgb


# ==========================================================
# RETRIEVE AND CLEAN DATA
# ==========================================================

def fill_missing_prices(df):
    # 1. Ensure data is sorted by index_name and date to maintain chronological order
    df = df.sort_values(by=['index_name', 'tradedate']).reset_index(drop=False)

    # 2. We use 'close' from the previous row to fill 'open' in the current row
    # Shift(1) moves the closing prices down by one day
    df['prev_close'] = df.groupby('index_name')['close'].shift(1)

    # 3. Fill the 'open' column where it is null using the 'prev_close'
    df['open'] = df['open'].fillna(df['prev_close'])

    # 4. For the very first day of a stock's history (where no prev_close exists), filling 'open' with the current day's 'close' as a final fallback
    df['open'] = df['open'].fillna(df['close'])

    # Cleanup auxiliary column
    df = df.drop(columns=['prev_close'])

    df = df.sort_values(by='index')
    df = df.set_index('index')
    df.index.name = None

    df['high'] = df['high'].fillna(np.maximum(df['open'], df['close']))
    df['low'] = df['low'].fillna(np.minimum(df['open'], df['close']))
    df['tradedate'] = pd.to_datetime(df['tradedate'])

    # df.to_csv('filled_indices.csv', index=False)

    return df

# ==========================================================
# ML FEATURE ENGINEERING
# ==========================================================

def create_features(df):
    df_list = []
    for name, group in df.groupby('index_name'):
        group = group.copy()

        # Current Day Returns
        group['returns'] = group['close'].pct_change().dropna()

        # Volatility Adjusted Momentum
        vol_21 = group['returns'].rolling(21).std()
        group['sharpe_mom'] = group['returns'].rolling(21).mean() / (vol_21 + 1e-9)

        # Trend / Regime Filters
        group['sma_50'] = group['close'].rolling(window=50).mean()
        group['dist_sma_50'] = (group['close'] / group['sma_50']) - 1

        # RSI
        delta = group['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        group['rsi_14'] = 100 - (100 / rs)

        # Lagged Returns (Pattern Recognition)
        for lag in [1, 3, 5, 10]:
            group[f'ret_lag_{lag}'] = group['returns'].shift(lag)

        # Target shifted by -2 to simulate real-life 1-day execution delay
        group['target_return'] = group['returns'].shift(-2)

        df_list.append(group)

    full_df = pd.concat(df_list)

    # Critical Fix: Only dropna on features. DO NOT dropna on target_return
    feature_cols = ['sharpe_mom', 'dist_sma_50', 'rsi_14', 'ret_lag_1', 'ret_lag_3', 'ret_lag_5', 'ret_lag_10']
    full_df = full_df.dropna(subset=feature_cols)
    return full_df


def prepare_ranking_labels(df, target_col='target_return', n_bins=5):
    df = df.copy()

    def qcut_safe(x):
        valid = x.dropna()
        if len(valid) < n_bins:
            return pd.Series(np.nan, index=x.index)
        return pd.qcut(x, n_bins, labels=False, duplicates='drop')

    # Rank based on the REALISTIC target (T+2)
    df['rank_label'] = df.groupby('tradedate')[target_col].transform(qcut_safe)
    df['rank_label'] = df['rank_label'].fillna(n_bins // 2).astype(int)
    return df


def train_and_predict(df):
    features = [
        'sharpe_mom', 'dist_sma_50', 'rsi_14',
        'ret_lag_1', 'ret_lag_3', 'ret_lag_5', 'ret_lag_10'
    ]

    df = df.sort_values(['tradedate', 'index_name'])
    df = prepare_ranking_labels(df)

    # Dynamic Validation Split: Automatically uses first 70% of dates for training
    unique_dates = df['tradedate'].sort_values().unique()
    split_idx = int(len(unique_dates) * 0.7)
    split_date = unique_dates[split_idx]

    # Use .copy() to ensure we safely manipulate these slices later
    train = df[df['tradedate'] < split_date].copy()
    test = df[df['tradedate'] >= split_date].copy()

    # Fallback if the dataset is too small
    if train.empty:
        train = df.copy()
        test = df.copy()

    qids_train = train.groupby("tradedate")["tradedate"].count().to_numpy()
    qids_test = test.groupby("tradedate")["tradedate"].count().to_numpy()

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=100,
        learning_rate=0.04,
        max_depth=3,
        random_state=42,
        n_jobs=-1,
        label_gain=[0, 0, 1, 3, 8]
    )

    print("Training LightGBM Ranker Model...")
    model.fit(
        train[features],
        train['rank_label'],
        group=qids_train,
        eval_set=[(test[features], test['rank_label'])],
        eval_group=[qids_test],
        eval_at=[1]
    )

    # Predict on the full dataframe so we have scores for all data
    df['predicted_score'] = model.predict(df[features])

    return df

# ============= ML ended =============

# ==========================================================
# CONFIG
# ==========================================================

INITIAL_CAPITAL = 5_000_000
TRANSACTION_COST = 0.00268
MIN_HISTORY_DAYS = 100
BENCHMARK = "NSE500"

# ==========================================================
# INDICATORS
# ==========================================================

def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_bollinger(close, window=10, n_std=2):
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    return ma + n_std*std, ma - n_std*std

def compute_rsi(close, period=5):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean().replace(0, 1e-9)
    rs = gain / loss
    return 100 - (100/(rs))

def compute_macd(close):
    fast = compute_ema(close, 15)
    slow = compute_ema(close, 35)
    macd = fast - slow
    signal = macd.ewm(span=5, adjust=False).mean()
    return macd, signal

def compute_stochastic(high, low, close):
    lmin = low.rolling(14).min()
    hmax = high.rolling(14).max()
    k = 100*(close - lmin)/(hmax - lmin).replace(0,1e-9)
    d = k.rolling(3).mean()
    return k, d

#============================
#PERFORMANCE METRICS
#============================
def compute_drawdown(series):
    running_max = series.cummax()
    return series/running_max - 1

def compute_cagr(df):
    years = (df["Date"].iloc[-1]-df["Date"].iloc[0]).days/365.25
    return (df["PortfolioValue"].iloc[-1]/INITIAL_CAPITAL)**(1/years)-1

def compute_vol(df):
    return df["Ret"].std()*np.sqrt(252)

def compute_ir(df):
    excess = df["Ret"]-df["BRet"]
    return (excess.mean()*252)/(excess.std()*np.sqrt(252))

def compute_up_down_capture(df):
    up = df[df["BRet"]>0]
    down = df[df["BRet"]<0]
    up_cap = ((1+up["Ret"]).prod()-1)/((1+up["BRet"]).prod()-1)
    down_cap = ((1+down["Ret"]).prod()-1)/((1+down["BRet"]).prod()-1)
    return up_cap, down_cap

def compute_rolling_outperformance(df, window):
    strat = (1+df["Ret"]).rolling(window).apply(np.prod, raw=True)-1
    bench = (1+df["BRet"]).rolling(window).apply(np.prod, raw=True)-1
    outperf = strat-bench
    return outperf.mean(), outperf.min()

def compute_rolling_series(df, window):
    strat = (1 + df["Ret"]).rolling(window).apply(np.prod, raw=True) - 1
    bench = (1 + df["BRet"]).rolling(window).apply(np.prod, raw=True) - 1
    return strat - bench

# ==========================================================
# ANALYSIS PIPELINE
# ==========================================================

def analyze_performance(equity_curve, raw_data):

    bench = raw_data[raw_data["index_name"]==BENCHMARK][
        ["tradedate","close"]
    ].rename(columns={"tradedate":"Date","close":"BenchClose"})

    df = pd.merge(equity_curve,bench,on="Date").dropna()

    df["Ret"] = df["PortfolioValue"].pct_change()
    df["BRet"] = df["BenchClose"].pct_change()
    df = df.dropna()

    df["Drawdown"] = compute_drawdown(df["PortfolioValue"])

    up_cap, down_cap = compute_up_down_capture(df)

    metrics = {
        "Final Value":df["PortfolioValue"].iloc[-1],
        "CAGR":compute_cagr(df),
        "Volatility":compute_vol(df),
        "MaxDD":df["Drawdown"].min(),
        "Information Ratio":compute_ir(df),
        "UpCapture":up_cap,
        "DownCapture":down_cap,
        "1Y Avg Outperformance":compute_rolling_outperformance(df,252)[0],
        "1Y Worst Underperformance":compute_rolling_outperformance(df,252)[1],
        "3Y Avg Outperformance":compute_rolling_outperformance(df,756)[0],
        "3Y Worst Underperformance":compute_rolling_outperformance(df,756)[1],
        "5Y Avg Outperformance":compute_rolling_outperformance(df,1260)[0],
        "5Y Worst Underperformance":compute_rolling_outperformance(df,1260)[1]
    }

    return metrics, df
#========= performance and indiacators done==========

# ==========================================================
# SIGNALS
# ==========================================================

def generate_sigmoid_score(df):
    """
    Transforms the raw LightGBM ranking scores into bounded sigmoid probabilities.
    Returns the dataframe with the new 'sigmoid_score' column.
    """
    df = df.copy()

    # Apply Sigmoid function: 1 / (1 + e^-x)
    df['sigmoid_score'] = 1 / (1 + np.exp(-df['predicted_score']))

    # Ensure data is sorted for easy merging later
    df = df.sort_values(['tradedate', 'index_name']).reset_index(drop=True)

    return df


def generate_signals(df):

    df = df.copy()

    # ==========================================================
    # 1️⃣ FIRST: Generate Sigmoid Score from ML Predictions
    # ==========================================================
    df = generate_sigmoid_score(df)
    # Now df contains: 'sigmoid_score'

    # ==========================================================
    # 2️⃣ TECHNICAL INDICATORS
    # ==========================================================
    df["EMA20"] = compute_ema(df["close"], 20)
    df["EMA100"] = compute_ema(df["close"], 100)
    df["BB_up"], df["BB_low"] = compute_bollinger(df["close"])
    df["RSI"] = compute_rsi(df["close"])
    df["MACD"], df["MACD_sig"] = compute_macd(df["close"])
    df["%K"], df["%D"] = compute_stochastic(df["high"], df["low"], df["close"])

    # ==========================================================
    # 3️⃣ TECHNICAL BUY SCORE
    # ==========================================================
    bcond1 = (df["close"] >= df["EMA100"])
    bcond2 = (df["EMA20"] >= df["EMA100"])
    bcond3 = (df["RSI"] <= 25)
    bcond4 = ((df["%K"] <= 20) | (df["%D"] <= 20))
    bcond5 = (df["close"] <= df["BB_low"])

    tech_buy_score = (
        bcond1.astype(int)*(16/8) +
        bcond2.astype(int)*(8/8) +
        bcond3.astype(int)*1 +
        bcond4.astype(int)*1 +
        bcond5.astype(int)*(6/8)
    )

    # ==========================================================
    # 4️⃣ TECHNICAL SELL SCORE
    # ==========================================================
    scond1 = (df["EMA20"] < df["EMA100"])
    scond2 = (df["close"] >= df["BB_up"])
    scond3 = (df["RSI"] >= 80)
    scond4 = ((df["MACD"].shift(1) > df["MACD_sig"].shift(1)) |
              (df["MACD"] < df["MACD_sig"]))
    scond5 = (df["MACD"] < df["MACD_sig"])

    tech_sell_score = (
        scond1.astype(int)*(18/9) +
        scond2.astype(int)*1 +
        scond3.astype(int)*1 +
        scond4.astype(int)*(6/9) +
        scond5.astype(int)*(5/10)
    )

    # ==========================================================
    # 5️⃣ COMBINE WITH SIGMOID SCORE
    # ==========================================================
    df["total_buy_score"] = 1.5*tech_buy_score + 0.5*df["sigmoid_score"]
    df["total_sell_score"] = 1.5*tech_sell_score + 0.5*(1 - df["sigmoid_score"])

    # ==========================================================
    # 6️⃣ FINAL SIGNALS
    # ==========================================================
    df["Buy"] = df["total_buy_score"] >= 6
    df["Sell"] = df["total_sell_score"] >= 7.5

    return df


def run_backtest(data):

    cash = INITIAL_CAPITAL
    positions = {}
    equity_curve = []
    trade_log = []
    total_traded_value = 0
    MAX_POSITION_SIZE = .4
    
   
    dates = sorted(data["tradedate"].unique())

    for i in range(MIN_HISTORY_DAYS, len(dates)-1):

        today = dates[i]
        tomorrow = dates[i+1]

        df_today = data[data["tradedate"]==today]
        df_next = data[data["tradedate"]==tomorrow]

        # --- EXIT ---
        for sym in list(positions.keys()):
            row = df_today[df_today["index_name"]==sym]
            if not row.empty and row["Sell"].iloc[0]:

                row_next = df_next[df_next["index_name"]==sym]
                if not row_next.empty:

                    exec_price = row_next[["open","high","low","close"]].mean(axis=1).iloc[0]
                    sell_price = exec_price*(1-TRANSACTION_COST)

                    qty = positions.pop(sym)
                    traded_value = qty*exec_price
                    total_traded_value += traded_value

                    cash += qty*sell_price

                    trade_log.append([tomorrow,sym,"SELL",qty,exec_price])

        # --- ENTRY ---
        buys = df_today[
            df_today["Buy"] &
            ~df_today["index_name"].isin(positions)
        ]

        if not buys.empty:
            
            current_portfolio_value = cash
            for sym, qty in positions.items():
                sym_row = df_today[df_today["index_name"]==sym]
                if not sym_row.empty:
                    current_portfolio_value += qty * sym_row["close"].iloc[0]

            max_allocation = current_portfolio_value * MAX_POSITION_SIZE

            for sym in buys["index_name"]:
                if cash <= 0:
                    break 
                
                allocation = min(max_allocation, cash)

                row_next = df_next[df_next["index_name"]==sym]
                if not row_next.empty:

                    exec_price = row_next[["open","high","low","close"]].mean(axis=1).iloc[0]
                    buy_price = exec_price*(1+TRANSACTION_COST)

                    qty = allocation//buy_price
                    cost = qty*buy_price

                    if qty>0 and cost<=cash:

                        traded_value = qty*exec_price
                        total_traded_value += traded_value

                        positions[sym]=qty
                        cash-=cost

                        trade_log.append([tomorrow,sym,"BUY",qty,exec_price])

        # --- DAILY VALUE ---
        portfolio_value = cash
        for sym,qty in positions.items():
            close_price = df_today[df_today["index_name"]==sym]["close"].iloc[0]
            portfolio_value += qty*close_price

        equity_curve.append({
            "Date":today,
            "PortfolioValue":portfolio_value,
            "Positions":len(positions)
        })

    equity_curve = pd.DataFrame(equity_curve)
    trade_log = pd.DataFrame(trade_log,columns=["Date","Symbol","Side","Qty","ExecPrice"])

    turnover = total_traded_value / INITIAL_CAPITAL

    return equity_curve, trade_log, turnover


#=================
#MAIN
#================
def main():
# 1️⃣ Load data
    csv_name = input("Enter the name of the CSV file containing the index data (e.g., indexes.csv): ")
    data = pd.read_csv(csv_name)
    data = fill_missing_prices(data)

    data.sort_values(['index_name', 'tradedate'], inplace=True)
    data.reset_index(drop=True, inplace=True)

# 2️⃣ Feature engineering
    df_features = create_features(data)

# 3️⃣ Train & Predict
    df_with_scores = train_and_predict(df_features)

# 4️⃣ Filter after prediction
    df_with_scores = df_with_scores[
    df_with_scores['tradedate'] >= '2010-01-01'
]

# 5️⃣ Pass to generate_signals()
    processed = []

    for _, df_stock in df_with_scores.groupby("index_name"):
        if len(df_stock) >= MIN_HISTORY_DAYS:
            processed.append(generate_signals(df_stock))

    data_signals = pd.concat(processed).reset_index(drop=True)

    equity_curve, trade_log, turnover = run_backtest(data_signals)

    metrics, full_df = analyze_performance(equity_curve,data)

    # OUTPUTS REQUIRED
    equity_curve.to_csv("equity_curve.csv",index=False)
    trade_log.to_csv("trade_log.csv",index=False)
    full_df[["Date","Drawdown"]].to_csv("drawdown_curve.csv",index=False)

    # Export metrics
    pd.DataFrame(metrics, index=[0]).to_csv("performance_metrics.csv", index=False)

    # Export rolling table
    rolling_table = pd.DataFrame({
    "Date": full_df["Date"],
    "Roll_1Y": compute_rolling_series(full_df,252),
    "Roll_3Y": compute_rolling_series(full_df,756),
    "Roll_5Y": compute_rolling_series(full_df,1260)
})
    rolling_table.to_csv("rolling_outperformance.csv", index=False)

    # Export turnover
    pd.DataFrame({"Turnover":[turnover]}).to_csv("turnover.csv", index=False)

    print("\n===== PERFORMANCE METRICS =====")
    for k,v in metrics.items():
        print(f"{k}: {v:.6f}")

    print(f"\nPortfolio Turnover: {turnover:.4f}")

if __name__=="__main__":
    main()