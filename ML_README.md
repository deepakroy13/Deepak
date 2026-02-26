# Quant Equity Ranker: LightGBM Ensemble Scorer

A robust, quantitative machine learning pipeline designed to evaluate Indian equities, rank them based on predictive features, and output continuous 0-to-1 conviction scores for use in a larger multi-model ensemble strategy.

Unlike traditional scripts that output rigid True/False or Buy/Sell signals, this model generates normalized Daily Z-Scored Sigmoid probabilities. This prevents any single model from dominating the ensemble due to fluctuating market volatility.

## Key Features

* **Avoids Look-Ahead Bias:** Targets are deliberately shifted to T+2 to simulate a realistic T+1 execution delay, complying with strict institutional backtesting standards.
* **LambdaRank Architecture:** Uses LightGBM Ranker with an NDCG metric to perform pairwise ranking, learning which stock is better than another rather than trying to predict noisy absolute returns.
* **Daily Cross-Sectional Z-Scoring:** Raw log-odds outputs from LightGBM are standardized on a daily basis (Mean=0, Std=1) before being passed through a stretched sigmoid function. This guarantees scores perfectly utilize the [0, 1] bounds regardless of whether the market is crashing or rallying.
* **Edge-Case Handling:** Bulletproof math handles data gaps, single-stock trading days, and zero-variance scenarios gracefully without producing NaNs.

## Feature Engineering

The pipeline calculates classic quantitative factors before feeding them into the ML model:
1. **Sharpe Momentum (21-Day):** Volatility-adjusted return to identify smooth, high-conviction momentum.
2. **Trend Regime (SMA 50):** Distance from the 50-day Simple Moving Average to filter for structural uptrends/downtrends.
3. **Mean Reversion (RSI 14):** Standard Relative Strength Index to identify overbought/oversold patterns within the trend.
4. **Lagged Returns:** T-1, T-3, T-5, and T-10 returns to allow the tree-based model to recognize micro-patterns.

## Installation & Requirements

Ensure you have Python 3.8+ installed along with the following libraries:

    pip install pandas numpy lightgbm

## Usage

1. **Prepare your data:** The script expects a CSV file named `filled_indices.csv` in the root directory.
   * Required columns: `tradedate` (YYYY-MM-DD), `index_name` (Stock/Ticker), and `close` (Closing Price).
2. **Run the pipeline:**
    python main.py
3. **Integration:** The script will output a file named `model_1_sigmoid_scores.csv`. Pass this file to your master ensemble script to combine with other models and generate final portfolio weights.

## Output Format

The output file (`ml_approach.csv`) is intentionally kept lightweight for fast ingestion by your ensemble/backtesting engine. It includes:

| Column | Description |
| :--- | :--- |
| `tradedate` | The date the signal was generated. |
| `index_name` | The stock ticker/identifier. |
| `close` | Closing price for the day. |
| `predicted_score` | Raw unbounded log-odds score from the LightGBM Ranker. |
| `sigmoid_score` | The final ensemble weight component (0.0 to 1.0). |
| `dist_sma_50` | Included as an optional strict filter for the ensemble (e.g., set weight to 0 if < 0). |

## Important Note on Backtesting

This script is a score generator, not a standalone trading bot. The discrete Buy/Sell portfolio construction logic and direct backtester integration have been deliberately removed. To evaluate the strategy's CAGR, Max Drawdown, and Turnover, you must ingest the `sigmoid_score` output into your master backtesting engine.
