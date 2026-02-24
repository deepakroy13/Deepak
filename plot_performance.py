import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Use a standard built-in style that is available in most environments
plt.style.use('ggplot')

# Read data
ec = pd.read_csv('equity_curve.csv', parse_dates=['Date'])
draw = pd.read_csv('drawdown_curve.csv', parse_dates=['Date'])
roll = pd.read_csv('rolling_outperformance.csv', parse_dates=['Date'])
metrics = pd.read_csv('performance_metrics.csv')
filled = pd.read_csv('filled_indices.csv', parse_dates=['tradedate'])

# Prepare benchmark series (NSE500)
bench = filled[filled['index_name']=='NSE500'][['tradedate','close']].rename(columns={'tradedate':'Date','close':'BenchClose'})
bench = bench.drop_duplicates(subset='Date').sort_values('Date')

# Merge equity with benchmark
df = pd.merge(ec, bench, on='Date', how='left')

# Forward fill benchmark if needed
df['BenchClose'] = df['BenchClose'].ffill()

# Normalize series for comparison
start_idx = 0
start_port = df['PortfolioValue'].iloc[start_idx]
start_bench = df['BenchClose'].iloc[start_idx] if not pd.isna(df['BenchClose'].iloc[start_idx]) else np.nan

df['PortNorm'] = df['PortfolioValue'] / start_port
if not np.isnan(start_bench):
    df['BenchNorm'] = df['BenchClose'] / start_bench
else:
    df['BenchNorm'] = np.nan

# Daily returns
df['Ret'] = df['PortfolioValue'].pct_change()

# Annual returns (calendar year)
df.set_index('Date', inplace=True)
annual = df['PortfolioValue'].resample('Y').last().pct_change().dropna()

# Plot 1: Equity curve vs Benchmark (normalized)
plt.figure(figsize=(12,6))
plt.plot(df.index, df['PortNorm'], label='Strategy (normalized)')
if df['BenchNorm'].notna().any():
    plt.plot(df.index, df['BenchNorm'], label='NSE500 (normalized)', alpha=0.9)
plt.title('Equity Curve — Strategy vs Benchmark')
plt.ylabel('Normalized Value')
plt.xlabel('Date')
plt.legend()
plt.tight_layout()
plt.savefig('equity_vs_benchmark.png', dpi=150)
plt.close()

# Plot 2: Drawdown
plt.figure(figsize=(12,4))
plt.plot(draw['Date'], draw['Drawdown'], color='tab:red')
plt.fill_between(draw['Date'], draw['Drawdown'], 0, color='tab:red', alpha=0.2)
plt.title('Portfolio Drawdown')
plt.ylabel('Drawdown')
plt.xlabel('Date')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
plt.tight_layout()
plt.savefig('drawdown.png', dpi=150)
plt.close()

# Plot 3: Rolling outperformance
plt.figure(figsize=(12,6))
plt.plot(roll['Date'], roll['Roll_1Y'], label='1Y')
plt.plot(roll['Date'], roll['Roll_3Y'], label='3Y')
plt.plot(roll['Date'], roll['Roll_5Y'], label='5Y')
plt.title('Rolling Outperformance (Strategy - Benchmark)')
plt.ylabel('Outperformance')
plt.xlabel('Date')
plt.legend()
plt.tight_layout()
plt.savefig('rolling_outperformance.png', dpi=150)
plt.close()

# Plot 4: Annual returns bar chart
plt.figure(figsize=(10,5))
annual.index = annual.index.year
plt.bar(annual.index.astype(str), annual.values, color='tab:blue')
plt.title('Annual Portfolio Returns')
plt.ylabel('Return')
plt.xlabel('Year')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('annual_returns.png', dpi=150)
plt.close()

# Plot 5: Daily returns distribution
plt.figure(figsize=(8,4))
plt.hist(df['Ret'].dropna(), bins=100, color='tab:gray', alpha=0.9)
plt.title('Distribution of Daily Returns')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.2%}'.format(x)))
plt.tight_layout()
plt.savefig('daily_returns_hist.png', dpi=150)
plt.close()

# Summary image (text) with key metrics
from PIL import Image, ImageDraw, ImageFont

metrics_dict = metrics.iloc[0].to_dict()
lines = [f"Final Value: ₹{metrics_dict['Final Value']:.2f}",
         f"CAGR: {metrics_dict['CAGR']*100:.2f}%",
         f"Volatility: {metrics_dict['Volatility']*100:.2f}%",
         f"Max Drawdown: {metrics_dict['MaxDD']*100:.2f}%",
         f"Information Ratio: {metrics_dict['Information Ratio']:.3f}",
         f"Turnover: see turnover.csv"]

img = Image.new('RGB', (800,200), color='white')
d = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype('DejaVuSans.ttf', 14)
except Exception:
    font = ImageFont.load_default()

y = 10
for line in lines:
    d.text((10,y), line, fill='black', font=font)
    y += 24

img.save('metrics_summary.png')

print('Saved charts: equity_vs_benchmark.png, drawdown.png, rolling_outperformance.png, annual_returns.png, daily_returns_hist.png, metrics_summary.png')
