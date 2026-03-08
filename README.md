# Quant Market Movement Prediction & Signal Generation (Long‑Term Strategy)

This repository contains a Python-based quantitative research project built as part of an **inter-hostel competition**. The **problem statement was provided by Beyond IRR** (see `Kriti2026_quant.pdf`). The project focuses on **predicting market movement** and **generating long-term Buy/Sell signals**.

The work explores three approaches:

1. **Pure Quantitative (rule/indicator-based) approach**
2. **Machine Learning (ML) approach**
3. **Integrated approach** combining Quant + ML

The repository is organized so that **file names clearly indicate the approach used**, and **all evaluation outputs/metrics are stored in the `performance_stats/` folder** with descriptive names.

---

## Project Objective

- Analyze index/market data
- Build long-term trading signals (Buy/Sell)
- Evaluate strategies using relevant performance metrics (returns, drawdowns, risk-adjusted measures, etc.)
- Compare outcomes across:
  - Quant-only logic
  - ML-only model
  - Integrated Quant+ML pipeline

---

## Repository Structure

### Core approaches

- **`quantitative_approach.py`**
  - Implements a **traditional quantitative strategy**, typically based on engineered indicators, rules, trend logic, or statistical signals.
  - Produces Buy/Sell signals and/or positions for long-term holding periods.

- **`ML.py`**
  - Implements the **machine learning approach**.
  - Includes feature engineering and model training/inference logic to predict market direction/movement and translate predictions into signals.

- **`integrated_approach.py`**
  - Implements the **hybrid approach**, integrating quantitative features/signals with ML predictions (or using quant logic as filters/confirmations).
  - Designed to test whether combining both improves robustness and performance.

---

## Data / Inputs

- **`indexes.csv`**
  - Primary dataset used for the analysis (index/market values and related fields).
  - Used across quant, ML, and integrated workflows.

- **`Kriti2026_quant.pdf`**
  - Official problem statement shared by **Beyond IRR** for the competition.

---

## Evaluation & Results

- **`performance_stats/`**
  - Contains **all performance evaluation metrics and outputs**, stored with meaningful filenames.
  - This folder is the primary place to look for backtest summaries, metric tables, and evaluation artifacts.

- **`PERFORMANCE_AND_ML_REPORT.md`**
  - Consolidated write-up/report describing performance results and ML observations.

- **`plot_performance.py`**
  - Utility for generating plots/visualizations of performance and/or equity curves from computed results.

---

## Additional Notes

- **`ML_README.md`**
  - ML-specific notes and documentation (features, models, pipeline details, etc.)

---

## How to Run (Typical Workflow)

> Exact commands may vary depending on how each script is implemented, but a common workflow is:

1. **Run Quant strategy**
   - `python quantitative_approach.py`

2. **Run ML strategy**
   - `python ML.py`

3. **Run Integrated strategy**
   - `python integrated_approach.py`

4. **Generate plots**
   - `python plot_performance.py`

After running, check:
- `performance_stats/` for saved metrics/results
- `PERFORMANCE_AND_ML_REPORT.md` for summarized findings

---

## Expected Outputs

Depending on the approach, the scripts typically generate:
- Predicted market movement labels/scores (ML / integrated)
- Buy/Sell signals or position series
- Backtest performance metrics (saved under `performance_stats/`)
- Visualizations (via `plot_performance.py`)

---

## Disclaimer

This project is for educational/research purposes (inter-hostel competition). It does not constitute financial advice. Trading and investing involve risk, and past performance does not guarantee future results.