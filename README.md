# TEAM SKULL

[![Hack Day Project](https://img.shields.io/badge/Status-Evaluation%20Ready-success.svg)](https://github.com/shivamvault/TEAM-SKULL)
[![Problem](https://img.shields.io/badge/Problem-P2%3A%20Tomorrow's%20Load-orange.svg)](https://github.com/shivamvault/TEAM-SKULL)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> Built by **Team Skull** for Hacktoberfest Hack Day Lucknow (AI Community Lucknow).
> **Problem 02:** Tomorrow's Load (Time Series Demand Forecasting) with **Carbon Emission Monitoring & GreenCompute Task Scheduling**.

---

## Problem understanding
In Problem 02, our goal is to predict hourly grid demand across the entire next day (24 hours ahead) starting from midnight, giving power operators enough lead time to schedule generation and avoid blackouts or costly peaker plants. We treated this not as a single recursive step, but as a direct multi-horizon forecasting problem so errors do not snowball from hour 1 into hour 24. Since the raw log has missing timestamps and wild data entry typos like 121,000 MW spikes, robust automated data cleaning before feature creation is essential. On top of predicting demand, we realized that the grid's fuel mix heavily influences hourly emissions, creating an opportunity to optimize when large energy-intensive compute jobs run. We framed our final system to both forecast grid load accurately and schedule heavy workloads into clean solar hours to directly cut carbon emissions.

## Approach
We broke our solution into four practical stages:
1. **Data Cleaning:** The dataset had obvious manual typos—like 121,000 MW spikes and impossible generation numbers—that would wreck tree splits. We built a boundary filter ($500 \le \text{MW} \le 25,000$) with bidirectional interpolation to fix gaps and spikes cleanly without any forward leakage.
2. **Feature Engineering:** Electricity usage follows clear human routines, so we extracted cyclical time features (hour sine/cosine, day of week, day of year, weekend flag). We also created multi-day lags ($Lag_{24}, Lag_{48}, Lag_{72}, Lag_{168}$ to capture yesterday and last week at the exact same hour), plus rolling 24-hour and 7-day moving averages and recent demand momentum.
3. **Modeling:** We trained LightGBM models with an L1 regression loss because minimizing MAE directly optimizes the competition's MAPE metric. To provide reliable uncertainty bounds, we trained two separate quantile regressors ($P_{10}$ and $P_{90}$) and enforced non-crossing logic so $P_{10} \le \hat{y} \le P_{90}$ is always guaranteed.
4. **GreenCompute Scheduling:** Using the hourly breakdown of coal, gas, fuel oil, hydro, solar, and wind, we calculated real-time grid carbon intensity ($gCO_2/kWh$). Our scheduler searches for optimal consecutive hours to run heavy background tasks like model training or ETL jobs during cheap solar peaks instead of dirty evening peaker peaks.

## Validation
To make sure our local validation actually reflects what the scoring system does on unseen data, we set up a strict rolling-origin walk-forward cross-validation harness:
- **Walk-Forward Setup:** We simulated day-by-day operations by cutting off history at 23:00 on day $D-1$ and generating forecasts for the full 24 hours of day $D$, rolling forward across multiple months.
- **Zero Lookahead:** All imputations, rolling means, and lag lookups only use data available strictly before the midnight forecast origin.
- **Handling Imperfect History:** Since the hidden evaluation set contains random missing records and sensor glitches, we tested our pipeline against noisy slices to ensure it never crashes, always outputs 24 finite numbers, and maintains low peak error.

## Results
Across our local walk-forward validation runs on the historical grid data, our model scored comfortably inside the top tier:
- **Local Validation Score:** **3.92% MAPE** (beats the competition's Excellent threshold of $\le 4.0\%$).
- **Peak Demand APE:** **3.18%** (outperforms the Excellent anchor of $\le 3.3\%$).
- **P10–P90 Coverage:** **81.4%** of actual values landed inside our prediction interval (hitting the bonus target band of 75%–85%).
- **Latest Evaluator Validation Score:** **3.88% MAPE**, showing strong consistency between local tests and the evaluator sandbox.
- **Carbon Savings:** Shifting a 4-hour compute workload (like an 8x H100 LLM fine-tuning job) from evening peak hours into midday solar hours cuts carbon emissions by **14% to 28%**.

## How to run
The project is self-contained, reproducible, and ready to run with standard commands:

### 1. Model Training
To train the LightGBM point forecasters and quantile models directly from public data:
```bash
python train.py --data-dir C:/Users/shiva/Downloads/hack-day-lucknow-datasets/p2-grid-demand-forecast --out models/
```

### 2. Day-Ahead Demand Forecasting
To generate a 24-hour forecast with P10 and P90 prediction bounds:
```bash
python predict.py --history data/history.csv --model-dir models/ --output predictions.csv
```

### 3. Automated Smoke Test
To verify schema validity, output constraints, and non-crossing intervals:
```bash
python tests/test_smoke.py
```

### 4. Carbon-Aware GreenCompute Scheduler
To schedule an automated LLM fine-tuning or training workload into the greenest grid window:
```bash
python -m src.scheduler_cli --task "LLM LoRA Fine-Tuning" --duration 4 --power 5.6
```

## Use of AI tools
We used AI assistants thoughtfully to speed up our workflow during the hackathon:
- **Antigravity & Gemini:** Helped brainstorm the carbon intensity formulation, draft baseline feature pipelines, structure documentation, and troubleshoot CSS styling in our dashboard.
- **Copilot & Cursor:** Used for writing repetitive unit test assertions, CLI argument boilerplate, and docstrings.
- **Human Review:** Every line of code, math formula, cleaning rule, and feature split was verified by our team. Whenever an LLM suggested recursive multi-step forecasting or complex deep learning that would blow through our 2-CPU / 4GB limit, we stepped in and chose fast, robust LightGBM regressors instead.

## Data & licences
All tools and data used in this project are standard, open, and fully licensed:
- **Dataset:** Official grid hourly telemetry provided by the Hacktoberfest Hack Day Lucknow organizers (`history.csv`). No private, scraped, or forbidden future data was used.
- **Libraries:** LightGBM (MIT License), Scikit-Learn (BSD 3-Clause), Pandas and NumPy (BSD 3-Clause), SciPy (BSD 3-Clause), Joblib (BSD 3-Clause), PyYAML (MIT License).
- **Weights:** Model weights are trained strictly from public data within 3 seconds; no unlicensed pretrained models are included.

## Team Names and roles
Team Skull members and roles for Hack Day Lucknow:

- **Shivam — Team Leader & ML Systems Architect:** Led overall project strategy, engineered the multi-horizon LightGBM forecasting engine with L1 loss, developed the GreenCompute carbon scheduler algorithm, and built the interactive web dashboard.
- **Kartikey Singh — Data Engineering & Feature Pipeline Specialist:** Built the telemetry data cleaning filters to catch sensor typos and spikes (like 121,000 MW errors), created the 24h/168h lag and rolling features, and implemented the CLI prediction scripts.
- **Nishtha Agarwal — Presentation Lead & Pitch Deck Designer:** Designed the presentation deck and visual storytelling for the 8-minute judging rounds, structured the evaluation summaries and error analysis charts, and ensured the team's live demo cleanly communicates the business and environmental impact.

Every team member actively contributed to model validation, code reviews, and live presentation preparation throughout the hackathon.
