# Multi-Agent Trading Volume Forecasting

A 5-agent time-series forecasting system that decomposes daily stock trading volume into trend, seasonal, and residual components, assigns a specialized forecasting agent to each component, and dynamically selects the best forecast family using rolling validation.

## Objective

Build a robust multi-agent forecasting system for daily trading volume, then quantify whether explicit decomposition plus agent-level specialization can reduce forecasting error against persistence, moving-average, seasonal-naive, and Prophet baselines.

Trading volume is used instead of price because volume has recurring structure: long-run liquidity shifts, trading-week seasonality, and event-driven residual spikes around news, earnings, options expiry, and rebalancing days.

## Approach — the 5 agents

| # | Agent | Role |
|---|---|---|
| 1 | **Decomposition Agent** | Runs robust STL on cleaned volume data to split the signal into trend, seasonal, and residual components. |
| 2 | **Trend Agent** | Dynamically chooses between drift, exponential smoothing, Holt, damped Holt, and ETS-style trend forecasts using validation error. |
| 3 | **Seasonal Agent** | Learns recurring trading-week effects using weekday/cycle patterns and chooses the best seasonal pattern on validation data. |
| 4 | **Residual Agent** | Models leftover shocks using stable lag-regression candidates and can shrink residual forecasts to zero when they do not help. |
| 5 | **Coordinator Agent** | Recombines component forecasts, learns optional non-negative component weights, and dynamically chooses between additive multi-agent, weighted multi-agent, seasonal-naive, EWMA, and moving-average forecasts. |

Each component follows a `.fit()` / `.predict()` interface, so the forecasting logic can be tested, replaced, or extended independently.

## Project structure

```text
multi-agent-volume-forecast/
├── agents/
│   ├── decomposition_agent.py
│   ├── trend_agent.py
│   ├── seasonal_agent.py
│   ├── residual_agent.py
│   └── coordinator.py
├── baselines/
│   └── baselines.py
├── evaluation/
│   ├── metrics.py
│   └── backtest.py
├── notebooks/
│   └── 01_eda_and_decomposition.ipynb
├── data/
├── results/
├── data_loader.py
├── run_pipeline.py
├── update_readme_results.py
├── requirements.txt
└── README.md
```

## Outcome

The optimized pipeline now saves metrics and plots directly into the repository and injects the latest results into this README after every run.

<!-- RESULTS_START -->
Run `python run_pipeline.py --ticker SPY --start 2018-01-01` to generate the metrics and plots.
<!-- RESULTS_END -->

## Setup

```bash
git clone https://github.com/technocat00/5-Multi-Agent-Forecasting.git
cd 5-Multi-Agent-Forecasting
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

For PowerShell, use:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Usage

Run the full pipeline with Prophet included:

```bash
python run_pipeline.py --ticker SPY --start 2018-01-01
```

Run a faster development check without Prophet:

```bash
python run_pipeline.py --ticker SPY --start 2018-01-01 --no_prophet
```

Run on a local CSV:

```bash
python run_pipeline.py --input_csv data/my_volume.csv --date_col Date --value_col Volume --no_prophet
```

The pipeline writes:

- `results/decomposition.png`
- `results/backtest_mape.png`
- `results/forecast_comparison.png`
- `results/backtest_results.csv`
- `results/summary_metrics.csv`
- an updated README outcome section with metrics and graph previews

## Why this is a meaningful comparison

Naive persistence and moving averages are the minimum viable forecasting baselines. Seasonal naive tests whether the model captures recurring trading-week structure. Prophet is a stronger additive baseline with built-in trend and seasonality. The multi-agent system is useful only if it beats or closely matches these baselines while giving clear decomposition, ablation, and model-selection diagnostics.

## Notes / limitations

- Forecasting volume is inherently noisy around event days, so the coordinator uses validation-based model selection rather than forcing the decomposed model every fold.
- Residual forecasts are intentionally shrunk when validation shows that residual extrapolation adds noise.
- The current seasonal period defaults to 5 trading days. Monthly or quarterly effects can be added later through multiple seasonal decomposition.
