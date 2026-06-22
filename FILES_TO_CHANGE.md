# Files to replace in your GitHub repo

Copy these files/folders into the root of `technocat00/5-Multi-Agent-Forecasting`:

- `agents/decomposition_agent.py`
- `agents/trend_agent.py`
- `agents/seasonal_agent.py`
- `agents/residual_agent.py`
- `agents/coordinator.py`
- `baselines/baselines.py`
- `evaluation/metrics.py`
- `evaluation/backtest.py`
- `data_loader.py`
- `run_pipeline.py`
- `update_readme_results.py`
- `README.md`
- `requirements.txt`
- `deploy_results.sh`

Then run:

```bash
python run_pipeline.py --ticker SPY --start 2018-01-01
```

For a quick check before Prophet:

```bash
python run_pipeline.py --ticker SPY --start 2018-01-01 --no_prophet
```

The pipeline will generate:

- `results/decomposition.png`
- `results/backtest_mape.png`
- `results/forecast_comparison.png`
- `results/backtest_results.csv`
- `results/summary_metrics.csv`

It will also rewrite the README section between:

```text
<!-- RESULTS_START -->
<!-- RESULTS_END -->
```

so the graphs and metrics show directly on GitHub.
