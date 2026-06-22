# Files to replace

Copy these files/folders into the root of your GitHub repo:

```text
agents/decomposition_agent.py
agents/trend_agent.py
agents/seasonal_agent.py
agents/residual_agent.py
agents/coordinator.py
baselines/baselines.py
evaluation/metrics.py
evaluation/backtest.py
data_loader.py
run_pipeline.py
update_readme_results.py
README.md
requirements.txt
deploy_results.sh
```

# Exact run

```bash
cd 5-Multi-Agent-Forecasting
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
python run_pipeline.py --ticker SPY --start 2018-01-01 --no_prophet
python run_pipeline.py --ticker SPY --start 2018-01-01
```

# Push

```bash
git add agents/ baselines/ evaluation/ data_loader.py run_pipeline.py update_readme_results.py README.md requirements.txt deploy_results.sh results/
git commit -m "Add dynamic forecasting coordinator and update results"
git push origin main
```

# Important

Your old screenshot still shows the old pipeline text:

```text
=== Average metrics across all folds ===
```

This patch prints:

```text
=== Average metrics across folds ===
```

If you still see `across all folds`, your terminal is not running the replaced `run_pipeline.py`.
