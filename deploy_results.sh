#!/usr/bin/env bash
set -e
python run_pipeline.py --ticker SPY --start 2018-01-01
git add agents/ baselines/ evaluation/ data_loader.py run_pipeline.py update_readme_results.py README.md results/
git commit -m "Optimize multi-agent forecasting and update results"
git push origin main
