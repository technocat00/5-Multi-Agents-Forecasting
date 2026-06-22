set -e
python run_pipeline.py --ticker SPY --start 2018-01-01
python - <<'PY'
import pandas as pd
s = pd.read_csv('results/summary_metrics.csv', index_col=0).iloc[:, 0]
print('\nHeadline metrics:')
for key in ['multi_agent_mape', 'moving_avg_mape', 'naive_mape', 'prophet_mape']:
    if key in s.index:
        print(f'{key}: {s[key]:.4f}')
PY
git status --short
