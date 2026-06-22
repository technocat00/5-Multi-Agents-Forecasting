from pathlib import Path
import pandas as pd

RESULTS_START = "<!-- RESULTS_START -->"
RESULTS_END = "<!-- RESULTS_END -->"


def _fmt(x):
    if pd.isna(x):
        return "-"
    try:
        return f"{float(x):,.3f}"
    except Exception:
        return str(x)


def build_results_block(results_dir: str = "results") -> str:
    results_path = Path(results_dir)
    summary_path = results_path / "summary_metrics.csv"
    backtest_path = results_path / "backtest_results.csv"
    if not summary_path.exists():
        return f"{RESULTS_START}\nRun `python run_pipeline.py --ticker SPY --start 2018-01-01` to generate the metrics and plots.\n{RESULTS_END}"
    summary = pd.read_csv(summary_path, index_col=0).iloc[:, 0]
    models = ["multi_agent", "naive", "moving_avg", "seasonal_naive", "prophet"]
    rows = []
    for model in models:
        if f"{model}_mape" in summary.index:
            rows.append(
                f"| {model.replace('_', ' ').title()} | {_fmt(summary.get(f'{model}_mape'))} | "
                f"{_fmt(summary.get(f'{model}_rmse'))} | {_fmt(summary.get(f'{model}_mae'))} | "
                f"{_fmt(summary.get(f'{model}_runtime_sec'))} |"
            )
    agent_rows = []
    for agent in ["trend", "seasonal", "residual"]:
        key = f"{agent}_agent_contribution"
        if key in summary.index:
            agent_rows.append(f"| {agent.title()} Agent | {_fmt(summary.get(key))} |")
    selected_block = ""
    if backtest_path.exists():
        bt = pd.read_csv(backtest_path)
        if "selected_model" in bt.columns:
            counts = bt["selected_model"].value_counts(normalize=True).mul(100).round(1)
            selected_block = "\n**Dynamic model selection frequency**\n\n" + "\n".join(
                [f"- `{name}`: {pct:.1f}% of folds" for name, pct in counts.items()]
            ) + "\n"
    return "\n".join([
        RESULTS_START,
        "### Latest generated results",
        "",
        "| Model | MAPE ↓ | RMSE ↓ | MAE ↓ | Runtime sec ↓ |",
        "|---|---:|---:|---:|---:|",
        *rows,
        "",
        "### Agent contribution ablation",
        "",
        "Positive RMSE contribution means removing that component worsened the forecast.",
        "",
        "| Component | Avg RMSE degradation ↑ |",
        "|---|---:|",
        *agent_rows,
        selected_block,
        "### Generated plots",
        "",
        "![STL decomposition](results/decomposition.png)",
        "",
        "![Backtest MAPE by fold](results/backtest_mape.png)",
        "",
        "![Latest holdout forecast](results/forecast_comparison.png)",
        RESULTS_END,
    ])


def update_readme(readme_path: str = "README.md", results_dir: str = "results") -> None:
    path = Path(readme_path)
    text = path.read_text(encoding="utf-8") if path.exists() else "# Multi-Agent Trading Volume Forecasting\n"
    block = build_results_block(results_dir)
    if RESULTS_START in text and RESULTS_END in text:
        before = text.split(RESULTS_START)[0]
        after = text.split(RESULTS_END)[1]
        text = before + block + after
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    update_readme()
