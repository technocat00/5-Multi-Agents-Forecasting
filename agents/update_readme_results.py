from pathlib import Path
import pandas as pd


START = "<!-- RESULTS_START -->"
END = "<!-- RESULTS_END -->"


def update_readme(readme_path="README.md", results_dir="results"):
    readme = Path(readme_path)
    results = Path(results_dir)
    if not readme.exists() or not (results / "summary_metrics.csv").exists():
        return
    summary = pd.read_csv(results / "summary_metrics.csv", index_col=0).iloc[:, 0]
    backtest_path = results / "backtest_results.csv"
    selected_text = ""
    if backtest_path.exists():
        bt = pd.read_csv(backtest_path)
        if "selected_model" in bt.columns:
            counts = bt["selected_model"].value_counts(normalize=True).mul(100).round(1)
            selected_text = "\n\n**Dynamic selection mix:**\n\n" + "\n".join([f"- `{idx}`: {val:.1f}% of folds" for idx, val in counts.items()])
    rows = []
    labels = [
        ("Dynamic coordinator", "multi_agent"),
        ("Naive persistence", "naive"),
        ("Moving average 5", "moving_avg"),
        ("Moving average 20", "moving_avg_20"),
        ("EWMA", "ewma"),
        ("Seasonal naive", "seasonal_naive"),
        ("Prophet", "prophet"),
    ]
    for label, prefix in labels:
        mape_key = f"{prefix}_mape"
        rmse_key = f"{prefix}_rmse"
        mae_key = f"{prefix}_mae"
        if mape_key in summary.index:
            rows.append((label, summary.get(mape_key), summary.get(rmse_key), summary.get(mae_key)))
    table = ["| Model | MAPE ↓ | RMSE ↓ | MAE ↓ |", "|---|---:|---:|---:|"]
    for label, mape, rmse, mae in rows:
        table.append(f"| {label} | {mape:.2f}% | {rmse:,.0f} | {mae:,.0f} |")
    best_mape = min((v for k, v in summary.items() if k.endswith("_mape") and pd.notna(v)), default=float("nan"))
    dynamic_mape = summary.get("multi_agent_mape", float("nan"))
    block = f"""{START}

## Latest Results

The current run reports **{dynamic_mape:.2f}% MAPE** for the dynamic coordinator. The best MAPE across all tracked models in this run is **{best_mape:.2f}%**.

{"\n".join(table)}{selected_text}

### Output Graphs

![STL decomposition](results/decomposition.png)

![Rolling backtest MAPE](results/backtest_mape.png)

![Latest holdout forecast](results/forecast_comparison.png)

{END}"""
    text = readme.read_text(encoding="utf-8")
    if START in text and END in text:
        before = text.split(START)[0]
        after = text.split(END, 1)[1]
        text = before + block + after
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    readme.write_text(text, encoding="utf-8")
