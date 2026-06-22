import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd

from agents.coordinator import Coordinator
from agents.decomposition_agent import DecompositionAgent
from data_loader import fetch_volume_series, load_volume_csv
from evaluation.backtest import rolling_backtest
from update_readme_results import update_readme


def plot_decomposition(series: pd.Series, seasonal_period: int, ticker: str):
    decomp_agent = DecompositionAgent(period=seasonal_period)
    components = decomp_agent.decompose(series)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    series.plot(ax=axes[0], title=f"{ticker} Daily Trading Volume", linewidth=0.8)
    components["trend"].plot(ax=axes[1], title="Trend Component")
    components["seasonal"].plot(ax=axes[2], title="Seasonal Component")
    components["resid"].plot(ax=axes[3], title="Residual Component", linewidth=0.6)
    plt.tight_layout()
    plt.savefig("results/decomposition.png", dpi=160)
    plt.close(fig)


def plot_backtest_mape(backtest_results: pd.DataFrame):
    mape_cols = [c for c in backtest_results.columns if c.endswith("_mape")]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = pd.to_datetime(backtest_results["fold_end"])
    for col in mape_cols:
        ax.plot(x, backtest_results[col], label=col.replace("_mape", ""))
    ax.set_title("Rolling-origin MAPE by fold")
    ax.set_xlabel("Fold end date")
    ax.set_ylabel("MAPE (%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/backtest_mape.png", dpi=160)
    plt.close(fig)


def plot_holdout_forecast(series: pd.Series, seasonal_period: int, ticker: str):
    holdout = min(max(20, seasonal_period * 4), max(5, len(series) // 5))
    train = series.iloc[:-holdout]
    test = series.iloc[-holdout:]
    coordinator = Coordinator(seasonal_period=seasonal_period, dynamic_selection=True)
    out = coordinator.fit_predict(train, holdout)
    pred = pd.Series(out["combined"], index=test.index, name="forecast")
    fig, ax = plt.subplots(figsize=(12, 6))
    series.iloc[-min(len(series), 120):].plot(ax=ax, label="actual")
    pred.plot(ax=ax, label=f"forecast: {out['selected_model']}")
    ax.axvline(test.index[0], linestyle="--", linewidth=1)
    ax.set_title(f"Latest holdout forecast for {ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/forecast_comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--input_csv", default=None)
    parser.add_argument("--date_col", default=None)
    parser.add_argument("--value_col", default=None)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--seasonal_period", type=int, default=5)
    parser.add_argument("--no_prophet", action="store_true")
    parser.add_argument("--skip_readme_update", action="store_true")
    args = parser.parse_args()
    os.makedirs("results", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    if args.input_csv:
        print(f"Loading volume data from {args.input_csv}...")
        series = load_volume_csv(args.input_csv, date_col=args.date_col, value_col=args.value_col)
        ticker_label = "CSV"
    else:
        print(f"Fetching volume data for {args.ticker}...")
        series = fetch_volume_series(ticker=args.ticker, start=args.start, end=args.end)
        ticker_label = args.ticker
        series.to_csv(f"data/{args.ticker.lower()}_volume.csv")
    print(f"Loaded {len(series)} trading days from {series.index[0].date()} to {series.index[-1].date()}")
    print("Saving decomposition plot...")
    plot_decomposition(series, args.seasonal_period, ticker_label)
    print("Running rolling-origin backtest...")
    initial_train_size = max(252, len(series) // 3)
    backtest_results = rolling_backtest(
        series,
        initial_train_size=initial_train_size,
        horizon=args.horizon,
        step=args.step,
        seasonal_period=args.seasonal_period,
        run_prophet=not args.no_prophet,
    )
    backtest_results.to_csv("results/backtest_results.csv", index=False)
    metric_cols = [c for c in backtest_results.columns if any(m in c for m in ["mape", "rmse", "mae", "runtime", "contribution"])]
    summary = backtest_results[metric_cols].mean(numeric_only=True)
    summary.to_csv("results/summary_metrics.csv")
    print("\n=== Average metrics across folds ===")
    print(summary.to_string())
    print("\nSaving result plots...")
    plot_backtest_mape(backtest_results)
    plot_holdout_forecast(series, args.seasonal_period, ticker_label)
    if not args.skip_readme_update:
        update_readme("README.md", "results")
        print("Updated README.md with metrics and plots.")
    print("Done. Commit README.md, results/, and the modified Python files.")


if __name__ == "__main__":
    main()
