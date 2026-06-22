import time
import numpy as np
import pandas as pd

from agents.coordinator import Coordinator
from baselines.baselines import ewma_forecast, moving_average, naive_persistence, prophet_forecast, seasonal_naive
from evaluation.metrics import agent_error_contribution, mae, mape, rmse


def rolling_backtest(
    series: pd.Series,
    initial_train_size: int,
    horizon: int = 5,
    step: int = 5,
    seasonal_period: int = 5,
    run_prophet: bool = True,
) -> pd.DataFrame:
    results = []
    n = len(series)
    start = int(initial_train_size)
    while start + horizon <= n:
        train = series.iloc[:start]
        test = series.iloc[start:start + horizon]
        y_true = test.values.astype(float)
        t0 = time.perf_counter()
        coordinator = Coordinator(seasonal_period=seasonal_period, dynamic_selection=True)
        agent_out = coordinator.fit_predict(train, horizon)
        ma_runtime = time.perf_counter() - t0
        ma_pred = agent_out["combined"]
        naive_pred = naive_persistence(train, horizon)
        mavg_pred = moving_average(train, horizon, window=seasonal_period)
        mavg20_pred = moving_average(train, horizon, window=20)
        ewma_pred = ewma_forecast(train, horizon, span=10)
        seasonal_pred = seasonal_naive(train, horizon, period=seasonal_period)
        fold_result = {
            "fold_end": train.index[-1],
            "test_start": test.index[0],
            "test_end": test.index[-1],
            "selected_model": agent_out.get("selected_model", "dynamic"),
            "ensemble_weights": str(agent_out.get("ensemble_weights", {})),
            "multi_agent_mape": mape(y_true, ma_pred),
            "multi_agent_rmse": rmse(y_true, ma_pred),
            "multi_agent_mae": mae(y_true, ma_pred),
            "multi_agent_runtime_sec": ma_runtime,
            "naive_mape": mape(y_true, naive_pred),
            "naive_rmse": rmse(y_true, naive_pred),
            "naive_mae": mae(y_true, naive_pred),
            "moving_avg_mape": mape(y_true, mavg_pred),
            "moving_avg_rmse": rmse(y_true, mavg_pred),
            "moving_avg_mae": mae(y_true, mavg_pred),
            "moving_avg_20_mape": mape(y_true, mavg20_pred),
            "moving_avg_20_rmse": rmse(y_true, mavg20_pred),
            "moving_avg_20_mae": mae(y_true, mavg20_pred),
            "ewma_mape": mape(y_true, ewma_pred),
            "ewma_rmse": rmse(y_true, ewma_pred),
            "ewma_mae": mae(y_true, ewma_pred),
            "seasonal_naive_mape": mape(y_true, seasonal_pred),
            "seasonal_naive_rmse": rmse(y_true, seasonal_pred),
            "seasonal_naive_mae": mae(y_true, seasonal_pred),
        }
        if run_prophet:
            try:
                t0 = time.perf_counter()
                prophet_pred = prophet_forecast(train, horizon)
                prophet_runtime = time.perf_counter() - t0
                fold_result["prophet_mape"] = mape(y_true, prophet_pred)
                fold_result["prophet_rmse"] = rmse(y_true, prophet_pred)
                fold_result["prophet_mae"] = mae(y_true, prophet_pred)
                fold_result["prophet_runtime_sec"] = prophet_runtime
            except Exception:
                fold_result["prophet_mape"] = np.nan
                fold_result["prophet_rmse"] = np.nan
                fold_result["prophet_mae"] = np.nan
                fold_result["prophet_runtime_sec"] = np.nan
        contributions = agent_error_contribution(y_true, agent_out)
        fold_result["trend_agent_contribution"] = contributions["trend"]
        fold_result["seasonal_agent_contribution"] = contributions["seasonal"]
        fold_result["residual_agent_contribution"] = contributions["residual"]
        results.append(fold_result)
        start += step
    return pd.DataFrame(results)
