import numpy as np
import pandas as pd


def naive_persistence(train_series: pd.Series, n_periods: int) -> np.ndarray:
    values = pd.Series(train_series).astype(float).dropna()
    return np.full(int(n_periods), float(values.iloc[-1]), dtype=float)


def moving_average(train_series: pd.Series, n_periods: int, window: int = 5) -> np.ndarray:
    values = pd.Series(train_series).astype(float).dropna()
    window = max(1, min(int(window), len(values)))
    return np.full(int(n_periods), float(values.tail(window).mean()), dtype=float)


def moving_median(train_series: pd.Series, n_periods: int, window: int = 5) -> np.ndarray:
    values = pd.Series(train_series).astype(float).dropna()
    window = max(1, min(int(window), len(values)))
    return np.full(int(n_periods), float(values.tail(window).median()), dtype=float)


def ewma_forecast(train_series: pd.Series, n_periods: int, span: int = 10) -> np.ndarray:
    values = pd.Series(train_series).astype(float).dropna()
    span = max(2, min(int(span), len(values)))
    return np.full(int(n_periods), float(values.ewm(span=span, adjust=False).mean().iloc[-1]), dtype=float)


def seasonal_naive(train_series: pd.Series, n_periods: int, period: int = 5) -> np.ndarray:
    values = pd.Series(train_series).astype(float).dropna()
    period = max(1, min(int(period), len(values)))
    pattern = values.tail(period).values.astype(float)
    return np.asarray([pattern[i % period] for i in range(int(n_periods))], dtype=float)


def drift_forecast(train_series: pd.Series, n_periods: int) -> np.ndarray:
    values = pd.Series(train_series).astype(float).dropna().values
    if len(values) < 2:
        return np.full(int(n_periods), float(values[-1] if len(values) else 0.0))
    slope = (values[-1] - values[0]) / max(len(values) - 1, 1)
    steps = np.arange(1, int(n_periods) + 1)
    return values[-1] + slope * steps


def prophet_forecast(train_series: pd.Series, n_periods: int) -> np.ndarray:
    from prophet import Prophet
    df = pd.DataFrame({"ds": pd.to_datetime(train_series.index), "y": pd.Series(train_series).astype(float).values})
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=int(n_periods), freq="B")
    forecast = model.predict(future)
    return forecast["yhat"].values[-int(n_periods):]
