import numpy as np
import pandas as pd


class TrendAgent:
    def __init__(self):
        self.model_name_ = "last"
        self.params_ = None
        self.values_ = None

    def fit(self, trend_series: pd.Series) -> "TrendAgent":
        values = pd.Series(trend_series).astype(float).dropna().values
        if len(values) == 0:
            values = np.array([0.0])
        self.values_ = values
        if len(values) < 20:
            self.model_name_ = "last"
            self.params_ = (float(values[-1]),)
            return self
        val_size = min(max(5, len(values) // 5), max(1, len(values) // 3))
        train = values[:-val_size]
        val = values[-val_size:]
        candidates = {
            "last": self._last(train, val_size),
            "mean_10": self._level(train, val_size, 10),
            "mean_30": self._level(train, val_size, 30),
            "linear_full": self._linear(train, val_size, len(train)),
            "linear_60": self._linear(train, val_size, min(60, len(train))),
            "damped_drift": self._damped_drift(train, val_size),
        }
        self.model_name_ = min(candidates, key=lambda k: self._mae(val, candidates[k]))
        self.params_ = self._fit_params(values, self.model_name_)
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        if self.values_ is None:
            raise RuntimeError("TrendAgent must be fit before predict.")
        return self._predict_from_params(self.values_, self.model_name_, self.params_, int(n_periods))

    def _fit_params(self, values: np.ndarray, name: str):
        if name == "last":
            return (float(values[-1]),)
        if name == "mean_10":
            return (float(np.mean(values[-min(10, len(values)):])) ,)
        if name == "mean_30":
            return (float(np.mean(values[-min(30, len(values)):])) ,)
        if name == "linear_full":
            return self._linear_params(values, len(values))
        if name == "linear_60":
            return self._linear_params(values, min(60, len(values)))
        return self._damped_params(values)

    def _predict_from_params(self, values: np.ndarray, name: str, params, n_periods: int) -> np.ndarray:
        if name in {"last", "mean_10", "mean_30"}:
            return np.full(n_periods, float(params[0]))
        if name in {"linear_full", "linear_60"}:
            slope, intercept, start_x, train_len = params
            x = np.arange(train_len, train_len + n_periods, dtype=float) - start_x
            return intercept + slope * x
        level, drift = params
        steps = np.arange(1, n_periods + 1, dtype=float)
        damping = 0.80 ** steps
        return level + drift * np.cumsum(damping)

    @staticmethod
    def _last(values: np.ndarray, n_periods: int) -> np.ndarray:
        return np.full(n_periods, float(values[-1]))

    @staticmethod
    def _level(values: np.ndarray, n_periods: int, window: int) -> np.ndarray:
        return np.full(n_periods, float(np.mean(values[-min(window, len(values)):])) )

    @classmethod
    def _linear(cls, values: np.ndarray, n_periods: int, window: int) -> np.ndarray:
        params = cls._linear_params(values, window)
        slope, intercept, start_x, train_len = params
        x = np.arange(train_len, train_len + n_periods, dtype=float) - start_x
        return intercept + slope * x

    @staticmethod
    def _linear_params(values: np.ndarray, window: int):
        train_len = len(values)
        window = max(2, min(window, train_len))
        y = values[-window:]
        x = np.arange(train_len - window, train_len, dtype=float)
        start_x = float(x[0])
        slope, intercept = np.polyfit(x - start_x, y, deg=1)
        return float(slope), float(intercept), start_x, train_len

    @classmethod
    def _damped_drift(cls, values: np.ndarray, n_periods: int) -> np.ndarray:
        level, drift = cls._damped_params(values)
        steps = np.arange(1, n_periods + 1, dtype=float)
        damping = 0.80 ** steps
        return level + drift * np.cumsum(damping)

    @staticmethod
    def _damped_params(values: np.ndarray):
        diffs = np.diff(values[-min(30, len(values)):])
        drift = float(np.median(diffs)) if len(diffs) else 0.0
        return float(values[-1]), drift

    @staticmethod
    def _mae(y_true, y_pred) -> float:
        return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))
