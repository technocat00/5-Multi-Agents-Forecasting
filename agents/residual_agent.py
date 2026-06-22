import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


class ResidualAgent:
    def __init__(self, lags=None, alpha: float = 10.0):
        self.lags = list(lags or [1, 2, 3, 5, 10])
        self.alpha = float(alpha)
        self.model_ = None
        self.history_ = None
        self.shrink_ = 0.0

    def fit(self, residual_series: pd.Series) -> "ResidualAgent":
        values = pd.Series(residual_series).astype(float).dropna().values
        if len(values) == 0:
            values = np.array([0.0])
        max_lag = max(self.lags)
        self.history_ = list(values[-max_lag:]) if len(values) >= max_lag else list(np.pad(values, (max_lag - len(values), 0)))
        if len(values) < max_lag + 20:
            self.model_ = None
            self.shrink_ = 0.0
            return self
        val_size = min(max(10, len(values) // 5), max(1, len(values) // 3))
        train = values[:-val_size]
        val = values[-val_size:]
        X, y = self._lag_matrix(train)
        if len(y) == 0:
            self.model_ = None
            self.shrink_ = 0.0
            return self
        model = Ridge(alpha=self.alpha).fit(X, y)
        pred = self._recursive_predict(model, list(train[-max_lag:]), val_size)
        zero_score = self._mae(val, np.zeros(val_size))
        model_score = self._mae(val, pred)
        if model_score >= zero_score:
            self.model_ = None
            self.shrink_ = 0.0
            return self
        X_full, y_full = self._lag_matrix(values)
        self.model_ = Ridge(alpha=self.alpha).fit(X_full, y_full)
        improvement = max(0.0, zero_score - model_score) / max(zero_score, 1e-9)
        self.shrink_ = float(np.clip(improvement, 0.0, 0.5))
        self.history_ = list(values[-max_lag:])
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        if self.model_ is None:
            return np.zeros(int(n_periods), dtype=float)
        return self.shrink_ * self._recursive_predict(self.model_, list(self.history_), int(n_periods))

    def _lag_matrix(self, values: np.ndarray):
        max_lag = max(self.lags)
        rows, target = [], []
        for t in range(max_lag, len(values)):
            rows.append([values[t - lag] for lag in self.lags])
            target.append(values[t])
        return np.asarray(rows, dtype=float), np.asarray(target, dtype=float)

    def _recursive_predict(self, model, history, n_periods: int) -> np.ndarray:
        preds = []
        for _ in range(n_periods):
            row = np.asarray([[history[-lag] for lag in self.lags]], dtype=float)
            pred = float(model.predict(row)[0])
            preds.append(pred)
            history.append(pred)
        return np.asarray(preds, dtype=float)

    @staticmethod
    def _mae(y_true, y_pred) -> float:
        return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
