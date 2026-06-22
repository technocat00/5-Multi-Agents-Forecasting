import numpy as np
import pandas as pd


class SeasonalAgent:
    def __init__(self, period: int = 5):
        self.period = int(period)
        self.pattern_ = None
        self.n_train_ = 0

    def fit(self, seasonal_series: pd.Series) -> "SeasonalAgent":
        values = pd.Series(seasonal_series).astype(float).dropna().values
        if len(values) == 0:
            values = np.array([0.0])
        self.n_train_ = len(values)
        period_scores = []
        for period in sorted(set([self.period, 5, 10, 21])):
            if len(values) >= 3 * period:
                score = self._validation_score(values, period)
                period_scores.append((score, period))
        if period_scores:
            self.period = min(period_scores)[1]
        positions = np.arange(len(values)) % self.period
        pattern = pd.Series(values, index=positions).groupby(level=0).median().reindex(range(self.period)).fillna(0.0)
        pattern = pattern - pattern.mean()
        self.pattern_ = pattern.values.astype(float)
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        if self.pattern_ is None:
            raise RuntimeError("SeasonalAgent must be fit before predict.")
        positions = np.arange(self.n_train_, self.n_train_ + int(n_periods)) % self.period
        return self.pattern_[positions]

    def _validation_score(self, values: np.ndarray, period: int) -> float:
        val_size = min(max(period, 5), max(1, len(values) // 3))
        train = values[:-val_size]
        val = values[-val_size:]
        positions = np.arange(len(train)) % period
        pattern = pd.Series(train, index=positions).groupby(level=0).median().reindex(range(period)).fillna(0.0)
        pattern = pattern - pattern.mean()
        future_positions = np.arange(len(train), len(train) + val_size) % period
        pred = pattern.values[future_positions]
        return float(np.mean(np.abs(val - pred)))
