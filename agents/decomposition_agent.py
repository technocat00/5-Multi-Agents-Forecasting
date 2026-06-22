import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


class DecompositionAgent:
    def __init__(self, period: int = 5):
        self.period = int(period)
        self.trend_ = None
        self.seasonal_ = None
        self.resid_ = None

    def decompose(self, series: pd.Series, robust: bool = True) -> dict:
        series = pd.Series(series).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            raise ValueError("Input series is empty.")
        if len(series) < 2 * self.period:
            trend = pd.Series(np.full(len(series), float(series.mean())), index=series.index)
            seasonal = pd.Series(np.zeros(len(series)), index=series.index)
            resid = series - trend
        else:
            result = STL(series, period=self.period, robust=robust).fit()
            trend = result.trend
            seasonal = result.seasonal
            resid = result.resid
        self.trend_ = trend
        self.seasonal_ = seasonal
        self.resid_ = resid
        return {"trend": trend, "seasonal": seasonal, "resid": resid}
