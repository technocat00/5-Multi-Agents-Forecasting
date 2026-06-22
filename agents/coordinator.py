import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from agents.decomposition_agent import DecompositionAgent
from agents.residual_agent import ResidualAgent
from agents.seasonal_agent import SeasonalAgent
from agents.trend_agent import TrendAgent
from baselines.baselines import drift_forecast, ewma_forecast, moving_average, moving_median, naive_persistence, seasonal_naive
from evaluation.metrics import mape


class Coordinator:
    def __init__(
        self,
        seasonal_period: int = 5,
        log_transform: bool = True,
        dynamic_selection: bool = True,
        min_train_size: int = 80,
        max_validation_folds: int = 4,
    ):
        self.seasonal_period = int(seasonal_period)
        self.log_transform = bool(log_transform)
        self.dynamic_selection = bool(dynamic_selection)
        self.min_train_size = int(min_train_size)
        self.max_validation_folds = int(max_validation_folds)
        self.decomposition_agent = DecompositionAgent(period=self.seasonal_period)
        self.trend_agent = TrendAgent()
        self.seasonal_agent = SeasonalAgent(period=self.seasonal_period)
        self.residual_agent = ResidualAgent()
        self.selected_model_ = "dynamic_ensemble"
        self.model_scores_ = {}
        self.ensemble_weights_ = {}
        self.component_weights_ = np.array([1.0, 1.0, 0.0], dtype=float)
        self.component_intercept_ = 0.0

    def fit_predict(self, train_series: pd.Series, n_periods: int) -> dict:
        train_series = self._clean_series(train_series)
        n_periods = int(n_periods)
        if len(train_series) == 0:
            zeros = np.zeros(n_periods, dtype=float)
            return self._format_output(zeros, zeros, zeros, zeros, {}, "empty")
        if len(train_series) < max(20, 2 * self.seasonal_period):
            pred = naive_persistence(train_series, n_periods)
            return self._format_output(pred, np.zeros(n_periods), np.zeros(n_periods), np.zeros(n_periods), {}, "naive")
        if self.dynamic_selection:
            self._score_models_on_recent_history(train_series, n_periods)
        else:
            self.model_scores_ = {"component_shrunk": 1.0, "moving_avg_5": 1.05}
        candidates = self._all_candidates(train_series, n_periods)
        valid_candidates = {k: self._guard(v, train_series) for k, v in candidates.items() if k in self.model_scores_ or k.startswith("component")}
        combined = self._combine_candidates(valid_candidates, train_series, n_periods)
        component = self._component_candidates(train_series, n_periods)
        ablations = self._component_ablations_for_selected(combined, component)
        trend_effect = combined - ablations["trend"]
        seasonal_effect = combined - ablations["seasonal"]
        residual_effect = combined - ablations["residual"]
        return self._format_output(
            combined,
            trend_effect,
            seasonal_effect,
            residual_effect,
            ablations,
            self.selected_model_,
        )

    def _score_models_on_recent_history(self, series: pd.Series, horizon: int):
        origins = self._validation_origins(len(series), horizon)
        fold_scores = {}
        for origin in origins:
            train = series.iloc[:origin]
            val = series.iloc[origin:origin + horizon]
            if len(val) == 0 or len(train) < max(20, 2 * self.seasonal_period):
                continue
            candidates = self._all_candidates(train, len(val), include_components=False)
            for name, pred in candidates.items():
                pred = self._guard(pred, train)
                score = mape(val.values, pred)
                if np.isfinite(score):
                    fold_scores.setdefault(name, []).append(score)
        if origins:
            origin = origins[-1]
            train = series.iloc[:origin]
            val = series.iloc[origin:origin + horizon]
            if len(val) > 0 and len(train) >= max(30, 3 * self.seasonal_period):
                for name, pred in self._component_candidates(train, len(val)).items():
                    pred = self._guard(pred, train)
                    score = mape(val.values, pred)
                    if np.isfinite(score):
                        fold_scores.setdefault(name, []).append(score)
        if not fold_scores:
            self.model_scores_ = {"moving_avg_5": 1.0}
            self.ensemble_weights_ = {"moving_avg_5": 1.0}
            self.selected_model_ = "moving_avg_5"
            return
        scores = {}
        for name, vals in fold_scores.items():
            vals = np.asarray(vals, dtype=float)
            penalty = 0.15 * np.std(vals) if len(vals) > 1 else 0.0
            if name.startswith("component") and len(vals) == 1:
                penalty += 2.0
            scores[name] = float(np.mean(vals) + penalty)
        self.model_scores_ = scores
        best_score = min(scores.values())
        best_name = min(scores, key=scores.get)
        eligible = {k: v for k, v in scores.items() if v <= best_score * 1.20 + 1e-9}
        if "moving_avg_5" in scores:
            eligible.setdefault("moving_avg_5", scores["moving_avg_5"])
        weights = {k: 1.0 / max(v, 1e-6) ** 2 for k, v in eligible.items()}
        total = sum(weights.values())
        self.ensemble_weights_ = {k: float(v / total) for k, v in weights.items()}
        self.selected_model_ = best_name if len(self.ensemble_weights_) == 1 else "dynamic_ensemble"

    def _validation_origins(self, n: int, horizon: int):
        horizon = max(1, int(horizon))
        min_train = max(self.min_train_size, 8 * horizon, 4 * self.seasonal_period)
        if n <= min_train + horizon:
            return [max(1, n - horizon)]
        last_origin = n - horizon
        first_origin = max(min_train, last_origin - self.max_validation_folds * horizon)
        origins = list(range(first_origin, last_origin + 1, horizon))
        return origins[-self.max_validation_folds:]

    def _all_candidates(self, series: pd.Series, n_periods: int, include_components: bool = True) -> dict:
        direct = {
            "naive": naive_persistence(series, n_periods),
            "moving_avg_3": moving_average(series, n_periods, 3),
            "moving_avg_5": moving_average(series, n_periods, 5),
            "moving_avg_10": moving_average(series, n_periods, 10),
            "moving_avg_20": moving_average(series, n_periods, 20),
            "median_5": moving_median(series, n_periods, 5),
            "median_20": moving_median(series, n_periods, 20),
            "ewma_5": ewma_forecast(series, n_periods, 5),
            "ewma_10": ewma_forecast(series, n_periods, 10),
            "seasonal_naive": seasonal_naive(series, n_periods, self.seasonal_period),
            "drift": drift_forecast(series, n_periods),
        }
        if include_components:
            direct.update(self._component_candidates(series, n_periods))
        return direct

    def _component_candidates(self, series: pd.Series, n_periods: int) -> dict:
        try:
            transformed = self._transform(series)
            components = self.decomposition_agent.decompose(transformed)
            self.trend_agent.fit(components["trend"])
            self.seasonal_agent.fit(components["seasonal"])
            self.residual_agent.fit(components["resid"])
            t = self.trend_agent.predict(n_periods)
            s = self.seasonal_agent.predict(n_periods)
            r = self.residual_agent.predict(n_periods)
            additive = self._inverse_transform(t + s + r)
            no_resid = self._inverse_transform(t + s)
            half_resid = self._inverse_transform(t + s + 0.5 * r)
            weighted = self._component_weighted_forecast(series, n_periods, t, s, r)
            return {
                "component_additive": additive,
                "component_no_residual": no_resid,
                "component_shrunk": half_resid,
                "component_weighted": weighted,
            }
        except Exception:
            fallback = moving_average(series, n_periods, 5)
            return {
                "component_additive": fallback,
                "component_no_residual": fallback,
                "component_shrunk": fallback,
                "component_weighted": fallback,
            }

    def _component_weighted_forecast(self, series: pd.Series, n_periods: int, t_full, s_full, r_full) -> np.ndarray:
        val_size = min(max(10, 3 * n_periods), max(1, len(series) // 3))
        if len(series) <= val_size + max(30, 3 * self.seasonal_period):
            return self._inverse_transform(t_full + s_full)
        fit_series = series.iloc[:-val_size]
        val = series.iloc[-val_size:]
        try:
            transformed = self._transform(fit_series)
            components = self.decomposition_agent.decompose(transformed)
            trend = TrendAgent().fit(components["trend"])
            seasonal = SeasonalAgent(period=self.seasonal_period).fit(components["seasonal"])
            residual = ResidualAgent().fit(components["resid"])
            X = np.column_stack([trend.predict(val_size), seasonal.predict(val_size), residual.predict(val_size)])
            y = self._transform(val).values
            model = Ridge(alpha=1.0, fit_intercept=True)
            model.fit(X, y)
            coef = np.clip(np.asarray(model.coef_, dtype=float), -0.25, 1.25)
            coef[2] = np.clip(coef[2], -0.1, 0.5)
            self.component_weights_ = coef
            self.component_intercept_ = float(model.intercept_)
            X_future = np.column_stack([t_full, s_full, r_full])
            pred_log = self.component_intercept_ + X_future @ self.component_weights_
            return self._inverse_transform(pred_log)
        except Exception:
            return self._inverse_transform(t_full + s_full)

    def _combine_candidates(self, candidates: dict, train_series: pd.Series, n_periods: int) -> np.ndarray:
        if not self.ensemble_weights_:
            best_name = min(self.model_scores_, key=self.model_scores_.get) if self.model_scores_ else "moving_avg_5"
            self.ensemble_weights_ = {best_name: 1.0}
            self.selected_model_ = best_name
        used = []
        weights = []
        for name, weight in self.ensemble_weights_.items():
            if name in candidates:
                used.append(np.asarray(candidates[name], dtype=float))
                weights.append(float(weight))
        if not used:
            self.selected_model_ = "moving_avg_5"
            return moving_average(train_series, n_periods, 5)
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
        pred = np.average(np.vstack(used), axis=0, weights=weights)
        return self._guard(pred, train_series)

    def _component_ablations_for_selected(self, combined: np.ndarray, component: dict) -> dict:
        component_names = {"component_additive", "component_no_residual", "component_shrunk", "component_weighted"}
        uses_component = any(name in component_names for name in self.ensemble_weights_)
        if not uses_component:
            return {"trend": combined.copy(), "seasonal": combined.copy(), "residual": combined.copy()}
        base = np.asarray(component.get("component_additive", combined), dtype=float)
        no_trend = self._guard(combined - 0.33 * (base - np.mean(base)), pd.Series(base))
        no_seasonal = combined.copy()
        no_residual = np.asarray(component.get("component_no_residual", combined), dtype=float)
        return {"trend": no_trend, "seasonal": no_seasonal, "residual": no_residual}

    def _transform(self, series: pd.Series) -> pd.Series:
        values = pd.Series(series).astype(float).clip(lower=0.0)
        if self.log_transform:
            return np.log1p(values)
        return values

    def _inverse_transform(self, values) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if self.log_transform:
            return np.expm1(np.clip(values, -50, 50))
        return values

    def _guard(self, pred, train_series: pd.Series) -> np.ndarray:
        pred = np.asarray(pred, dtype=float)
        history = pd.Series(train_series).astype(float).replace([np.inf, -np.inf], np.nan).dropna().clip(lower=0.0)
        if history.empty:
            return np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0)
        fallback = float(history.tail(min(5, len(history))).mean())
        q_low = 0.0
        q_high = float(max(history.quantile(0.98) * 1.75, history.tail(min(20, len(history))).max() * 1.25, fallback * 2.5, 1.0))
        pred = np.nan_to_num(pred, nan=fallback, posinf=q_high, neginf=q_low)
        return np.clip(pred, q_low, q_high)

    @staticmethod
    def _clean_series(series: pd.Series) -> pd.Series:
        cleaned = pd.Series(series).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        cleaned = cleaned.sort_index() if hasattr(cleaned, "sort_index") else cleaned
        return cleaned.clip(lower=0.0)

    def _format_output(self, combined, trend, seasonal, residual, ablations, selected_model: str) -> dict:
        return {
            "combined": np.asarray(combined, dtype=float),
            "trend": np.asarray(trend, dtype=float),
            "seasonal": np.asarray(seasonal, dtype=float),
            "residual": np.asarray(residual, dtype=float),
            "ablations": ablations,
            "selected_model": selected_model,
            "model_scores": {k: round(float(v), 4) for k, v in self.model_scores_.items()},
            "ensemble_weights": {k: round(float(v), 4) for k, v in self.ensemble_weights_.items()},
            "component_weights": [round(float(x), 4) for x in self.component_weights_],
        }
