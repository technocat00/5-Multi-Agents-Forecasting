import numpy as np


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-12
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def agent_error_contribution(y_true: np.ndarray, agent_forecasts: dict) -> dict:
    combined = np.asarray(agent_forecasts.get("combined"), dtype=float)
    full_rmse = rmse(y_true, combined)
    ablations = agent_forecasts.get("ablations", {}) or {}
    contributions = {}
    for name in ["trend", "seasonal", "residual"]:
        if name in ablations:
            ablated = np.asarray(ablations[name], dtype=float)
        else:
            ablated = combined
        contributions[name] = rmse(y_true, ablated) - full_rmse
    return contributions
