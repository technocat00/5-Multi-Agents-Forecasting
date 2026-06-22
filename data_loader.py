import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None


def fetch_volume_series(ticker: str = "SPY", start: str = "2018-01-01", end: str = None) -> pd.Series:
    if yf is None:
        raise ImportError("yfinance is required when --input_csv is not provided. Install requirements or pass --input_csv.")
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if data.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")
    volume = data["Volume"].copy()
    if isinstance(volume, pd.DataFrame):
        volume = volume.iloc[:, 0]
    volume.index = pd.to_datetime(volume.index)
    volume = volume.dropna().astype(float)
    volume.name = "volume"
    return volume


def load_volume_csv(path: str, date_col: str = None, value_col: str = None) -> pd.Series:
    df = pd.read_csv(path)
    if value_col is None:
        candidates = [c for c in df.columns if c.lower() in {"volume", "vol", "y", "value"}]
        value_col = candidates[0] if candidates else df.columns[-1]
    if date_col is None:
        candidates = [c for c in df.columns if c.lower() in {"date", "ds", "datetime", "timestamp"}]
        date_col = candidates[0] if candidates else df.columns[0]
    series = pd.Series(df[value_col].values, index=pd.to_datetime(df[date_col]), name="volume")
    return series.dropna().astype(float).sort_index()
