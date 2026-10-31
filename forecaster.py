import pandas as pd
from src.model import DemandForecaster

# Global forecaster instance
_FORECASTER = DemandForecaster(model_dir="models")

def load(model_dir: str) -> None:
    """
    Optional initialization hook called once by the evaluator with model_dir path.
    """
    global _FORECASTER
    _FORECASTER.load(model_dir)

def forecast(history: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """
    Generates 24-hour ahead electricity demand forecast.
    
    Parameters:
    - history: pandas.DataFrame with historical records up to 23:00 of the previous day.
    - horizon: Forecast horizon in hours (default 24).
    
    Returns:
    - pandas.DataFrame with columns: timestamp, demand_mw, p10, p90.
    """
    global _FORECASTER
    # Automatically attempt loading if not yet loaded
    if not _FORECASTER.is_loaded:
        _FORECASTER.load("models")
        
    return _FORECASTER.predict_24h(history, horizon=horizon)

