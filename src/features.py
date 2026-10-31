import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts temporal and lag features for day-ahead demand forecasting.
    """
    df = df.copy()
    if 'timestamp' in df.columns:
        dt = pd.to_datetime(df['timestamp'])
        df['hour'] = dt.dt.hour
        df['dayofweek'] = dt.dt.dayofweek
        df['month'] = dt.dt.month
        df['dayofyear'] = dt.dt.dayofyear
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        # Cyclical transforms
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
        df['day_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)

    if 'demand_mw' in df.columns:
        s = df['demand_mw']
        # Critical day-ahead lags (must be >= 24 hours ahead)
        df['lag_24'] = s.shift(24)
        df['lag_48'] = s.shift(48)
        df['lag_72'] = s.shift(72)
        df['lag_168'] = s.shift(168)
        
        # Rolling statistics on available lags
        df['rolling_mean_24'] = s.shift(24).rolling(window=24, min_periods=1).mean()
        df['rolling_std_24'] = s.shift(24).rolling(window=24, min_periods=1).std().fillna(0)
        df['rolling_mean_168'] = s.shift(24).rolling(window=168, min_periods=1).mean()
        
        # Momentum / delta
        df['delta_24_48'] = df['lag_24'] - df['lag_48']
        df['ratio_24_168'] = (df['lag_24'] + 1.0) / (df['lag_168'] + 1.0)
        
    return df

