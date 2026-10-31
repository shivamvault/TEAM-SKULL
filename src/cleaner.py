import pandas as pd
import numpy as np

def clean_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw history DataFrame from the national grid log:
    - Parses and sorts by timestamp
    - Removes unrealistic typo-like demand spikes (> 25000 MW or < 500 MW)
    - Clips fuel generation outliers
    - Interpolates short missing intervals
    """
    df = df.copy()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Handle target spikes
    if 'demand_mw' in df.columns:
        # Flag spikes and impossible dropouts
        bad_mask = (df['demand_mw'] > 25000) | (df['demand_mw'] < 500)
        if bad_mask.any():
            df.loc[bad_mask, 'demand_mw'] = np.nan
            # Linear interpolation bounded by neighboring hours
            df['demand_mw'] = df['demand_mw'].interpolate(method='linear', limit=12)
            df['demand_mw'] = df['demand_mw'].bfill().ffill()
    
    # Clean fuel columns if present
    fuel_cols = ['gas', 'liquid_fuel', 'coal', 'hydro', 'solar', 'wind']
    for col in fuel_cols:
        if col in df.columns:
            # Replace negative values with 0
            df[col] = df[col].clip(lower=0)
            # Liquid fuel typo spike cleaning (e.g. 29,222,897 MW)
            if col == 'liquid_fuel':
                df.loc[df[col] > 20000, col] = np.nan
                df[col] = df[col].interpolate().bfill().ffill()
            elif col == 'gas':
                df.loc[df[col] > 30000, col] = np.nan
                df[col] = df[col].interpolate().bfill().ffill()
    
    return df

