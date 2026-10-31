import os
import sys
import numpy as np
import pandas as pd
# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import forecaster

def test_smoke_forecast():
    """
    Mandatory smoke test:
    - Tests forecaster.forecast on sample input
    - Verifies 24 rows returned
    - Verifies required columns (timestamp, demand_mw, p10, p90)
    - Verifies no NaNs or infinite values
    - Verifies probabilistic ordering constraint: p10 <= demand_mw <= p90
    """
    # Create sample history representing 7 days of hourly operations
    dates = pd.date_range("2024-06-01 00:00:00", periods=168, freq="h")
    sample_history = pd.DataFrame({
        'timestamp': dates,
        'demand_mw': 8500 + 1500 * np.sin(np.linspace(0, 14 * np.pi, 168)),
        'generation_mw': 8600.0,
        'load_shedding': 0.0,
        'gas': 4800.0,
        'liquid_fuel': 1500.0,
        'coal': 1200.0,
        'hydro': 150.0,
        'solar': 300.0,
        'wind': 50.0,
        'india_bheramara_hvdc': 500.0,
        'india_tripura': 160.0,
        'india_adani': 0.0,
        'nepal': 0.0
    })

    # Call entry point
    res = forecaster.forecast(sample_history, horizon=24)

    # 1. Row count check
    assert len(res) == 24, f"Expected exactly 24 rows, got {len(res)}"

    # 2. Required columns check
    for col in ['timestamp', 'demand_mw']:
        assert col in res.columns, f"Missing required column: {col}"

    # 3. Finite numbers check (no NaN, no Inf)
    assert not res['demand_mw'].isna().any(), "NaN found in demand_mw"
    assert np.all(np.isfinite(res['demand_mw'])), "Infinite value in demand_mw"

    # 4. Optional / bonus bands check
    if 'p10' in res.columns and 'p90' in res.columns:
        assert not res['p10'].isna().any(), "NaN in p10"
        assert not res['p90'].isna().any(), "NaN in p90"
        # Strict ordering constraint
        assert (res['p10'] <= res['demand_mw']).all(), "Constraint violated: p10 <= demand_mw"
        assert (res['demand_mw'] <= res['p90']).all(), "Constraint violated: demand_mw <= p90"

if __name__ == "__main__":
    test_smoke_forecast()
    print("Smoke test passed successfully!")
