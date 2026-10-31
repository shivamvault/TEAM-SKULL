import pandas as pd
import numpy as np
from typing import Dict, Any, List

# Emission Factors in gCO2 / kWh (equivalent to kgCO2 / MWh)
# Reference: IPCC & Central Electricity Authority / Grid Emission Standards
EMISSION_FACTORS = {
    'coal': 950.0,
    'liquid_fuel': 720.0,
    'gas': 450.0,
    'india_bheramara_hvdc': 500.0,
    'india_tripura': 500.0,
    'india_adani': 850.0,
    'nepal': 50.0,          # Nepal import is predominantly run-of-the-river hydro
    'solar': 35.0,
    'hydro': 15.0,
    'wind': 12.0
}

def compute_historical_carbon_intensity(df: pd.DataFrame) -> pd.Series:
    """
    Computes grid-average carbon intensity (gCO2 / kWh) from historical fuel generation mix.
    """
    total_gen = pd.Series(0.0, index=df.index)
    total_emissions = pd.Series(0.0, index=df.index)
    
    for fuel, ef in EMISSION_FACTORS.items():
        if fuel in df.columns:
            gen = df[fuel].fillna(0).clip(lower=0)
            total_gen += gen
            total_emissions += gen * ef
            
    # Default fallback to grid average ~ 550 gCO2/kWh if generation data is missing
    carbon_intensity = np.where(total_gen > 100, total_emissions / total_gen, 550.0)
    return pd.Series(carbon_intensity, index=df.index, name="carbon_intensity_g_kwh")

def forecast_carbon_intensity(demand_forecast: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """
    Forecasts hourly carbon intensity (gCO2 / kWh) for the 24-hour demand forecast horizon.
    Utilizes empirical dispatch merit order:
    - Base renewables (hydro, solar, wind)
    - Coal & Gas intermediate load
    - Liquid fuel peaking plants during peak demand hours
    """
    df = demand_forecast.copy()
    hours = pd.to_datetime(df['timestamp']).dt.hour
    
    # Estimate typical diurnal carbon intensity profile from recent 7 days
    recent = history.tail(168).copy()
    if 'timestamp' in recent.columns:
        recent['hour'] = pd.to_datetime(recent['timestamp']).dt.hour
        recent_ci = compute_historical_carbon_intensity(recent)
        hourly_ci_map = recent_ci.groupby(recent['hour']).mean().to_dict()
    else:
        hourly_ci_map = {}
        
    predicted_ci = []
    for h, d in zip(hours, df['demand_mw']):
        base_ci = hourly_ci_map.get(h, 540.0)
        # Carbon intensity increases during high demand peaks due to liquid fuel peaking plants
        demand_factor = (d / 9000.0)
        # Solar dip during mid-day hours (10:00 - 15:00)
        solar_benefit = 40.0 if (10 <= h <= 15) else 0.0
        # Evening peak surge (18:00 - 22:00)
        peak_penalty = 60.0 if (18 <= h <= 22) else 0.0
        
        ci = float(np.clip(base_ci * demand_factor - solar_benefit + peak_penalty, 300.0, 850.0))
        predicted_ci.append(round(ci, 2))
        
    df['carbon_intensity_g_kwh'] = predicted_ci
    return df

class GreenComputeScheduler:
    """
    Carbon-Aware Compute Scheduler for planning long/automated workloads
    (e.g., LLM Training, LoRA fine-tuning, Batch ETL, Machine Learning inference)
    to minimize carbon footprint.
    """
    def __init__(self, forecast_df: pd.DataFrame):
        self.forecast_df = forecast_df.copy()
        if 'carbon_intensity_g_kwh' not in self.forecast_df.columns:
            raise ValueError("Forecast DataFrame must contain 'carbon_intensity_g_kwh'.")
            
    def plan_task(self, 
                  task_name: str, 
                  duration_hours: int, 
                  power_kw: float = 5.0, 
                  deadline_hours: int = 24) -> Dict[str, Any]:
        """
        Finds the optimal time window to execute an automated compute job.
        
        Parameters:
        - task_name: Descriptive name of the job
        - duration_hours: Runtime in hours (1 to 24)
        - power_kw: Average power draw in kilowatts (e.g. 8x H100 GPU cluster ~ 5.6 kW)
        - deadline_hours: Execution deadline window
        """
        duration = min(max(1, duration_hours), 24)
        max_start = min(deadline_hours, 24) - duration
        if max_start < 0:
            max_start = 0
            duration = min(deadline_hours, 24)
            
        ci = self.forecast_df['carbon_intensity_g_kwh'].values
        timestamps = self.forecast_df['timestamp'].values
        
        window_emissions = []
        for start_idx in range(max_start + 1):
            window_ci = ci[start_idx : start_idx + duration]
            # Total emissions in kg CO2 = Sum(Power kW * 1 hr * CI g/kWh) / 1000
            total_kg_co2 = float(np.sum(power_kw * 1.0 * window_ci) / 1000.0)
            avg_ci = float(np.mean(window_ci))
            window_emissions.append({
                'start_idx': start_idx,
                'start_time': str(timestamps[start_idx]),
                'end_time': str(timestamps[start_idx + duration - 1]),
                'total_kg_co2': round(total_kg_co2, 3),
                'avg_ci_g_kwh': round(avg_ci, 2)
            })
            
        # Best (Greenest), Worst (Peak), and Immediate (Now)
        sorted_windows = sorted(window_emissions, key=lambda x: x['total_kg_co2'])
        best_window = sorted_windows[0]
        worst_window = sorted_windows[-1]
        immediate_window = window_emissions[0]
        
        # Calculate savings
        savings_vs_worst = round(worst_window['total_kg_co2'] - best_window['total_kg_co2'], 3)
        pct_savings_vs_worst = round((savings_vs_worst / max(worst_window['total_kg_co2'], 1e-3)) * 100.0, 1)
        
        savings_vs_immediate = round(immediate_window['total_kg_co2'] - best_window['total_kg_co2'], 3)
        pct_savings_vs_immediate = round((savings_vs_immediate / max(immediate_window['total_kg_co2'], 1e-3)) * 100.0, 1)
        
        return {
            'task_name': task_name,
            'duration_hours': duration,
            'power_kw': power_kw,
            'optimal_window': best_window,
            'immediate_window': immediate_window,
            'worst_window': worst_window,
            'co2_saved_kg': max(0.0, savings_vs_immediate),
            'co2_reduction_pct': max(0.0, pct_savings_vs_immediate),
            'max_potential_reduction_pct': pct_savings_vs_worst,
            'recommendation': (
                f"Schedule '{task_name}' at {best_window['start_time']} to save "
                f"{pct_savings_vs_immediate}% CO2 emissions ({savings_vs_immediate} kg CO2) "
                f"compared to immediate dispatch."
            )
        }

