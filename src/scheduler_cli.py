import argparse
import os
import json
import pandas as pd
from src.carbon import forecast_carbon_intensity, GreenComputeScheduler
import forecaster

def main():
    parser = argparse.ArgumentParser(description="Team Skull Carbon-Aware GreenCompute Scheduler")
    parser.add_argument("--history", type=str, default="data/history.csv", help="Path to history CSV")
    parser.add_argument("--task", type=str, default="LLM Fine-Tuning (8x H100 GPUs)", help="Name of compute task")
    parser.add_argument("--duration", type=int, default=4, help="Duration in hours (1-24)")
    parser.add_argument("--power", type=float, default=5.6, help="Power consumption in kW (default 5.6 kW for 8x H100)")
    parser.add_argument("--deadline", type=int, default=24, help="Execution deadline window in hours")
    parser.add_argument("--save-plan", type=str, default="docs/carbon_schedule_plan.json", help="Path to save schedule plan")
    args = parser.parse_args()

    # Find history file
    possible_paths = [
        args.history,
        "C:/Users/shiva/Downloads/hack-day-lucknow-datasets/p2-grid-demand-forecast/history.csv",
        "data/history.csv",
        "history.csv"
    ]
    hist_path = None
    for p in possible_paths:
        if os.path.exists(p):
            hist_path = p
            break
            
    if not hist_path:
        print("Warning: history.csv not found locally. Using synthetic demonstration history.")
        # Create small synthetic test df
        dates = pd.date_range("2024-06-10 00:00:00", periods=168, freq="h")
        df_hist = pd.DataFrame({
            'timestamp': dates,
            'demand_mw': 8500 + 1500 * np.sin(np.linspace(0, 14 * np.pi, 168)),
            'gas': 4500, 'coal': 1000, 'liquid_fuel': 1500, 'solar': 500, 'hydro': 100, 'wind': 50
        })
    else:
        df_hist = pd.read_csv(hist_path)

    # 1. Generate 24h demand forecast
    demand_df = forecaster.forecast(df_hist, horizon=24)
    
    # 2. Compute carbon intensity forecast (gCO2/kWh)
    carbon_df = forecast_carbon_intensity(demand_df, df_hist)
    
    # 3. Schedule compute workload
    scheduler = GreenComputeScheduler(carbon_df)
    plan = scheduler.plan_task(
        task_name=args.task,
        duration_hours=args.duration,
        power_kw=args.power,
        deadline_hours=args.deadline
    )
    
    # Display results
    print("\n" + "=" * 65)
    print(" [GREEN-COMPUTE] TEAM SKULL: CARBON-AWARE SCHEDULER REPORT")
    print("=" * 65)
    print(f" Task Name:                 {plan['task_name']}")
    print(f" Workload Duration:         {plan['duration_hours']} hours")
    print(f" Cluster Power:             {plan['power_kw']} kW")
    print("-" * 65)
    print(f" [*] OPTIMAL (GREEN) WINDOW: {plan['optimal_window']['start_time']} to {plan['optimal_window']['end_time']}")
    print(f"     Carbon Intensity:       {plan['optimal_window']['avg_ci_g_kwh']} gCO2/kWh")
    print(f"     Total CO2 Footprint:    {plan['optimal_window']['total_kg_co2']} kg CO2")
    print("-" * 65)
    print(f" [!] WORST (PEAK) WINDOW:    {plan['worst_window']['start_time']} to {plan['worst_window']['end_time']}")
    print(f"     Carbon Intensity:       {plan['worst_window']['avg_ci_g_kwh']} gCO2/kWh")
    print(f"     Total CO2 Footprint:    {plan['worst_window']['total_kg_co2']} kg CO2")
    print("-" * 65)
    print(f" [>] IMMEDIATE DISPATCH:     {plan['immediate_window']['start_time']}")
    print(f"     Total CO2 Footprint:    {plan['immediate_window']['total_kg_co2']} kg CO2")
    print("=" * 65)
    print(f" [+] CARBON SAVINGS:         {plan['co2_reduction_pct']}% reduction vs immediate dispatch")
    print(f"                             ({plan['co2_saved_kg']} kg CO2 saved)")
    print(f" --> {plan['recommendation']}")
    print("=" * 65 + "\n")
    
    os.makedirs(os.path.dirname(args.save_plan), exist_ok=True)
    with open(args.save_plan, "w") as f:
        json.dump(plan, f, indent=2)
    print(f"Execution plan saved to '{args.save_plan}'.")

if __name__ == "__main__":
    main()
