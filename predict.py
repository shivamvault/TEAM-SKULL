import argparse
import pandas as pd
import forecaster

def main():
    parser = argparse.ArgumentParser(description="P2 Demand Forecasting CLI")
    parser.add_argument("--history", type=str, default="data/history.csv", help="Path to history CSV")
    parser.add_argument("--model-dir", type=str, default="models", help="Path to models directory")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output path for forecast CSV")
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon (hours)")
    args = parser.parse_args()

    print(f"Loading history from {args.history}...")
    df_history = pd.read_csv(args.history)
    
    print(f"Loading model from {args.model_dir}...")
    forecaster.load(args.model_dir)
    
    print(f"Generating {args.horizon}-hour ahead forecast...")
    forecast_df = forecaster.forecast(df_history, horizon=args.horizon)
    
    forecast_df.to_csv(args.output, index=False)
    print(f"Forecast successfully exported to {args.output} ({len(forecast_df)} rows).")
    print(forecast_df.head())

if __name__ == "__main__":
    main()

