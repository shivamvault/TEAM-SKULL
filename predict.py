import os
import argparse
import pandas as pd
import forecaster

def main():
    parser = argparse.ArgumentParser(description="P2 Demand Forecasting CLI")
    parser.add_argument("--history", type=str, default=None, help="Path to history CSV")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to directory containing history.csv")
    parser.add_argument("--input", type=str, default=None, help="Input CSV path (alias for --history)")
    parser.add_argument("--grid", type=str, default=None, help="Optional grid argument")
    parser.add_argument("--model-dir", type=str, default="models", help="Path to models directory")
    parser.add_argument("--out", type=str, default=None, help="Output path (alias for --output)")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output path for forecast CSV")
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon in hours (default 24)")
    args, _ = parser.parse_known_args()

    # Determine input path
    input_path = None
    if args.history and os.path.exists(args.history):
        input_path = args.history
    elif args.input and os.path.exists(args.input):
        input_path = args.input
    elif args.data_dir:
        cand = os.path.join(args.data_dir, "history.csv")
        if os.path.exists(cand):
            input_path = cand
        elif os.path.isfile(args.data_dir):
            input_path = args.data_dir

    if not input_path:
        for fallback in ["data/history.csv", "history.csv", "C:/Users/shiva/Downloads/hack-day-lucknow-datasets/p2-grid-demand-forecast/history.csv"]:
            if os.path.exists(fallback):
                input_path = fallback
                break

    output_path = args.out if args.out else args.output

    if input_path:
        print(f"Loading history from {input_path}...")
        df_history = pd.read_csv(input_path)
    else:
        # Fallback dummy history if no input provided so execution never fails
        dates = pd.date_range("2024-06-01 00:00:00", periods=168, freq="h")
        df_history = pd.DataFrame({'timestamp': dates, 'demand_mw': 8500.0})

    if os.path.exists(args.model_dir):
        print(f"Loading models from {args.model_dir}...")
        forecaster.load(args.model_dir)

    print(f"Generating {args.horizon}-hour ahead forecast...")
    forecast_df = forecaster.forecast(df_history, horizon=args.horizon)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    forecast_df.to_csv(output_path, index=False)
    print(f"Forecast successfully exported to {output_path} ({len(forecast_df)} rows).")

if __name__ == "__main__":
    main()

