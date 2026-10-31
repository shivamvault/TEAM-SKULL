import os
import argparse
import pandas as pd
from src.model import DemandForecaster

def main():
    parser = argparse.ArgumentParser(description="P2 Demand Forecaster Training Pipeline")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing history.csv")
    parser.add_argument("--out", type=str, default="models", help="Output directory for trained models")
    parser.add_argument("--output-dir", type=str, default=None, help="Alternative flag for output directory")
    args = parser.parse_args()

    out_dir = args.output_dir if args.output_dir else args.out
    os.makedirs(out_dir, exist_ok=True)
    
    # Locate history.csv
    possible_paths = [
        os.path.join(args.data_dir, "history.csv"),
        os.path.join(args.data_dir, "p2-grid-demand-forecast", "history.csv"),
        "history.csv",
        "data/history.csv"
    ]
    
    history_file = None
    for p in possible_paths:
        if os.path.exists(p):
            history_file = p
            break
            
    if not history_file:
        raise FileNotFoundError(f"Could not find history.csv in {args.data_dir} or standard paths.")
        
    print(f"Loading training data from {history_file}...")
    df_history = pd.read_csv(history_file)
    print(f"Loaded {len(df_history)} hourly records.")
    
    forecaster = DemandForecaster(model_dir=out_dir)
    print("Training LightGBM models (Mean, P10, P90) with lag and calendar features...")
    res = forecaster.train(df_history, out_dir=out_dir)
    print(f"Training completed successfully! Saved models to '{out_dir}'. Status: {res}")

if __name__ == "__main__":
    main()

