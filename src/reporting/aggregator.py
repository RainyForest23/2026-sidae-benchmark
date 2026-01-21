
import os
import pandas as pd
import glob
import config
from src.evaluation.metrics import calculate_accuracy

def aggregate_results(results_dir=None):
    if results_dir is None:
        results_dir = config.RESULTS_DIR
        
    all_files = glob.glob(os.path.join(results_dir, "*.csv"))
    
    aggregated_records = []
    
    for filename in all_files:
        if "leaderboard" in filename or "aggregated" in filename:
            continue
            
        try:
            df = pd.read_csv(filename)
            if df.empty:
                continue
                
            # Assume strict columns exist if produced by our runner
            if "model" not in df.columns:
                # Fallback or skip
                continue
                
            model = df["model"].iloc[0]
            benchmark = df["benchmark"].iloc[0]
            task = df["task"].iloc[0]
            
            # Calculate score
            # Determine metric based on benchmark?
            # For now, default to accuracy.
            score = calculate_accuracy(df["prediction"].astype(str).tolist(), df["reference"].astype(str).tolist())
            
            aggregated_records.append({
                "Model": model,
                "Benchmark": benchmark,
                "Task": task,
                "Score": score,
                "Samples": len(df)
            })
            
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            
    return pd.DataFrame(aggregated_records)
