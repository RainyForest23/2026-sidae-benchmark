
import pandas as pd
import os
import config
from .aggregator import aggregate_results

def generate_leaderboard(results_dir=None):
    df = aggregate_results(results_dir)
    if df.empty:
        print("No results to generate leaderboard.")
        return
        
    # Pivot: Model x Benchmark (Average Score)
    # First, average across tasks within a benchmark if needed?
    # Usually we want average score per benchmark.
    
    # Group by Model, Benchmark -> mean(Score)
    benchmark_scores = df.groupby(["Model", "Benchmark"])["Score"].mean().reset_index()
    
    # Pivot
    leaderboard = benchmark_scores.pivot(index="Model", columns="Benchmark", values="Score")
    
    # Add an Average column
    leaderboard["Average"] = leaderboard.mean(axis=1)
    
    # Sort by Average
    leaderboard = leaderboard.sort_values("Average", ascending=False)
    
    # Save
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{config.RESULTS_DIR}/leaderboard_{timestamp}.csv"
    leaderboard.to_csv(output_path)
    print(f"Leaderboard saved to {output_path}")
    print(leaderboard)
