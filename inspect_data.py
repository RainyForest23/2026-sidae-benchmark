import os
import json
from datasets import load_dataset
import pandas as pd

# Define benchmark mapping
BENCHMARKS = {
    "kobest": "skt/kobest_v1",
    "kmmlu": "HAERAE-HUB/KMMLU",
    "haerae": "HAERAE-HUB/HAE_RAE_BENCH_1.1",
    "logickor": "maywell/LogicKor"
}

OUTPUT_DIR = "data/samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def inspect_kobest():
    print(f"Inspecting KoBEST ({BENCHMARKS['kobest']})...")
    # KoBEST has sub-tasks/configs: boolq, copa, hellaswag, sentineg, wic
    tasks = ['boolq', 'copa', 'hellaswag', 'sentineg', 'wic']
    
    for task in tasks:
        try:
            ds = load_dataset(BENCHMARKS['kobest'], task) # stop_on_empty=False just in case
            print(f"  - {task}: {ds}")
            if 'test' in ds:
                sample = ds['test'][0]
                with open(f"{OUTPUT_DIR}/kobest_{task}_sample.json", "w", encoding="utf-8") as f:
                    json.dump(sample, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  ! Failed to load {task}: {e}")

def inspect_kmmlu():
    print(f"Inspecting KMMLU ({BENCHMARKS['kmmlu']})...")
    # KMMLU usually has many subsets
    try:
        # Load list of configs if possible, or just try one
        ds = load_dataset(BENCHMARKS['kmmlu'], "Accounting") # Try one subset
        print(f"  - Accounting: {ds}")
        if 'test' in ds:
            sample = ds['test'][0]
            with open(f"{OUTPUT_DIR}/kmmlu_sample.json", "w", encoding="utf-8") as f:
                json.dump(sample, f, ensure_ascii=False, indent=2)
        
        # Get all configs names if possible - logic might vary
        # For now, just confirming it loads
    except Exception as e:
        print(f"  ! Failed to load KMMLU: {e}")

def inspect_haerae():
    print(f"Inspecting HAE-RAE ({BENCHMARKS['haerae']})...")
    try:
        ds = load_dataset(BENCHMARKS['haerae'])
        print(f"  - Default: {ds}")
        # Check splits
        for split in ds.keys():
            sample = ds[split][0]
            with open(f"{OUTPUT_DIR}/haerae_{split}_sample.json", "w", encoding="utf-8") as f:
                json.dump(sample, f, ensure_ascii=False, indent=2)
            break 
    except Exception as e:
        print(f"  ! Failed to load HAE-RAE: {e}")

def inspect_logickor():
    print(f"Inspecting LogicKor ({BENCHMARKS['logickor']})...")
    try:
        ds = load_dataset(BENCHMARKS['logickor'])
        print(f"  - Default: {ds}")
        for split in ds.keys():
            sample = ds[split][0]
            with open(f"{OUTPUT_DIR}/logickor_{split}_sample.json", "w", encoding="utf-8") as f:
                json.dump(sample, f, ensure_ascii=False, indent=2)
            break
    except Exception as e:
        print(f"  ! Failed to load LogicKor: {e}")

def main():
    inspect_kobest()
    inspect_kmmlu()
    inspect_haerae()
    inspect_logickor()
    print(f"\nInspection complete. Samples saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
