import os
import glob
import pandas as pd

def analyze_results():
    results_dir = "results"
    files = glob.glob(os.path.join(results_dir, "*.csv"))
    
    data = []
    
    print(f"Found {len(files)} result files.")
    
    for file in files:
        try:
            # Filename format: YYYYMMDD_HHMMSS_model_benchmark_task.csv
            # But model names can contain hyphens/underscores.
            # We know the specific naming from runner.py: 
            # f"{timestamp}_{model.model_name}_{benchmark_prefix}_{task}.csv"
            # It's safer to just split by known models or known benchmarks?
            # Or just assume standard format: timestamp (2 parts) + rest.
            
            basename = os.path.basename(file)
            parts = basename.replace(".csv", "").split("_")
            
            # parts[0]: date, parts[1]: time
            timestamp = f"{parts[0]}_{parts[1]}"
            
            # The rest is model_benchmark_task.
            # Models: helpy-pro, gpt-oss-20b, gpt-5.2 (some have hyphens)
            # Benchmarks: kobest, kmmlu
            # Tasks: boolq, copa, Accounting, etc.
            
            # Let's rely on the file content or reconstruct. 
            # Actually, the 'results' in runner.py stores 'model', 'benchmark', 'task'.
            # Reading the CSV is the most reliable way.
            
            df = pd.read_csv(file)
            if df.empty:
                continue
                
            model = df.iloc[0]['model']
            benchmark = df.iloc[0]['benchmark']
            task = df.iloc[0]['task']
            
            # Calculate Score (Accuracy) with Regex Extraction
            preds = df['prediction'].fillna("").astype(str)
            refs = df['reference'].fillna("").astype(str)
            
            import re
            
            def extract_answer(text, task_name):
                # Normalize
                text = text.strip()
                if not text: return ""
                
                # KoBEST BoolQ (0/1) - strictly look for specific keywords if needed, 
                # but stats showed it was already working well (0.97). 
                # Just keep exact match fallback if simple extraction fails.
                
                # KMMLU / Multiple Choice (1,2,3,4,5 or A,B,C,D,E)
                # Patterns observed: "정답: **2**", "정답은 2번", "A", "2"
                
                # 1. Explicit patterns with "Answer" or "정답"
                match = re.search(r'(?:정답|Answer|답)[:\s]*(?:\*\*|\[)?([1-5A-E])(?:\*\*|\])?', text, re.IGNORECASE)
                if match:
                    val = match.group(1).upper()
                    # Convert number to letter if ref is letter? Or vice versa?
                    # KMMLU refs seem to be 1,2,3,4 (based on debug log).
                    return val

                # 2. "XX번" pattern (Korean "Number XX")
                match = re.search(r'([1-5])\s*번', text)
                if match:
                    return match.group(1)

                # 3. Simple start anchor (most likely if it just spits out the answer)
                match = re.match(r'^([1-5A-E])(?:\.|\)|:|$)', text, re.IGNORECASE)
                if match:
                    return match.group(1).upper()
                
                # 4. Yes/No/True/False (for WiC/BoolQ)
                lower_text = text.lower()
                if lower_text.startswith("yes") or lower_text.startswith("true") or "정답: 예" in text or "정답: 참" in text:
                    return "1"
                if lower_text.startswith("no") or lower_text.startswith("false") or "정답: 아니오" in text or "정답: 거짓" in text:
                    return "0"

                # Fallback: returns original text (for BoolQ which seemed to work match-wise)
                return text

            # Apply extraction
            cleaned_preds = [extract_answer(p, task) for p in preds]
            
            # Additional normalization for comparison (e.g. A vs 1, 1-based vs 0-based)
            def normalize_match(pred, ref, task_name):
                # map A->1, B->2, etc if needed.
                mapping = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5', 'TRUE': '1', 'FALSE': '0', 'YES': '1', 'NO': '0'}
                pred_norm = mapping.get(pred.upper(), pred)
                ref_norm = mapping.get(str(ref).upper(), str(ref))
                
                # Check for 0-index vs 1-index mismatch
                # KoBest HellaSwag & COPA are often 0-indexed refs (0,1,2,3)
                # But models output 1,2,3,4
                if "hellaswag" in task_name or "copa" in task_name:
                    try:
                        # If Pred is 1, Ref is 0 -> Match
                        p_int = int(pred_norm)
                        r_int = int(ref_norm)
                        if p_int - 1 == r_int:
                            return True
                    except:
                        pass
                
                return pred_norm == ref_norm

            correct = sum(normalize_match(p, r, task) for p, r in zip(cleaned_preds, refs))
            total = len(df)
            score = correct / total if total > 0 else 0
            
            data.append({
                "Model": model,
                "Benchmark": benchmark,
                "Task": task,
                "Score": score,
                "Samples": total
            })
            
        except Exception as e:
            print(f"Error processing {file}: {e}")

    if not data:
        print("No results found.")
        return

    df_results = pd.DataFrame(data)
    
    # Pivot table for better view
    # Index: Task, Columns: Model, Values: Score
    pivot = df_results.pivot_table(index=["Benchmark", "Task"], columns="Model", values="Score")
    
    print("\n=== Benchmark Results Summary ===")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(pivot)
    
    # Debug: Check one failing case
    print("\n=== DEBUG: Inspecting first 5 rows of a failed task ===")
    target_debug = "wic"
    
    # Re-scan for debug
    debug_files = [f for f in files if target_debug in f]
    if debug_files:
        debug_file = debug_files[0]
        print(f"Inspecting {debug_file}...")
        df_debug = pd.read_csv(debug_file)
        print(df_debug[['prediction', 'reference']].head(5))

if __name__ == "__main__":
    analyze_results()
