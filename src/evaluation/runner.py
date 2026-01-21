
import os
import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import config
from src.models import HelpyProModel, HelpyEduModel, MLApiModel, OpenAIModel
from src.benchmarks import load_kobest, load_kmmlu, load_haerae, load_logickor
from src.evaluation import metrics, prompts

def get_model(model_name):
    if model_name == "helpy-pro":
        return HelpyProModel(model_name)
    elif model_name == "helpy-edu":
        return HelpyEduModel(model_name)
    elif model_name in ["gpt-oss-20b", "gpt-5.2"]:
        return MLApiModel(model_name)
    elif "gemini" in model_name:
        raise ValueError(f"Gemini models are temporarily disabled: {model_name}")
    else:
        # Fallback to OpenAI for unknown models
        return OpenAIModel(model_name)

def evaluate_task(model, task_name, dataset, prompt_func, metric_func=metrics.calculate_accuracy):
    results = []
    references = []
    predictions = []
    
    print(f"Evaluating {model.model_name} on {task_name} ({len(dataset)} samples)...")
    
    for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
        prompt = prompt_func(sample)
        prediction = model.generate(prompt)
        
        # Determine reference (ground truth)
        # This varies by dataset. 
        # KoBEST: label (0/1 or index)
        # KMMLU: answer (A/B/C/D)
        # Needs specific handling or consistent dataset formatting.
        
        reference = str(sample.get("label", sample.get("answer", "")))
        
        # Post-process prediction for metric
        # e.g. extract "A" from "The answer is A"
        # For numeric labels (KoBEST), we might need to map prediction to 0/1.
        
        # For simplicity in this first pass, we store raw and let metric handle or refine later.
        predictions.append(prediction)
        references.append(reference)
        
        if metric_func:
            # We assume metric_func calculates based on lists, but here we might want per-item scoring?
            # For simplicity, we calculate global score later.
            pass

        results.append({
            "model": model.model_name,
            "benchmark": task_name.split("_")[0], # kobest, kmmlu, etc.
            "task": task_name,
            "prompt": prompt,
            "prediction": prediction,
            "reference": reference,
            "full_sample": str(sample)
        })
        
        # Rate limiting or sleep could be added here
        
    # score = metric_func(predictions, references) # Deprecated
    print(f"Score: {score:.4f}")
    
    return results, score

from src.benchmarks.kmmlu import CATEGORIES as KMMLU_CATEGORIES
from src.evaluation.prompts import format_logickor, format_haerae

def run_evaluation():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Initialize models
    models = []
    for model_name in config.ENABLED_MODELS:
        try:
            models.append(get_model(model_name))
        except Exception as e:
            print(f"Failed to initialize {model_name}: {e}")
            
    if not models:
        print("No models initialized. Exiting.")
        return

    # Run Benchmarks
    if "kobest" in config.ENABLED_BENCHMARKS:
        for task in ['boolq', 'copa', 'hellaswag', 'sentineg', 'wic']:
            try:
                ds = load_kobest(task, config.SAMPLE_SIZE)['test']
                prompt_func = getattr(prompts, f"format_kobest_{task}")
                
                for model in models:
                    results, score = evaluate_task(model, f"kobest_{task}", ds, prompt_func)
                    
                    df = pd.DataFrame(results)
                    output_path = f"{config.RESULTS_DIR}/{timestamp}_{model.model_name}_kobest_{task}.csv"
                    df.to_csv(output_path, index=False)
                    print(f"Saved results to {output_path}")
            except Exception as e:
                print(f"Error evaluating kobest_{task}: {e}")

    if "kmmlu" in config.ENABLED_BENCHMARKS:
        # Evaluate all or subset based on config? For now, iterate all if not specified otherwise.
        # But for brevity in this run, we might want to limit.
        # Let's use the full list but catch errors.
        for cat in KMMLU_CATEGORIES:
            try:
                ds = load_kmmlu(cat, config.SAMPLE_SIZE)['test']
                for model in models:
                    results, score = evaluate_task(model, f"kmmlu_{cat}", ds, prompts.format_kmmlu)
                    
                    df = pd.DataFrame(results)
                    output_path = f"{config.RESULTS_DIR}/{timestamp}_{model.model_name}_kmmlu_{cat}.csv"
                    df.to_csv(output_path, index=False)
            except Exception as e:
                print(f"Error evaluating kmmlu_{cat}: {e}")

    if "logickor" in config.ENABLED_BENCHMARKS:
        try:
             ds = load_logickor(config.SAMPLE_SIZE)['train'] # LogicKor uses train usually as it's small
             for model in models:
                # LogicKor needs special metric or judge. For now just generating answers.
                # Metric func can be dummy or None.
                results, score = evaluate_task(model, "logickor", ds, format_logickor, metric_func=lambda p, r: 0.0)
                
                df = pd.DataFrame(results)
                output_path = f"{config.RESULTS_DIR}/{timestamp}_{model.model_name}_logickor.csv"
                df.to_csv(output_path, index=False)
        except Exception as e:
            print(f"Error evaluating logickor: {e}")

    if "haerae" in config.ENABLED_BENCHMARKS:
        tasks = [
            'correct_definition_matching', 'csat_geo', 'csat_law', 'csat_socio', 
            'date_understanding', 'general_knowledge', 'history', 'loan_words', 
            'lyrics_denoising', 'proverbs_denoising', 'rare_words', 
            'standard_nomenclature', 'reading_comprehension'
        ]
        for task in tasks:
            try:
                ds = load_haerae(task, config.SAMPLE_SIZE).get('test', load_haerae(task, config.SAMPLE_SIZE).get('train'))
                # HAE-RAE structure check needed. Using generic prompt.
                for model in models:
                    results, score = evaluate_task(model, f"haerae_{task}", ds, format_haerae)
                    
                    df = pd.DataFrame(results)
                    output_path = f"{config.RESULTS_DIR}/{timestamp}_{model.model_name}_haerae_{task}.csv"
                    df.to_csv(output_path, index=False)
            except Exception as e:
                print(f"Error evaluating haerae_{task}: {e}")
    
    print("Evaluation complete.")

if __name__ == "__main__":
    run_evaluation()
