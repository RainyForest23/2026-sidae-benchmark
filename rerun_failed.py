#!/usr/bin/env python3
"""
재평가 스크립트 - 실패한 벤치마크만 다시 실행
"""

import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

import config
from src.models import HelpyProModel, MLApiModel
from src.benchmarks import load_kmmlu, load_haerae, load_logickor
from src.evaluation import prompts, metrics
from src.evaluation.prompts import format_logickor, format_haerae

# 실패한 카테고리/태스크 목록
FAILED_KMMLU = [
    'Criminal-Law', 'Ecology', 'Economics', 'Education', 'Electrical-Engineering',
    'Electronics-Engineering', 'Energy-Management', 'Environmental-Science',
    'Fashion', 'Food-Processing', 'Industrial-Engineer', 'Information-Technology',
    'Korean-History', 'Law', 'Management', 'Maritime-Engineering', 'Marketing',
    'Materials-Engineering', 'Math', 'Mechanical-Engineering', 'Nondestructive-Testing',
    'Patent', 'Political-Science-and-Sociology', 'Psychology', 'Public-Safety',
    'Railway-and-Automotive-Engineering', 'Real-Estate', 'Refrigerating-Machinery',
    'Social-Welfare', 'Taxation', 'Telecommunications-and-Wireless-Technology'
]

FAILED_HAERAE = [
    'correct_definition_matching', 'csat_geo', 'csat_law', 'csat_socio',
    'date_understanding', 'general_knowledge', 'history', 'loan_words',
    'lyrics_denoising', 'proverbs_denoising', 'rare_words',
    'standard_nomenclature', 'reading_comprehension'
]


def get_model(model_name):
    if model_name == "helpy-pro":
        return HelpyProModel(model_name)
    elif model_name in ["gpt-oss-20b", "gpt-5.2"]:
        return MLApiModel(model_name)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def evaluate_task(model, task_name, dataset, prompt_func):
    results = []
    predictions = []
    references = []

    print(f"Evaluating {model.model_name} on {task_name} ({len(dataset)} samples)...")

    for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
        prompt = prompt_func(sample)
        prediction = model.generate(prompt)

        reference = str(sample.get("label", sample.get("answer", "")))

        predictions.append(prediction)
        references.append(reference)

        results.append({
            "model": model.model_name,
            "benchmark": task_name.split("_")[0],
            "task": task_name,
            "prompt": prompt,
            "prediction": prediction,
            "reference": reference,
            "full_sample": str(sample)
        })

    return results


def run_failed_evaluations():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # 모델 초기화
    models = []
    for model_name in config.ENABLED_MODELS:
        try:
            models.append(get_model(model_name))
            print(f"✅ {model_name} initialized")
        except Exception as e:
            print(f"❌ Failed to initialize {model_name}: {e}")

    if not models:
        print("No models initialized. Exiting.")
        return

    # 1. KMMLU 재평가
    print("\n" + "="*50)
    print("KMMLU 재평가 시작")
    print("="*50)

    for cat in FAILED_KMMLU:
        print(f"\n[KMMLU] {cat}")
        try:
            ds = load_kmmlu(cat, config.SAMPLE_SIZE)['test']
            for model in models:
                results = evaluate_task(model, f"kmmlu_{cat}", ds, prompts.format_kmmlu)

                df = pd.DataFrame(results)
                output_path = f"{config.RESULTS_DIR}/{timestamp}_{model.model_name}_kmmlu_{cat}.csv"
                df.to_csv(output_path, index=False)
                print(f"  Saved: {output_path}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    # 2. HAE-RAE 재평가
    print("\n" + "="*50)
    print("HAE-RAE 재평가 시작")
    print("="*50)

    for task in FAILED_HAERAE:
        print(f"\n[HAE-RAE] {task}")
        try:
            ds_loaded = load_haerae(task, config.SAMPLE_SIZE)
            ds = ds_loaded.get('test', ds_loaded.get('train'))

            for model in models:
                results = evaluate_task(model, f"haerae_{task}", ds, format_haerae)

                df = pd.DataFrame(results)
                output_path = f"{config.RESULTS_DIR}/{timestamp}_{model.model_name}_haerae_{task}.csv"
                df.to_csv(output_path, index=False)
                print(f"  Saved: {output_path}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    # 3. LogicKor 재평가
    print("\n" + "="*50)
    print("LogicKor 재평가 시작")
    print("="*50)

    try:
        ds = load_logickor(config.SAMPLE_SIZE)['train']
        for model in models:
            results = evaluate_task(model, "logickor", ds, format_logickor)

            df = pd.DataFrame(results)
            output_path = f"{config.RESULTS_DIR}/{timestamp}_{model.model_name}_logickor.csv"
            df.to_csv(output_path, index=False)
            print(f"Saved: {output_path}")
    except Exception as e:
        print(f"❌ LogicKor Error: {e}")

    print("\n" + "="*50)
    print("재평가 완료!")
    print("="*50)


if __name__ == "__main__":
    run_failed_evaluations()
