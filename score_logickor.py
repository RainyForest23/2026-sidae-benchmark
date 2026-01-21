#!/usr/bin/env python3
"""
LogicKor LLM Judge Scoring Script

Uses GPT-5.2 as judge to score LogicKor responses on a 1-5 scale.
Based on the original LogicKor evaluation methodology.
"""

import os
import re
import json
import time
import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Judge model configuration (using GPT-5.2 via mlapi.run)
JUDGE_UUID = "664ce153-d45c-42a7-903c-d9119cc55b69"
JUDGE_MODEL = "openai/gpt-5.2"
API_KEY = os.getenv("ELICE_API_KEY")

# LogicKor categories
CATEGORIES = {
    "추론(Reasoning)": "reasoning",
    "수학(Math)": "math",
    "글쓰기(Writing)": "writing",
    "코딩(Coding)": "coding",
    "이해(Comprehension)": "comprehension",
    "문법(Grammar)": "grammar"
}

# Judge prompt template
JUDGE_PROMPT = """당신은 한국어 AI 응답의 품질을 평가하는 전문가입니다.

## 평가 기준
다음 기준에 따라 응답을 1-5점으로 평가해주세요:

**5점 (탁월함)**: 질문의 모든 측면에 완벽하게 답변. 논리적이고 창의적이며 깊이 있는 분석 제공.
**4점 (우수함)**: 질문에 잘 답변했으나 약간의 개선 여지가 있음. 대체로 정확하고 유용함.
**3점 (보통)**: 기본적인 답변을 제공하나 깊이가 부족하거나 일부 누락이 있음.
**2점 (미흡함)**: 질문에 부분적으로만 답변하거나 상당한 오류가 있음.
**1점 (부적절함)**: 질문과 관련 없거나 완전히 잘못된 답변.

## 질문
{question}

## AI의 응답
{response}

## 평가 지시
위 응답을 평가하고, 반드시 다음 형식으로만 답변하세요:
점수: [1-5 중 숫자 하나]
이유: [한 문장으로 간단한 평가 이유]
"""

def call_judge_api(question: str, response: str) -> dict:
    """Call the judge model to score a response."""
    url = f"https://mlapi.run/{JUDGE_UUID}/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = JUDGE_PROMPT.format(question=question, response=response)

    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_completion_tokens": 200,
        "temperature": 0.1  # Low temperature for consistent scoring
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()

        if "choices" in result:
            content = result["choices"][0]["message"].get("content", "")
            return {"success": True, "content": content}
        return {"success": False, "error": "No choices in response"}

    except Exception as e:
        return {"success": False, "error": str(e)}


def extract_score(judge_response: str) -> int:
    """Extract score (1-5) from judge response."""
    if not judge_response:
        return None

    patterns = [
        r'점수[:\s]*([1-5])',
        r'([1-5])\s*점',
        r'^([1-5])[\.\s]',
    ]

    for pattern in patterns:
        match = re.search(pattern, judge_response)
        if match:
            return int(match.group(1))

    return None


def score_logickor_file(filepath: str, output_path: str = None):
    """Score all responses in a LogicKor result file."""
    print(f"\nScoring: {filepath}")

    df = pd.read_csv(filepath)

    # Filter valid predictions
    valid_mask = df['prediction'].notna() & ~df['prediction'].str.contains('Error:', na=False)
    valid_df = df[valid_mask].copy()

    print(f"Valid responses: {len(valid_df)}/{len(df)}")

    if len(valid_df) == 0:
        print("No valid responses to score.")
        return None

    scores = []
    reasons = []

    for idx, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Scoring"):
        question = row['prompt']
        response = row['prediction']

        # Call judge
        result = call_judge_api(question, response)

        if result['success']:
            score = extract_score(result['content'])
            scores.append(score)
            reasons.append(result['content'])
        else:
            scores.append(None)
            reasons.append(f"Error: {result['error']}")

        # Rate limiting
        time.sleep(1)

    # Add scores to dataframe
    valid_df['judge_score'] = scores
    valid_df['judge_reason'] = reasons

    # Calculate statistics
    valid_scores = [s for s in scores if s is not None]

    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        print(f"\nResults:")
        print(f"  Average Score: {avg_score:.2f}/5.00")
        print(f"  Scored: {len(valid_scores)}/{len(valid_df)}")

        # Score distribution
        dist = {i: valid_scores.count(i) for i in range(1, 6)}
        print(f"  Distribution: {dist}")

    # Save results
    if output_path is None:
        output_path = filepath.replace('.csv', '_scored.csv')

    valid_df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

    return {
        'file': filepath,
        'total': len(df),
        'valid': len(valid_df),
        'scored': len(valid_scores),
        'avg_score': avg_score if valid_scores else None,
        'distribution': dist if valid_scores else None
    }


def main():
    """Score all LogicKor result files."""
    print("=" * 60)
    print("LogicKor LLM Judge Scoring")
    print("=" * 60)

    # Find LogicKor files
    result_files = [
        'results/20260118_175403_helpy-pro_logickor.csv',
        'results/20260118_175403_gpt-oss-20b_logickor.csv',
        'results/20260118_175403_gpt-5.2_logickor.csv',
    ]

    all_results = []

    for filepath in result_files:
        if os.path.exists(filepath):
            result = score_logickor_file(filepath)
            if result:
                all_results.append(result)
        else:
            print(f"File not found: {filepath}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for r in all_results:
        model = r['file'].split('_')[-2]
        if r['avg_score']:
            print(f"{model}: {r['avg_score']:.2f}/5.00 ({r['scored']} scored)")
        else:
            print(f"{model}: No valid scores")


if __name__ == "__main__":
    main()
