# Korean LLM Benchmark Project

## Overview
This project evaluates the performance of 6 Large Language Models (LLMs) on 4 standard Korean benchmarks.
The goal is to analyze their capabilities in Korean language understanding, reasoning, and domain-specific knowledge.

## Models Evaluated
- **Elice**: Helpy Pro, Helpy Edu
- **OpenAI**: gpt-oss-20b (via Elice), GPT 5.2
- **Google**: Gemini 3 Pro, Gemini 3 Flash

## Benchmarks
1. **KoBEST**: Korean Balanced Evaluation of Significant Tasks
2. **KMMLU**: Korean Massive Multitask Language Understanding
3. **HAE-RAE Bench**: Korean cultural and linguistic benchmarks
4. **LogicKor**: Logic and reasoning tasks

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. specific API keys in `.env` (copy from `.env.template`)
3. Run evaluation: `python main.py`
