#!/usr/bin/env python3
"""
LLMIP Financial Pipeline - Simple Version
Uses OpenRouter API with z-ai/glm-4.7
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
load_dotenv("/home/l1zle/LLMIP/.env")
import requests
import time

PROJECT_DIR = Path("/home/l1zle/LLMIP")
FIN_DIR = PROJECT_DIR / "data" / "financial"
RESULTS_DIR = PROJECT_DIR / "results"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = "z-ai/glm-4.7"
BASE_URL = "https://openrouter.ai/api/v1"

def call_llm(prompt, max_tokens=500, temperature=0.3):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://llmip.research",
        "X-Title": "LLMIP Research"
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    try:
        resp = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload, timeout=120)
        if resp.status_code == 200:
            result = resp.json()
            msg = result["choices"][0]["message"]
            return msg.get("content") or msg.get("reasoning") or ""
        print(f"Error: {resp.status_code}")
    except Exception as e:
        print(f"Exception: {e}")
    return None

def extract_direction(text):
    """Extract 0 or 1 from response."""
    if not text:
        return None
    if '1' in text and text.index('1') < text.index('0') if '0' in text else True:
        return 1
    elif '0' in text:
        return 0
    return None

# Load data
print("Loading data...")
train = pd.read_csv(FIN_DIR / "sp500_train.csv")
test = pd.read_csv(FIN_DIR / "sp500_test.csv")
train_targets = train['target_direction']
test_targets = test['target_direction']
exclude = ['target_direction', 'next_return', 'date']
feature_cols = [c for c in train.columns if c not in exclude]
train_features = train[feature_cols]
test_features = test[feature_cols]

print(f"Train: {train_features.shape}, Test: {test_features.shape}")

# Phase 1: Training Analysis
print("\n" + "="*60)
print("PHASE 1: TRAINING ANALYSIS")
print("="*60)

n_train_samples = 10
subset = train_features.iloc[:n_train_samples]
target_subset = train_targets.iloc[:n_train_samples]

prompt = f"""Analyze market data to predict next-day return direction (0=DOWN, 1=UP).

Samples (features + target):
"""

for i in range(n_train_samples):
    row = train_features.iloc[i]
    t = train_targets.iloc[i]
    prompt += f"\n{i+1}. rsi={row['rsi']:.1f}, ret1d={row['return_1d']:.3f}, vol5={row['volatility_5d']:.4f}, bb_pos={row['bb_position']:.2f} -> {int(t)}\n"

prompt += "\nGive me 3 rules to predict direction. Format: IF [condition] THEN [0 or 1]"

print("Analyzing training data...")
analysis = call_llm(prompt, max_tokens=400)

if analysis:
    with open(RESULTS_DIR / "financial_phase1_analysis.txt", "w") as f:
        f.write(analysis)
    print(f"Phase 1 complete! ({len(analysis)} chars)")
    print(analysis[:300])
else:
    print("Phase 1 failed!")
    analysis = "No analysis available."

# Phase 2: Predictions
print("\n" + "="*60)
print("PHASE 2: PREDICTIONS")
print("="*60)

n_test = 10
predictions = {}

for i in range(n_test):
    row = test_features.iloc[i]
    
    prompt = f"""Predict next-day market direction (0=DOWN, 1=UP).

Features:
- rsi: {row['rsi']:.1f}
- return_1d: {row['return_1d']:.4f}
- volatility_5d: {row['volatility_5d']:.4f}
- bb_position: {row['bb_position']:.2f}
- return_5d: {row['return_5d']:.4f}
- momentum: {row['momentum']:.4f}

Output ONLY the number 0 or 1."""

    resp = call_llm(prompt, max_tokens=10, temperature=0.0)
    pred = extract_direction(resp) if resp else None
    
    actual = int(test_targets.iloc[i])
    correct = (pred == actual) if pred is not None else False
    
    predictions[str(i)] = {"prediction": pred, "actual": actual, "raw": resp or ""}
    
    print(f"  Sample {i}: pred={pred}, actual={actual}, {'OK' if correct else 'WRONG'}")
    
    time.sleep(1)

# Calculate accuracy
correct_count = sum(1 for p in predictions.values() if p["prediction"] == p["actual"])
print(f"\nOriginal LLM Accuracy: {correct_count}/{n_test} = {correct_count/n_test:.0%}")

with open(RESULTS_DIR / "financial_phase2_predictions.json", "w") as f:
    json.dump(predictions, f, indent=2)

# Phase 3: Rulebook
print("\n" + "="*60)
print("PHASE 3: RULEBOOK EXTRACTION")
print("="*60)

comparison = "Predictions vs Actuals:\n"
for i, p in predictions.items():
    comparison += f"Sample {i}: predicted={p['prediction']}, actual={p['actual']}\n"

prompt = f"""Extract decision rules from these predictions.

{comparison}

Give me 4 simple IF-THEN rules:
Format: Rule: IF [condition] THEN [0 or 1]

Keep it simple and actionable."""

rulebook = call_llm(prompt, max_tokens=400)

if rulebook:
    with open(RESULTS_DIR / "financial_phase3_rulebook.txt", "w") as f:
        f.write(rulebook)
    print(f"Phase 3 complete! ({len(rulebook)} chars)")
    print(rulebook[:400])
else:
    print("Phase 3 failed!")
    rulebook = "No rulebook available."

# Phase 4: Replicability
print("\n" + "="*60)
print("PHASE 4: REPLICABILITY TEST")
print("="*60)

replicability_results = {"samples": [], "domain": "financial"}

for i in range(min(10, n_test)):
    row = test_features.iloc[i]
    
    prompt = f"""Apply these rules to predict market direction.

Rules:
{rulebook}

Features:
- rsi: {row['rsi']:.1f}
- return_1d: {row['return_1d']:.4f}
- volatility_5d: {row['volatility_5d']:.4f}

Output ONLY the number 0 or 1."""

    resp = call_llm(prompt, max_tokens=10, temperature=0.0)
    fresh_pred = extract_direction(resp) if resp else None
    
    orig_pred = predictions[str(i)]["prediction"]
    actual = predictions[str(i)]["actual"]
    
    replicability_results["samples"].append({
        "index": i,
        "original": orig_pred,
        "fresh": fresh_pred,
        "actual": actual,
        "original_correct": orig_pred == actual if orig_pred is not None else None,
        "fresh_correct": fresh_pred == actual if fresh_pred is not None else None
    })
    
    print(f"  Sample {i}: orig={orig_pred}, fresh={fresh_pred}, actual={actual}")
    
    time.sleep(1)

# Calculate scores
n = len(replicability_results["samples"])
orig_correct = sum(1 for s in replicability_results["samples"] if s["original_correct"])
fresh_correct = sum(1 for s in replicability_results["samples"] if s["fresh_correct"])

replicability_results["original_accuracy"] = orig_correct / n
replicability_results["replicability_score"] = fresh_correct / n
replicability_results["n_samples"] = n

print(f"\nOriginal LLM Accuracy: {orig_correct}/{n} = {orig_correct/n:.0%}")
print(f"Fresh LLM Accuracy (Replicability Score): {fresh_correct}/{n} = {fresh_correct/n:.0%}")

with open(RESULTS_DIR / "financial_replicability_test.json", "w") as f:
    json.dump(replicability_results, f, indent=2)

print("\n" + "="*60)
print("FINANCIAL PIPELINE COMPLETE")
print("="*60)
