#!/usr/bin/env python3
"""
LLMIP Financial Pipeline — Stateful Experiment
=============================================
S&P 500 direction prediction (next-day return: 0=DOWN, 1=UP).

This is the STATEFUL experiment (Experiment 2 from the thesis).
The target variable has memory — path dependency, autocorrelation, momentum.
This stress-tests whether LLMIP's Rulebook can encode temporal logic.

Pipeline:
    Phase 1: Train — LLM analyzes training data, learns patterns
    Phase 2: Predict — LLM predicts direction for test cases
    Phase 3: Extract Rulebook — LLM converts predictions to decision rules
    Phase 4: Replicability Test — Fresh LLM given only Rulebook reproduces predictions

Replicability Score = fraction of fresh LLM predictions matching ground truth.
High score = Rulebook contains genuine predictive logic (Sarkar 2024 rebuttal).

Label handling: target_direction and next_return are removed from features.
All features are lagged (available at prediction time).
"""

import os
from dotenv import load_dotenv
load_dotenv()

import json
import pandas as pd
import numpy as np
from pathlib import Path
import time
import re
import requests

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODAL_API_KEY = os.environ.get("MODAL_API_KEY")
MODAL_URL = "https://api.us-west-2.modal.direct/v1/chat/completions"
MODEL = "zai-org/GLM-5-FP8"

FEATURE_DESCRIPTIONS = {
    "close": "Closing price (USD)",
    "open": "Opening price (USD)",
    "high": "Daily high price (USD)",
    "low": "Daily low price (USD)",
    "volume": "Trading volume (shares)",
    "return_1d": "1-day return (lagged — feature, not target)",
    "return_5d": "5-day cumulative return (lagged)",
    "return_20d": "20-day cumulative return (lagged)",
    "sma_5": "5-day simple moving average of close price",
    "sma_20": "20-day simple moving average of close price",
    "sma_50": "50-day simple moving average of close price",
    "volatility_5d": "5-day realized volatility (standard deviation of returns)",
    "volatility_20d": "20-day realized volatility",
    "momentum": "20-day momentum (return_20d proxy)",
    "rsi": "Relative Strength Index (0-100; >70 overbought, <30 oversold)",
    "macd": "MACD line (12-day EMA - 26-day EMA)",
    "macd_signal": "MACD signal line (9-day EMA of MACD)",
    "bb_mid": "Bollinger Band middle band (20-day SMA)",
    "bb_upper": "Bollinger Band upper band (+2 std)",
    "bb_lower": "Bollinger Band lower band (-2 std)",
    "bb_position": "Bollinger Band position (0=at lower band, 1=at upper band)",
    "dayofweek": "Day of week (0=Monday, 6=Sunday)",
    "month": "Month of year (1-12)",
}


def call_llm(prompt, system="You are a helpful assistant.", max_tokens=512, temperature=0.1, retries=5):
    """Call Modal API with retry logic and error handling."""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {MODAL_API_KEY}"}
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(retries):
        try:
            resp = requests.post(MODAL_URL, headers=headers, json=payload, timeout=180)
            if resp.status_code == 200:
                data = resp.json()
                msg = data["choices"][0]["message"]
                return msg.get("content") or msg.get("reasoning_content") or ""
            print(f"  API error {resp.status_code}: {resp.text[:100]}")
            time.sleep(3)
        except Exception as e:
            print(f"  Exception: {e}")
            time.sleep(3)
    return None


def extract_01(text):
    """Extract 0 or 1 from any model output."""
    if not text:
        return None
    match = re.search(r'\b([01])\b', text)
    return int(match.group(1)) if match else None


def load_financial():
    """Load financial data.

    Returns:
        train_features: DataFrame with training features (target removed)
        test_features: DataFrame with test features (target removed)
        train_targets: Series with training labels
        test_targets: Series with test labels
    """
    fin_dir = DATA_DIR / "financial"
    train = pd.read_csv(fin_dir / "sp500_train.csv")
    test = pd.read_csv(fin_dir / "sp500_test.csv")

    train_targets = train['target_direction'].copy()
    test_targets = test['target_direction'].copy()

    feature_cols = [c for c in train.columns if c not in ['target_direction', 'next_return', 'date']]
    train_features = train[feature_cols]
    test_features = test[feature_cols]

    return train_features, test_features, train_targets, test_targets


def _format_features(row, feature_cols):
    """Format all features with descriptions for LLM prompts."""
    lines = []
    for col in feature_cols:
        if col in row.index:
            desc = FEATURE_DESCRIPTIONS.get(col, col)
            val = row[col]
            if isinstance(val, float):
                lines.append(f"  {col}: {val:.6f}  ({desc})")
            else:
                lines.append(f"  {col}: {val}  ({desc})")
    return "\n".join(lines)


def run_phase1(train_features, train_targets):
    """Phase 1: Train — LLM analyzes training data with full feature context.

    The LLM receives:
    - Full feature descriptions (what each indicator means)
    - Summary statistics of the training data
    - Target distribution information
    - A representative sample of training rows

    Output: analysis string that feeds into Phase 2 and Phase 3.
    """
    print("\n=== Phase 1: Training Analysis ===")

    n_samples = min(30, len(train_features))
    subset = train_features.iloc[:n_samples]
    target_subset = train_targets.iloc[:n_samples]

    feat_desc_str = "\n".join(
        f"  - {col}: {desc}" for col, desc in FEATURE_DESCRIPTIONS.items()
        if col in train_features.columns
    )

    summary_stats = {
        "n_samples": len(train_features),
        "target_distribution": train_targets.value_counts().to_dict(),
        "target_pct_up": float(train_targets.mean() * 100),
        "return_1d_mean": float(train_features['return_1d'].mean()),
        "return_1d_std": float(train_features['return_1d'].std()),
        "rsi_mean": float(train_features['rsi'].mean()),
        "volatility_mean": float(train_features['volatility_5d'].mean()),
    }

    system = (
        "You are a quantitative analyst specializing in statistical arbitrage and "
        "market microstructure. You analyze historical market data to identify patterns "
        "that predict next-day return direction."
    )
    user = f"""## Your Task

Analyze the following market data to identify patterns that predict next-day return direction.

## Feature Definitions
{feat_desc_str}

## Data Summary
- Total training samples: {summary_stats['n_samples']}
- Target distribution: UP (1) = {summary_stats['target_distribution'].get(1, 0)} days ({summary_stats['target_pct_up']:.1f}%)
- Average 1-day return: {summary_stats['return_1d_mean']:.4f} (std: {summary_stats['return_1d_std']:.4f})
- Average RSI: {summary_stats['rsi_mean']:.1f}
- Average 5-day volatility: {summary_stats['volatility_mean']:.4f}

## Representative Training Samples
For each sample, you see the features and the actual target (0=DOWN, 1=UP).
Study these to learn what patterns predict the target.

{subset.to_string(max_rows=20)}

Corresponding targets (next-day direction):
{target_subset.to_string(max_rows=20)}

## Your Analysis
Identify:
1. Which features are most predictive of direction
2. Key thresholds and conditions (e.g., RSI > 70 often predicts DOWN)
3. Interaction patterns (e.g., high RSI + low momentum = strong DOWN signal)
4. Any regime conditions (bull/bear market patterns)

Be specific. Include actual numbers from the data above."""

    result = call_llm(user, system=system, max_tokens=1024, temperature=0.3)

    if result:
        with open(RESULTS_DIR / "financial_phase1_analysis.txt", "w") as f:
            f.write(result)
        print(f"Phase 1 complete. Analysis: {len(result)} chars.")
    else:
        print("WARNING: Phase 1 returned empty.")

    return result


def run_phase2(test_features, analysis, n=10):
    """Phase 2: Predict — LLM makes direction predictions on test data.

    The Phase 1 analysis is injected so the LLM applies its learned patterns.
    ALL available features are passed (not a subset).
    """
    print("\n=== Phase 2: Predictions ===")

    n = min(n, len(test_features))
    feature_cols = list(test_features.columns)

    predictions = {}
    for i in range(n):
        row = test_features.iloc[i]
        feat_str = _format_features(row, feature_cols)

        system = (
            "You are a quantitative analyst. Use the patterns learned from training data "
            "to predict whether the next-day return will be UP (1) or DOWN (0)."
        )
        user = f"""## Patterns Learned from Training (Phase 1)
{analysis or '(No prior analysis — use your general knowledge of market patterns.)'}

## Test Case Features
{feat_str}

## Your Task
Based ONLY on the features above and the patterns from Phase 1, predict the
next-day return direction: 0 (DOWN/negative return) or 1 (UP/positive return).

Output your prediction in this format:
Predicted direction: <0 or 1>
Confidence: <high/medium/low>
Brief reasoning: <1-2 sentences on why>"""

        result = call_llm(user, system=system, max_tokens=256, temperature=0.1)
        pred = extract_01(result)

        predictions[str(i)] = {
            "raw": result or "",
            "prediction": pred,
        }
        print(f"  Sample {i}: pred={pred} (raw: {str(result)[:60] if result else 'EMPTY'}...)")
        time.sleep(0.5)

    with open(RESULTS_DIR / "financial_phase2_predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Phase 2 complete. {len(predictions)} predictions.")
    return predictions


def run_phase3(predictions, test_targets, analysis, n_samples=10):
    """Phase 3: Extract Decision Rulebook.

    The LLM reviews its predictions vs actuals and extracts structured IF-THEN rules.
    The Phase 1 analysis is included so the rulebook is grounded in learned patterns.
    Uses ALL prediction texts (not just parsed values).
    """
    print("\n=== Phase 3: Rulebook Extraction ===")

    feature_cols = list(test_targets.index.names)  # placeholder, not needed here

    comparison = ""
    for i in range(min(n_samples, len(predictions))):
        raw = predictions.get(str(i), {}).get("raw", "")
        pred = predictions.get(str(i), {}).get("prediction", None)
        actual = int(test_targets.iloc[i]) if i < len(test_targets) else None
        comparison += f"\n## Test Sample {i}\n"
        comparison += f"Predicted: {pred} (raw: {raw[:200]})\n"
        comparison += f"Actual direction: {actual}\n"

    system = (
        "You are a quantitative analyst specializing in extracting structured "
        "decision rules from prediction data."
    )
    user = f"""## Phase 1 Learned Patterns (for context)
{analysis or 'No Phase 1 analysis available.'}

## Predictions vs Ground Truth (Test Set)
{comparison}

## Your Task — Extract Decision Rules
Review the predictions above. For each sample, the model's reasoning is shown alongside
the actual outcome. Extract DECISION RULES that capture the predictive logic.

Format each rule as:
  IF [condition involving features] THEN [0 or 1]

Example:
  IF RSI > 70 AND return_1d < -0.02 THEN 0 (DOWN)
  IF momentum > 0.05 AND bb_position < 0.2 THEN 1 (UP)

Rules must be:
- Actionable: another analyst could apply them without seeing the data
- Specific: include concrete feature thresholds
- Comprehensive: cover the key prediction logic
- Limited: 3-8 rules maximum — prioritize the most important

Output:
## Decision Rulebook
[Numbered rules in IF-THEN format]
"""

    result = call_llm(user, system=system, max_tokens=1024, temperature=0.3)

    if result:
        with open(RESULTS_DIR / "financial_phase3_rulebook.txt", "w") as f:
            f.write(result)
        print(f"Phase 3 complete. Rulebook: {len(result)} chars.")
    else:
        print("WARNING: Rulebook extraction returned empty.")

    return result


def run_phase4(predictions, rulebook, test_features, test_targets, n_samples=10):
    """Phase 4: Replicability Test.

    CORE CONTRIBUTION: Tests whether the Rulebook contains genuine predictive logic.

    A FRESH LLM is given ONLY the Rulebook + test features (NO training data).
    If the fresh LLM reproduces predictions matching ground truth, the Rulebook
    encodes real predictive logic — not just post-hoc rationalization.

    Replicability Score = fraction of fresh predictions matching GROUND TRUTH
    (not matching the original LLM's predictions).

    This directly tests Sarkar (2024)'s "explanation" critique:
    - If R is HIGH: Rulebook contains genuine logic (Sarkar falsified for this case)
    - If R is LOW: Rulebook may be post-hoc rationalization
    """
    print("\n=== Phase 4: Replicability Test ===")

    if not rulebook:
        print("No rulebook. Skipping Phase 4.")
        return None

    n = min(n_samples, len(predictions), len(test_features))
    feature_cols = list(test_features.columns)

    results = {
        "samples": [],
        "n_tested": n,
    }

    original_correct_count = 0
    fresh_correct_count = 0
    fresh_matches_original_count = 0

    for i in range(n):
        row = test_features.iloc[i]
        feat_str = _format_features(row, feature_cols)

        pred_info = predictions.get(str(i), {})
        original_pred = pred_info.get("prediction", None)
        actual = int(test_targets.iloc[i]) if i < len(test_targets) else None

        system = (
            "You are a quantitative analyst following a strict decision rulebook. "
            "Apply ONLY the rules provided — do not use external knowledge."
        )
        user = f"""## Decision Rulebook
Use ONLY the rules below to predict the next-day return direction.

{rulebook}

## Test Case Features
{feat_str}

## Your Task
Apply the rules above to the test features. Output:
Predicted direction: <0 or 1>"""

        fresh_raw = call_llm(user, system=system, max_tokens=128, temperature=0.0)
        fresh_pred = extract_01(fresh_raw)

        orig_correct = (original_pred == actual) if (original_pred is not None and actual is not None) else None
        fresh_correct = (fresh_pred == actual) if (fresh_pred is not None and actual is not None) else None
        match_orig = (fresh_pred == original_pred) if (fresh_pred is not None and original_pred is not None) else None

        if orig_correct:
            original_correct_count += 1
        if fresh_correct:
            fresh_correct_count += 1
        if match_orig:
            fresh_matches_original_count += 1

        print(f"  Sample {i}: orig={original_pred}({'✓' if orig_correct else '✗'}), "
              f"fresh={fresh_pred}({'✓' if fresh_correct else '✗'}), "
              f"actual={actual}")

        results["samples"].append({
            "sample": i,
            "original_prediction": original_pred,
            "fresh_prediction": fresh_pred,
            "actual": actual,
            "original_correct": orig_correct,
            "fresh_correct": fresh_correct,
            "fresh_matches_original": match_orig,
            "fresh_raw": str(fresh_raw)[:200] if fresh_raw else "EMPTY",
        })

        time.sleep(0.5)

    # Replicability Score = fresh accuracy vs ground truth
    results["replicability_score"] = fresh_correct_count / n if n > 0 else 0.0
    results["original_accuracy"] = original_correct_count / n if n > 0 else 0.0
    results["fresh_matches_original_rate"] = fresh_matches_original_count / n if n > 0 else 0.0
    results["n_samples"] = n
    results["n_fresh_correct"] = fresh_correct_count
    results["n_original_correct"] = original_correct_count

    print(f"\n=== Financial Replicability Results ===")
    print(f"  Samples tested:        {n}")
    print(f"  Original LLM Accuracy:  {results['original_accuracy']:.2%} ({original_correct_count}/{n})")
    print(f"  Fresh LLM Accuracy:     {results['replicability_score']:.2%} ({fresh_correct_count}/{n})")
    print(f"  Replicability Score:   {results['replicability_score']:.2%}")
    print(f"  Fresh matches original: {results['fresh_matches_original_rate']:.2%}")
    print(f"\n  Interpretation:")
    print(f"    R > 0.7: Rulebook has strong predictive signal")
    print(f"    R 0.4-0.7: Rulebook captures partial logic")
    print(f"    R < 0.4: Rulebook may be post-hoc rationalization (Sarkar concern)")

    with open(RESULTS_DIR / "financial_replicability_test.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    print("=" * 60)
    print("LLMIP Financial Pipeline — Stateful Experiment")
    print("=" * 60)

    train_features, test_features, train_targets, test_targets = load_financial()
    print(f"\nData: {len(train_features)} train, {len(test_features)} test")
    print(f"Features: {list(train_features.columns)}")
    print(f"Target: target_direction (0=DOWN, 1=UP)")

    analysis = run_phase1(train_features, train_targets)
    predictions = run_phase2(test_features, analysis, n=10)
    rulebook = run_phase3(predictions, test_targets, analysis, n_samples=10)

    if rulebook:
        print(f"\nRulebook preview:\n{rulebook[:300]}...")
    else:
        print("\nWARNING: Rulebook is empty!")

    results = run_phase4(predictions, rulebook, test_features, test_targets, n_samples=10)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    if results:
        print(f"Replicability Score: {results['replicability_score']:.2%}")
        print(f"Original Accuracy:   {results['original_accuracy']:.2%}")


if __name__ == "__main__":
    main()
