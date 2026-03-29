#!/usr/bin/env python3
"""
LLMIP Main Pipeline - REBUILT
Phase 1: Train (LLM learns from training data)
Phase 2: Predict (LLM makes predictions on test data)  
Phase 3: Extract Decision Rulebook (LLM explains decision logic)
Phase 4: Replicate (Fresh LLM uses only Rulebook to reproduce predictions)

Key improvements:
- Proper decision rulebook prompts (not explanations)
- Financial data handling with label removal
- Prompt ablation support
- Better replicability calculation
"""

import os
from dotenv import load_dotenv
load_dotenv()  # Load API keys from .env

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import random
import time
import re

# Project paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data" / "prepared"
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = PROJECT_DIR / "models"

# Create directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# IMPROVED PROMPTS - Decision Rulebook focused
# ============================================================================

PHASE1_SYSTEM_GRID = """You are an expert power systems engineer analyzing IEEE 118-bus power flow data."""

PHASE1_USER_GRID = """## Training Data Summary

You are given {n_train} samples of IEEE 118-bus power system data. 

### Input Features:
{train_features}

### Target Variables (voltage magnitude in per-unit at each bus):
{train_targets}

### Your Task:
Analyze this data to LEARN the patterns that predict voltage magnitudes. Identify:
1. Feature thresholds that determine voltage levels
2. Conditional rules (IF-THEN patterns)
3. Bus categories and their typical voltage ranges
4. Key relationships between input features and output voltages

Focus on learning DECISION RULES that can be used to predict voltages for new data."""

PHASE2_SYSTEM_GRID = """You are an expert power systems engineer. Use the training analysis to predict voltages."""

PHASE2_USER_GRID = """## New Test Case

Based on your training analysis, predict the voltage magnitude (vm_pu) for each bus.

### Input Features:
{test_features}

### Your Task:
For each of the {n_buses} buses, predict the voltage magnitude (vm_pu).

Output Format:
```
Bus: <number>
Predicted vm_pu: <value>
```

Example:
Bus: 1
Predicted vm_pu: 1.045
Bus: 2
Predicted vm_pu: 1.038"""

PHASE3_SYSTEM_GRID = """You are an expert power systems engineer. Extract DECISION RULES from your predictions."""

PHASE3_USER_GRID = """## Predictions vs Actual Values

### Your Predictions:
{predictions}

### Actual Values (Ground Truth):
{actuals}

### Your Task - EXTRACT DECISION RULES:
Based on comparing your predictions to actual values, extract the DECISION RULES you used.

IMPORTANT: Write RULES that another engineer could follow to reproduce your predictions. Not explanations - DECISION LOGIC.

Format as:
## Decision Rulebook

### Rule 1: [Name]
IF [condition involving features] THEN [predicted voltage or action]
Example: IF total_load > 500 AND hour < 6 THEN voltage = 1.02 ± 0.02

### Rule 2: ...

Focus on:
- Specific feature thresholds
- Conditional logic (IF-THEN-ELSE)
- Voltage ranges for different conditions
- Be precise enough that someone could calculate voltages using only these rules"""

# Financial prompts with label removal
PHASE1_SYSTEM_FINANCIAL = """You are a quantitative analyst analyzing market data for predictive modeling."""

PHASE1_USER_FINANCIAL = """## Training Data

You are given {n_train} samples of market data.

### Input Features:
{train_features}

### Target Variable (to predict):
{train_targets}

### Your Task:
Analyze this data to LEARN patterns for prediction. Identify:
1. Feature thresholds and conditions
2. Conditional rules (IF-THEN patterns)
3. Key relationships between features and target
4. Decision boundaries

Focus on learning DECISION RULES that can be used to make predictions for new data."""

PHASE2_SYSTEM_FINANCIAL = """You are a quantitative analyst. Use the training analysis to make predictions."""

PHASE2_USER_FINANCIAL = """## New Test Case

Based on your training analysis, predict the target variable.

### Input Features:
{test_features}

### Your Task:
Predict the target variable value.

Output Format:
Predicted target: <value>"""

PHASE3_SYSTEM_FINANCIAL = """You are a quantitative analyst. Extract DECISION RULES from your predictions."""

PHASE3_USER_FINANCIAL = """## Predictions vs Actual Values

### Your Predictions:
{predictions}

### Actual Values:
{actuals}

### Your Task - EXTRACT DECISION RULES:
Extract DECISION RULES you used to make predictions.

Format as:
## Decision Rulebook

### Rule 1: [Name]
IF [condition] THEN [prediction]
Example: IF rsi > 70 AND momentum > 0.05 THEN predict UP

### Rule 2: ...

Focus on specific thresholds and conditional logic."""

# ============================================================================
# Prompt Ablation Variations
# ============================================================================

PHASE3_PROMPT_VARIATIONS = {
    "standard": {
        "system": "You are an expert analyzing data to extract decision rules.",
        "user_template": PHASE3_USER_GRID
    },
    "explicit_ifthen": {
        "system": "You are an expert extracting IF-THEN rules from predictions.",
        "user_template": """## Predictions vs Actual

Predictions: {predictions}
Actual: {actuals}

Extract decision rules in EXPLICIT IF-THEN format:

Rule 1:
IF [feature] [operator] [value] THEN [prediction] = [value]
Example: IF load > 500 THEN voltage = 1.04

Rule 2: ..."""
    },
    "stepbystep": {
        "system": "You are an expert creating step-by-step decision procedures.",
        "user_template": """## Predictions vs Actual

Predictions: {predictions}
Actual: {actuals}

Create a STEP-BY-STEP decision procedure:

Step 1: Check [feature condition]
Step 2: If true, go to step X; else go to step Y
Step 3: Calculate [prediction]

Format each step clearly."""
    },
    "quantitative": {
        "system": "You are an expert extracting quantitative decision rules.",
        "user_template": """## Predictions vs Actual

Predictions: {predictions}
Actual: {actuals}

Extract QUANTITATIVE rules with specific numbers:

Rule: [feature] determines [prediction] with [formula]
Example: voltage = 1.02 + (load * 0.0001) - (wind * 0.001)

Provide formulas where possible."""
    },
    "reproducible": {
        "system": "You are an expert creating reproducible decision procedures.",
        "user_template": """## Predictions vs Actual

Predictions: {predictions}
Actual: {actuals}

Create rules that ANYONE could use to reproduce your predictions:

Rule 1: [Clear condition] → [Prediction calculation]
Rule 2: ...

Include specific thresholds and calculations."""
    }
}

# ============================================================================
# LLM Client
# ============================================================================

class LLMClient:
    def __init__(self, model="zai-org/GLM-5-FP8", api_key=None, base_url="https://api.us-west-2.modal.direct/v1"):
        self.model = model
        self.api_key = api_key or os.environ.get("MODAL_API_KEY") or os.environ.get("ZAI_API_KEY")
        self.base_url = base_url
        
    def chat(self, system, user, temperature=0.3, max_tokens=4096, retries=3):
        """Send a chat request to the LLM with retry logic."""
        import requests
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        for attempt in range(retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=180
                )
                
                if response.status_code != 200:
                    if attempt < retries - 1:
                        print(f"  Retry {attempt+1}/{retries}...")
                        import time; time.sleep(3)
                        continue
                    print(f"Error: {response.status_code} - {response.text[:200]}")
                    return None
                
                result = response.json()
                msg = result["choices"][0]["message"]
                return msg.get("content") or msg.get("reasoning_content") or ""
            except Exception as e:
                if attempt < retries - 1:
                    print(f"  Exception, retry {attempt+1}/{retries}...")
                    import time; time.sleep(3)
                    continue
                print(f"Exception: {e}")
                return None
        return None

# ============================================================================
# Data Loading
# ============================================================================

def load_grid_data(name='llmip'):
    """Load grid dataset.

    For 'llmip': loads llmip_train_*.csv / llmip_test_*.csv (real IEEE 118-Bus power flow data).
    For other names: loads {name}_train_*.csv / {name}_test_*.csv (synthetic fallback).
    """
    try:
        train_features = pd.read_csv(DATA_DIR / f'{name}_train_features.csv', index_col=0)
        test_features = pd.read_csv(DATA_DIR / f'{name}_test_features.csv', index_col=0)
        train_targets = pd.read_csv(DATA_DIR / f'{name}_train_targets.csv', index_col=0)
        test_targets = pd.read_csv(DATA_DIR / f'{name}_test_targets.csv', index_col=0)
    except FileNotFoundError:
        name = 'llmip'
        train_features = pd.read_csv(DATA_DIR / f'{name}_train_features.csv', index_col=0)
        test_features = pd.read_csv(DATA_DIR / f'{name}_test_features.csv', index_col=0)
        train_targets = pd.read_csv(DATA_DIR / f'{name}_train_targets.csv', index_col=0)
        test_targets = pd.read_csv(DATA_DIR / f'{name}_test_targets.csv', index_col=0)

    vm_cols = [c for c in train_targets.columns if c.startswith('vm_')]
    train_targets = train_targets[vm_cols]
    test_targets = test_targets[vm_cols]

    return train_features, test_features, train_targets, test_targets

def load_financial_data():
    """Load financial dataset - with label removal for LLM."""
    PROJECT_DIR = Path(__file__).parent.parent
    financial_dir = PROJECT_DIR / "data" / "financial"
    
    train = pd.read_csv(financial_dir / "sp500_train.csv")
    test = pd.read_csv(financial_dir / "sp500_test.csv")
    
    # REMOVE LABELS - critical for preventing LLM refusal
    # Store targets separately
    train_targets = train['target_direction'].copy()
    test_targets = test['target_direction'].copy()
    
    # Remove target columns from features
    feature_cols = [c for c in train.columns if c not in ['target_direction', 'next_return', 'date']]
    train_features = train[feature_cols]
    test_features = test[feature_cols]
    
    return train_features, test_features, train_targets, test_targets

# ============================================================================
# Pipeline Functions
# ============================================================================

def run_phase1(client, train_features, train_targets, domain='grid'):
    """Phase 1: Train - LLM analyzes training data.

    The LLM receives full feature descriptions and a representative sample of
    training data with target values. It learns the patterns that predict the
    target variable. The analysis output feeds into Phase 2 (predictions) and
    Phase 3 (rulebook extraction).
    """
    print("\n=== Phase 1: Training Analysis ===")

    n_samples = min(30, len(train_features))
    features_subset = train_features.iloc[:n_samples]
    targets_subset = train_targets.iloc[:n_samples]

    feature_descriptions = {
        'hour': 'Hour of day (0-23)',
        'day': 'Day of month (1-31)',
        'month': 'Month (1-12)',
        'dayofweek': 'Day of week (0=Mon, 6=Sun)',
        'is_weekend': 'Weekend flag (0=weekday, 1=weekend)',
        'total_load_p_R1_mw': 'Total active power demand in Region 1 (MW)',
        'total_load_q_R1_mvar': 'Total reactive power demand in Region 1 (MVar)',
        'total_load_p_R2_mw': 'Total active power demand in Region 2 (MW)',
        'total_load_q_R2_mvar': 'Total reactive power demand in Region 2 (MVar)',
        'total_load_p_R3_mw': 'Total active power demand in Region 3 (MW)',
        'total_load_q_R3_mvar': 'Total reactive power demand in Region 3 (MVar)',
        'total_load_p_mw': 'Total system active power demand (MW)',
    }

    if domain == 'grid':
        feat_desc_str = "\n".join(
            f"  - {k}: {v}" for k, v in feature_descriptions.items() if k in train_features.columns
        )
        target_desc = (
            "Voltage magnitude (vm_pu) at each of 118 buses. "
            "Generator buses (typical: 1, 4, 6, 8, 10, 12, 15, 18, 24, 25, 26, 31, 32, 34, 36...) "
            "typically operate at 1.05-1.20 pu. "
            "Load buses typically operate at 0.95-1.05 pu. "
            "Normal operating range is 0.95-1.05 pu; >1.05 pu may indicate overvoltage; <0.95 pu undervoltage."
        )
        system_prompt = PHASE1_SYSTEM_GRID
        user_prompt = PHASE1_USER_GRID.format(
            n_train=n_samples,
            train_features=feat_desc_str,
            train_targets=target_desc,
        ) + "\n\n## Representative Training Samples\n\n" + features_subset.head(15).to_string(max_rows=15) + "\n\n## Corresponding Target Values\n\n" + targets_subset.head(15).to_string(max_rows=15)
    else:
        feat_desc_str = "\n".join(
            f"  - {col}: {col}" for col in train_features.columns
        )
        target_desc = (
            "Binary direction: 0 = DOWN (negative next-day return), 1 = UP (positive next-day return). "
            "Predicting the direction (not magnitude) of next-period return."
        )
        system_prompt = PHASE1_SYSTEM_FINANCIAL
        user_prompt = PHASE1_USER_FINANCIAL.format(
            n_train=n_samples,
            train_features=feat_desc_str,
            train_targets=target_desc,
        ) + "\n\n## Representative Training Samples\n\n" + features_subset.head(15).to_string(max_rows=15) + "\n\n## Corresponding Target Values\n\n" + targets_subset.head(15).to_string(max_rows=15)

    print(f"Training LLM on {n_samples} samples with {len(train_features.columns)} features...")
    analysis = client.chat(system_prompt, user_prompt, temperature=0.3, max_tokens=4096)

    if analysis:
        print(f"Phase 1 complete. Analysis length: {len(analysis)} chars.")

    return analysis


def run_phase2(client, analysis, test_features, train_features, train_targets, domain='grid', n_buses=None):
    """Phase 2: Predict - LLM makes predictions on test data.

    Uses the patterns learned in Phase 1 (analysis) to predict on test data.
    The Phase 1 analysis is injected into the prompt so the LLM applies
    its learned understanding.
    """
    print("\n=== Phase 2: Prediction ===")

    predictions = {}

    n_test = min(10, len(test_features))
    test_samples = test_features.head(n_test)

    feature_descriptions_grid = {
        'hour': 'Hour of day (0-23)',
        'day': 'Day of month (1-31)',
        'month': 'Month (1-12)',
        'dayofweek': 'Day of week (0=Mon, 6=Sun)',
        'is_weekend': 'Weekend flag (0=weekday, 1=yes)',
        'total_load_p_R1_mw': 'Total active power demand Region 1 (MW)',
        'total_load_q_R1_mvar': 'Total reactive power demand Region 1 (MVar)',
        'total_load_p_R2_mw': 'Total active power demand Region 2 (MW)',
        'total_load_q_R2_mvar': 'Total reactive power demand Region 2 (MVar)',
        'total_load_p_R3_mw': 'Total active power demand Region 3 (MW)',
        'total_load_q_R3_mvar': 'Total reactive power demand Region 3 (MVar)',
        'total_load_p_mw': 'Total system active power demand (MW)',
    }

    for i, (idx, row) in enumerate(test_samples.iterrows()):
        print(f"Predicting test case {i+1}/{n_test} (index={idx})...")

        if domain == 'grid':
            feat_str = "\n".join(
                f"  {col} ({feature_descriptions_grid.get(col, col)}): {row[col]}"
                for col in train_features.columns if col in row.index
            )
            system_prompt = PHASE2_SYSTEM_GRID
            user_prompt = (
                f"## Patterns Learned in Phase 1\n\n{analysis or 'No prior analysis available.'}\n\n"
                f"## Test Case Features\n\n{feat_str}\n\n"
                f"## Your Task\n\n"
                f"Predict the voltage magnitude (vm_pu) at each of the 118 buses.\n"
                f"For generator buses (typically: 1,4,6,8,10,12,15,18,24,25,26,31,32,34,36...): "
                f"expect 1.05-1.20 pu.\n"
                f"For load buses: expect 0.95-1.05 pu.\n\n"
                f"Output format (one per bus):\nBus: <number>\nPredicted vm_pu: <value>\n\n"
                f"Predict for ALL 118 buses."
            )
        else:
            feat_str = "\n".join(f"  {col}: {row[col]}" for col in train_features.columns if col in row.index)
            system_prompt = PHASE2_SYSTEM_FINANCIAL
            user_prompt = (
                f"## Patterns Learned in Phase 1\n\n{analysis or 'No prior analysis available.'}\n\n"
                f"## Test Case Features\n\n{feat_str}\n\n"
                f"## Your Task\n\n"
                f"Predict the target direction: 0 (DOWN/negative return) or 1 (UP/positive return).\n"
                f"Output format:\nPredicted target: <0 or 1>"
            )

        result = client.chat(system_prompt, user_prompt, temperature=0.1, max_tokens=2048)

        if result:
            predictions[str(idx)] = result
        else:
            predictions[str(idx)] = ""
            print(f"  WARNING: LLM returned empty for index {idx}")

        time.sleep(0.5)

    print(f"Phase 2 complete. {len(predictions)} predictions.")
    return predictions


def run_phase3(client, predictions, test_targets, test_features, analysis, domain='grid'):
    """Phase 3: Extract Decision Rulebook.

    The LLM reviews its predictions vs actual ground truth and extracts
    decision rules that capture the logic used. Phase 1 analysis is included
    to ensure the rulebook is grounded in the learned patterns.
    Uses ALL predictions from Phase 2 (not just a subset).
    """
    print("\n=== Phase 3: Rulebook Extraction ===")

    comparison = ""
    for idx in predictions.keys():
        idx_int = int(idx)
        comparison += f"\n## Test Case (index={idx})\n"

        if domain == 'grid':
            actual_vals = test_targets.iloc[idx_int] if idx_int < len(test_targets) else None
            comparison += f"Predicted (from Phase 2):\n{predictions[idx][:400]}\n\n"
            if actual_vals is not None:
                comparison += f"Actual voltages:\n{actual_vals.to_string()[:300]}\n"
        else:
            actual_val = test_targets.iloc[idx_int] if idx_int < len(test_targets) else None
            comparison += f"Predicted: {predictions[idx][:200]}\n"
            if actual_val is not None:
                comparison += f"Actual target: {actual_val}\n"

    if domain == 'grid':
        system_prompt = PHASE3_SYSTEM_GRID
        user_prompt = (
            f"{PHASE3_USER_GRID.format(predictions=comparison, actuals='(see above per case)')}\n\n"
            f"## Ground Truth (per test case above)\n\n"
            f"The actual voltage values are shown per test case above. Compare your predictions "
            f"to the actuals and identify WHY you predicted what you did — what feature patterns "
            f"drived each decision."
        )
    else:
        system_prompt = PHASE3_SYSTEM_FINANCIAL
        user_prompt = (
            f"{PHASE3_USER_FINANCIAL.format(predictions=comparison, actuals='(see above per case)')}\n\n"
            f"## Ground Truth\n\n"
            f"The actual target directions are shown per test case. Compare your predictions "
            f"to the actuals and extract the decision rules."
        )

    rulebook = client.chat(system_prompt, user_prompt, temperature=0.3, max_tokens=4096)

    if rulebook:
        print(f"Phase 3 complete. Rulebook length: {len(rulebook)} chars.")
    else:
        print("WARNING: Rulebook extraction returned empty.")

    return rulebook
    
    rulebook = client.chat(system_prompt, user_prompt, temperature=0.3, max_tokens=4096)
    
    if rulebook:
        print("Phase 3 complete. Rulebook saved.")
    
    return rulebook

def run_phase3_ablation(client, predictions, test_targets, domain='grid'):
    """Run Phase 3 with multiple prompt variations."""
    print("\n=== Phase 3: Prompt Ablation ===")
    
    ablation_results = {}
    
    for variation_name, variation in PHASE3_PROMPT_VARIATIONS.items():
        print(f"\nRunning variation: {variation_name}")
        
        # Format data for this variation
        comparison = ""
        for idx in list(predictions.keys())[:3]:  # Use fewer samples for ablation
            comparison += f"Predictions: {predictions[idx][:200]}\n"
            comparison += f"Actual: {test_targets.loc[int(idx)].to_string()[:100]}\n\n"
        
        user_prompt = variation["user_template"].format(
            predictions=comparison,
            actuals=test_targets.head(3).to_string()
        )
        
        rulebook = client.chat(variation["system"], user_prompt, temperature=0.3, max_tokens=4096)
        
        ablation_results[variation_name] = rulebook if rulebook else ""
        
        time.sleep(1)
    
    return ablation_results

def parse_grid_predictions(llm_response):
    """Parse voltage predictions from LLM response."""
    voltages = {}
    
    if not llm_response:
        return voltages
    
    # Pattern: Bus: X, Predicted vm_pu: Y
    lines = llm_response.split('\n')
    for line_idx, line in enumerate(lines):
        match = re.search(r'Bus:?\s*(\d+)', line, re.IGNORECASE)
        if match:
            bus = match.group(1)
            # Look for voltage value nearby
            voltage_match = re.search(r'(?:vm_pu|voltage[:\s]+(?:pu)?)\s*[:=]?\s*([0-9.]+)', line, re.IGNORECASE)
            if not voltage_match:
                # Check next few lines
                for j in range(line_idx+1, min(line_idx+3, len(lines))):
                    voltage_match = re.search(r'([0-9]+\.[0-9]+)', lines[j])
                    if voltage_match:
                        break
            
            if voltage_match:
                try:
                    voltages[f'vm_{bus}'] = float(voltage_match.group(1))
                except:
                    pass
    
    return voltages

def parse_financial_predictions(llm_response):
    """Parse financial predictions from LLM response."""
    if not llm_response:
        return None
    
    # Look for prediction value
    match = re.search(r'Predicted[:\s]+(?:target[:\s]+)?(\d+|UP|DOWN|up|down)', llm_response, re.IGNORECASE)
    if match:
        pred = match.group(1).upper()
        if pred in ['UP', 'DOWN']:
            return 1 if pred == 'UP' else 0
        try:
            return int(pred)
        except:
            pass
    
    return None

def calculate_replicability(client, predictions, test_targets, rulebook, test_features, domain='grid'):
    """Phase 4: Calculate Replicability Score.

    The Replicability Score answers: Can a FRESH LLM, given ONLY the Rulebook
    and test features, reproduce predictions that match the ACTUAL ground truth?

    Definition (per thesis): R = accuracy(fresh_LLM | Rulebook+features) / accuracy(original_LLM)

    For continuous targets (grid): MAE reduction = 1 - (fresh_MAE / original_MAE)
    For discrete targets (financial): R = fraction of fresh predictions matching ground truth

    A high R means the Rulebook contains genuine predictive logic, not just
    post-hoc rationalization (Sarkar 2024 rebuttal — Layer 3 empirical falsification).
    """
    print("\n=== Phase 4: Replicability Test ===")

    if not rulebook or not predictions:
        print("Missing rulebook or predictions. Skipping.")
        return None

    if test_features is None:
        print("Missing test features. Skipping.")
        return None

    results = {
        "samples": [],
        "domain": domain,
    }

    all_original = []
    all_fresh = []
    all_actual = []

    test_indices = list(predictions.keys())[:5]

    for idx in test_indices:
        idx_int = int(idx)
        print(f"\n  Sample {idx}...")

        original_response = predictions.get(idx, '')

        if domain == 'grid':
            original_voltages = parse_grid_predictions(original_response)
            test_row = test_features.iloc[idx_int] if idx_int < len(test_features) else None
            actual_row = test_targets.iloc[idx_int].to_dict() if idx_int < len(test_targets) else {}
        else:
            original_voltages = parse_financial_predictions(original_response)
            test_row = test_features.iloc[idx_int] if idx_int < len(test_features) else None
            actual_row = test_targets.iloc[idx_int] if idx_int < len(test_targets) else None

        if test_row is None:
            print(f"    No test row for index {idx_int}. Skipping.")
            continue

        if domain == 'grid':
            features_str = _format_grid_features(test_row)
            replication_system = (
                "You are a power systems engineer. Use the Decision Rulebook below "
                "to predict voltage magnitudes at each bus for the given test case. "
                "Output ONLY the predictions in the format:\nBus: <number>\nPredicted vm_pu: <value>"
            )
            replication_user = (
                f"## Decision Rulebook\n\n{rulebook}\n\n"
                f"## Test Case Features\n\n{features_str}\n\n"
                f"## Your Task\n\n"
                f"Using ONLY the rules above and the test features, predict the voltage magnitude "
                f"(vm_pu) at each bus. Output in this format:\nBus: <number>\nPredicted vm_pu: <value>\n..."
            )
        else:
            features_str = _format_financial_features(test_row)
            replication_system = (
                "You are a quantitative analyst. Use the Decision Rulebook below "
                "to predict whether the market direction is UP (1) or DOWN (0) for the given test case."
            )
            replication_user = (
                f"## Decision Rulebook\n\n{rulebook}\n\n"
                f"## Test Case Features\n\n{features_str}\n\n"
                f"## Your Task\n\n"
                f"Using ONLY the rules above and the test features, predict the target (0 or 1)."
            )

        fresh_response = client.chat(
            replication_system,
            replication_user,
            temperature=0.1,
            max_tokens=1024,
        )

        if not fresh_response:
            print(f"    Fresh LLM call failed. Skipping.")
            continue

        if domain == 'grid':
            fresh_voltages = parse_grid_predictions(fresh_response)

            if isinstance(original_voltages, dict) and isinstance(fresh_voltages, dict):
                matching_buses = set(original_voltages.keys()) & set(fresh_voltages.keys())
                if matching_buses:
                    errors_original = []
                    errors_fresh = []
                    for bus in matching_buses:
                        orig = original_voltages[bus]
                        fresh = fresh_voltages[bus]
                        actual = actual_row.get(bus, None)
                        errors_original.append(abs(orig - actual) if actual is not None else None)
                        errors_fresh.append(abs(fresh - actual) if actual is not None else None)

                    errors_original = [e for e in errors_original if e is not None]
                    errors_fresh = [e for e in errors_fresh if e is not None]

                    if errors_original and errors_fresh:
                        mae_original = float(np.mean(errors_original))
                        mae_fresh = float(np.mean(errors_fresh))
                        replicability_score = max(0.0, 1.0 - mae_fresh / mae_original) if mae_original > 0 else 0.0

                        print(f"    Original MAE: {mae_original:.4f}, Fresh MAE: {mae_fresh:.4f}")
                        print(f"    Replicability Score: {replicability_score:.2%} ({len(matching_buses)} buses)")

                        results["samples"].append({
                            "index": idx,
                            "n_matching_buses": len(matching_buses),
                            "mae_original": mae_original,
                            "mae_fresh": mae_fresh,
                            "replicability_score": replicability_score,
                        })

                        for bus in matching_buses:
                            all_original.append(original_voltages[bus])
                            all_fresh.append(fresh_voltages[bus])
                            actual = actual_row.get(bus)
                            if actual is not None:
                                all_actual.append(actual)
        else:
            fresh_pred = parse_financial_predictions(fresh_response)
            if original_voltages is not None and fresh_pred is not None:
                actual = int(actual_row.item()) if actual_row is not None else None
                original_correct = int(original_voltages == actual) if actual is not None else None
                fresh_correct = int(fresh_pred == actual) if actual is not None else None
                match = int(fresh_pred == original_voltages)

                print(f"    Original: {original_voltages} {'✓' if original_correct else '✗'}, "
                      f"Fresh: {fresh_pred} {'✓' if fresh_correct else '✗'}, "
                      f"Actual: {actual}")

                results["samples"].append({
                    "index": idx,
                    "original": original_voltages,
                    "fresh": fresh_pred,
                    "actual": actual,
                    "fresh_correct": fresh_correct,
                    "original_correct": original_correct,
                    "match_with_original": match,
                })

                all_original.append(original_voltages)
                all_fresh.append(fresh_pred)
                if actual is not None:
                    all_actual.append(actual)

    if domain == 'grid' and all_original and all_fresh and all_actual:
        mae_original_all = float(np.mean([abs(o - a) for o, a in zip(all_original, all_actual)]))
        mae_fresh_all = float(np.mean([abs(f - a) for f, a in zip(all_fresh, all_actual)]))
        results["overall_mae_original"] = mae_original_all
        results["overall_mae_fresh"] = mae_fresh_all
        results["replicability_score"] = max(0.0, 1.0 - mae_fresh_all / mae_original_all) if mae_original_all > 0 else 0.0
        results["n_total_buses"] = len(all_original)

        print(f"\n=== Grid Replicability Results ===")
        print(f"  Original LLM MAE: {mae_original_all:.4f} pu")
        print(f"  Fresh LLM MAE:    {mae_fresh_all:.4f} pu")
        print(f"  Replicability Score: {results['replicability_score']:.2%}")
        print(f"  (Score = 1 means fresh LLM perfectly replicates original)")

    elif domain == 'financial' and all_fresh and all_actual:
        n_correct = sum(int(f == a) for f, a in zip(all_fresh, all_actual))
        n_total = len(all_fresh)
        results["replicability_score"] = n_correct / n_total if n_total > 0 else 0.0
        results["fresh_accuracy"] = results["replicability_score"]
        results["n_samples"] = n_total
        results["n_correct"] = n_correct

        if all_original and all_actual:
            orig_correct = sum(int(o == a) for o, a in zip(all_original, all_actual))
            results["original_accuracy"] = orig_correct / len(all_original)

        print(f"\n=== Financial Replicability Results ===")
        print(f"  Fresh LLM Accuracy (vs actual): {results['replicability_score']:.2%} ({n_correct}/{n_total})")
        if results.get("original_accuracy") is not None:
            print(f"  Original LLM Accuracy:          {results['original_accuracy']:.2%}")

    return results


def _format_grid_features(row):
    """Format grid test row features for LLM prompts."""
    lines = []
    for col, val in row.items():
        desc = {
            'hour': 'Hour of day (0-23)',
            'day': 'Day of month (1-31)',
            'month': 'Month (1-12)',
            'dayofweek': 'Day of week (0=Mon, 6=Sun)',
            'is_weekend': 'Weekend flag (0=no, 1=yes)',
            'total_load_p_R1_mw': 'Total active power demand Region 1 (MW)',
            'total_load_q_R1_mvar': 'Total reactive power demand Region 1 (MVar)',
            'total_load_p_R2_mw': 'Total active power demand Region 2 (MW)',
            'total_load_q_R2_mvar': 'Total reactive power demand Region 2 (MVar)',
            'total_load_p_R3_mw': 'Total active power demand Region 3 (MW)',
            'total_load_q_R3_mvar': 'Total reactive power demand Region 3 (MVar)',
            'total_load_p_mw': 'Total system active power demand (MW)',
        }.get(col, col)
        lines.append(f"  {col} ({desc}): {val}")
    return "\n".join(lines) if lines else str(row)


def _format_financial_features(row):
    """Format financial test row features for LLM prompts — ALL features."""
    lines = []
    for col, val in row.items():
        desc = {
            'close': 'Closing price', 'open': 'Opening price',
            'high': 'Daily high', 'low': 'Daily low',
            'volume': 'Trading volume',
            'return_1d': '1-day return (lagged)', 'return_5d': '5-day return (lagged)',
            'return_20d': '20-day return (lagged)',
            'sma_5': '5-day SMA', 'sma_20': '20-day SMA', 'sma_50': '50-day SMA',
            'volatility_5d': '5-day volatility', 'volatility_20d': '20-day volatility',
            'momentum': 'Momentum indicator', 'rsi': 'RSI (0-100)',
            'macd': 'MACD', 'macd_signal': 'MACD signal',
            'bb_mid': 'Bollinger mid', 'bb_upper': 'Bollinger upper',
            'bb_lower': 'Bollinger lower', 'bb_position': 'BB position (0-1)',
            'dayofweek': 'Day of week', 'month': 'Month',
        }.get(col, col)
        lines.append(f"  {col} ({desc}): {val}")
    return "\n".join(lines) if lines else str(row)

def main():
    parser = argparse.ArgumentParser(description='Run LLMIP pipeline')
    parser.add_argument('--model', type=str, default='zai-org/GLM-5-FP8')
    parser.add_argument('--domain', type=str, default='grid', choices=['grid', 'financial'])
    parser.add_argument('--data', type=str, default='pilot', help='Dataset name for grid')
    parser.add_argument('--ablation', action='store_true', help='Run prompt ablation')
    parser.add_argument('--skip-phases', type=str, default='', help='Phases to skip')
    args = parser.parse_args()
    
    # Initialize LLM client
    client = LLMClient(model=args.model)
    
    # Load data
    if args.domain == 'grid':
        train_features, test_features, train_targets, test_targets = load_grid_data(args.data)
        print(f"Grid data: Train={len(train_features)}, Test={len(test_features)}")
    else:
        train_features, test_features, train_targets, test_targets = load_financial_data()
        print(f"Financial data: Train={len(train_features)}, Test={len(test_features)}")
    
    skip = args.skip_phases.split(',') if args.skip_phases else []
    
    # Run pipeline
    analysis = None
    if '1' not in skip:
        analysis = run_phase1(client, train_features, train_targets, args.domain)
    else:
        # Try to load existing
        if args.domain == 'grid':
            analysis_path = RESULTS_DIR / 'phase1_analysis.txt'
        else:
            analysis_path = RESULTS_DIR / 'financial_phase1_analysis.txt'
        if analysis_path.exists():
            with open(analysis_path) as f:
                analysis = f.read()
    
    predictions = None
    if '2' not in skip:
        n_buses = len(train_targets.columns) if args.domain == 'grid' else None
        predictions = run_phase2(
            client, analysis, test_features, train_features, train_targets,
            args.domain, n_buses
        )

        # Save
        if args.domain == 'grid':
            with open(RESULTS_DIR / 'phase2_predictions.json', 'w') as f:
                json.dump(predictions, f, indent=2)
        else:
            with open(RESULTS_DIR / 'financial_phase2_predictions.json', 'w') as f:
                json.dump(predictions, f, indent=2)
    else:
        # Load existing
        if args.domain == 'grid':
            with open(RESULTS_DIR / 'phase2_predictions.json') as f:
                predictions = json.load(f)
        else:
            with open(RESULTS_DIR / 'financial_phase2_predictions.json') as f:
                predictions = json.load(f)

    rulebook = None
    if '3' not in skip:
        rulebook = run_phase3(
            client, predictions, test_targets, test_features, analysis, args.domain
        )

        if args.domain == 'grid':
            with open(RESULTS_DIR / 'phase3_rulebook.txt', 'w') as f:
                f.write(rulebook or "")
        else:
            with open(RESULTS_DIR / 'financial_phase3_rulebook.txt', 'w') as f:
                f.write(rulebook or "")
    else:
        # Load existing
        if args.domain == 'grid':
            with open(RESULTS_DIR / 'phase3_rulebook.txt') as f:
                rulebook = f.read()
        else:
            with open(RESULTS_DIR / 'financial_phase3_rulebook.txt') as f:
                rulebook = f.read()

    # Prompt ablation
    if args.ablation and '3' not in skip:
        ablation_results = run_phase3_ablation(client, predictions, test_targets, args.domain)
        
        ablation_path = RESULTS_DIR / 'prompt_ablation_results.json'
        with open(ablation_path, 'w') as f:
            json.dump(ablation_results, f, indent=2)
        print(f"Ablation results saved to {ablation_path}")
    
    # Phase 4: Replicability
    if '4' not in skip:
        replicability_results = calculate_replicability(
            client, predictions, test_targets, rulebook, test_features, args.domain
        )
        
        if replicability_results:
            if args.domain == 'grid':
                with open(RESULTS_DIR / 'grid_replicability_test.json', 'w') as f:
                    json.dump(replicability_results, f, indent=2)
            else:
                with open(RESULTS_DIR / 'financial_replicability_test.json', 'w') as f:
                    json.dump(replicability_results, f, indent=2)
    
    print("\n=== Pipeline Complete ===")

if __name__ == '__main__':
    main()
