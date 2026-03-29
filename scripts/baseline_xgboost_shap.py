#!/usr/bin/env python3
"""
XGBoost + SHAP Baseline Comparison
Compares LLMIP predictions against XGBoost with SHAP explanations.

This baseline is designed to match the LLMIP pipeline exactly:
- Grid: predicts per-bus voltage magnitudes (all 118 buses), same as LLM
- Financial: predicts direction classification, same as LLM
- SHAP attributions are compared against LLMIP Rulebook feature importance
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import xgboost as xgb
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.multioutput import MultiOutputRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data" / "prepared"
RESULTS_DIR = PROJECT_DIR / "results"

FEATURE_DESCRIPTIONS = {
    "hour": "Hour of day (0-23)",
    "day": "Day of month (1-31)",
    "month": "Month of year (1-12)",
    "dayofweek": "Day of week (0=Monday, 6=Sunday)",
    "is_weekend": "Weekend flag (0=weekday, 1=weekend)",
    "total_load_p_R1_mw": "Total active power demand in Region 1 (MW)",
    "total_load_q_R1_mvar": "Total reactive power demand in Region 1 (MVar)",
    "total_load_p_R2_mw": "Total active power demand in Region 2 (MW)",
    "total_load_q_R2_mvar": "Total reactive power demand in Region 2 (MVar)",
    "total_load_p_R3_mw": "Total active power demand in Region 3 (MW)",
    "total_load_q_R3_mvar": "Total reactive power demand in Region 3 (MVar)",
    "total_load_p_mw": "Total system active power demand (MW)",
    "mean_vm_pu": "Mean voltage magnitude across all buses (pu) — NORMALIZED TARGET",
}

GENERATOR_BUSES = [1, 4, 6, 8, 10, 12, 15, 18, 24, 25, 26, 31, 32, 34, 36, 40, 42, 46, 49, 54, 55, 56, 59, 61, 65, 66, 69, 70, 72, 73, 74, 76, 77, 79, 80, 82, 85, 87, 89, 91, 92, 99, 100, 103, 104, 105, 107, 110, 111, 112, 113, 116]
LOAD_BUSES = [b for b in range(1, 119) if b not in GENERATOR_BUSES]

def load_grid_data(name='llmip'):
    """Load grid dataset with full feature descriptions.

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

def load_financial_data(include_next_return=False):
    """Load financial dataset.

    Args:
        include_next_return: If True, include next_return as a feature.
            The thesis requires the LLM to have temporal context for stateful reasoning.
            However, this is the TARGET variable, so we only include lagged versions.
    """
    PROJECT_DIR = Path(__file__).parent.parent
    financial_dir = PROJECT_DIR / "data" / "financial"

    train = pd.read_csv(financial_dir / "sp500_train.csv")
    test = pd.read_csv(financial_dir / "sp500_test.csv")

    train_targets = train['target_direction']
    test_targets = test['target_direction']

    exclude_cols = ['target_direction', 'date']
    if not include_next_return:
        exclude_cols.append('next_return')

    feature_cols = [c for c in train.columns if c not in exclude_cols]
    train_features = train[feature_cols]
    test_features = test[feature_cols]

    return train_features, test_features, train_targets, test_targets

def get_feature_description_string(feature_names):
    """Build a feature description string for the report."""
    lines = []
    for col in feature_names:
        desc = FEATURE_DESCRIPTIONS.get(col, f"Feature: {col}")
        lines.append(f"  - {col}: {desc}")
    return "\n".join(lines)

def run_xgboost_grid():
    """Run XGBoost baseline for grid (voltage prediction).

    Trains multi-output XGBoost to predict voltage magnitude at ALL 118 buses
    simultaneously — exactly matching what the LLMIP Phase 2 pipeline predicts.

    Per-bus MAE, per-bus SHAP, and aggregated metrics are computed so results
    can be directly compared against the LLMIP Rulebook feature attributions.
    """
    print("\n=== XGBoost Grid Baseline (Multi-Output) ===")

    train_features, test_features, train_targets, test_targets = load_grid_data()

    print(f"Train: {len(train_features)} samples, {len(train_features.columns)} features")
    print(f"Test:  {len(test_features)} samples, {len(test_targets.columns)} buses")
    print(f"Generator buses: {len(GENERATOR_BUSES)}, Load buses: {len(LOAD_BUSES)}")

    print("\n[1/4] Training multi-output XGBoost...")
    base_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        tree_method='hist',
    )
    model = MultiOutputRegressor(base_model)
    model.fit(train_features, train_targets)

    print("[2/4] Predicting on test set...")
    predictions = model.predict(test_features)
    predictions_df = pd.DataFrame(predictions, columns=train_targets.columns, index=test_targets.index)

    print("[3/4] Computing per-bus metrics...")
    errors = np.abs(predictions - test_targets.values)
    mae_per_bus = pd.Series(errors.mean(axis=0), index=train_targets.columns, dtype=np.float64)
    rmse_per_bus = pd.Series(np.sqrt((errors ** 2).mean(axis=0)), index=train_targets.columns, dtype=np.float64)

    overall_mae = mae_per_bus.mean()
    overall_rmse = np.sqrt((errors ** 2).mean())
    mae_by_bus_type = {
        "generator_buses": mae_per_bus[[f"vm_{b}" for b in GENERATOR_BUSES if f"vm_{b}" in mae_per_bus]].mean(),
        "load_buses": mae_per_bus[[f"vm_{b}" for b in LOAD_BUSES if f"vm_{b}" in mae_per_bus]].mean(),
    }

    print(f"\n  Overall MAE:  {overall_mae:.4f} pu")
    print(f"  Overall RMSE: {overall_rmse:.4f} pu")
    print(f"  Generator bus MAE:  {mae_by_bus_type['generator_buses']:.4f} pu")
    print(f"  Load bus MAE:       {mae_by_bus_type['load_buses']:.4f} pu")
    print(f"  Best bus MAE:  {mae_per_bus.min():.4f} pu  ({mae_per_bus.idxmin()})")
    print(f"  Worst bus MAE: {mae_per_bus.max():.4f} pu  ({mae_per_bus.idxmax()})")

    print(f"\n  Per-bus MAE statistics:")
    print(f"    Median: {mae_per_bus.median():.4f}")
    print(f"    Std:    {mae_per_bus.std():.4f}")
    print(f"    P25:    {mae_per_bus.quantile(0.25):.4f}")
    print(f"    P75:    {mae_per_bus.quantile(0.75):.4f}")

    print("\n[4/4] Computing SHAP values (per-bus and aggregated)...")
    n_shap_samples = min(50, len(test_features))
    shap_X = test_features.head(n_shap_samples)
    shap_values_all = []
    for estimator in model.estimators_:
        shap_vals = shap.TreeExplainer(estimator).shap_values(shap_X)
        shap_values_all.append(shap_vals)
    shap_values_arr = np.array(shap_values_all)
    shap_abs = np.abs(shap_values_arr)

    shap_aggregated = shap_abs.mean(axis=1).mean(axis=0)
    top_features = pd.Series(shap_aggregated, index=train_features.columns).sort_values(ascending=False)

    shap_per_bus = {}
    for bus_idx, bus_col in enumerate(train_targets.columns):
        bus_shap = shap_abs[bus_idx].mean(axis=0)
        shap_per_bus[bus_col] = pd.Series(bus_shap, index=train_features.columns).sort_values(ascending=False).to_dict()

    print("\n  Top 10 Features (aggregated across all buses):")
    for feat, val in top_features.head(10).items():
        print(f"    {feat}: {val:.4f}")

    print("\n  Top Features by Bus Type (generator vs load):")
    gen_shap = pd.DataFrame(shap_per_bus).loc[:, [f"vm_{b}" for b in GENERATOR_BUSES if f"vm_{b}" in train_targets.columns]].mean(axis=1).sort_values(ascending=False)
    load_shap = pd.DataFrame(shap_per_bus).loc[:, [f"vm_{b}" for b in LOAD_BUSES if f"vm_{b}" in train_targets.columns]].mean(axis=1).sort_values(ascending=False)
    print("    Generator buses — top 3:", gen_shap.head(3).to_dict())
    print("    Load buses — top 3:", load_shap.head(3).to_dict())

    feature_desc_str = get_feature_description_string(train_features.columns)

    results = {
        "domain": "grid",
        "model": "MultiOutputRegressor(XGBRegressor)",
        "n_buses": int(len(train_targets.columns)),
        "n_train": int(len(train_features)),
        "n_test": int(len(test_features)),
        "n_features": int(len(train_features.columns)),
        "generator_buses": GENERATOR_BUSES,
        "load_buses": LOAD_BUSES,
        "metrics": {
            "mae_overall": float(overall_mae),
            "rmse_overall": float(overall_rmse),
            "mae_generator_buses": float(mae_by_bus_type['generator_buses']),
            "mae_load_buses": float(mae_by_bus_type['load_buses']),
            "mae_per_bus": mae_per_bus.to_dict(),
            "rmse_per_bus": rmse_per_bus.to_dict(),
            "mae_best_bus": mae_per_bus.idxmin(),
            "mae_worst_bus": mae_per_bus.idxmax(),
            "mae_median": float(mae_per_bus.median()),
            "mae_std": float(mae_per_bus.std()),
            "mae_p25": float(mae_per_bus.quantile(0.25)),
            "mae_p75": float(mae_per_bus.quantile(0.75)),
        },
        "shap": {
            "n_samples_used": n_shap_samples,
            "top_features_aggregated": top_features.head(20).to_dict(),
            "top_features_generator_buses": gen_shap.head(10).to_dict(),
            "top_features_load_buses": load_shap.head(10).to_dict(),
            "per_bus": shap_per_bus,
        },
        "feature_descriptions": feature_desc_str,
    }

    with open(RESULTS_DIR / 'xgboost_grid_shap.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {RESULTS_DIR / 'xgboost_grid_shap.json'}")

    shap_fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    mae_sorted = mae_per_bus.sort_values()
    colors = ['#f39c12' if int(str(m).replace('vm_', '')) in GENERATOR_BUSES else '#3498db'
              for m in mae_sorted.index]
    axes[0].barh(range(len(mae_sorted)), mae_sorted.values, color=colors)
    axes[0].set_yticks([])
    axes[0].set_xlabel('MAE (pu)')
    axes[0].set_title('XGBoost MAE per Bus\n(orange=generator, blue=load)')
    axes[0].axvline(overall_mae, color='red', linestyle='--', label=f'Mean={overall_mae:.4f}')
    axes[0].legend()

    top_n = 10
    shap_top = top_features.head(top_n)
    axes[1].barh(range(top_n), shap_top.values[::-1], color='#9b59b6')
    axes[1].set_yticks(range(top_n))
    axes[1].set_yticklabels(shap_top.index[::-1], fontsize=8)
    axes[1].set_xlabel('Mean |SHAP|')
    axes[1].set_title('Top 10 Features by SHAP (aggregated)')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'xgboost_grid_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figures saved to {RESULTS_DIR / 'xgboost_grid_analysis.png'}")

    return results

FINANCIAL_FEATURE_DESCRIPTIONS = {
    "close": "Closing price",
    "open": "Opening price",
    "high": "Daily high price",
    "low": "Daily low price",
    "volume": "Trading volume",
    "return_1d": "1-day return (lagged, feature)",
    "return_5d": "5-day return (lagged, feature)",
    "return_20d": "20-day return (lagged, feature)",
    "sma_5": "5-day simple moving average",
    "sma_20": "20-day simple moving average",
    "sma_50": "50-day simple moving average",
    "volatility_5d": "5-day realized volatility",
    "volatility_20d": "20-day realized volatility",
    "momentum": "Price momentum indicator",
    "rsi": "Relative Strength Index (0-100)",
    "macd": "MACD line value",
    "macd_signal": "MACD signal line",
    "bb_mid": "Bollinger Band middle band",
    "bb_upper": "Bollinger Band upper band",
    "bb_lower": "Bollinger Band lower band",
    "bb_position": "Bollinger Band position (0-1)",
    "dayofweek": "Day of week (0-6)",
    "month": "Month (1-12)",
}

def run_xgboost_financial():
    """Run XGBoost baseline for financial (direction prediction).

    Trains XGBoost classifier to predict next-day return direction (0=down, 1=up).
    Uses the FULL feature set — the same features the LLM receives in Phase 2.
    This is the correct comparison for the stateful experiment.

    Note: next_return is the target variable and is NOT included as a feature.
    All features used are lagged values — they are available at prediction time.
    """
    print("\n=== XGBoost Financial Baseline ===")

    train_features, test_features, train_targets, test_targets = load_financial_data(
        include_next_return=False
    )

    print(f"Train: {len(train_features)} samples, {len(train_features.columns)} features")
    print(f"Test:  {len(test_features)} samples")
    print(f"Target distribution — Train: {train_targets.value_counts().to_dict()}")
    print(f"Target distribution — Test:  {test_targets.value_counts().to_dict()}")
    print(f"\nFeatures used ({len(train_features.columns)}):")
    for col in train_features.columns:
        desc = FINANCIAL_FEATURE_DESCRIPTIONS.get(col, col)
        print(f"  - {col}: {desc}")

    print("\n[1/3] Training XGBoost classifier...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
    )
    model.fit(train_features, train_targets)

    print("[2/3] Predicting on test set...")
    predictions = model.predict(test_features)
    pred_proba = model.predict_proba(test_features)

    accuracy = accuracy_score(test_targets, predictions)
    print(f"\n  Accuracy: {accuracy:.2%}")
    print(f"  Correct: {sum(predictions == test_targets)} / {len(test_targets)}")

    print("[3/3] Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    n_shap = min(100, len(test_features))
    shap_values = explainer.shap_values(test_features.head(n_shap))

    shap_abs = np.abs(shap_values)
    feature_importance = shap_abs.mean(axis=0)
    top_features = pd.Series(feature_importance, index=train_features.columns).sort_values(ascending=False)

    print("\n  Top 10 Features by SHAP:")
    for feat, val in top_features.head(10).items():
        print(f"    {feat}: {val:.4f}")

    feature_desc_str = "\n".join(
        f"  - {col}: {FINANCIAL_FEATURE_DESCRIPTIONS.get(col, col)}"
        for col in train_features.columns
    )

    results = {
        "domain": "financial",
        "model": "XGBClassifier",
        "n_train": int(len(train_features)),
        "n_test": int(len(test_features)),
        "n_features": int(len(train_features.columns)),
        "target": "next-day return direction (0=down, 1=up)",
        "metrics": {
            "accuracy": float(accuracy),
            "correct_predictions": int(sum(predictions == test_targets)),
            "total_predictions": int(len(test_targets)),
        },
        "shap": {
            "n_samples_used": n_shap,
            "top_features": top_features.head(20).to_dict(),
            "all_features": top_features.to_dict(),
        },
        "feature_descriptions": feature_desc_str,
    }

    with open(RESULTS_DIR / 'xgboost_financial_shap.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {RESULTS_DIR / 'xgboost_financial_shap.json'}")

    shap_fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    top_n = 15
    shap_top = top_features.head(top_n)
    axes[0].barh(range(top_n), shap_top.values[::-1], color='#2c3e50')
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels(shap_top.index[::-1], fontsize=9)
    axes[0].set_xlabel('Mean |SHAP|')
    axes[0].set_title('XGBoost — Top Features by SHAP')

    conf_matrix = pd.crosstab(
        pd.Series(test_targets.values, name='Actual'),
        pd.Series(predictions, name='Predicted'),
        rownames=['Actual'],
        colnames=['Predicted'],
    )
    im = axes[1].imshow(conf_matrix.values, cmap='Blues')
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(['Down (0)', 'Up (1)'])
    axes[1].set_yticklabels(['Down (0)', 'Up (1)'])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_title(f'Confusion Matrix\nAccuracy={accuracy:.2%}')
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(conf_matrix.values[i, j]), ha='center', va='center', fontsize=14)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'xgboost_financial_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figures saved to {RESULTS_DIR / 'xgboost_financial_analysis.png'}")

    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='grid', choices=['grid', 'financial'])
    args = parser.parse_args()
    
    if args.domain == 'grid':
        run_xgboost_grid()
    else:
        run_xgboost_financial()
    
    print("\n=== XGBoost Baseline Complete ===")

if __name__ == '__main__':
    main()
