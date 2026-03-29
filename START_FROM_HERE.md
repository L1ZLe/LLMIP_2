# LLMIP - START FROM HERE

## 1. PROJECT OVERVIEW

### Research Domain
LLMIP (**LLM-as-Interpretable-Predictor**) is an AI/ML research framework at the intersection of:
- Large Language Models (LLMs) for tabular prediction
- Explainable AI (XAI) for critical infrastructure
- Novel evaluation metrics for interpretability

### Core Problem Being Solved
The project addresses a fundamental limitation in AI-powered forecasting: **black-box models** (including LLMs) produce predictions that cannot be inspected, audited, or understood by domain experts. In safety-critical domains like electricity grid operations, this opacity is unacceptable—operators must understand *why* a prediction was made to trust and act on it.

### Core Research Hypothesis
Can LLM-generated explanations ("Decision Rulebooks") contain genuine predictive logic, or are they merely "post-hoc rationalization" as argued by Sarkar (2024)?

### Key Innovation: Replicability Score
The **Replicability Score** is the central novel contribution—a metric that tests whether a *fresh* LLM can reproduce predictions using **only** the extracted Rulebook (no training data). A high score falsifies the Sarkar critique: the Rulebook must contain genuine predictive logic.

### Target Applications
1. **Primary**: Electricity Grid State Forecasting (IEEE 118-Bus Power System) — stateless, physics-grounded
2. **Secondary**: Financial Market Prediction (S&P 500) — stateful, path-dependent

> **IMPORTANT:** For understanding the codebase, read the **Jupyter notebooks** (`notebooks/`) first — they contain detailed analysis, explanations, and execution traces. The Python scripts in `scripts/` are the underlying implementation but are harder to follow.

---

## 2. REPOSITORY STRUCTURE

```
LLMIP/
├── .env                          # API keys (GLM 4.7, OpenRouter, ZAI, OpenCode, GitHub)
├── TODO.md                       # Project status, results, task tracking
├── LLMIP_Thesis.txt              # Full research thesis/mission brief (500+ lines)
├── LLMIP_Thesis.docx             # Word document version
├── START_FROM_HERE.md            # THIS FILE
├── Technical Presentation...pptx # Project presentation deck
│
├── scripts/                      # Main execution pipelines
│   ├── run_llmip_rebuilt.py     # Grid experiment (4-phase pipeline)
│   ├── run_financial_fixed.py   # Financial experiment (4-phase pipeline)
│   ├── baseline_xgboost_shap.py # XGBoost + SHAP baselines
│   └── prepare_grid_data.py    # Raw grid data → CSV conversion
│
├── notebooks/                    # Jupyter notebooks (analysis & exploration)
│   ├── 01_data_preparation.ipynb     # IEEE 118-bus data exploration
│   ├── 02_baseline_grid.ipynb       # XGBoost grid baseline
│   ├── 03_llmip_grid.ipynb          # LLMIP grid experiment
│   ├── 04_baseline_financial.ipynb  # XGBoost financial baseline
│   ├── 05_llmip_financial.ipynb     # LLMIP financial experiment
│   └── 06_analysis_comparison.ipynb # Results comparison
│
├── data/                        # All datasets
│   ├── grid/                    # IEEE 118-bus power flow data
│   │   ├── samples/             # 8760 hourly JSON snapshots (2024)
│   │   │   └── *.json          # Power flow state snapshots
│   │   ├── buses.csv            # Bus metadata (118 buses)
│   │   ├── gens.csv             # Generator metadata
│   │   ├── loads.csv            # Load metadata
│   │   ├── branches.csv         # Transmission line data
│   │   ├── gens_ts.csv          # Generator time series (⚠️ 184MB - NOT in git)
│   │   └── loads_ts.csv         # Load time series (⚠️ 72MB - NOT in git)
│   │
│   ├── financial/               # S&P 500 data
│   │   ├── sp500_train.csv     # Training set (~3000 samples)
│   │   └── sp500_test.csv      # Test set (~700 samples)
│   │
│   └── prepared/                # Processed feature/target CSVs
│       ├── llmip_train_features.csv
│       ├── llmip_train_targets.csv
│       ├── llmip_test_features.csv
│       ├── llmip_test_targets.csv
│       └── llmip_stats.json
│
├── results/                     # Experiment outputs
│   ├── phase1_analysis.txt      # Phase 1: Training analysis
│   ├── phase2_predictions.json  # Phase 2: LLM predictions
│   ├── phase3_rulebook.txt      # Phase 3: Decision Rulebook
│   ├── grid_replicability_test.json
│   ├── financial_replicability_test.json
│   ├── xgboost_grid_shap.json
│   ├── xgboost_financial_shap.json
│   ├── xgboost_grid_analysis.png
│   ├── xgboost_financial_analysis.png
│   └── physical_consistency_check.md
│
├── save/                        # Snapshot scripts (push to GitHub)
│   ├── llmip-ercot-snapshot.sh
│   └── llmip-snapshot-ai.py
│
├── myenv/                       # Python virtual environment
│   └── bin/activate            # Activate with: source myenv/bin/activate
│
└── .git/                        # Git repository
```

---

## 3. FILE-BY-FILE BREAKDOWN

### Core Scripts

#### `scripts/run_llmip_rebuilt.py` (983 lines)
- **Purpose:** Main LLMIP pipeline implementing all 4 phases
- **Role:** Primary execution script for both grid and financial experiments
- **Key Logic:**
  - Phase 1: Train (LLM analyzes training data, learns patterns)
  - Phase 2: Predict (LLM makes predictions on test data)
  - Phase 3: Extract Decision Rulebook (LLM explains reasoning)
  - Phase 4: Replicability Test (Fresh LLM uses only Rulebook)
- **Dependencies:** `dotenv`, `pandas`, `numpy`, `requests`
- **Notes:** Uses GLM 4.7 API as primary LLM; configured in `.env`

#### `scripts/baseline_xgboost_shap.py` (420 lines)
- **Purpose:** XGBoost + SHAP baseline for comparison
- **Role:** Provides benchmark to compare LLMIP against standard ML
- **Key Logic:**
  - Grid: MultiOutputRegressor for 118-bus voltage prediction
  - Financial: XGBClassifier for direction prediction
  - SHAP computation for feature importance
- **Dependencies:** `xgboost`, `shap`, `sklearn`, `matplotlib`

#### `scripts/run_financial_fixed.py`
- **Purpose:** Financial domain pipeline with label removal
- **Role:** Handles S&P 500 experiment
- **Key Logic:** Removes target labels before LLM processing to prevent refusal

#### `scripts/prepare_grid_data.py`
- **Purpose:** Converts raw IEEE 118-bus JSON snapshots to CSV features/targets
- **Role:** Data preprocessing pipeline for grid data

### Configuration & Environment

#### `.env`
- **Purpose:** API keys for LLM services
- **Contains:**
  - `GitHub_key` — GitHub PAT for repository push
  - `MODAL_API_KEY` — Primary LLM API
  - `ZAI_API_KEY` — Fallback LLM API
  - `OPENROUTER_API_KEY` — Secondary fallback
  - `OPENCODE_API_KEY` — Code assistance
- **Notes:** ⚠️ Never commit to public repos

#### `myenv/`
- **Purpose:** Python virtual environment with all dependencies
- **Activation:** `source myenv/bin/activate`
- **Key packages:** pandas, numpy, xgboost, shap, requests, python-dotenv

### Documentation

#### `LLMIP_Thesis.txt` (500+ lines)
- **Purpose:** Complete research thesis/mission brief
- **Role:** Defines methodology, experimental design, conference targeting
- **Key Content:**
  - Sarkar (2024) counter-argument (3 layers)
  - Dual-environment experimental design
  - Publication roadmap
  - Technical environment setup

#### `TODO.md`
- **Purpose:** Project status and task tracking
- **Current Results:**
  - Grid: LLM MAE 0.032, XGBoost MAE 0.002, Replicability 35.2%
  - Financial: LLM 60%, XGBoost 50%, Replicability 70%

### Data Files

#### `data/grid/samples/*.json` (8760 files)
- **Purpose:** Hourly IEEE 118-bus power flow snapshots for 2024
- **Format:** JSON with bus voltages, generator outputs, load demands
- **Source:** Derived from IEEE 118-bus test case power flow simulations

#### `data/financial/sp500_*.csv`
- **Purpose:** S&P 500 historical data with technical indicators
- **Features:** OHLCV, SMA, RSI, MACD, Bollinger Bands, momentum
- **Target:** `target_direction` (0=DOWN, 1=UP)

#### `data/prepared/llmip_*.csv`
- **Purpose:** Pre-processed features and targets for LLM pipeline
- **Features:** Temporal + load/generation features (12 columns)
- **Targets:** 118 voltage columns (`vm_1` to `vm_118`)

---

## 4. SYSTEM ARCHITECTURE & FLOW

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LLMIP PIPELINE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   PHASE 1    │ →  │   PHASE 2    │ →  │   PHASE 3    │          │
│  │   Training   │    │  Prediction   │    │ Rulebook Ext │          │
│  │   Analysis   │    │              │    │              │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         ↓                    ↓                   ↓                 │
│    [Analysis]           [Predictions]        [Rulebook]              │
│                                             ↓                       │
│                                    ┌──────────────┐                 │
│                                    │   PHASE 4    │                 │
│                                    │ Replicability│                 │
│                                    │    Test      │                 │
│                                    └──────────────┘                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         BASELINE COMPARISON                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   LLMIP (Rulebook)  ←──────────────────────→  XGBoost + SHAP        │
│        ↓                                           ↓                 │
│   Replicability Score                      Feature Importance        │
│        ↓                                           ↓                 │
│   [Fresh LLM | Rulebook]                 [TreeExplainer]           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                          DATA FLOW                                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  RAW DATA SOURCES                                                     │
│       │                                                               │
│       ├── IEEE 118-Bus Power Flow (data/grid/samples/*.json)        │
│       │   └── prepare_grid_data.py → CSV features/targets            │
│       │                                                               │
│       └── S&P 500 (data/financial/*.csv)                            │
│           └── Label removal preprocessing                            │
│                                                                       │
│  PREPROCESSED → data/prepared/llmip_*.csv                           │
│       │                                                               │
│       ├── llmip_train_features.csv (temporal + load features)       │
│       ├── llmip_train_targets.csv (118 voltage values)              │
│       ├── llmip_test_features.csv                                   │
│       └── llmip_test_targets.csv                                     │
│                                                                       │
│  PIPELINE EXECUTION                                                  │
│       │                                                               │
│       ├── Phase 1: LLM analyzes 30 training samples                 │
│       │         → Analysis string (feeds Phases 2 & 3)               │
│       │                                                               │
│       ├── Phase 2: LLM predicts 10 test cases                        │
│       │         → Predictions JSON                                   │
│       │                                                               │
│       ├── Phase 3: LLM extracts IF-THEN decision rules               │
│       │         → Rulebook.txt (human-readable)                     │
│       │                                                               │
│       └── Phase 4: Fresh LLM uses ONLY Rulebook                     │
│               → Replicability Score (R)                              │
│               R = 1.0 - (fresh_MAE / original_MAE) [grid]           │
│               R = correct_fresh / total [financial]                  │
│                                                                       │
│  RESULTS → results/                                                   │
│       ├── XGBoost baselines for comparison                           │
│       ├── Physical consistency checks                                 │
│       └── Final metrics: MAE, Accuracy, Replicability Score          │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Module Interaction Map

```
run_llmip_rebuilt.py
    ├── LLMClient class
    │   └── calls: GLM 4.7 API endpoint
    ├── load_grid_data()
    │   └── reads: data/prepared/llmip_*.csv
    ├── load_financial_data()
    │   └── reads: data/financial/sp500_*.csv
    ├── run_phase1() → analysis string
    ├── run_phase2() → predictions dict
    ├── run_phase3() → rulebook string
    ├── run_phase3_ablation() → prompt variations
    ├── parse_grid_predictions() → voltage dict
    ├── parse_financial_predictions() → 0/1
    └── calculate_replicability() → score

baseline_xgboost_shap.py
    ├── run_xgboost_grid()
    │   ├── XGBRegressor (MultiOutput)
    │   └── shap.TreeExplainer
    └── run_xgboost_financial()
        ├── XGBClassifier
        └── shap.TreeExplainer
```

---

## 5. RESEARCH GOALS & CURRENT STATUS

### Stated Research Objectives
1. **Novel Framework**: LLM-as-Interpretable-Predictor for tabular data
2. **Replicability Score**: New metric to validate if explanations contain genuine logic
3. **Dual-Environment Testing**: Stateless (grid) vs Stateful (financial)
4. **Sarkar (2024) Falsification**: Empirical test of "explanation vs. rationalization"

### Completed & Functional

| Experiment | Metric | Value | Status |
|------------|--------|-------|--------|
| Grid (IEEE 118-Bus) - Stateless | LLM MAE | 0.032 pu | ✅ Done |
| | XGBoost MAE | 0.002 pu | ✅ Done |
| | Replicability Score | **35.2%** | ✅ Done |
| | Physical Consistency | PASS | ✅ Done |
| Financial (S&P 500) - Stateful | LLM Accuracy | 60% | ✅ Done |
| | XGBoost Accuracy | 50% | ✅ Done |
| | Replicability Score | **70%** | ✅ Done |
| | Data Labels Removed | ✅ | ✅ Done |

### In Progress / Planned

- [ ] Re-run grid replicability with GLM 4.7 API
- [ ] Run prompt ablation (5 variations)
- [ ] Human evaluation (5-10 domain experts)
- [ ] Paper updates for submission

### Known Limitations

1. **Prediction Accuracy**: XGBoost significantly outperforms LLM on grid data
2. **Large Files Not in Git**: `gens_ts.csv` (184MB) and `loads_ts.csv` (72MB) exceed GitHub limits
3. **Context Window**: Full training data cannot fit in single prompt; uses 30-sample subset
4. **Prompt Sensitivity**: Results may vary with different prompt formulations

### Key Findings

1. **Grid**: XGBoost significantly outperforms LLM (0.002 vs 0.032 MAE)
2. **Financial**: Both near random (60% vs 50%) — market is hard to predict
3. **Replicability**: 35% (grid) vs 70% (financial) — rulebooks capture some logic
4. **Financial Labels**: Removed successfully — LLM no longer refuses

---

## 6. HOW TO GET STARTED

### Environment Setup

```bash
# Activate Python environment (REQUIRED)
source myenv/bin/activate

# Verify environment
which python  # Should point to myenv/bin/python

# Check .env exists and has keys
cat .env
```

### Run Grid Experiment

```bash
# Full 4-phase pipeline
python scripts/run_llmip_rebuilt.py --domain grid

# Skip phases (if re-running from cached results)
python scripts/run_llmip_rebuilt.py --domain grid --skip-phases 1,2

# Run with prompt ablation
python scripts/run_llmip_rebuilt.py --domain grid --ablation
```

### Run Financial Experiment

```bash
python scripts/run_llmip_rebuilt.py --domain financial
```

### Run Baselines

```bash
# XGBoost + SHAP for grid
python scripts/baseline_xgboost_shap.py --domain grid

# XGBoost + SHAP for financial
python scripts/baseline_xgboost_shap.py --domain financial
```

### Push to GitHub

```bash
# Using the snapshot script
source myenv/bin/activate
./save/llmip-ercot-snapshot.sh save "Your commit message"
```

### Required Data

All required data is already in the repository:
- `data/prepared/llmip_*.csv` — processed grid features/targets
- `data/financial/sp500_*.csv` — S&P 500 data
- `data/grid/samples/*.json` — raw power flow snapshots (optional)

### API Configuration

The pipeline uses GLM 4.7 API by default. Check `.env` for configuration:
```
MODAL_API_KEY=modalresearch_...
MODEL=zai-org/GLM-5-FP8
```

Fallback APIs (ZAI, OpenRouter) are available in the code.

---

## 7. KEY CONCEPTS & DESIGN DECISIONS

### Replicability Score (Core Innovation)

**Definition:**
```
R = 1 - (fresh_MAE / original_MAE)     [for continuous targets]
R = correct_fresh / total_samples       [for discrete targets]
```

**Interpretation:**
- R = 1.0: Fresh LLM perfectly replicates original predictions using only Rulebook
- R = 0.0: Rulebook contains no useful predictive logic
- R > 0.5: Rulebook captures substantial predictive logic

**Why It Matters:**
- Directly falsifies Sarkar (2024) claim that LLM explanations are "exoplanations"
- Provides empirical evidence for/against genuine in-context reasoning

### Stateless vs. Stateful Contrast

| Property | Grid (Stateless) | Financial (Stateful) |
|----------|-------------------|----------------------|
| Target memory | None (optimization resets daily) | Yes (path dependency) |
| Autocorrelation | Low | High |
| Physics-grounded | Yes (power flow equations) | No (market dynamics) |
| Expected Replicability | Higher | Lower (degradation expected) |

### Label Removal for Financial Data

The pipeline removes target labels (`target_direction`, `next_return`) from financial features before LLM processing. This prevents the LLM from refusing to make predictions on "financial advice" grounds—critical for obtaining meaningful results.

### Phase 3 Prompt Variations

The pipeline supports 5 prompt variations for Phase 3 (Rulebook extraction):
1. `standard` — Basic decision rules
2. `explicit_ifthen` — Explicit IF-THEN format
3. `stepbystep` — Step-by-step procedures
4. `quantitative` — Rules with formulas
5. `reproducible` — Focus on reproducibility

---

## 8. GLOSSARY

| Term | Definition |
|------|------------|
| **LLMIP** | LLM-as-Interpretable-Predictor — framework for transparent, self-documenting AI prediction |
| **Decision Rulebook** | Structured IF-THEN decision rules extracted from LLM predictions in natural language |
| **Replicability Score (R)** | Fraction by which a fresh LLM can reproduce predictions using only the Rulebook |
| **Stateless** | Prediction environment where target variable has no memory (grid optimization) |
| **Stateful** | Prediction environment with path dependency (financial markets) |
| **Exoplanation** | Sarkar (2024) term for post-hoc rationalization — explanation disconnected from actual reasoning |
| **IEEE 118-Bus** | Standard power system test case with 118 buses, 54 generators, 99 loads |
| **vm_pu** | Voltage magnitude in per-unit (pu) — normalized voltage measurement |
| **SHAP** | SHapley Additive exPlanations — feature importance attribution method |
| **MAE** | Mean Absolute Error — continuous prediction metric |
| **Accuracy** | Fraction correct — classification metric |
| **GLM 4.7 API** | Primary LLM API endpoint used for all pipeline phases |
| **Phase 1** | LLM training analysis — learns patterns from training data |
| **Phase 2** | LLM prediction — makes predictions on test data |
| **Phase 3** | Rulebook extraction — extracts decision rules |
| **Phase 4** | Replicability test — validates Rulebook quality |

### Variable Naming Conventions

- `vm_X` — voltage magnitude at bus X (e.g., `vm_1`, `vm_118`)
- `total_load_p_R*_mw` — total active power demand by region (MW)
- `total_load_q_R*_mvar` — total reactive power demand by region (MVar)
- `target_direction` — 0=DOWN, 1=UP for financial prediction

---

## QUICK REFERENCE

### One-Line Commands

```bash
# Activate environment
source myenv/bin/activate

# Run grid experiment
python scripts/run_llmip_rebuilt.py --domain grid

# Run baseline
python scripts/baseline_xgboost_shap.py --domain grid

# Push to GitHub
./save/llmip-ercot-snapshot.sh save "Your message"
```

### File Locations

| What | Where |
|------|-------|
| Main pipeline | `scripts/run_llmip_rebuilt.py` |
| Baseline | `scripts/baseline_xgboost_shap.py` |
| Results | `results/` |
| Data (prepared) | `data/prepared/` |
| Data (raw) | `data/grid/`, `data/financial/` |
| Thesis | `LLMIP_Thesis.txt` |
| Environment | `myenv/` |

---

*Last Updated: 2026-03-28*
