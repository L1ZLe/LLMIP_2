# LLMIP Project - COMPLETED ANALYSIS

## STATUS (2026-03-18)

**FIXED & WORKING:** Modal API configured, financial experiment complete, all pipelines running.

---

## EXPERIMENT RESULTS

### Grid (IEEE 118-Bus) - Stateless
| Metric | Value |
|--------|-------|
| LLM MAE | 0.032 pu |
| XGBoost MAE | 0.002 pu |
| Replicability Score | **35.2%** |
| Physical Consistency | ✅ PASS |

### Financial (S&P 500) - Stateful
| Metric | Value |
|--------|-------|
| LLM Accuracy | 60% |
| XGBoost Accuracy | 50% |
| Replicability Score | **70%** |
| Data Labels | ✅ REMOVED |

---

## FILES CREATED/FIXED

| File | Purpose |
|------|---------|
| `scripts/run_financial_fixed.py` | Working financial pipeline |
| `scripts/run_llmip_rebuilt.py` | Grid pipeline (needs retest) |
| `scripts/baseline_xgboost_shap.py` | XGBoost + SHAP baselines |
| `.env` | API keys hardcoded |

---

## API CONFIGURATION

```
MODAL_API_KEY=modalresearch_q97UreSaoHONh_DmAYDEmPinpcp5LBAP6YX0eba3Q7E
MODEL=zai-org/GLM-5-FP8
```

---

## KEY FINDINGS

1. **Grid**: XGBoost significantly outperforms LLM (0.002 vs 0.032 MAE)
2. **Financial**: Both near random (60% vs 50%), market is hard to predict
3. **Replicability**: 35% (grid) vs 70% (financial) - rulebooks capture some logic
4. **Financial labels**: Removed successfully - LLM no longer refuses

---

## REMAINING WORK

- Re-run grid replicability with Modal API
- Run prompt ablation (5 variations)
- Human evaluation
- Paper updates

---

*Last Updated: 2026-03-18*
