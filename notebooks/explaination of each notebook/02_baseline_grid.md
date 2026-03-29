# Code Explanation
**Generated:** March 27, 2026, 02:07:20 AM
**Language:** Python

---

This notebook establishes a baseline machine learning model to predict voltage magnitudes across an electrical grid (specifically the IEEE 118-bus system). It uses the **XGBoost** algorithm to learn the relationship between grid conditions and bus voltages, then utilizes **SHAP** (SHapley Additive exPlanations) to identify which features are most influential in making those predictions.

### Logical Block Breakdown

*   **Setup and Data Loading**
    The code imports standard data science libraries (`pandas`, `numpy`) and specialized ML tools (`xgboost`, `shap`). It loads prepared CSV files containing features (grid inputs) and targets (bus voltages). It specifically filters the target data to focus only on `vm_pu` (voltage magnitude in per-unit), which is a standard way to measure grid health.

*   **Bus Categorization**
    Buses in a power grid behave differently: **Generator buses** usually have controlled, stable voltages, while **Load buses** are passive and fluctuate based on demand. The code manually defines these categories to allow for more granular performance analysis later on.

*   **Multi-Output XGBoost Training**
    Since the grid has 118 different buses, the code uses a `MultiOutputRegressor`. This is a wrapper that effectively trains **118 individual XGBoost models**—one for every bus. Each model learns to predict the voltage for its specific location based on the global state of the grid.

*   **Performance Evaluation**
    The model's accuracy is measured using **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)**. The code calculates these metrics for the entire system and also breaks them down by bus type (Generator vs. Load). This helps identify if the model is more accurate in predicting controlled generator voltages versus more volatile load voltages.

*   **SHAP Interpretability**
    To avoid the "black box" nature of machine learning, the code uses `shap.TreeExplainer`. It iterates through each of the 118 models to calculate SHAP values, which quantify how much each input feature (like wind power or thermal generation) contributed to a specific voltage prediction. These values are then averaged to find the "top 10" most important features for the entire grid.

*   **Visualizations and Data Export**
    The code generates a multi-panel figure showing:
    1.  A ranking of error (MAE) across all buses.
    2.  The most important features according to SHAP.
    3.  A distribution of prediction errors.
    4.  A "Perfect vs. Actual" scatter plot for a sample bus.
    Finally, all metrics and feature rankings are saved to a JSON file for comparison with future experiments.

### Key Ideas
*   **Multi-Output Regression:** Trains a unique model for every location (bus) in the electrical grid simultaneously.
*   **Explainable AI:** Uses SHAP values to reveal which power sources (wind, solar, thermal) most impact grid voltage.
*   **Domain-Aware Analysis:** Evaluates performance differently for generator buses versus load buses to reflect real-world physics.