# Code Explanation
**Generated:** March 27, 2026, 02:05:43 AM
**Language:** Python

---

This notebook, `01_data_preparation.ipynb`, is designed to explore and prepare the **IEEE 118-Bus system power flow dataset** for a machine learning project called LLMIP. It acts as a bridge between raw simulation outputs (complex JSON files) and a structured format suitable for training models, ensuring the data is valid and well-understood.

### Logical Blocks

*   **Environment Setup and Path Management:**
    The notebook begins by importing essential libraries like `pandas` and `pathlib` to handle data and file systems. It defines specific directory paths for raw data, samples, and prepared output, ensuring that the script can locate the 8,760 hourly simulation samples and their corresponding metadata files.

*   **Raw Data Inventory:**
    This section scans the `data/grid` directory to list the available files and their sizes. It specifically filters for JSON files containing power flow results. This block is crucial for verifying that the dataset is complete (e.g., checking if all 8,760 hours of the year are present) and that the files aren't corrupted.

*   **Grid Metadata Exploration (Buses, Generators, Loads):**
    The code loads and analyzes CSV files representing the static structure of the power grid.
    *   **Buses:** Identifies the 118 buses, their regions, and which bus acts as the "Slack" bus (the reference point for the grid).
    *   **Generators:** Categorizes power sources by type (solar, wind, biomass, etc.) and inspects their operating limits.
    *   **Loads:** Maps electricity consumption points to specific buses.

*   **Nested JSON Parsing Logic:**
    The raw simulation samples are stored in a specialized, nested JSON format (likely exported from the `pandapower` library). This block defines a helper function, `parse_df`, which extracts specific tables (like bus voltages or generator outputs) from a complex "split-format" string inside the JSON and converts them back into usable Pandas DataFrames.

*   **Results Validation and Statistics:**
    Once the JSON is parsed, the notebook inspects the `res_bus` (bus results) and `res_gen` (generator results) tables. It calculates statistics for voltage magnitudes (`vm_pu`) to ensure they stay within stable limits (typically around 0.8 to 1.2 per-unit) and confirms that the power flow simulation actually "converged" (solved successfully) before the data is used for training.

*   **Prepared Data Verification:**
    Finally, the notebook checks for a `llmip_stats.json` file. This file stores metadata about the processed dataset, such as the number of features, the number of training/testing samples, and the total number of buses, providing a quick summary of the dataset's final state.

---

### Key Ideas

*   **Data Wrangling:** Converting complex, nested JSON simulation results into flat, tabular data for machine learning.
*   **Grid Topography:** Understanding how 118 buses, various generator types (solar/wind), and loads are connected.
*   **Sanity Checking:** Validating that the electrical simulations are physically realistic and mathematically solved.