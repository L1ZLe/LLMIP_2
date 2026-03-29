# Code Explanation
**Generated:** March 27, 2026, 01:59:59 AM
**Language:** Python

---

This script processes raw IEEE 118-Bus power flow simulation data to create a structured dataset for machine learning. It transforms complex grid snapshots into a clean format with input features (like regional demand and time of day) and target outputs (voltage magnitudes for all 118 buses).

### Logical Block Explanation

*   **Metadata Extraction & Mapping (`extract_bus_metadata`, `extract_gen_metadata`, etc.)**  
    These functions load static CSV files that define the "skeleton" of the power grid. They map each bus to a specific region (R1, R2, or R3) and categorize generators by fuel type (solar, wind, thermal, etc.). This provides the context needed to group individual bus data into meaningful regional aggregates.

*   **PandaPower JSON Parsing (`parse_pandapower_json`)**  
    The simulation data is stored in a complex, nested JSON format used by the `pandapower` library. This block implements a custom parser to "unpack" these JSON objects into standard Pandas DataFrames, specifically extracting power flow results for buses, loads, and generators.

*   **Feature Engineering (`extract_features_and_targets`)**  
    This is the core logic that builds the model's inputs. It calculates:
    *   **Temporal Features:** Converts timestamps into sine/cosine waves so the model understands that 11:00 PM is close to 1:00 AM.
    *   **Regional Aggregates:** Sums up real and reactive power demand and generation by region and fuel type.
    *   **Data Leakage Prevention:** Crucially, it excludes voltage statistics from the features, as these are the values the model is meant to predict.

*   **Stratified Sampling (`run` - sampling section)**  
    Instead of picking snapshots at random, the script uses stratified sampling to ensure the 200 chosen snapshots are spread evenly across different months of the year. This ensures the final dataset captures seasonal variations in power usage and generation.

*   **Data Partitioning & Export (`run` - final section)**  
    After processing, the script splits the data into a **Training Set (80%)** and a **Test Set (20%)**. It then saves these as CSV files along with a `stats.json` file that provides a high-level summary of the dataset's characteristics (e.g., total regions, bus counts, and voltage ranges).

---

### Key Ideas

*   **Smart Data Compression:** Simplifies a massive grid of 118 buses into regional "summaries" that are easier for AI models to process.
*   **Cyclical Time Encoding:** Uses math (sine/cosine) to help the AI understand that time is a repeating cycle, not just a linear number.
*   **Real-World Reliability:** Ensures the data is physically "valid" by checking if the power flow simulation actually converged before including it in the dataset.