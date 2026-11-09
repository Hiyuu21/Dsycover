# Explainable ML Framework for Non-Invasive Prediabetes and Diabetes Risk Prediction

This project builds an automated, non-invasive triage screening model to identify undiagnosed dysglycemia using CDC NHANES survey data.

To prevent data leakage, a centralized, object-oriented data pipeline has been established. **Please read this document carefully before writing any model code.**

## 📂 Repository Structure

The core pipeline has been refactored into a modular Object-Oriented Programming (OOP) architecture to ensure consistency and scalability across the team:

1. **`dataConfig.py`**: The centralized configuration dataclass. It acts as the single source of truth for our random seed, feature blocks, data typing, and impossible-value cleaning rules.


2. **`load_data.py`**: Handles the raw SAS files, applies base inclusion/exclusion criteria, builds the target labels, and caches the data into fast `.feather` files. *(Note: Do not import from this file directly).*


3. **`preprocessing.py`**: The engine of our data pipeline. It contains the `Loader`, `Preprocessor`, and `FeatureExtracter` classes, which dynamically handle leakage guards, missing value imputation, and feature scaling.


4. **`test_run.py`**: The master execution script used to load the data, generate impossible-value cleaning reports, execute strict sanity checks, and trigger the automated EDA suite.


5. **`eda.py`**: The Exploratory Data Analysis module. It generates prevalence tables, missingness charts, and correlation heatmaps (saving them directly to `./figures/`). It strictly calculates outcomes on the *train* cycle only to prevent leakage.


6. **`baseline.py`**: The standardized evaluation loop. It utilizes a 15-fold `RepeatedStratifiedKFold` (5 folds × 3 repeats) to test both standalone feature blocks and cumulative tiers. **This is the file you will edit to run your assigned models.**



## ⚙️ Setup & Requirements

Ensure you have the required libraries installed. We are using `scikit-learn >= 1.2` and `pandas >= 1.5`. You will also need `pyarrow` to read the cached `.feather` files.
Use this to download the dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn pyarrow

```

or

```bash
py -m pip install pandas numpy scikit-learn matplotlib seaborn pyarrow

```

## 🚨 The Rules of the Pipeline

To ensure our methodology is flawless for the presentation, we have a few strict rules hard-coded into the pipeline:

* **The Test Set is Locked:** The 2021-2023 test cycle is touched exactly once at the end of Phase 4. We do not use it for EDA outcome prevalence, and we do not tune hyperparameters on it.


* **No Manual Data Cleaning in Scripts:** `preprocessing.py` dynamically handles missing values (Median/Mode imputation) and categorical encoding *inside* the cross-validation folds using Scikit-Learn Pipelines. This prevents data leakage.


* **Dependency Injection:** We pass the shared `dataConfig` object into our `Loader`, `Preprocessor`, and `FeatureExtracter` classes so the entire pipeline operates on the exact same rules.


* **Never extract X and y manually:** Always use `extractor.get_xy()`. It has built-in assert checks that will crash the program if a forbidden lab test (like HbA1c) accidentally leaks into your features.



## 🛠️ Instructions for Part A Modeling

For Part A, we are scoring our 8 assigned models with their **default hyperparameters** across standalone blocks and cumulative feature blocks (Tiers 1 through 5) to find the most *parsimonious* model.

### Our Assignments:

* **TF:** Logistic Regression (LR), Naive Bayes
* **PJ:** k-Nearest Neighbors (k-NN), Support Vector Machine (SVM)
* **Wayne:** Decision Tree, Random Forest
* **JY:** Boosting (XGBoost/LightGBM), Multi-Layer Perceptron (MLP)

### How to Run Your Models:

1. Open `baseline.py`.
2. Locate the `# TODO` sections.
3. Import your assigned models, instantiate them with default parameters (setting `random_state=config.random_state` where applicable), and specify if they need `"standard"` scaling or `None` (for trees).


4. Run the script. It will evaluate your models across all blocks and tiers using our shared Repeated Stratified K-Folds.


5. Send the generated `my_models_scorecard.csv` to the team so we can aggregate the ROC-AUC and PR-AUC scores!

### Model Evaluation Methodology

To ensure our pipeline is clinically viable and robust, we evaluate models using a multi-faceted metric suite rather than relying solely on PR-AUC:

* **Recall (Sensitivity):** Prioritized to capture as many true positive cases as possible, as early diabetes screening demands a low tolerance for missed diagnoses.
* **Precision & Specificity:** Monitored to balance high recall by minimizing false positive alarms, preventing unnecessary clinical follow-ups, patient anxiety, and resource strain.
* **F1-Score:** Utilized to gauge balanced performance (the harmonic mean of precision and recall) across our imbalanced medical dataset.
* **ROC-AUC:** Measured as the primary metric for Part A to assess the model's global discriminative power across all possible decision thresholds.


* **Brier Score:** Assessed to guarantee that the risk probabilities output by our models are well-calibrated and accurately reflect real-world clinical risk for safe patient triage.

### AI Prompt for Framework Help

Use the following prompt exactly to get just the syntax you need for the `# TODO` blocks:

> "Act as a Scikit-Learn syntax assistant. I am working on a university machine learning assignment with strict rules against AI-generated complete solutions. I only need syntax help for my specific component. My assigned baseline models are: **[INSERT YOUR TWO MODELS HERE]**. Please provide ONLY: 1) The exact `import` statements. 2) The Python code to instantiate both using default hyperparameters (set random_state to a variable named `config.random_state`). 3) State whether each mathematically requires feature scaling (like StandardScaler) or if scaling should be `None`. DO NOT write a loop or data loading steps."

## 📊 Generating EDA Plots

To view the data distributions, missingness, impossible value cleaning reports, and cohort flow, run the master test script:

```bash
python test_run.py

```

Check the `./figures/` folder for the generated `.png` and `.csv` files.