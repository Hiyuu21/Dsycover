import pandas as pd
from preprocessing import load_clean, get_xy, run_sanity_checks
from eda import run_all_eda

"""
# 1. Load the cleaned data (this handles the raw merging, derivations, and impossible-value blanking all at once)
train_df, test_df, flow = load_clean(force_rebuild=False)

# 2. Extract X and y for your models (this automatically enforces the leakage checks)
# You can pass specific blocks, or leave it blank to get all features
X_train, y_train = get_xy(train_df)
X_test, y_test = get_xy(test_df)
"""

print("1. Loading and cleaning data...")
# force_rebuild=False will use your .feather cache if it exists
train_df, test_df, flow = load_clean(force_rebuild=True)
print(f"Loaded Train shape: {train_df.shape}, Test shape: {test_df.shape}\n")

print("2. Running strictly enforced sanity checks...")
# This will crash if there is leakage, NaN targets, or column mismatches
run_sanity_checks(train_df, test_df)
print("\n")

print("3. Testing get_xy extraction...")
# Extract all feature blocks by leaving the blocks argument blank
X_train, y_train = get_xy(train_df)
X_test, y_test = get_xy(test_df)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}\n")

print("--- First 5 rows of X_train (Features) ---")
# pd.set_option ensures pandas doesn't truncate the columns in the terminal
pd.set_option('display.max_columns', None)
print(X_train.head())
print("\n")

print("--- Target Distribution (y_train) ---")
print(y_train.value_counts(normalize=True).round(3))
print("\n")

print("--- Missingness in X_train (Before Imputation) ---")
# Shows the raw NaNs that the Pipeline will eventually impute
print(X_train.isna().sum()[X_train.isna().sum() > 0])
print("\n")

print("4. Running EDA suite...")
# This tests if the plotting functions work and saves to ./figures
eda_results = run_all_eda(train_df, test_df, flow=flow)
print("EDA complete. Check the ./figures/ directory.")