import pandas as pd
from preprocessing import Loader, Preprocessor
from eda import run_all_eda
from dataConfig import dataConfig

# ==========================================
# Initialize OOP Architecture
# ==========================================
config = dataConfig()
loader = Loader(config)
pre = Preprocessor(config)

print("1. Loading and cleaning data...")
# force_rebuild=True forces a fresh read of the SAS files and rebuilds the feather cache
train_df, test_df, flow, tr_report, te_report = loader.load_clean(force_rebuild=True)
print(f"Loaded Train shape: {train_df.shape}, Test shape: {test_df.shape}\n")

print("Exporting cleaned datasets to CSV...")
train_df.to_csv("clean_train.csv", index=False)
test_df.to_csv("clean_test.csv", index=False)
print("Saved 'clean_train.csv' and 'clean_test.csv' to your project folder.\n")

print("--- Impossible Value Cleaning Report (Train) ---")
print(tr_report[["rule", "n_values_changed"]].to_string(index=False))
print("\n")

print("2. Running strictly enforced sanity checks...")
# The sanity checks now test the OOP pipeline internally
pre.run_sanity_check(train_df, test_df)
print("\n")

print("3. Testing get_xy extraction...")
# Extract all feature blocks by leaving the blocks argument blank
X_train, y_train = pre.extractor.get_xy(train_df)
X_test, y_test = pre.extractor.get_xy(test_df)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}\n")

print("--- First 5 rows of X_train (Features) ---")
pd.set_option('display.max_columns', None)
print(X_train.head())
print("\n")

print("--- Target Distribution (y_train) ---")
print(y_train.value_counts(normalize=True).round(3))
print("\n")

print("--- Missingness in X_train (Before Imputation) ---")
print(X_train.isna().sum()[X_train.isna().sum() > 0])
print("\n")

print("4. Running EDA suite...")
# This uses the new dataConfig rules to generate the figures
eda_results = run_all_eda(train_df, test_df, flow=flow)
print("EDA complete. Check the ./figures/ directory.")