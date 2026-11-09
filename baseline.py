import os
import pandas as pd
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold

"""
Updated baseline.py - This script checks off all your requirements for the team handoff:

1. Repeated CV: Added RepeatedStratifiedKFold(n_splits=5, n_repeats=3) for much more stable scoring across 15 total folds.
2. Standalone Blocks: Added Demographics_Only, Body_Only, etc., to test each category's isolation power before doing the cumulative tiers.
3. Standard Deviations & ROC-AUC Primary: It now calculates the std() for both metrics and emphasizes ROC-AUC.
4. Updated Unpacking: It uses the new 5-variable load_clean() unpack so it doesn't crash.
"""

"""
- AI Prompt for Guidance
========================

Act as a Scikit-Learn syntax assistant. I am working on a university machine learning assignment with strict rules against AI-generated complete solutions. I only need syntax help for my specific component.
My assigned baseline models are: [INSERT YOUR TWO MODELS HERE, e.g., Decision Tree and Random Forest].

Please provide ONLY the following three things:
1. The exact import statements for these two models in scikit-learn.
2. The Python code to instantiate both models using their default hyperparameters. If the model has a random_state parameter, you MUST set it to a variable named RANDOM_STATE.
3. State briefly whether each model mathematically requires feature scaling (like StandardScaler) or if scaling should be set to None.

DO NOT write a loop, DO NOT write data loading steps, and DO NOT write evaluation code. Only provide the imports, the instantiation variables, and the scaling advice.
"""

# ==========================================
# TODO 1: Import your assigned models here
# ==========================================
# e.g., from sklearn.linear_model import LogisticRegression
# e.g., from sklearn.naive_bayes import GaussianNB


from dataConfig import dataConfig
from preprocessing import Loader, Preprocessor, FeatureExtracter

config = dataConfig()
loader = Loader(config)
pre = Preprocessor(config)
extractor = FeatureExtracter(config)

def run_member_evaluation():
    print("Loading strictly filtered training data...")
    train_df, test_df, flow, tr_report, te_report = loader.load_clean(force_rebuild=False)

    tiers = {
        "Block_Demographics": ["demographics"],
        "Block_Body": ["body"],
        "Block_BP": ["blood_pressure"],
        "Block_History": ["clinical_history"],
        "Block_Lifestyle": ["lifestyle"],
        
        "Tier1_Demo": ["demographics"],
        "Tier2_PlusBody": ["demographics", "body"],
        "Tier3_PlusBP": ["demographics", "body", "blood_pressure"],
        "Tier4_PlusHistory": ["demographics", "body", "blood_pressure", "clinical_history"],
        "Tier5_AllFeatures": ["demographics", "body", "blood_pressure", "clinical_history", "lifestyle"]
    }

    # ==========================================
    # TODO 2: Instantiate your two assigned models
    # Use default hyperparameters, but ALWAYS set random_state=RANDOM_STATE if the model accepts it.
    # ==========================================
    model_1 = None # Replace with your first model
    model_2 = None # Replace with your second model
    
    # ==========================================
    # TODO 3: Does your model need scaling? 
    # Linear/Distance/NN models need "standard" or "robust". Tree models can use None.
    # ==========================================
    scale_1 = "standard" 
    scale_2 = "standard" 

    my_models = [(model_1, scale_1), (model_2, scale_2)]
    results = []

    cv_strategy = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=config.random_state)

    for current_model, current_scale in my_models:
        if current_model is None: continue
            
        model_name = current_model.__class__.__name__
        print(f"\nEvaluating {model_name}...")
        
        for tier_name, block_list in tiers.items():
            X, y = extractor.get_xy(train_df, blocks=block_list)
            
            pipe = pre.make_pipeline(current_model, blocks=block_list, scale=current_scale, missing_indicators=True)
            
            cv_scores = cross_validate(
                pipe, X, y, 
                cv=cv_strategy, 
                scoring=["roc_auc", "average_precision"], 
                n_jobs= max(1, os.cpu_count() - 2)
            )
            
            roc_auc_mean = cv_scores["test_roc_auc"].mean()
            roc_auc_std = cv_scores["test_roc_auc"].std()
            pr_auc_mean = cv_scores["test_average_precision"].mean()
            pr_auc_std = cv_scores["test_average_precision"].std()
            
            print(f"  {tier_name}: ROC-AUC = {roc_auc_mean:.4f} (±{roc_auc_std:.4f}) | PR-AUC = {pr_auc_mean:.4f} (±{pr_auc_std:.4f})")
            
            results.append({
                "Model": model_name,
                "Tier": tier_name,
                "ROC_AUC_mean": roc_auc_mean,
                "ROC_AUC_std": roc_auc_std,
                "PR_AUC_mean": pr_auc_mean,
                "PR_AUC_std": pr_auc_std
            })

    # Save to CSV for team aggregation
    if results:
        scorecard = pd.DataFrame(results)
        scorecard = scorecard.sort_values(by=['Model', 'ROC_AUC_mean'], ascending=[True, False])
        scorecard.to_csv("my_models_scorecard.csv", index=False)
        print("\nSaved 'my_models_scorecard.csv'.")

if __name__ == "__main__":
    run_member_evaluation()