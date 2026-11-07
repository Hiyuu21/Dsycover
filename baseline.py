import pandas as pd
from sklearn.model_selection import cross_validate


"""
AI Prompt for Guidance
=========================

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


from preprocessing import load_clean, get_xy, make_pipeline, get_cv, RANDOM_STATE

def run_member_evaluation():
    print("Loading strictly filtered training data...")
    train_df, test_df, flow = load_clean(force_rebuild=False)

    tiers = {
        "T1_Demographics": ["demographics"],
        "T2_PlusBody": ["demographics", "body"],
        "T3_PlusBP": ["demographics", "body", "blood_pressure"],
        "T4_PlusHistory": ["demographics", "body", "blood_pressure", "clinical_history"],
        "T5_AllFeatures": ["demographics", "body", "blood_pressure", "clinical_history", "lifestyle"]
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

    for current_model, current_scale in my_models:
        if current_model is None:
            continue
            
        model_name = current_model.__class__.__name__
        print(f"\nEvaluating {model_name}...")
        
        for tier_name, block_list in tiers.items():
            X, y = get_xy(train_df, blocks=block_list)
            
            # Build pipeline: dynamically imputes and scales within CV folds to prevent leakage
            pipe = make_pipeline(current_model, blocks=block_list, scale=current_scale, missing_indicators=True)
            
            cv_scores = cross_validate(
                pipe, X, y, 
                cv=get_cv(n=5), 
                scoring=["roc_auc", "average_precision"], 
                n_jobs=-1
            )
            
            mean_pr_auc = cv_scores["test_average_precision"].mean()
            mean_roc_auc = cv_scores["test_roc_auc"].mean()
            
            print(f"  {tier_name}: PR-AUC = {mean_pr_auc:.4f} | ROC-AUC = {mean_roc_auc:.4f}")
            
            results.append({
                "Model": model_name,
                "Tier": tier_name,
                "PR_AUC": mean_pr_auc,
                "ROC_AUC": mean_roc_auc
            })

    # Save to CSV for team aggregation
    if results:
        scorecard = pd.DataFrame(results)
        scorecard.to_csv("my_models_scorecard.csv", index=False)
        print("\nSaved 'my_models_scorecard.csv'. Please share this with the team!")

if __name__ == "__main__":
    run_member_evaluation()