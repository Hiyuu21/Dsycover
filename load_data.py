import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier

# =============================================
# Columns
# =============================================

MANDATORY_COLUMNS = {
    'DEMO': ['SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'DMDEDUC2', 'INDFMPIR', 'RIDEXPRG'], 
    'BMX':  ['SEQN', 'BMXWT', 'BMXHT', 'BMXBMI', 'BMXWAIST'],
    'BPXO': ['SEQN', 'BPXOSY1', 'BPXOSY2', 'BPXOSY3', 'BPXODI1', 'BPXODI2', 'BPXODI3'],
    'BPQ':  ['SEQN', 'BPQ020', 'BPQ080'], 
    'SMQ':  ['SEQN', 'SMQ020', 'SMQ040'],
    'ALQ':  ['SEQN', 'ALQ111', 'ALQ121', 'ALQ130'],
    'PAQ':  ['SEQN', 'PAD680'],
    'SLQ':  ['SEQN', 'SLD012'],
    'MCQ':  ['SEQN', 'MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F'],
    'DIQ':  ['SEQN', 'DIQ010', 'DIQ050', 'DIQ070', 'DIQ160'],
    'GHB':  ['SEQN', 'LBXGH'],
    'GLU':  ['SEQN', 'LBXGLU'],
    'FASTQX': ['SEQN', 'PHAFSTHR']
}

# Optional columns (Weights, design variables, and updated questionnaires)
CYCLE_SPECIFIC_COLUMNS = {
    'DEMO': ['DMDMARTL', 'DMDMARTZ', 'WTINT2YR', 'WTMEC2YR', 'WTINTPRP', 'WTMECPRP', 'SDMVPSU', 'SDMVSTRA'],
    'BPQ':  ['BPQ050A', 'BPQ150', 'BPQ100D', 'BPQ101D', 'BPQ040A', 'BPQ090D']
}

FILE_MAP_1720 = {
    'DEMO': 'P_DEMO.XPT',
    'BMX':  'P_BMX.XPT',
    'BPXO': 'P_BPXO.XPT',
    'BPQ':  'P_BPQ.XPT',
    'SMQ':  'P_SMQ.XPT',
    'ALQ':  'P_ALQ.XPT',
    'PAQ':  'P_PAQ.XPT',
    'SLQ':  'P_SLQ.XPT',
    'MCQ':  'P_MCQ.XPT',
    'DIQ':  'P_DIQ.XPT',
    'GHB':  'P_GHB.XPT',
    'GLU':  'P_GLU.XPT',
    'FASTQX': 'P_FASTQX.XPT'
}

FILE_MAP_2123 = {
    'DEMO': 'DEMO_L.XPT',
    'BMX':  'BMX_L.XPT',
    'BPXO': 'BPXO_L.XPT',
    'BPQ':  'BPQ_L.XPT',
    'SMQ':  'SMQ_L.XPT',
    'ALQ':  'ALQ_L.XPT',
    'PAQ':  'PAQ_L.XPT',
    'SLQ':  'SLQ_L.XPT',
    'MCQ':  'MCQ_L.XPT',
    'DIQ':  'DIQ_L.XPT',
    'GHB':  'GHB_L.XPT',
    'GLU':  'GLU_L.XPT',
    'FASTQX': 'FASTQX_L.XPT'
}

# Define the dictionary to map original NHANES feature names to human-readable names
RENAME_DICT = {
    # Demographics
    "RIDAGEYR": "age",
    "RIAGENDR": "sex",
    "RIDRETH3": "race_ethnicity",
    "is_asian": "is_asian_descent",
    "DMDEDUC2": "education_level",
    "marital_status": "marital_status",
    "INDFMPIR": "income_poverty_ratio",
    
    # Body Measures
    "BMXBMI": "body_mass_index",
    "BMXWAIST": "waist_circumference_cm",
    "waist_height_ratio": "waist_height_ratio",
    
    # Vitals & Medical History
    "sbp_mean": "systolic_bp_mean",
    "dbp_mean": "diastolic_bp_mean",
    "BPQ020": "history_high_bp",             
    "bp_med": "taking_bp_meds",
    "BPQ080": "history_high_cholesterol",    
    "chol_med": "taking_cholesterol_meds",
    "cvd_history": "history_cvd",
    
    # Lifestyle & Habits
    "smoking_status": "smoking_status",
    "alcohol_drinks_on_drinking_days": "alcohol_drinks_on_drinking_days",
    "PAD680": "sedentary_minutes_per_day",
    "SLD012": "sleep_hours_weekday",         
    
    # Target
    "target": "undiagnosed_dysglycemia" 
}

FEATURE_BLOCKS = {
    'demographics': ['age', 'sex', 'education_level', 'income_poverty_ratio', 'marital_status'],
    'body':         ['body_mass_index', 'waist_circumference_cm', 'waist_height_ratio'],
    'blood_pressure': ['systolic_bp_mean', 'diastolic_bp_mean'],
    'clinical_history': ['history_high_bp', 'taking_bp_meds', 'history_high_cholesterol',
                         'taking_cholesterol_meds', 'history_cvd'],
    'lifestyle':    ['smoking_status', 'alcohol_drinks_on_drinking_days',
                     'sedentary_minutes_per_day', 'sleep_hours_weekday'],
}
TARGET = 'undiagnosed_dysglycemia'

# =============================================
# Data Class
# =============================================

class NHANES:

    def __init__(self):
        self.data_dir = Path('./data/raw')
        self.cache_dir = Path('./data/cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_1720 = self.cache_dir / 'df_1720.feather'
        self.cache_2123 = self.cache_dir / 'df_2123.feather'
        self.flow = []
       
    def _log(self, cycle, step, df):
        self.flow.append({'cycle': cycle, 'step': step, 'n': len(df)})

    def read_path(self, path, file_type):
        try:
            df = pd.read_sas(self.data_dir / path)
        except FileNotFoundError:
            raise FileNotFoundError(f'Missing raw file: {self.data_dir / path}.')
        df.columns = [col.decode('utf-8') if isinstance(col, bytes) else str(col) for col in df.columns]
        num = df.select_dtypes('number').columns
        df[num] = df[num].mask(df[num].abs() < 1e-30, 0.0)
        missing_mandatory = [col for col in MANDATORY_COLUMNS[file_type] if col not in df.columns]
        if missing_mandatory: raise ValueError(f"Missing mandatory columns in {path}: {missing_mandatory}")

        keep_cols = MANDATORY_COLUMNS[file_type].copy()
    
        if file_type in CYCLE_SPECIFIC_COLUMNS:
            for col in CYCLE_SPECIFIC_COLUMNS[file_type]:
                if col in df.columns:
                    keep_cols.append(col)
    
        return df[keep_cols].copy()

    def replace_missing_codes(self, df):
        """Converts NHANES Refused (7) and Don't Know (9) codes to np.nan."""
        # 1-digit codes
        cols_1_digit = ['DMDEDUC2', 'BPQ020', 'BPQ050A', 'BPQ150', 'BPQ080', 'BPQ100D', 'BPQ101D','BPQ090D', 'BPQ040A', 
                        'SMQ020', 'SMQ040', 'ALQ111', 'MCQ160B', 'MCQ160C', 
                        'MCQ160D', 'MCQ160E', 'MCQ160F', 'DIQ010', 'DIQ050', 'DIQ070', 'DIQ160']
        for col in cols_1_digit:
            if col in df.columns:
                df[col] = df[col].replace({7.0: np.nan, 9.0: np.nan})
                
        # 2-digit codes
        cols_2_digit = ['ALQ121', 'SLD012', 'DMDMARTL', 'DMDMARTZ']
        for col in cols_2_digit:
            if col in df.columns:
                df[col] = df[col].replace({77.0: np.nan, 99.0: np.nan})
                
        # 3-digit and 4-digit codes
        if 'ALQ130' in df.columns:
            df['ALQ130'] = df['ALQ130'].replace({777.0: np.nan, 999.0: np.nan})
        if 'PAD680' in df.columns:
            df['PAD680'] = df['PAD680'].replace({7777.0: np.nan, 9999.0: np.nan})
            
        return df

    def merge_files(self, dfs_dict):
        """Left merges all dataframes sequentially onto the DEMO base file."""
        df_master = dfs_dict['DEMO'].copy()
        
        for key, df_right in dfs_dict.items():
            if key == 'DEMO':
                continue
            # Strict 1-to-1 Left Join ensures no row duplication and no orphaned records
            df_master = df_master.merge(df_right, on='SEQN', how='left', validate='one_to_one')
            
        return df_master

    def derive_demographics(self, df):
        # 6:Non-hispanic Asian
        df['is_asian'] = np.where(df['RIDRETH3'] == 6, 1, 0) 
        df['waist_height_ratio'] = df['BMXWAIST'] / df['BMXHT']

        if 'DMDMARTZ' in df.columns:
            df['marital_status'] = df['DMDMARTZ']
        elif 'DMDMARTL' in df.columns:
            # Map 2015-2016 6-category code to the 3-category standard
            mapping = {1:1, 6:1, 2:2, 3:2, 4:2, 5:3}
            df['marital_status'] = df['DMDMARTL'].map(mapping)
        else:
            df['marital_status'] = np.nan

        return df

    def derive_medical_history(self, df):
        # Blood Pressure Means
        df['sbp_mean'] = df[['BPXOSY1', 'BPXOSY2', 'BPXOSY3']].mean(axis=1)
        df['dbp_mean'] = df[['BPXODI1', 'BPXODI2', 'BPXODI3']].mean(axis=1)

        # Composite cardiovascular (CVD) history (1 if any are 1)
        cvd_cols = ['MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F']
        all_cvd_missing = df[cvd_cols].isna().all(axis=1)
        any_cvd_yes = (df[cvd_cols] == 1).any(axis=1)

        df['cvd_history'] = np.select(
            [all_cvd_missing, any_cvd_yes],
            [np.nan, 1.0],
            default=0.0 # Not missing AND Not True
        )
        return df

    def derive_medical_condition(self, df):
        # Blood Pressure Medication
        bp_med_cond = [
            (df['BPQ020'] == 2), # Unmedicated (Never told)
            (df['cycle'] == '2021-2023') & (df['BPQ150'] == 1),
            (df['cycle'] == '2021-2023') & (df['BPQ150'] == 2),
            (df['cycle'] == '2017-2020') & (df['BPQ050A'] == 1),
            (df['cycle'] == '2017-2020') & ((df['BPQ040A'] == 2) | (df['BPQ050A'] == 2))
        ]
        df['bp_med'] = np.select(bp_med_cond, [0, 1, 0, 1, 0], default=np.nan)
    
        # Cholesterol Medication
        chol_med_cond = [
            (df['BPQ080'] == 2), # Unmedicated (Never told)
            (df['cycle'] == '2021-2023') & (df['BPQ101D'] == 1),
            (df['cycle'] == '2021-2023') & (df['BPQ101D'] == 2),
            (df['cycle'] == '2017-2020') & (df['BPQ100D'] == 1),
            (df['cycle'] == '2017-2020') & ((df['BPQ090D'] == 2) | (df['BPQ100D'] == 2))
        ]
        df['chol_med'] = np.select(chol_med_cond, [0, 1, 0, 1, 0], default=np.nan)

        return df

    def derive_habit(self, df):
        # Smoking Status
        smk_conditions = [
            (df['SMQ020'] == 1) & (df['SMQ040'].isin([1, 2])),
            (df['SMQ020'] == 1) & (df['SMQ040'] == 3),
            (df['SMQ020'] == 2)
        ]
        df['smoking_status'] = np.select(smk_conditions, [1, 2, 3], default=np.nan)
    
        # Alcohol
        df['alcohol_drinks_on_drinking_days'] = np.where((df['ALQ111'] == 2) | (df['ALQ121'] == 0), 0.0, df['ALQ130'])
        return df

    def derive_targets(self, df):
        # 1. Define Laboratory Truths
        lab_dm = (df['LBXGH'] >= 6.5) | ((df['LBXGLU'] >= 126) & (df['PHAFSTHR'] >= 8))
        lab_pre = (~lab_dm) & (
            ((df['LBXGH'] >= 5.7) & (df['LBXGH'] < 6.5)) | 
            ((df['LBXGLU'] >= 100) & (df['LBXGLU'] < 126) & (df['PHAFSTHR'] >= 8))
        )
        
        # 2. Create the Binary Target: 1 = Undiagnosed Dysglycemia, 0 = Normal
        # We only assign 1 or 0 if they explicitly answered "No" to having diabetes (DIQ010 == 2)
        eligible = (df['DIQ010'] == 2) & (df['DIQ160'] != 1) & (df['DIQ050'] != 1) & (df['DIQ070'] != 1)
        conditions = [
            eligible & (lab_dm | lab_pre), # 1: Undiagnosed (Pre-DM or DM)
            eligible & (~lab_dm) & (~lab_pre) # 0: Strictly Normal
        ]
        
        # Call it 'target' (or 'y_undx_dysgly' if you prefer)
        df['target'] = np.select(conditions, [1.0, 0.0], default=np.nan)
        
        # 3. Drop the known cases and people with missing DIQ010 responses
        # By dropping NaNs in 'target', we isolate the SCREEN-ELIGIBLE population!
        cycle = df['cycle'].iloc[0]
        df = df.dropna(subset=['target']).copy()
        self._log(cycle, 5, df) 
        return df

    def derive_all(self, df):
        df = self.derive_demographics(df)
        df = self.derive_medical_history(df)
        df = self.derive_medical_condition(df)
        df = self.derive_habit(df)
        df = self.derive_targets(df)
        df = df.rename(columns=RENAME_DICT)
        return df

    def load_raw_data(self, path_dict, cycle):
        dfs = {}
        for name, path in path_dict.items():
            dfs[name] = self.read_path(path, name)

        # Merge all dataframes into a single master df
        df = self.merge_files(dfs)
        df['cycle'] = cycle
        df = self.replace_missing_codes(df)
        self._log(cycle, 1, df)

        # Excluding respondents under 18 and pregnant respondents
        df = df[df['RIDAGEYR'] >= 18].copy()
        self._log(cycle, 2, df)

        if 'RIDEXPRG' in df.columns:
            df = df[df['RIDEXPRG'] != 1.0].copy()
        self._log(cycle, 3, df)

        # Drop missing HbA1c
        df = df.dropna(subset=['LBXGH']).copy()
        self._log(cycle, 4, df)
        
        # Drop pregnancy flag as it is no longer needed
        df = df.drop(columns=['RIDEXPRG'], errors='ignore')

        for cols in CYCLE_SPECIFIC_COLUMNS.values():
            for col in cols:
                if col not in df.columns:
                    df[col] = np.nan
        
        # Cache the result
        if cycle == '2017-2020':
            df.reset_index(drop=True).to_feather(self.cache_1720)
        elif cycle == '2021-2023':
            df.reset_index(drop=True).to_feather(self.cache_2123)
        else:
            raise ValueError(f"Unknown cycle: {cycle}")

        return df

    def get_nhanes(self, force_rebuild=False):
        if self.cache_1720.exists() and not force_rebuild:
            print("--- Loading 2017-2020 data from cache")
            df_1720 = pd.read_feather(self.cache_1720)
            self.flow = pd.read_json(self.cache_dir / 'flow.json').to_dict('records')

        else:
            print("--- Loading 2017-2020 RAW data")
            df_1720 = self.load_raw_data(FILE_MAP_1720, '2017-2020')

        if self.cache_2123.exists() and not force_rebuild:
            print("--- Loading 2021-2023 data from cache")
            df_2123 = pd.read_feather(self.cache_2123)
            self.flow = pd.read_json(self.cache_dir / 'flow.json').to_dict('records')

        else:
            print("--- Loading 2021-2023 RAW data")
            df_2123 = self.load_raw_data(FILE_MAP_2123, '2021-2023')

        # Apply derivations sequentially to 2017-2020 & 2021-2023
        df_1720 = self.derive_all(df_1720)
        df_2123 = self.derive_all(df_2123)

        pd.DataFrame(self.flow).drop_duplicates(['cycle','step'], keep='last').to_json(self.cache_dir/'flow.json', orient='records')
        return df_1720, df_2123

    """def get_xy(self, df, blocks):
        cols = [c for b in blocks for c in FEATURE_BLOCKS[b]]
        return df[cols], df[TARGET]"""


if __name__ == "__main__":
    nh = NHANES()
    df_tr, df_te = nh.get_nhanes(force_rebuild=False)

    print("\n" + "="*65)
    print("1. COHORT FLOW (Row counts at each filtering step)")
    print("="*65)
    
    # Map the step numbers from your code to readable descriptions
    step_names = {
        1: "1. Merged Raw Files (All ages)",
        2: "2. Adults (Age >= 18)",
        3: "3. Excluded Pregnant",
        4: "4. Dropped Missing HbA1c",
        5: "5. Target Eligible (No known DM)"
    }
    
    flow_df = pd.DataFrame(nh.flow).drop_duplicates(['cycle', 'step'], keep='last')
    # Create a new column with the readable names
    flow_df['step_desc'] = flow_df['step'].map(step_names)
    
    # Pivot using the descriptive column instead of the number
    print(flow_df.pivot(index='step_desc', columns='cycle', values='n'))
    print("\n")


    print("="*65)
    print("2. TARGET BALANCE & ASIAN SUBGROUP")
    print("="*65)
    for name, d in [('TRAIN (2017-2020)', df_tr), ('TEST (2021-2023)', df_te)]:
        print(f"--- {name} ---")
        print("Target counts:")
        print(d[TARGET].value_counts(dropna=False))
        print(f"Positive rate: {d[TARGET].mean().round(3)}")
        print("Asian subgroup breakdown (0=Non-Asian, 1=Asian):")
        print(d.groupby('is_asian_descent')[TARGET].agg(['count', 'sum']))
        print("")
    print("\n")

    print("="*65)
    print("3. MISSINGNESS PER FEATURE (Proportion of NaNs)")
    print("="*65)
    feats = [c for b in FEATURE_BLOCKS.values() for c in b]
    missingness = pd.concat([df_tr[feats].isna().mean(), df_te[feats].isna().mean()], axis=1, keys=['train', 'test'])
    print(missingness.round(3))
    print("\n")


    print("="*65)
    print("4. LEARNING CURVE (Training Size vs Validation AUC)")
    print("="*65)
    from sklearn.model_selection import learning_curve, StratifiedKFold
    from sklearn.ensemble import HistGradientBoostingClassifier
    import numpy as np

    X, y = nh.get_xy(df_tr, list(FEATURE_BLOCKS))   
    
    print("Running learning curve (this may take a minute or two)...")
    sizes, tr_sc, va_sc = learning_curve(
        HistGradientBoostingClassifier(random_state=0), X, y,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=StratifiedKFold(5, shuffle=True, random_state=0),
        scoring='roc_auc', n_jobs=-1
    )
    
    print("\nResults:")
    print(f"{'Train Size':>12} | {'Train AUC':>10} | {'Validation AUC':>15}")
    print("-" * 43)
    for n, tr_s, va_s in zip(sizes, tr_sc.mean(axis=1), va_sc.mean(axis=1)):
        print(f"{int(n):>12,} | {tr_s:>10.4f} | {va_s:>15.4f}")
    print("="*65 + "\n")

    print("--- Data Types ---")
    print(df_tr.dtypes)

    print("\n--- Descriptive Statistics (Continuous & Ordinal - Train) ---")
    all_features = [feat for block in FEATURE_BLOCKS.values() for feat in block]
    print(df_tr[all_features].describe().T.round(4))   

    cat_features = [
        'sex', 'education_level', 'marital_status', 'smoking_status',
        'history_high_bp', 'taking_bp_meds', 'history_high_cholesterol',
        'taking_cholesterol_meds', 'history_cvd', 'is_asian_descent'
    ]
    for col in cat_features:
        print(f"\n--- {col} ---")
        # Get counts for train and test
        train_counts = df_tr[col].value_counts(dropna=False).rename('Train')
        test_counts = df_te[col].value_counts(dropna=False).rename('Test')
        
        # Concatenate them side-by-side (fillna(0) handles if a category exists in one but not the other)
        comparison = pd.concat([train_counts, test_counts], axis=1).fillna(0).astype(int)
        print(comparison)