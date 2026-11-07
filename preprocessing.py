"""
preprocessing.py
================
Cleaning and preprocessing for the NHANES undiagnosed-dysglycaemia project.

Design rules (be ready to explain these in the presentation)
------------------------------------------------------------
1. The 2021-2023 TEST cycle is locked. Nothing in this file learns a value from
   it. Every fitted step (imputation values, scaling, one-hot categories) lives
   inside an sklearn Pipeline, so it is fitted on training folds only.
2. Rule-based cleaning (impossible values, binary recoding) uses fixed domain
   limits, not statistics from the data. That is why it may be applied to both
   cycles before modelling without leaking anything.
3. Lab columns (LBXGH, LBXGLU, PHAFSTHR) and DIQ* columns build the label and
   must never reach the feature matrix. get_xy() refuses to build X if they do.
4. race_ethnicity and is_asian_descent are kept ONLY for subgroup analysis, so
   they are not in FEATURE_BLOCKS and never enter X.

Assumptions about the environment: pandas >= 1.5, scikit-learn >= 1.2
(needs OneHotEncoder(sparse_output=...) and set_output).
"""
from __future__ import annotations

from typing import Iterable, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from load_data import NHANES, FEATURE_BLOCKS, TARGET

RANDOM_STATE = 42  # one fixed seed for anything random in this project

# =============================================================================
# 1. Feature typing
# =============================================================================
# Every model feature needs a type so the ColumnTransformer knows how to treat
# it. If a feature is added to FEATURE_BLOCKS and not listed here,
# resolve_features() stops with a clear message instead of guessing.
FEATURE_TYPES: dict[str, str] = {
    "age": "numeric",
    "sex": "binary",
    "education_level": "ordinal",            # codes 1-5, ordered low -> high
    "income_poverty_ratio": "numeric",
    "marital_status": "nominal",             # 1 married/partner, 2 wid/div/sep, 3 never
    "body_mass_index": "numeric",
    "waist_circumference_cm": "numeric",
    "waist_height_ratio": "numeric",
    "systolic_bp_mean": "numeric",
    "diastolic_bp_mean": "numeric",
    "history_high_bp": "binary",
    "taking_bp_meds": "binary",
    "history_high_cholesterol": "binary",
    "taking_cholesterol_meds": "binary",
    "history_cvd": "binary",
    "smoking_status": "nominal",             # 1 current, 2 former, 3 never
    "alcohol_drinks_on_drinking_days": "numeric",
    "sedentary_minutes_per_day": "numeric",
    "sleep_hours_weekday": "numeric",
}

# Binary columns arrive with mixed codings (NHANES 1=Yes/2=No for the BPQ items,
# 1=Male/2=Female for sex, 0/1 for the derived columns). Recoding everything to
# 0/1 means every binary column means the same thing ("1 = the flag is present").
# The dict doubles as a whitelist: any other value raises an error.
BINARY_ENCODING: dict[str, dict[float, float]] = {
    "sex": {1.0: 0.0, 2.0: 1.0},                       # 1 = female after recoding
    "history_high_bp": {1.0: 1.0, 2.0: 0.0},           # 1 = told high BP
    "history_high_cholesterol": {1.0: 1.0, 2.0: 0.0},  # 1 = told high cholesterol
    "taking_bp_meds": {0.0: 0.0, 1.0: 1.0},
    "taking_cholesterol_meds": {0.0: 0.0, 1.0: 1.0},
    "history_cvd": {0.0: 0.0, 1.0: 1.0},
}

# Only these two get an optional "was missing" flag: they have the most missing
# values (income ~14%, alcohol ~6-14% and different between cycles), so the fact
# that they are missing might itself carry information.
MISSING_FLAG_FEATURES = [
    "income_poverty_ratio", 
    "alcohol_drinks_on_drinking_days",
    "education_level",  # Added to catch structural missingness for ages 18-19
    "marital_status"    # Added to catch structural missingness for ages 18-19
    ]

# Columns that must never be model inputs (label construction or subgroup only).
FORBIDDEN_EXACT = {"LBXGH", "LBXGLU", "PHAFSTHR", TARGET, "race_ethnicity",
                   "is_asian_descent", "SEQN", "cycle"}
FORBIDDEN_PREFIXES = ("DIQ",)


def _require_columns(df: pd.DataFrame, cols: Iterable[str], where: str) -> None:
    """Raise a readable error if columns are missing (instead of a bare KeyError)."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"[{where}] missing column(s): {missing}. "
                       f"Available columns start with: {list(df.columns)[:8]} ...")


def _assert_no_leakage(features: Iterable[str]) -> None:
    """Stop if a lab, DIQ, target or subgroup column is about to become a feature."""
    bad = [f for f in features
           if f in FORBIDDEN_EXACT or f.startswith(FORBIDDEN_PREFIXES)]
    if bad:
        raise ValueError(f"Leakage guard: {bad} must not be model features.")


def resolve_features(blocks: Optional[Iterable[str]] = None) -> list[str]:
    """Turn block names into a validated, ordered list of feature names.

    blocks=None means all blocks. Validation happens here, once, so every other
    function can trust the list it receives.
    """
    names = list(FEATURE_BLOCKS) if blocks is None else list(blocks)
    unknown = [b for b in names if b not in FEATURE_BLOCKS]
    if unknown:
        raise KeyError(f"Unknown block(s) {unknown}. Choose from {list(FEATURE_BLOCKS)}.")
    features = [f for b in names for f in FEATURE_BLOCKS[b]]
    untyped = [f for f in features if f not in FEATURE_TYPES]
    if untyped:
        raise KeyError(f"No type assigned for {untyped}. Add them to FEATURE_TYPES.")
    _assert_no_leakage(features)
    return features


# =============================================================================
# 2. X / y extraction
# =============================================================================
def get_xy(df: pd.DataFrame, blocks: Optional[Iterable[str]] = None
           ) -> tuple[pd.DataFrame, pd.Series]:
    """Return the feature matrix X and the integer target y for the chosen blocks.

    X is deliberately left raw (NaNs and original codes kept): imputation and
    encoding are done by the Pipeline so they are fitted on training folds only.
    """
    features = resolve_features(blocks)
    _require_columns(df, features + [TARGET], "get_xy")
    y = df[TARGET]
    if y.isna().any():
        raise ValueError(f"{TARGET} has {int(y.isna().sum())} NaN. "
                         "Only screen-eligible rows should reach modelling.")
    if not set(y.unique()) <= {0.0, 1.0}:
        raise ValueError(f"{TARGET} must be 0/1, found {sorted(y.unique())}.")
    return df[features].copy(), y.astype(int)


# =============================================================================
# 3. Impossible-value cleaning
# =============================================================================
# (lower, upper, reason). Limits come from physical/biological possibility, NOT
# from the data distribution, so the same fixed rules apply to both cycles.
# They are deliberately generous: extreme-but-possible values (e.g. a BMI of 60,
# a very high blood pressure) are kept because they are real patients.
IMPOSSIBLE_RULES: dict[str, tuple[float, float, str]] = {
    "age": (18, 80,
            "Cohort is adults only; NHANES top-codes age at 80."),
    "income_poverty_ratio": (0, 5,
            "A ratio cannot be negative; NHANES top-codes it at 5."),
    "body_mass_index": (10, 100,
            "BMI below 10 is incompatible with life; above 100 is beyond any recorded measurement."),
    "waist_circumference_cm": (40, 200,
            "Adult waist circumference outside 40-200 cm is a recording error."),
    "waist_height_ratio": (0.25, 1.5,
            "Waist/height outside 0.25-1.5 means waist or height is wrong."),
    "systolic_bp_mean": (60, 260,
            "Mean systolic BP outside 60-260 mmHg is not compatible with a standing examinee."),
    "diastolic_bp_mean": (30, 160,
            "Mean diastolic BP outside 30-160 mmHg is a device or entry error."),
    "alcohol_drinks_on_drinking_days": (0, 36,
            "More than 36 drinks on an average drinking day is not credible. Zero is valid (non-drinkers)."),
    "sedentary_minutes_per_day": (0, 1440,
            "A day has 24 x 60 = 1440 minutes."),
    "sleep_hours_weekday": (2, 16,
            "Usual sleep outside 2-16 hours is a coding error (NHANES records 2-14)."),
}


def clean_impossible_values(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Set physically impossible values to NaN and report how many each rule changed.

    Values are set to NaN, never dropped: the row still carries valid information
    in its other columns, and the Pipeline imputes the gap later (training-fold
    statistics only).

    Returns (cleaned copy, report).
    """
    _require_columns(df, list(IMPOSSIBLE_RULES), "clean_impossible_values")
    out = df.copy()
    records: list[dict] = []

    # (a) Single-column range rules
    for col, (lo, hi, why) in IMPOSSIBLE_RULES.items():
        bad = out[col].notna() & ((out[col] < lo) | (out[col] > hi))
        out.loc[bad, col] = np.nan
        records.append({
            "rule": f"{lo} <= {col} <= {hi}", "column": col, "lower": lo, "upper": hi,
            "n_values_changed": int(bad.sum()),
            "pct_of_non_missing": round(100 * bad.sum() / max(int(df[col].notna().sum()), 1), 3),
            "reason": why})

    # (b) Cross-field rule: systolic must be higher than diastolic. If not, we
    # cannot tell which of the two readings is wrong, so both are blanked.
    sbp, dbp = "systolic_bp_mean", "diastolic_bp_mean"
    swapped = out[sbp].notna() & out[dbp].notna() & (out[sbp] <= out[dbp])
    out.loc[swapped, [sbp, dbp]] = np.nan
    records.append({
        "rule": "systolic > diastolic", "column": f"{sbp}, {dbp}", "lower": np.nan,
        "upper": np.nan, "n_values_changed": int(2 * swapped.sum()),
        "pct_of_non_missing": np.nan,
        "reason": "Pulse pressure cannot be zero or negative."})

    # (c) Derived-column consistency: waist/height is built from the waist value,
    # so if the waist was blanked the ratio built from it must be blanked too.
    stale = out["waist_circumference_cm"].isna() & out["waist_height_ratio"].notna()
    out.loc[stale, "waist_height_ratio"] = np.nan
    records.append({
        "rule": "waist_height_ratio follows cleaned waist", "column": "waist_height_ratio",
        "lower": np.nan, "upper": np.nan, "n_values_changed": int(stale.sum()),
        "pct_of_non_missing": np.nan,
        "reason": "A derived value must not survive when its input was removed."})

    return out, pd.DataFrame(records)


# =============================================================================
# 4. Binary recoding (stateless, so it is safe inside a Pipeline)
# =============================================================================
def recode_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Recode every binary column in BINARY_ENCODING to 0/1; leave other columns alone.

    Raises if a column holds a value outside its whitelist. That also catches the
    common mistake of recoding twice (sex would then contain 0, which is not a
    valid raw code).
    """
    out = df.copy()
    for col, mapping in BINARY_ENCODING.items():
        if col not in out.columns:
            continue
        unexpected = set(out[col].dropna().unique()) - set(mapping)
        if unexpected:
            raise ValueError(f"'{col}' has unexpected value(s) {sorted(unexpected)}; "
                             f"allowed raw codes are {sorted(mapping)}. Already recoded?")
        out[col] = out[col].map(mapping)
    return out


def _to_float(X):
    """Module-level (picklable) helper: cast boolean flags to float."""
    return X.astype(float)


# =============================================================================
# 5. Preprocessor and pipeline
# =============================================================================
def _make_scaler(scale: Optional[str]):
    """Fresh scaler each call so numeric and ordinal branches never share state."""
    if scale == "standard":
        return StandardScaler()
    if scale == "robust":
        return RobustScaler()      # uses median/IQR, so it resists the long BMI/alcohol tails
    if scale is None:
        return "passthrough"       # tree models do not need scaling
    raise ValueError("scale must be 'standard', 'robust' or None.")

def _make_imputer(kind):
    if kind in ("median", "mean"):
        return SimpleImputer(strategy=kind, keep_empty_features=True)
    if kind == "knn":
        return KNNImputer(n_neighbors=5)
    if kind == "none":
        return "passthrough"   # only for HistGB/XGBoost/LightGBM, which accept NaN

def build_preprocessor(blocks: Optional[Iterable[str]] = None,
                       scale: Optional[Literal["standard", "robust"]] = "standard",
                       impute: str = "median",
                       missing_indicators: bool = False) -> ColumnTransformer:
    """Build the ColumnTransformer for the chosen feature blocks.

    Branches (one per feature type):
      numeric  -> median impute -> optional scaling   (median resists skew/outliers)
      ordinal  -> mode impute   -> optional scaling   (codes 1-5 already carry the order)
      binary   -> recode to 0/1 -> mode impute
      nominal  -> mode impute   -> one-hot            (no order exists, so no integer codes)
      optional -> missing-indicator flags for income and alcohol

    Nothing is fitted here. Fitting happens when the Pipeline is fitted, which
    cross-validation does on training folds only (no leakage).
    """
    features = resolve_features(blocks)
    groups = {t: [f for f in features if FEATURE_TYPES[f] == t]
              for t in ("numeric", "ordinal", "binary", "nominal")}

    # keep_empty_features keeps the output shape stable even if a training fold
    # happens to have a column that is entirely missing.
    transformers: list[tuple] = []
    if groups["numeric"]:
        transformers.append(("numeric", Pipeline([
            ("impute", _make_imputer(impute)),
            ("scale", _make_scaler(scale))]), groups["numeric"]))
    if groups["ordinal"]:
        transformers.append(("ordinal", Pipeline([
            ("impute", _make_imputer(impute)),
            ("scale", _make_scaler(scale))]), groups["ordinal"]))
    if groups["binary"]:
        transformers.append(("binary", Pipeline([
            ("recode", FunctionTransformer(recode_binary, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True))]),
            groups["binary"]))
    if groups["nominal"]:
        transformers.append(("nominal", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
            groups["nominal"]))
    if missing_indicators:
        flag_cols = [c for c in MISSING_FLAG_FEATURES if c in features]
        if flag_cols:
            # features="all" gives one flag per column in every fold. The default
            # ("missing-only") would change the number of columns between folds.
            transformers.append(("missing_flags", Pipeline([
                ("flag", MissingIndicator(features="all")),
                ("to_float", FunctionTransformer(_to_float, feature_names_out="one-to-one"))]),
                flag_cols))

    # remainder="drop": a column not listed in blocks can never sneak in.
    prep = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    prep.set_output(transform="pandas")  # keep column names for SHAP / feature tables
    return prep


def make_pipeline(model, blocks: Optional[Iterable[str]] = None,
                  scale: Optional[Literal["standard", "robust"]] = "standard",
                  missing_indicators: bool = False) -> Pipeline:
    """Attach a model to the preprocessor: Pipeline([prep, model]).

    Use scale="standard" for LR/SVM/kNN/MLP, scale=None for tree-based models.
    Passing this whole object to cross_validate / RandomizedSearchCV is what
    guarantees imputation and scaling are re-fitted inside every training fold.
    """
    prep = build_preprocessor(blocks, scale=scale, missing_indicators=missing_indicators)
    return Pipeline([("prep", prep), ("model", model)])


# =============================================================================
# 6. Data quality report
# =============================================================================
def data_quality_report(train: pd.DataFrame, test: pd.DataFrame,
                        blocks: Optional[Iterable[str]] = None,
                        near_constant_share: float = 0.99) -> dict[str, pd.DataFrame]:
    """Collect the integrity checks into one dict of DataFrames.

    Keys: duplicate_seqn, seqn_overlap, missingness, impossible_values,
    near_constant. The test cycle is described, never used to make a decision:
    the near-constant flag is based on TRAIN only.
    """
    features = resolve_features(blocks)
    frames = {"train": train, "test": test}
    for name, df in frames.items():
        _require_columns(df, ["SEQN"] + features, f"data_quality_report ({name})")

    report: dict[str, pd.DataFrame] = {}

    # SEQN is the respondent ID: it must be unique inside a cycle...
    report["duplicate_seqn"] = pd.DataFrame([
        {"cycle": n, "n_rows": len(d), "n_duplicate_seqn": int(d["SEQN"].duplicated().sum())}
        for n, d in frames.items()])
    # ...and a person must not appear in both cycles (that would leak train into test).
    overlap = set(train["SEQN"].dropna()) & set(test["SEQN"].dropna())
    report["seqn_overlap"] = pd.DataFrame([{"n_seqn_in_both_cycles": len(overlap)}])

    miss = pd.concat([train[features].isna().mean(), test[features].isna().mean()],
                     axis=1, keys=["train", "test"])
    miss["abs_diff"] = (miss["train"] - miss["test"]).abs()
    report["missingness"] = miss.sort_values("train", ascending=False).round(4)

    _, rep_tr = clean_impossible_values(train)
    _, rep_te = clean_impossible_values(test)
    report["impossible_values"] = rep_tr[["rule", "n_values_changed"]].merge(
        rep_te[["rule", "n_values_changed"]], on="rule", suffixes=("_train", "_test"))

    rows = []
    for f in features:
        def top_share(s: pd.Series) -> tuple:
            vc = s.dropna().value_counts(normalize=True)
            return (vc.index[0], float(vc.iloc[0])) if len(vc) else (np.nan, np.nan)
        top_tr, share_tr = top_share(train[f])
        _, share_te = top_share(test[f])
        rows.append({"feature": f, "top_value_train": top_tr,
                     "top_share_train": round(share_tr, 4),
                     "top_share_test": round(share_te, 4),
                     "n_unique_train": int(train[f].nunique()),
                     "near_constant": bool(share_tr >= near_constant_share)})
    report["near_constant"] = pd.DataFrame(rows).sort_values("top_share_train", ascending=False)
    return report


# =============================================================================
# 7. Sanity checks
# =============================================================================
def run_sanity_checks(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Fail loudly if any project rule is broken. Run before every modelling session."""
    from sklearn.linear_model import LogisticRegression

    X_tr, y_tr = get_xy(train)
    X_te, y_te = get_xy(test)

    # 1. No lab / DIQ / subgroup / target columns inside X.
    for X in (X_tr, X_te):
        assert not [c for c in X.columns
                    if c in FORBIDDEN_EXACT or c.startswith(FORBIDDEN_PREFIXES)], \
            "Leakage: forbidden column in X"

    # 2. Train and test expose exactly the same feature columns, in the same order.
    assert list(X_tr.columns) == list(X_te.columns), "Train/test feature columns differ"

    # 3. After the pipeline there is no NaN, whatever the options. The test cycle is
    #    only transformed (using training statistics), never used to fit anything.
    for scale in ("standard", "robust", None):
        for flags in (False, True):
            prep = build_preprocessor(scale=scale, missing_indicators=flags)
            Z_tr = prep.fit_transform(X_tr)
            Z_te = prep.transform(X_te)
            assert not Z_tr.isna().any().any(), f"NaN left in train (scale={scale}, flags={flags})"
            assert not Z_te.isna().any().any(), f"NaN left in test (scale={scale}, flags={flags})"
            assert list(Z_tr.columns) == list(Z_te.columns), "Processed columns differ"

    # 4. Block subsets work and the pipeline trains end to end.
    pipe = make_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                         blocks=["demographics", "body"])
    pipe.fit(*get_xy(train, ["demographics", "body"]))
    proba = pipe.predict_proba(get_xy(test, ["demographics", "body"])[0])[:, 1]
    assert proba.shape == (len(test),) and np.isfinite(proba).all(), "Bad predictions"

    # 5. Target is strictly binary.
    assert set(y_tr.unique()) <= {0, 1} and set(y_te.unique()) <= {0, 1}
    print("All sanity checks passed.")


# To ensure Single Data Entry point
def load_clean(force_rebuild=False):
    """Single entry point for loaded, validated, and impossibles-cleaned data."""
    # Imported locally to prevent circular dependency with load_data.py
    nh = NHANES()
    tr, te = nh.get_nhanes(force_rebuild)
    return clean_impossible_values(tr)[0], clean_impossible_values(te)[0], nh.flow

def get_cv(n=5):
    """Shared cross-validation generator to ensure model scoring parity."""
    return StratifiedKFold(n, shuffle=True, random_state=RANDOM_STATE)

if __name__ == "__main__":

    nh = NHANES()
    df_train, df_test = nh.get_nhanes()
    df_train, rep_train = clean_impossible_values(df_train)
    df_test, rep_test = clean_impossible_values(df_test)
    print(rep_train[["rule", "n_values_changed"]].to_string(index=False))
    for key, table in data_quality_report(df_train, df_test).items():
        print(f"\n--- {key} ---\n{table.to_string()}")
    run_sanity_checks(df_train, df_test)