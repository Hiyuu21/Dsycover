"""
preprocessing.py
================

Cleaning and preprocessing for the NHANES undiagnosed-dysglycaemia project.

Design rules
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

* for Preprocessor._make_imputer, the knn option will be considered at Part B due to its complexity for scaling.

"""

from __future__ import annotations
from typing import Iterable, Literal, Optional
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from load_data import NHANES
from dataConfig import dataConfig

class FeatureExtracter:
    def __init__(self, config: dataConfig = dataConfig()):
        self.config = config
        self.target = self.config.target_col

    def _require_columns(self, df:pd.DataFrame, cols:Iterable[str], where:str) -> None:
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise KeyError(f"[{where}] missing column(s): {missing}."
                               f"Available columns start with : {list(df.columns)[:8]} ...")
    
    def _forbid_columns(self, features: Iterable[str]) -> None:
        bad = [f for f in features if f in self.config.forbidden_exact or f.startswith(self.config.forbidden_prefixes)]
        if bad: raise ValueError(f"Possible Leakage: {bad} must noe be model features.")

    def resolve_features(self, blocks: Optional[Iterable[str]] = None) -> list[str]:
        names = list(self.config.feature_blocks) if blocks is None else list(blocks)
        unknown = [b for b in names if b not in self.config.feature_blocks]

        if unknown: raise KeyError(f"Unknown block(s) {unknown}. Choose from {list(self.config.feature_blocks)}.")
        features = [f for b in names for f in self.config.feature_blocks[b]]
        untyped = [f for f in features if f not in self.config.feature_types]

        if untyped: raise KeyError(f"No type assigned for {untyped}. Add them to FEATURE_TYPES.")
        self._forbid_columns(features)
        return features

    def get_xy(self, df: pd.DataFrame, blocks:Optional[Iterable[str]] = None) -> tuple[pd.DataFrame, pd.Series]:
        features = self.resolve_features(blocks)
        self._require_columns(df, features + [self.target], "get_xy")
        y = df[self.target]
        if y.isna().any():
            raise ValueError(f"{self.target} has {int(y.isna().sum())} NaN.")
        if not set (y.unique()) <= {0.0, 1.0}:
            raise ValueError(f"{self.target} must be 0/1, found {sorted(y.unique())}.")
        return df[features].copy(), y.astype(int)

    def recode_binary(self, df: pd.DataFrame) -> pd.DataFrame:
            """
            Recode every binary column in dataConfig.binary_encoding to 0/1 (one-hot) and leave other columns alone.
            Raises error is a column holds a value out of its whitelist.
            """
    
            out = df.copy()
            for col, mapping in self.config.binary_encoding.items():
                if col not in out.columns:
                    continue
                unexpected = set(out[col].dropna().unique()) - set(mapping)
                if unexpected:
                    raise ValueError(f"'{col}' has unexpected value(s) {sorted(unexpected)}."
                                     f"allowed raw codes are {sorted(mapping)}.")
                out[col] = out[col].map(mapping)
    
            return out
    
    def _to_float(self, X):
        return X.astype(float)



class Preprocessor:
    def __init__(self, config:dataConfig = dataConfig()):
        self.report_ = None
        self.config = config
        self.extractor = FeatureExtracter(config)
        self.rules = self.config.impossible_rules
        self.target = 'undiagnosed_dysglycemia'

    def clean(self, df:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert the physically impossible values to NaN and report how many values are changed.
        The converted NaN values will then be imputed with training-fold statistics.
        Return: cleaned copy, report
        """

        self.extractor._require_columns(df, list(self.rules), "clean_impossible_values")
        out = df.copy()
        records = []

        # (a) single-column range rules
        for col, (lo, hi, why) in self.rules.items():
            if col in out.columns:
                bad = out[col].notna() & ((out[col] < lo) | (out[col] > hi))
                out.loc[bad, col] = np.nan
                records.append({
                    "rule": f"{lo} <= {col} <= {hi}",
                    "column" : col,
                    "lower":lo,
                    "upper":hi,
                    "n_values_changed": int(bad.sum()),
                    "pct_of_non_missing": round(100 * bad.sum() / max(int(df[col].notna().sum()), 1), 3),
                    "reason": why
                })

        # (b) Cross-field rule: systolic must > diastolic.
        sbp, dbp = "systolic_bp_mean", "diastolic_bp_mean"
        swapped = out[sbp].notna() & out[dbp].notna() & (out[sbp] <= out[dbp])
        out.loc[swapped, [sbp, dbp]] = np.nan
        records.append({
            "rule": "systolic > diastolic", 
            "column":f"{sbp}, {dbp}",
            "lower":np.nan,
            "upper":np.nan,
            "n_values_changed": int(2 * swapped.sum()),
            "pct_of_non_missing": np.nan,
            "reason": "Pulse pressure cannot be zero or negative."
        })

        # (c) Derived-column check: waist/hright is built from the waist value -> if the waist is blank, the ratio is blank too.
        stale = out["waist_circumference_cm"].isna() & out["waist_height_ratio"].notna()
        out.loc[stale, "waist_height_ratio"] = np.nan
        records.append({
            "rule": "waist_height_ratio follows cleaned waist", 
            "column": "waist_height_ratio",
            "lower": np.nan, 
            "upper": np.nan, 
            "n_values_changed": int(stale.sum()),
            "pct_of_non_missing": np.nan,
            "reason": "A derived value will not exist when its input was removed."})
    
        self.report_ = pd.DataFrame(records)
        return out, self.report_

    def _make_scaler(self, scale: Optional[str]):
        if scale == "standard": return StandardScaler()
        if scale == "robust": return RobustScaler()
        if scale is None: return "passthrough"  # tree models do not need scaling
        raise ValueError("Scale must be 'standard','robust', or 'None'.")

    def _make_imputer(self, kind):
        if kind in ("median", "mean"): return SimpleImputer(strategy=kind, keep_empty_features=True)
        # if kind == "knn": return KNNImputer(n_neighbors=5)
        if kind == "none": return "passthrough" # only for HistGB/XGBoost/LightGBM, which accept NaN
        raise ValueError(f"Unknown imputer kind: '{kind}'. Choose from 'median', 'mean', or 'none'.")

    def preprocessor(self, 
                     blocks: Optional[Iterable[str]] = None,
                     scale: Optional[Literal["standard", "robust"]] = "standard",
                     impute: str = "median",
                     missing_indicators: bool = False) -> ColumnTransformer:
        """Build the ColumnTransformer for the chosen feature blocks.
        
        Branches (one per feature type):
            numeric  -> median impute -> optional scaling   (median resists skew/outliers unlike mean)
            ordinal  -> mode impute   -> optional scaling   (codes 1-5 already carry the order)
            binary   -> recode to 0/1 -> mode impute
            nominal  -> mode impute   -> one-hot            (no order exists, so no integer codes)
            optional -> missing-indicator flags for income and alcohol
    
        Nothing is fitted here. Fitting happens when the Pipeline is fitted, which
        cross-validation does on training folds only (no leakage).
        """
        features = self.extractor.resolve_features(blocks)
        groups = {t: [f for f in features if self.config.feature_types[f] == t] 
                  for t in ("numeric", "ordinal", "binary", "nominal")}

        transformers: list[tuple] = []
        if groups["numeric"]:
                transformers.append(("numeric", Pipeline([
                    ("impute", self._make_imputer(impute)),
                    ("scale", self._make_scaler(scale))]), groups["numeric"]))
        if groups["ordinal"]:
            transformers.append(("ordinal", Pipeline([
                ("impute", self._make_imputer(impute)),
                ("scale", self._make_scaler(scale))]), groups["ordinal"]))
        if groups["binary"]:
            transformers.append(("binary", Pipeline([
                ("recode", FunctionTransformer(self.extractor.recode_binary, feature_names_out="one-to-one")),
                ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True))]),
                groups["binary"]))
        if groups["nominal"]:
            transformers.append(("nominal", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                groups["nominal"]))
        if missing_indicators:
            flag_cols = [c for c in self.config.missing_flag_features if c in features]
            if flag_cols:
                # features="all" gives one flag per column in every fold. The default
                # ("missing-only") would change the number of columns between folds.
                transformers.append(("missing_flags", Pipeline([
                    ("flag", MissingIndicator(features="all")),
                    ("to_float", FunctionTransformer(self.extractor._to_float, feature_names_out="one-to-one"))]),
                    flag_cols))

        # remainder="drop": a column not listed in blocks can never sneak in
        prep = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
        prep.set_output(transform="pandas")
        return prep

    def make_pipeline(self, model,
                      blocks: Optional[Iterable[str]]= None,
                      scale: Optional[Literal["standard", "robust"]] = "standard",
                      impute: str = 'median',
                      missing_indicators: bool = False) -> Pipeline:
        """
        Attach a model to the preprocessor: Pipeline([prep, model])
        Use scale="standard" for LR/SVM/kNN/MLP, scale=None for tree-based models.
        Passing this whole object to cross_validate / RandomizedSearchCV is what
        guarantees imputation and scaling are re-fitted inside every training fold.
        """
        prep = self.preprocessor(blocks, scale=scale, impute=impute, missing_indicators=missing_indicators)
        return Pipeline([("prep", prep), ("model", model)])

    def data_quality_report(self, train, test,
                            blocks:Optional[Iterable[str]] = None,
                            near_constant_share: float = 0.99) -> dict[str, pd.DataFrame]:
        
        features = self.extractor.resolve_features(blocks)
        frames = {"train": train, "test": test}
        for name, df in frames.items():
            self.extractor._require_columns(df, ["SEQN"] + features, f"data_quality_report ({name})")
        report: dict[str, pd.DataFrame] = {}

        # SEQN is the respondent ID: it must be unique inside a cycle...
        report['duplicate_seqn'] = pd.DataFrame([
            {"cycle": n, "n_rows":len(d), "n_duplicate_seqn":int(d['SEQN'].duplicated().sum())}
            for n, d in frames.items()])
        # ...and a person must not appear in both cycles (that would leak train into test).
        overlap = set(train['SEQN'].dropna()) & set(test["SEQN"].dropna())
        report["seqn_overlap"] = pd.DataFrame([{"n_seqn_in_both_cycles" : len(overlap)}])

        miss = pd.concat([train[features].isna().mean(), test[features].isna().mean()], axis=1, keys=["train", "test"])
        miss["abs_diff"] = (miss["train"] - miss["test"]).abs()
        report["missingness"] = miss.sort_values("train", ascending=False).round(4)

        _, rep_tr = self.clean(train)
        _, rep_te = self.clean(test)
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

    def run_sanity_check(self, train, test) -> None:
        from sklearn.linear_model import LogisticRegression
        X_tr, y_tr = self.extractor.get_xy(train)
        X_te, y_te = self.extractor.get_xy(test)

        for X in (X_tr, X_te):
            assert not [c for c in X.columns
                        if c in self.config.forbidden_exact or c.startswith(self.config.forbidden_prefixes)], \
                "Leakage: forbidden column in X"

        assert list(X_tr.columns) == list(X_te.columns), "Train/test feature columns differ"
        for scale in ("standard", "robust", None):
            for flags in (False, True):
                prep = self.preprocessor(scale=scale, missing_indicators=flags)
                Z_tr = prep.fit_transform(X_tr)
                Z_te = prep.transform(X_te)
                assert not Z_tr.isna().any().any(), f"NaN left in train (scale={scale}, flags={flags})"
                assert not Z_te.isna().any().any(), f"NaN left in test (scale={scale}, flags={flags})"
                assert list(Z_tr.columns) == list(Z_te.columns), "Processed columns differ"
    
        # 4. Block subsets work and the pipeline trains end to end.
        pipe = self.make_pipeline(LogisticRegression(max_iter=1000, random_state=self.config.random_state),
                                blocks=["demographics", "body"])
        subset_features = self.extractor.resolve_features(["demographics", "body"])
        pipe.fit(X_tr[subset_features], y_tr)
        proba = pipe.predict_proba(X_te[subset_features])[:, 1]
        assert proba.shape == (len(test),) and np.isfinite(proba).all(), "Bad predictions"
    
        # 5. Target is strictly binary.
        assert set(y_tr.unique()) <= {0, 1} and set(y_te.unique()) <= {0, 1}
        print("All sanity checks passed.")

            

class Loader:
    def __init__(self, config:dataConfig = dataConfig()):
        self.config = config
        self.pre = Preprocessor(config)
        self.nh = NHANES()

    def load_clean(self, force_rebuild=False):
        tr, te = self.nh.get_nhanes(force_rebuild)
        tr_clean, tr_report = self.pre.clean(tr)
        te_clean, te_report = self.pre.clean(te)
        return tr_clean, te_clean, self.nh.flow, tr_report, te_report

    def get_cv(self, n=5):
        return StratifiedKFold(n, shuffle=True, random_state=self.config.random_state)

if __name__ == "__main__":
    config = dataConfig()
    loader = Loader(config)
    tr, te, flow, tr_report, te_report = loader.load_clean(force_rebuild=True)
    
    pre = Preprocessor(config)
    for key, table in pre.data_quality_report(tr, te).items():
        print(f"\n--- {key} ---\n{table.to_string()}")
    pre.run_sanity_check(tr, te)