from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from load_data import FEATURE_BLOCKS, TARGET

@dataclass
class dataConfig:
    random_state: int = 42
    target_col: str = "undiagnosed_dysglycemia"

    feature_blocks: Dict[str, List[str]] = field(default_factory=lambda: FEATURE_BLOCKS)

    # Every model feature needs a type so the ColumnTransformer knows how to treat it.
    # If a feature is added to FEATURE_BLOCKS and not listed here,
    # resolve_features() stops with a clear message instead of guessing.
    feature_types: Dict[str, str] = field(default_factory=lambda:{
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
    })

    # Find impossible values and return error message for fix.
    impossible_rules: Dict[str, Tuple[float, float, str]] = field(default_factory=lambda:{
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
    })

    forbidden_exact: frozenset = frozenset({"LBXGH", "LBXGLU", "PHAFSTHR", TARGET, "race_ethnicity","is_asian_descent", "SEQN", "cycle"})
    forbidden_prefixes: tuple = ("DIQ",)

    # Binary columns arrive with mixed codings (NHANES 1=Yes/2=No for the BPQ items,
    # 1=Male/2=Female for sex, 0/1 for the derived columns). Recoding everything to
    # 0/1 means every binary column means the same thing ("1 = the flag is present").
    # The dict doubles as a whitelist: any other value raises an error.
    binary_encoding: dict[str, dict[float, float]] = field(default_factory=lambda:{
        "sex": {1.0: 0.0, 2.0: 1.0},                       # 1 = female after recoding
        "history_high_bp": {1.0: 1.0, 2.0: 0.0},           # 1 = told high BP
        "history_high_cholesterol": {1.0: 1.0, 2.0: 0.0},  # 1 = told high cholesterol
        "taking_bp_meds": {0.0: 0.0, 1.0: 1.0},
        "taking_cholesterol_meds": {0.0: 0.0, 1.0: 1.0},
        "history_cvd": {0.0: 0.0, 1.0: 1.0},
    })

    # Only these two get an optional "was missing" flag: they have the most missing
    # values (income ~14%, alcohol ~6-14% and different between cycles), so the fact
    # that they are missing might itself carry information.
    missing_flag_features: List[str] = field(default_factory=lambda:[
        "income_poverty_ratio", 
        "alcohol_drinks_on_drinking_days",
        "education_level",  # Added to catch structural missingness for ages 18-19
        "marital_status"    # Added to catch structural missingness for ages 18-19
    ])