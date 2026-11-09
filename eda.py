"""
eda.py
======
Exploratory data analysis for the NHANES undiagnosed-dysglycaemia project.

Conventions (be ready to explain these in the presentation)
-----------------------------------------------------------
* Every function returns DataFrames and saves a figure to ./figures/ (dpi 200).
  Nothing calls plt.show(), so it runs the same on a laptop and on a server.
* Anything that involves the TARGET (prevalence, feature-vs-target, correlations,
  subgroup plots) uses the TRAIN cycle only. The test cycle is locked: looking at
  its outcome patterns would let it influence our choices.
* Describing train vs test (missingness, distributions, drift) is allowed because
  it uses no outcome and drives no decision. Drift is reported, never used to
  select or drop features.
* Prevalence is shown with Wilson 95% intervals and n on every bar, because the
  Asian subgroup is small and a bare percentage would hide how uncertain it is.

Expects the cleaned frames (after preprocessing.clean_impossible_values).
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")  # headless backend: figures are only saved, never shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dataConfig import dataConfig
from preprocessing import FeatureExtracter

config = dataConfig()
extractor = FeatureExtracter(config=config)

FIG_DIR = Path("./figures")
DPI = 200
TRAIN_LABEL, TEST_LABEL = "Train 2017-2020", "Test 2021-2023"
COLORS = {TRAIN_LABEL: "#2B6CB0", TEST_LABEL: "#DD6B20",
          "Asian": "#2F855A", "Non-Asian": "#718096"}
MIN_N = 30  # bars with fewer people than this are hatched ("interpret with care")

# Human-readable level names (from the NHANES codebook / data dictionary).
# Binary features are shown AFTER recoding to 0/1, so 1 always means "yes/present".
LEVEL_LABELS: dict[str, dict[float, str]] = {
    "sex": {0.0: "Male", 1.0: "Female"},
    "education_level": {1.0: "<9th grade", 2.0: "9-11th", 3.0: "HS/GED",
                        4.0: "Some college", 5.0: "College+"},
    "marital_status": {1.0: "Married/partner", 2.0: "Wid./div./sep.", 3.0: "Never married"},
    "smoking_status": {1.0: "Current", 2.0: "Former", 3.0: "Never"},
}
BINARY_NO_YES = {0.0: "No", 1.0: "Yes"}

# BMI band systems. WHO is the standard; Asia-Pacific uses lower cut-offs (23, 27.5)
# because Asian adults carry metabolic risk at lower BMI. Confirm the exact
# cut-offs against the Malaysian clinical guideline before citing them.
BMI_BANDS: dict[str, tuple[list[float], list[str]]] = {
    "WHO": ([0, 18.5, 25, 30, np.inf], ["<18.5", "18.5-24.9", "25-29.9", ">=30"]),
    "Asia-Pacific": ([0, 18.5, 23, 27.5, np.inf], ["<18.5", "18.5-22.9", "23-27.4", ">=27.5"]),
}
AGE_BINS = [18, 30, 45, 60, 70, 81]
AGE_LABELS = ["18-29", "30-44", "45-59", "60-69", "70-80"]


# =============================================================================
# Helpers
# =============================================================================
def _style() -> None:
    """One consistent look for every figure."""
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.9)
    plt.rcParams.update({"figure.dpi": DPI, "savefig.dpi": DPI,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid.axis": "y",  # horizontal guide lines only; vertical ones cut through bars
                         "axes.titleweight": "bold"})


def _save(fig: plt.Figure, name: str) -> Path:
    """Save at dpi 200 into ./figures/ and close the figure to free memory."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def wilson_ci(k, n, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """Wilson 95% interval for a proportion k/n.

    Why Wilson and not the textbook p +/- 1.96*se: it stays inside [0, 1] and is
    reliable for small n or proportions near 0 or 1, which is exactly our Asian
    subgroup situation.
    """
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    low = np.where(n > 0, centre - half, np.nan)
    high = np.where(n > 0, centre + half, np.nan)
    return low, high


def _prevalence_table(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Rows = groups defined by `by`; columns n, positives, prevalence, Wilson CI."""
    extractor._require_columns(df, by + [config.target_col], "_prevalence_table")
    g = (df.groupby(by, observed=True)[config.target_col]
           .agg(n="count", positives="sum").reset_index())
    g["prevalence"] = g["positives"] / g["n"]
    g["ci_low"], g["ci_high"] = wilson_ci(g["positives"], g["n"])
    return g


def _stack(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Stack both cycles with a readable label column named 'cycle_label'."""
    return pd.concat([train.assign(cycle_label=TRAIN_LABEL),
                      test.assign(cycle_label=TEST_LABEL)], ignore_index=True)


def _grouped_bars(ax: plt.Axes, tbl: pd.DataFrame, level_col: str, hue_col: str,
                  levels: list, hues: list, title: str, ylabel: str = "Prevalence") -> None:
    """Grouped bars with Wilson error bars and n labels; hatch bars with n < MIN_N."""
    width = 0.8 / len(hues)
    x = np.arange(len(levels))
    top = 0.0
    for i, hue in enumerate(hues):
        sub = tbl[tbl[hue_col] == hue].set_index(level_col).reindex(levels)
        pos = x - 0.4 + width * (i + 0.5)
        prev = sub["prevalence"].to_numpy(dtype=float)
        err = np.vstack([np.nan_to_num(prev - sub["ci_low"].to_numpy(dtype=float)),
                         np.nan_to_num(sub["ci_high"].to_numpy(dtype=float) - prev)])
        bars = ax.bar(pos, np.nan_to_num(prev), width, yerr=err, capsize=3,
                      color=COLORS.get(hue, "#4A5568"), label=str(hue), error_kw={"lw": 1})
        for bar, n, hi in zip(bars, sub["n"], sub["ci_high"]):
            if pd.notna(n):
                if n < MIN_N:
                    bar.set_hatch("//")
                ax.text(bar.get_x() + bar.get_width() / 2, (hi if pd.notna(hi) else 0) + 0.015,
                        f"n={int(n)}", ha="center", va="bottom", fontsize=6.5)
                top = max(top, hi if pd.notna(hi) else 0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in levels])
    ax.set_ylim(0, min(1.0, top + 0.15))
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _level_name(feature: str, value: float) -> str:
    """Readable name for a category level (recoded binaries show No/Yes)."""
    if feature in LEVEL_LABELS:
        return LEVEL_LABELS[feature].get(float(value), str(value))
    if config.feature_types.get(feature) == "binary":
        return BINARY_NO_YES.get(float(value), str(value))
    return str(value)


def _split_by_type(features: list[str]) -> tuple[list[str], list[str]]:
    """Numeric features vs everything categorical (binary, ordinal, nominal)."""
    numeric = [f for f in features if config.feature_types[f] == "numeric"]
    categorical = [f for f in features if config.feature_types[f] != "numeric"]
    return numeric, categorical


# =============================================================================
# 1. Cohort flow and prevalence overview
# =============================================================================
def cohort_flow_table(flow: list[dict]) -> pd.DataFrame:
    """Turn NHANES.flow (a list of dicts) into a readable step-by-cycle table.

    NOTE: the loader logs steps 1-4 only when it rebuilds from the raw files. If it
    loaded from cache you will only see step 5. Run get_nhanes(force_rebuild=True)
    once to log the full flow.
    """
    if not flow:
        raise ValueError("Cohort flow is empty. Call NHANES.get_nhanes(force_rebuild=True) first.")
    names = {1: "1. Merged raw files (all ages)", 2: "2. Adults (age >= 18)",
             3: "3. Not pregnant", 4: "4. HbA1c present",
             5: "5. Screen-eligible (no known diabetes)"}
    df = pd.DataFrame(flow).drop_duplicates(["cycle", "step"], keep="last")
    if df["step"].nunique() < len(names):
        print("Warning: cohort flow is incomplete (cache was used). "
              "Rebuild with force_rebuild=True to log steps 1-4.")
    table = df.pivot(index="step", columns="cycle", values="n").sort_index()
    table.index = table.index.map(names)
    table.index.name = "step"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(FIG_DIR / "01_cohort_flow.csv")
    return table


def prevalence_overview(train: pd.DataFrame, test: pd.DataFrame,
                        save_name: str = "01_prevalence_overview.png") -> pd.DataFrame:
    """Target prevalence by age band, sex and Asian descent (TRAIN only). Test n is reported."""
    extractor._require_columns(train, [config.target_col, "age", "sex", "is_asian_descent"], "prevalence_overview (train)")
    _style()
    
    # Process train for plotting
    train = train.copy()
    train["Cohort"] = TRAIN_LABEL # Dummy hue column so _grouped_bars knows what color to draw
    train["All adults"] = "All adults"
    train["Age band"] = pd.cut(train["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    train["Sex"] = train["sex"].map({1.0: "Male", 2.0: "Female"})
    train["Asian descent"] = train["is_asian_descent"].map({0: "Non-Asian", 1: "Asian"})

    dims = [("All adults", ["All adults"]), ("Age band", AGE_LABELS),
            ("Sex", ["Male", "Female"]), ("Asian descent", ["Non-Asian", "Asian"])]
    
    tables = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (dim, levels) in zip(axes.ravel(), dims):
        t = _prevalence_table(train.dropna(subset=[dim]), [dim, "Cohort"])
        # Plot grouped bars just for TRAIN
        _grouped_bars(ax, t, dim, "Cohort", levels, [TRAIN_LABEL], 
                      f"Prevalence by {dim.lower()}" if dim != "All adults" else "Overall prevalence")
        tables.append(t.rename(columns={dim: "level"}).assign(dimension=dim))
        
    _save(fig, save_name)
    
    # Generate a pure 'n' count for the test set to report demographics without leaking TARGET
    test_n = {"total": len(test), "n_asian": int(test["is_asian_descent"].sum())}
    print(f"Test cycle demographics (locked): {test_n}")
    
    out = pd.concat(tables, ignore_index=True)
    return out[["dimension", "level", "Cohort", "n", "positives", "prevalence", "ci_low", "ci_high"]]

# =============================================================================
# 2. Missingness
# =============================================================================
def missingness_summary(train: pd.DataFrame, test: pd.DataFrame,
                        blocks: Optional[Iterable[str]] = None,
                        save_name: str = "02_missingness.png") -> pd.DataFrame:
    """Proportion missing per MODEL feature, train vs test, sorted high to low."""
    features = extractor.resolve_features(blocks)
    extractor._require_columns(train, features, "missingness_summary (train)")
    extractor._require_columns(test, features, "missingness_summary (test)")
    _style()
    miss = pd.concat([train[features].isna().mean(), test[features].isna().mean()],
                     axis=1, keys=[TRAIN_LABEL, TEST_LABEL])
    miss["abs_diff"] = (miss[TRAIN_LABEL] - miss[TEST_LABEL]).abs()
    miss = miss.sort_values(TRAIN_LABEL, ascending=False)

    fig, ax = plt.subplots(figsize=(8, 0.35 * len(miss) + 1.5))
    y = np.arange(len(miss))
    ax.barh(y - 0.2, miss[TRAIN_LABEL], 0.4, color=COLORS[TRAIN_LABEL], label=TRAIN_LABEL)
    ax.barh(y + 0.2, miss[TEST_LABEL], 0.4, color=COLORS[TEST_LABEL], label=TEST_LABEL)
    ax.set_yticks(y)
    ax.set_yticklabels(miss.index)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Missing")
    ax.set_title("Missingness per model feature")
    ax.legend(frameon=False)
    _save(fig, save_name)
    return miss


def missing_vs_target(train: pd.DataFrame, blocks: Optional[Iterable[str]] = None,
                      min_missing: int = 20,
                      save_name: str = "02_missing_vs_target.png") -> pd.DataFrame:
    """Target prevalence for missing vs observed groups, per feature (TRAIN only).

    Why: if people with a missing value have a different outcome rate, the
    missingness is informative and a missing-indicator flag may help. Features
    with fewer than `min_missing` missing rows are skipped (too few to compare).
    """
    features = extractor.resolve_features(blocks)
    extractor._require_columns(train, features + [config.target_col], "missing_vs_target")
    _style()
    rows = []
    for f in features:
        is_missing = train[f].isna()
        if is_missing.sum() < min_missing:
            continue
        for label, mask in (("Missing", is_missing), ("Observed", ~is_missing)):
            rows.append({"feature": f, "group": label, "n": int(mask.sum()),
                         "positives": float(train.loc[mask, config.target_col].sum())})
    if not rows:
        print(f"No feature has >= {min_missing} missing rows in train; nothing to plot.")
        return pd.DataFrame(columns=["feature", "group", "n", "positives",
                                     "prevalence", "ci_low", "ci_high"])
    t = pd.DataFrame(rows)
    t["prevalence"] = t["positives"] / t["n"]
    t["ci_low"], t["ci_high"] = wilson_ci(t["positives"], t["n"])

    feats = list(dict.fromkeys(t["feature"]))
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(feats) + 1.5))
    for j, (grp, colour, off) in enumerate((("Missing", "#C53030", -0.12), ("Observed", "#2B6CB0", 0.12))):
        s = t[t["group"] == grp].set_index("feature").reindex(feats)
        ypos = np.arange(len(feats)) + off
        ax.errorbar(s["prevalence"], ypos,
                    xerr=[s["prevalence"] - s["ci_low"], s["ci_high"] - s["prevalence"]],
                    fmt="o", color=colour, capsize=3, label=grp)
    ax.set_yticks(np.arange(len(feats)))
    ax.set_yticklabels(feats)
    ax.invert_yaxis()
    ax.set_xlabel("Target prevalence (95% Wilson CI)")
    ax.set_title("Is missingness informative? (train)")
    ax.legend(frameon=False)
    _save(fig, save_name)
    return t


# =============================================================================
# 3. Distributions
# =============================================================================
def plot_distributions(train: pd.DataFrame, test: pd.DataFrame,
                       blocks: Optional[Iterable[str]] = None,
                       save_prefix: str = "03_distribution") -> dict[str, pd.DataFrame]:
    """Histograms for numeric features and proportion bars for categorical ones.

    Both cycles are drawn on the same axes with density scaling so different
    sample sizes do not distort the comparison. Missing values are left out here
    (they are covered by missingness_summary).
    """
    features = extractor.resolve_features(blocks)
    extractor._require_columns(train, features, "plot_distributions (train)")
    extractor._require_columns(test, features, "plot_distributions (test)")
    _style()
    numeric, categorical = _split_by_type(features)
    both = _stack(extractor.recode_binary(train), extractor.recode_binary(test))
    palette = {TRAIN_LABEL: COLORS[TRAIN_LABEL], TEST_LABEL: COLORS[TEST_LABEL]}

    # --- numeric: overlaid density histograms
    num_summary = (both.groupby("cycle_label")[numeric].describe().T
                   .round(3).reset_index().rename(columns={"level_0": "feature", "level_1": "stat"}))
    ncols = 3
    nrows = math.ceil(len(numeric) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.8 * nrows), squeeze=False)
    for ax, f in zip(axes.ravel(), numeric):
        sns.histplot(data=both, x=f, hue="cycle_label", stat="density", common_norm=False,
                     element="step", fill=False, bins=30, palette=palette, ax=ax, legend=False)
        ax.set_title(f)
        ax.set_xlabel("")
    for ax in axes.ravel()[len(numeric):]:
        ax.axis("off")
    fig.legend(handles=[plt.Line2D([0], [0], color=c, lw=2) for c in palette.values()],
               labels=list(palette), loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0), frameon=False)
    _save(fig, f"{save_prefix}_numeric.png")

    # --- categorical: proportion of each level within each cycle
    cat_rows = []
    ncols = 3
    nrows = math.ceil(len(categorical) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3 * nrows), squeeze=False)
    for ax, f in zip(axes.ravel(), categorical):
        prop = (both.dropna(subset=[f]).groupby("cycle_label")[f]
                    .value_counts(normalize=True).rename("proportion").reset_index())
        prop["level"] = prop[f].map(lambda v: _level_name(f, v))
        levels = [_level_name(f, v) for v in sorted(prop[f].unique())]
        prop["feature"] = f
        cat_rows.append(prop[["feature", "cycle_label", "level", "proportion"]])
        width = 0.4
        for i, cyc in enumerate((TRAIN_LABEL, TEST_LABEL)):
            s = prop[prop["cycle_label"] == cyc].set_index("level").reindex(levels)["proportion"]
            ax.bar(np.arange(len(levels)) + (i - 0.5) * width, s.fillna(0), width,
                   color=COLORS[cyc])
        ax.set_xticks(np.arange(len(levels)))
        ax.set_xticklabels(levels, rotation=25, ha="right", fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(f)
    for ax in axes.ravel()[len(categorical):]:
        ax.axis("off")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in palette.values()],
               labels=list(palette), loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0), frameon=False)
    _save(fig, f"{save_prefix}_categorical.png")

    return {"numeric_summary": num_summary,
            "categorical_proportions": pd.concat(cat_rows, ignore_index=True)}


# =============================================================================
# 4. Feature vs target
# =============================================================================
def prevalence_by_feature(train: pd.DataFrame, blocks: Optional[Iterable[str]] = None,
                          n_bins: int = 10,
                          save_prefix: str = "04_prevalence") -> pd.DataFrame:
    """Target prevalence by decile (numeric) or by category (categorical), TRAIN only.

    Deciles (equal-sized groups) are used instead of equal-width bins so every
    point rests on a similar number of people. Features with many tied values
    (e.g. alcohol: many zeros) produce fewer than 10 bins; that is expected.
    """
    features = extractor.resolve_features(blocks)
    extractor._require_columns(train, features + [config.target_col], "prevalence_by_feature")
    _style()
    numeric, categorical = _split_by_type(features)
    data = extractor.recode_binary(train)
    overall = data[config.target_col].mean()
    rows = []

    # --- numeric -> deciles
    ncols = 3
    nrows = math.ceil(len(numeric) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3 * nrows), squeeze=False)
    for ax, f in zip(axes.ravel(), numeric):
        d = data[[f, config.target_col]].dropna()
        d["bin"] = pd.qcut(d[f], q=n_bins, duplicates="drop")
        t = _prevalence_table(d, ["bin"])
        t["feature"], t["bin_order"] = f, np.arange(1, len(t) + 1)
        t["bin_range"] = t["bin"].astype(str)
        rows.append(t.drop(columns="bin").assign(kind="numeric_decile"))
        ax.errorbar(t["bin_order"], t["prevalence"],
                    yerr=[t["prevalence"] - t["ci_low"], t["ci_high"] - t["prevalence"]],
                    fmt="o-", color=COLORS[TRAIN_LABEL], capsize=3)
        ax.axhline(overall, color="grey", ls="--", lw=1)
        ax.set_xticks(t["bin_order"])
        ax.set_title(f)
        ax.set_xlabel("Decile (low to high)")
        ax.set_ylabel("Prevalence")
    for ax in axes.ravel()[len(numeric):]:
        ax.axis("off")
    _save(fig, f"{save_prefix}_numeric.png")

    # --- categorical -> by level
    nrows = math.ceil(len(categorical) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3 * nrows), squeeze=False)
    for ax, f in zip(axes.ravel(), categorical):
        d = data[[f, config.target_col]].dropna()
        t = _prevalence_table(d, [f])
        t["level"] = t[f].map(lambda v: _level_name(f, v))
        t["feature"] = f
        rows.append(t.drop(columns=f).assign(kind="category"))
        ax.bar(np.arange(len(t)), t["prevalence"],
               yerr=[t["prevalence"] - t["ci_low"], t["ci_high"] - t["prevalence"]],
               capsize=3, color=COLORS[TRAIN_LABEL])
        for i, (n, hi) in enumerate(zip(t["n"], t["ci_high"])):
            ax.text(i, hi + 0.01, f"n={int(n)}", ha="center", fontsize=6.5)
        ax.axhline(overall, color="grey", ls="--", lw=1)
        ax.set_xticks(np.arange(len(t)))
        ax.set_xticklabels(t["level"], rotation=25, ha="right", fontsize=7)
        ax.set_ylim(0, min(1.0, t["ci_high"].max() + 0.15))
        ax.set_title(f)
    for ax in axes.ravel()[len(categorical):]:
        ax.axis("off")
    _save(fig, f"{save_prefix}_categorical.png")

    out = pd.concat(rows, ignore_index=True)
    first = ["feature", "kind", "level", "bin_order", "bin_range"]
    cols = [c for c in first if c in out.columns] + \
           [c for c in out.columns if c not in first]
    return out[cols]


# =============================================================================
# 5. Spearman correlation per block
# =============================================================================
def spearman_heatmaps(train: pd.DataFrame, blocks: Optional[Iterable[str]] = None,
                      save_name: str = "05_spearman_by_block.png") -> dict[str, pd.DataFrame]:
    """Spearman correlation heatmap for each feature block (TRAIN only).

    Why Spearman: it needs no normality assumption and works for the skewed
    numeric features and for 0/1 or ordered codes. Nominal features (marital,
    smoking) are skipped because their integer codes have no order, so a
    correlation would be meaningless.
    """
    names = list(config.feature_blocks) if blocks is None else list(blocks)
    extractor.resolve_features(names)  # validation only
    _style()
    data = extractor.recode_binary(train)
    corr_by_block: dict[str, pd.DataFrame] = {}
    for block in names:
        feats = [f for f in config.feature_blocks[block] if config.feature_types[f] != "nominal"]
        extractor._require_columns(data, feats, f"spearman_heatmaps ({block})")
        if len(feats) >= 2:
            # pairwise-complete correlation: each pair uses the rows where both exist
            corr_by_block[block] = data[feats].corr(method="spearman")
    if not corr_by_block:
        raise ValueError("No block has two or more non-nominal features.")

    # Equal-sized panels in a grid (3 per row) stay readable; one long row would shrink the text
    ncols = 3
    nrows = math.ceil(len(corr_by_block) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.6 * nrows), squeeze=False)
    for ax, (block, corr) in zip(axes.ravel(), corr_by_block.items()):
        sns.heatmap(corr, ax=ax, vmin=-1, vmax=1, center=0, cmap="vlag", square=True,
                    annot=True, fmt=".2f", annot_kws={"size": 8}, cbar=False,
                    linewidths=0.5, linecolor="white")
        ax.set_title(block)
        ax.tick_params(axis="x", rotation=45)
        plt.setp(ax.get_xticklabels(), ha="right", fontsize=8)
        plt.setp(ax.get_yticklabels(), fontsize=8)
    for ax in axes.ravel()[len(corr_by_block):]:
        ax.axis("off")
    _save(fig, save_name)
    return corr_by_block

def global_correlation(train: pd.DataFrame, blocks: Optional[Iterable[str]] = None,
                       save_name: str = "05_global_spearman.png") -> pd.DataFrame:
    """Global Spearman correlation to catch collinearity across all blocks (TRAIN only)."""
    features = extractor.resolve_features(blocks)
    # Exclude nominal features as integer codes lack magnitude/order
    feats = [f for f in features if config.feature_types[f] != "nominal"]
    extractor._require_columns(train, feats, "global_correlation")
    _style()
    
    data = extractor.recode_binary(train)
    corr = data[feats].corr(method="spearman")
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, ax=ax, vmin=-1, vmax=1, center=0, cmap="vlag", square=True,
                annot=False, cbar=True, linewidths=0.5, linecolor="white")
    ax.set_title("Global Spearman Correlation")
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    _save(fig, save_name)
    
    # Extract pairs with |rho| > 0.8
    pairs = corr.unstack().reset_index()
    pairs.columns = ["feature_1", "feature_2", "spearman_rho"]
    # Remove self-correlations and duplicate pairs
    pairs = pairs[pairs["feature_1"] < pairs["feature_2"]]
    high_corr = pairs[pairs["spearman_rho"].abs() > 0.8].sort_values(by="spearman_rho", key=abs, ascending=False)
    
    return high_corr.reset_index(drop=True)

# =============================================================================
# 6. Temporal drift (descriptive only)
# =============================================================================
def temporal_drift(train: pd.DataFrame, test: pd.DataFrame,
                   blocks: Optional[Iterable[str]] = None,
                   save_name: str = "06_temporal_drift.png") -> pd.DataFrame:
    """Rank features by how much their distribution moved from train to test.

    * numeric features: standardised mean difference (SMD)
          (mean_test - mean_train) / sqrt((var_train + var_test) / 2)
      SMD divides by the spread, so features with different units are comparable.
    * categorical features: the largest absolute difference in any category's
      proportion (0 = identical mix, 1 = completely different).

    The two scales are not identical, so the bars are coloured by kind and the
    0.1 line is only a rough "small shift" rule of thumb. This chart DESCRIBES the
    shift for the report. It must not be used to drop or alter features, because
    that would let the locked test cycle influence modelling.
    """
    features = extractor.resolve_features(blocks)
    extractor._require_columns(train, features, "temporal_drift (train)")
    extractor._require_columns(test, features, "temporal_drift (test)")
    _style()
    tr, te = extractor.recode_binary(train), extractor.recode_binary(test)
    rows = []
    for f in features:
        a, b = tr[f].dropna(), te[f].dropna()
        if config.feature_types[f] == "numeric":
            pooled_sd = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            smd = (b.mean() - a.mean()) / pooled_sd if pooled_sd > 0 else 0.0
            rows.append({"feature": f, "kind": "numeric (SMD)", "drift": abs(smd),
                         "signed_detail": f"SMD = {smd:+.3f}",
                         "missing_diff": te[f].isna().mean() - tr[f].isna().mean()})
        else:
            pa = a.value_counts(normalize=True)
            pb = b.value_counts(normalize=True)
            diff = pb.sub(pa, fill_value=0.0)
            worst = diff.abs().idxmax()
            rows.append({"feature": f, "kind": "categorical (max |diff in proportion|)",
                         "drift": float(diff.abs().max()),
                         "signed_detail": f"{_level_name(f, worst)}: {diff[worst]:+.3f}",
                         "missing_diff": te[f].isna().mean() - tr[f].isna().mean()})
    out = pd.DataFrame(rows).sort_values("drift", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 0.35 * len(out) + 1.5))
    kind_colour = {"numeric (SMD)": "#2B6CB0", "categorical (max |diff in proportion|)": "#DD6B20"}
    ax.barh(out["feature"], out["drift"], color=out["kind"].map(kind_colour))
    ax.invert_yaxis()
    ax.axvline(0.1, color="grey", ls="--", lw=1)
    ax.set_xlabel("Drift (train -> test)")
    ax.set_title("Temporal drift of model features (descriptive only)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in kind_colour.values()],
              labels=list(kind_colour), frameon=False, loc="lower right", fontsize=8)
    _save(fig, save_name)
    return out


# =============================================================================
# 7. Asian subgroup by BMI band
# =============================================================================
def asian_bmi_prevalence(train: pd.DataFrame,
                         save_name: str = "07_asian_bmi_prevalence.png") -> pd.DataFrame:
    """Prevalence by BMI band, Asian vs non-Asian, under two band systems (TRAIN only).

    Two panels (WHO and Asia-Pacific cut-offs) show whether using the lower Asian
    cut-offs separates risk better inside the Asian subgroup. Hatched bars have
    fewer than MIN_N people: read them as hints, not findings.
    """
    extractor._require_columns(train, ["body_mass_index", "is_asian_descent", config.target_col], "asian_bmi_prevalence")
    _style()
    d = train[["body_mass_index", "is_asian_descent", config.target_col]].dropna().copy()
    d["Group"] = d["is_asian_descent"].map({0: "Non-Asian", 1: "Asian"})
    tables = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (system, (edges, labels)) in zip(axes, BMI_BANDS.items()):
        d["band"] = pd.cut(d["body_mass_index"], bins=edges, labels=labels, right=False)
        t = _prevalence_table(d, ["band", "Group"])
        _grouped_bars(ax, t, "band", "Group", labels, ["Non-Asian", "Asian"],
                      f"{system} BMI bands (hatched = n<{MIN_N})")
        ax.set_xlabel("BMI band (kg/m2)")
        tables.append(t.assign(band_system=system))
    axes[0].legend(frameon=False, fontsize=8)
    _save(fig, save_name)
    out = pd.concat(tables, ignore_index=True).rename(columns={"band": "bmi_band", "Group": "group"})
    return out[["band_system", "bmi_band", "group", "n", "positives",
                "prevalence", "ci_low", "ci_high"]]


def label_quality_check(train: pd.DataFrame, save_name: str = "08_label_quality.png") -> pd.DataFrame:
    """Prevalence for 'fasting glucose available' vs 'HbA1c only'."""
    extractor._require_columns(train, ["PHAFSTHR", "LBXGLU", config.target_col], "label_quality_check")
    _style()
    
    d = train[["PHAFSTHR", "LBXGLU", config.target_col]].copy()

    has_fasting = d['LBXGLU'].notna() & (d["PHAFSTHR"] >= 8)

    d["Label Source"] = has_fasting.map({True: "Fasting Glucose + HbA1c", False: "HbA1c only"})
    d["Cohort"] = "Train" # Dummy hue for the bar plot helper
    
    t = _prevalence_table(d, ["Label Source", "Cohort"])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    _grouped_bars(ax, t, "Label Source", "Cohort", 
                  ["Fasting Glucose + HbA1c", "HbA1c only"], ["Train"], 
                  "Prevalence by Label Source")
    ax.set_xlabel("Available Lab Tests")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, save_name)
    
    return t[["Label Source", "n", "positives", "prevalence", "ci_low", "ci_high"]]

# =============================================================================
# Run everything
# =============================================================================
def run_all_eda(train: pd.DataFrame, test: pd.DataFrame,
                flow: Optional[list[dict]] = None) -> dict[str, object]:
    """Run every EDA step in order and return all result tables in one dict."""
    results: dict[str, object] = {}
    if flow:
        results["cohort_flow"] = cohort_flow_table(flow)
    results["prevalence_overview"] = prevalence_overview(train, test)
    results["missingness"] = missingness_summary(train, test)
    results["missing_vs_target"] = missing_vs_target(train)
    results["distributions"] = plot_distributions(train, test)
    results["prevalence_by_feature"] = prevalence_by_feature(train)
    results["spearman"] = spearman_heatmaps(train)
    results["global_correlation"] = global_correlation(train)
    results["drift"] = temporal_drift(train, test)
    results["asian_bmi"] = asian_bmi_prevalence(train)
    results["label_quality"] = label_quality_check(train)
    return results


if __name__ == "__main__":
    from preprocessing import Loader

    loader = Loader(config)
    df_train, df_test, flow, _, _ = loader.load_clean(force_rebuild=False)

    run_all_eda(df_train, df_test, flow=flow)
    print("Saved figures:", sorted(p.name for p in FIG_DIR.glob("*.png")))