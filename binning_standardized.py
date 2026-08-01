from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path("/home/evo/McMaster/eye_gaze_proj")
DATASET_ROOT = PROJECT_ROOT / "exp1-gaze-interaction-dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TE_BLOCK_PATH = OUTPUT_DIR / (
    "histogram_te_block_results_diagnostic_all_modes_pairwise_2d_3x3_"
    "quantile_lag1_shuffle100.csv"
)

BLOCK_SUMMARY_PATH = DATASET_ROOT / "data" / "processed" / "balanced_subject_block_summary.csv"

OUT_MERGED = OUTPUT_DIR / "standardized_te_bias_noise_merged.csv"
OUT_CORR = OUTPUT_DIR / "standardized_te_bias_noise_correlations.csv"
OUT_SUMMARY = OUTPUT_DIR / "standardized_te_bias_noise_tracking_summary.csv"


# ------------------------------------------------------------
# Variables
# ------------------------------------------------------------

TE_METRICS = [
    "eye_to_head_te",
    "head_to_eye_te",
    "eye_to_head_corrected_te",
    "head_to_eye_corrected_te",
]

# Numeric bias/noise columns from the processed dataset.
# These are not binary condition labels.
BIAS_NOISE_COLUMNS = [
    "label_bias_magnitude_mean_deg",
    "label_bias_az_mean_deg",
    "label_bias_el_mean_deg",
    "label_jitter_2d_rms_deg",
]

# Try these merge keys in order.
MERGE_CANDIDATES = [
    ["condition_block_key"],
    ["participant_id", "layout", "tracking", "block_index"],
    ["participant_id", "layout", "tracking"],
]


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def zscore(series: pd.Series) -> pd.Series:
    """
    Z-score standardization:
        z = (x - mean(x)) / sd(x)

    Uses sample standard deviation with ddof=1.
    """
    sd = series.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        raise ValueError(
            f"Cannot z-score column '{series.name}': standard deviation is zero or NaN."
        )
    return (series - series.mean()) / sd


def safe_corr(x: pd.Series, y: pd.Series, method: str):
    """Compute correlation after dropping missing values."""
    temp = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(temp) < 3:
        return np.nan, np.nan, len(temp)

    if temp["x"].nunique() < 2 or temp["y"].nunique() < 2:
        return np.nan, np.nan, len(temp)

    if method == "pearson":
        r, p = pearsonr(temp["x"], temp["y"])
    elif method == "spearman":
        r, p = spearmanr(temp["x"], temp["y"])
    else:
        raise ValueError(f"Unknown method: {method}")

    return r, p, len(temp)


def choose_merge_keys(te_df: pd.DataFrame, block_df: pd.DataFrame):
    """Choose a valid set of merge keys shared by both dataframes."""
    for keys in MERGE_CANDIDATES:
        if all(key in te_df.columns for key in keys) and all(key in block_df.columns for key in keys):
            return keys

    raise KeyError(
        "Could not find valid merge keys.\n\n"
        f"TE columns:\n{te_df.columns.tolist()}\n\n"
        f"Block summary columns:\n{block_df.columns.tolist()}"
    )


def load_inputs():
    if not TE_BLOCK_PATH.exists():
        raise FileNotFoundError(f"Could not find TE block results:\n{TE_BLOCK_PATH}")

    if not BLOCK_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Could not find block summary:\n{BLOCK_SUMMARY_PATH}")

    te_df = pd.read_csv(TE_BLOCK_PATH)
    block_df = pd.read_csv(BLOCK_SUMMARY_PATH)

    if "status" in te_df.columns:
        te_df = te_df[te_df["status"].eq("ok")].copy()

    missing_te = [col for col in TE_METRICS if col not in te_df.columns]
    if missing_te:
        raise KeyError(f"Missing TE columns from TE results: {missing_te}")

    missing_bias_noise = [col for col in BIAS_NOISE_COLUMNS if col not in block_df.columns]
    if missing_bias_noise:
        raise KeyError(
            "Missing expected numeric bias/noise columns from block summary:\n"
            f"{missing_bias_noise}\n\n"
            "Check balanced_subject_block_summary.csv for the exact column names."
        )

    return te_df, block_df


def main():
    te_df, block_df = load_inputs()

    print(f"Loaded TE rows: {len(te_df)}")
    print(f"Loaded block summary rows: {len(block_df)}")

    merge_keys = choose_merge_keys(te_df, block_df)
    print(f"Using merge keys: {merge_keys}")

    keep_cols = merge_keys + BIAS_NOISE_COLUMNS

    # Add useful metadata/performance columns if available.
    for col in ["accuracy", "correct_count", "n_trials", "tracking", "layout", "participant_id"]:
        if col in block_df.columns and col not in keep_cols:
            keep_cols.append(col)

    block_small = block_df[keep_cols].drop_duplicates()

    merged = te_df.merge(
        block_small,
        on=merge_keys,
        how="left",
        validate="many_to_one",
    )

    # Check merge quality.
    for col in BIAS_NOISE_COLUMNS:
        missing_n = merged[col].isna().sum()
        if missing_n > 0:
            raise ValueError(
                f"After merging, {missing_n} rows are missing '{col}'. "
                "The merge keys may be wrong."
            )

    # Resolve duplicate accuracy columns if needed.
    if "accuracy_x" in merged.columns and "accuracy_y" in merged.columns:
        merged["accuracy"] = merged["accuracy_x"]
        merged = merged.drop(columns=["accuracy_x", "accuracy_y"])
    elif "accuracy_x" in merged.columns:
        merged = merged.rename(columns={"accuracy_x": "accuracy"})
    elif "accuracy_y" in merged.columns:
        merged = merged.rename(columns={"accuracy_y": "accuracy"})

    if "accuracy" in merged.columns and "error_rate" not in merged.columns:
        merged["error_rate"] = 1.0 - merged["accuracy"]

    # --------------------------------------------------------
    # Z-score standardization
    # --------------------------------------------------------

    standardized_columns = []

    # Z-score TE metrics.
    for col in TE_METRICS:
        z_col = f"{col}_z"
        merged[z_col] = zscore(merged[col])
        standardized_columns.append(z_col)

    # Z-score numeric bias/noise values.
    for col in BIAS_NOISE_COLUMNS:
        z_col = f"{col}_z"
        merged[z_col] = zscore(merged[col])
        standardized_columns.append(z_col)

    te_z_metrics = [f"{col}_z" for col in TE_METRICS]
    bias_noise_z_predictors = [f"{col}_z" for col in BIAS_NOISE_COLUMNS]

    # --------------------------------------------------------
    # Correlations:
    # standardized TE vs standardized bias/noise
    # --------------------------------------------------------

    rows = []

    for predictor in bias_noise_z_predictors:
        for outcome in te_z_metrics:
            pearson_r, pearson_p, n = safe_corr(
                merged[predictor],
                merged[outcome],
                method="pearson",
            )

            spearman_rho, spearman_p, _ = safe_corr(
                merged[predictor],
                merged[outcome],
                method="spearman",
            )

            rows.append(
                {
                    "analysis_family": "standardized_te_vs_standardized_bias_noise",
                    "predictor": predictor,
                    "outcome": outcome,
                    "n": n,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_rho": spearman_rho,
                    "spearman_p": spearman_p,
                }
            )

    # Optional: standardized bias/noise vs performance.
    # This directly checks whether the manipulation strength relates to accuracy/error.
    performance_metrics = []
    for col in ["accuracy", "error_rate"]:
        if col in merged.columns:
            z_col = f"{col}_z"
            merged[z_col] = zscore(merged[col])
            performance_metrics.append(z_col)

    for predictor in bias_noise_z_predictors:
        for outcome in performance_metrics:
            pearson_r, pearson_p, n = safe_corr(
                merged[predictor],
                merged[outcome],
                method="pearson",
            )

            spearman_rho, spearman_p, _ = safe_corr(
                merged[predictor],
                merged[outcome],
                method="spearman",
            )

            rows.append(
                {
                    "analysis_family": "standardized_performance_vs_standardized_bias_noise",
                    "predictor": predictor,
                    "outcome": outcome,
                    "n": n,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_rho": spearman_rho,
                    "spearman_p": spearman_p,
                }
            )

    corr_df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Tracking-mode summary for sanity checks
    # --------------------------------------------------------

    summary_cols = (
        BIAS_NOISE_COLUMNS
        + bias_noise_z_predictors
        + TE_METRICS
        + te_z_metrics
    )

    for col in ["accuracy", "error_rate", "accuracy_z", "error_rate_z"]:
        if col in merged.columns:
            summary_cols.append(col)

    if "tracking" in merged.columns:
        tracking_summary = (
            merged.groupby("tracking", as_index=False)
            .agg(
                n_blocks=("tracking", "size"),
                **{
                    f"{col}_mean": (col, "mean")
                    for col in summary_cols
                    if col in merged.columns
                },
                **{
                    f"{col}_std": (col, "std")
                    for col in summary_cols
                    if col in merged.columns
                },
            )
        )
    else:
        tracking_summary = pd.DataFrame()

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    merged.to_csv(OUT_MERGED, index=False)
    corr_df.to_csv(OUT_CORR, index=False)
    tracking_summary.to_csv(OUT_SUMMARY, index=False)

    print("\nSaved:")
    print(f"  {OUT_MERGED}")
    print(f"  {OUT_CORR}")
    print(f"  {OUT_SUMMARY}")

    print("\nStandardized TE vs standardized numeric bias/noise correlations:")
    main_corr = corr_df[
        corr_df["analysis_family"].eq("standardized_te_vs_standardized_bias_noise")
    ]
    print(main_corr.to_string(index=False, float_format=lambda x: f"{x:.6g}"))

    print("\nStandardized performance vs standardized numeric bias/noise correlations:")
    perf_corr = corr_df[
        corr_df["analysis_family"].eq("standardized_performance_vs_standardized_bias_noise")
    ]
    if len(perf_corr) > 0:
        print(perf_corr.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    else:
        print("No performance columns found.")


if __name__ == "__main__":
    main()