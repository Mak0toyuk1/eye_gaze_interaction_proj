from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


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

OUT_MERGED = OUTPUT_DIR / "friedman_te_level_tests_merged_fixed.csv"
OUT_AGG = OUTPUT_DIR / "friedman_te_level_tests_participant_level_fixed.csv"
OUT_GLOBAL = OUTPUT_DIR / "friedman_te_level_tests_global_fixed.csv"
OUT_PAIRWISE = OUTPUT_DIR / "friedman_te_level_tests_pairwise_wilcoxon_fixed.csv"
OUT_SUMMARY = OUTPUT_DIR / "friedman_te_level_tests_level_summary_fixed.csv"


# ------------------------------------------------------------
# Variables
# ------------------------------------------------------------

TE_METRICS = [
    "eye_to_head_te",
    "head_to_eye_te",
    "eye_to_head_corrected_te",
    "head_to_eye_corrected_te",
]

PERFORMANCE_METRICS = [
    "accuracy",
    "error_rate",
]

NUMERIC_MANIPULATION_COLUMNS = [
    "label_bias_magnitude_mean_deg",
    "label_bias_az_mean_deg",
    "label_bias_el_mean_deg",
    "label_jitter_2d_rms_deg",
]

MERGE_CANDIDATES = [
    ["condition_block_key"],
    ["participant_id", "layout", "tracking", "block_index"],
    ["participant_id", "layout", "tracking"],
]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def choose_merge_keys(te_df: pd.DataFrame, block_df: pd.DataFrame):
    for keys in MERGE_CANDIDATES:
        if all(k in te_df.columns for k in keys) and all(k in block_df.columns for k in keys):
            return keys

    raise KeyError(
        "Could not find valid merge keys.\n\n"
        f"TE columns:\n{te_df.columns.tolist()}\n\n"
        f"Block summary columns:\n{block_df.columns.tolist()}"
    )


def coalesce_merged_column(df: pd.DataFrame, base_name: str) -> pd.DataFrame:
    """
    After pandas merge, duplicate columns may become tracking_x/tracking_y.
    This restores them to tracking, participant_id, layout, etc.
    """
    x_col = f"{base_name}_x"
    y_col = f"{base_name}_y"

    if x_col in df.columns and y_col in df.columns:
        df[base_name] = df[x_col].combine_first(df[y_col])
        df = df.drop(columns=[x_col, y_col])
    elif x_col in df.columns:
        df = df.rename(columns={x_col: base_name})
    elif y_col in df.columns:
        df = df.rename(columns={y_col: base_name})

    return df


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR correction."""
    p = p_values.astype(float).to_numpy()
    n = len(p)

    order = np.argsort(p)
    ranked = p[order]

    adjusted = np.empty(n, dtype=float)
    cumulative_min = 1.0

    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        cumulative_min = min(cumulative_min, val)
        adjusted[order[i]] = cumulative_min

    return pd.Series(np.minimum(adjusted, 1.0), index=p_values.index)


def load_and_merge() -> pd.DataFrame:
    if not TE_BLOCK_PATH.exists():
        raise FileNotFoundError(f"Could not find TE block file:\n{TE_BLOCK_PATH}")

    if not BLOCK_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Could not find block summary file:\n{BLOCK_SUMMARY_PATH}")

    te_df = pd.read_csv(TE_BLOCK_PATH)
    block_df = pd.read_csv(BLOCK_SUMMARY_PATH)

    if "status" in te_df.columns:
        te_df = te_df[te_df["status"].eq("ok")].copy()

    missing_te = [c for c in TE_METRICS if c not in te_df.columns]
    if missing_te:
        raise KeyError(f"Missing TE metric columns from TE file: {missing_te}")

    missing_num = [c for c in NUMERIC_MANIPULATION_COLUMNS if c not in block_df.columns]
    if missing_num:
        raise KeyError(
            "Missing expected numeric bias/noise columns from block summary:\n"
            f"{missing_num}\n\n"
            "Check balanced_subject_block_summary.csv for the exact column names."
        )

    merge_keys = choose_merge_keys(te_df, block_df)
    print(f"Using merge keys: {merge_keys}")

    keep_cols = merge_keys + NUMERIC_MANIPULATION_COLUMNS

    for col in [
        "participant_id",
        "layout",
        "tracking",
        "block_index",
        "accuracy",
        "correct_count",
        "n_trials",
    ]:
        if col in block_df.columns and col not in keep_cols:
            keep_cols.append(col)

    block_small = block_df[keep_cols].drop_duplicates()

    merged = te_df.merge(
        block_small,
        on=merge_keys,
        how="left",
        validate="many_to_one",
    )

    # Fix columns like tracking_x/tracking_y created by merge.
    for col in [
        "participant_id",
        "layout",
        "tracking",
        "block_index",
        "accuracy",
        "correct_count",
        "n_trials",
    ]:
        merged = coalesce_merged_column(merged, col)

    for col in NUMERIC_MANIPULATION_COLUMNS:
        missing = merged[col].isna().sum()
        if missing > 0:
            raise ValueError(
                f"After merge, {missing} rows are missing {col}. "
                "The merge keys may be wrong."
            )

    required_after_merge = ["participant_id", "tracking"]
    missing_after_merge = [c for c in required_after_merge if c not in merged.columns]
    if missing_after_merge:
        raise KeyError(
            f"Missing required columns after merge: {missing_after_merge}\n"
            f"Columns available:\n{merged.columns.tolist()}"
        )

    if "accuracy" in merged.columns and "error_rate" not in merged.columns:
        merged["error_rate"] = 1.0 - merged["accuracy"]

    print(f"Rows after merge: {len(merged)}")
    print("Tracking modes:", sorted(merged["tracking"].dropna().unique().tolist()))
    print("Participants:", merged["participant_id"].nunique())

    return merged


def make_level_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Main 4-condition repeated-measures factor.
    df["tracking_level"] = df["tracking"].astype(str)

    # Numeric experimental levels. Rounding avoids tiny float differences.
    for col in NUMERIC_MANIPULATION_COLUMNS:
        level_col = f"{col}_level"
        df[level_col] = df[col].round(6).astype(str)

    return df


def participant_level_aggregate(
    df: pd.DataFrame,
    factor_col: str,
    outcomes: list[str],
) -> pd.DataFrame:
    """
    Aggregate to participant x level.

    This avoids treating all 384 block rows as independent.
    Each participant contributes one mean value per level.
    """
    existing_outcomes = [o for o in outcomes if o in df.columns]
    needed = ["participant_id", factor_col] + existing_outcomes

    temp = df[needed].dropna(subset=["participant_id", factor_col]).copy()

    agg = (
        temp.groupby(["participant_id", factor_col], as_index=False)
        .agg({outcome: "mean" for outcome in existing_outcomes})
    )

    return agg


def friedman_for_factor(agg: pd.DataFrame, factor_col: str, outcome: str) -> dict:
    wide = agg.pivot(index="participant_id", columns=factor_col, values=outcome)

    # Friedman requires complete repeated-measures rows.
    wide_complete = wide.dropna(axis=0, how="any")

    n_participants = wide_complete.shape[0]
    levels = list(wide_complete.columns)
    k_levels = len(levels)

    if n_participants < 2 or k_levels < 3:
        return {
            "factor": factor_col,
            "outcome": outcome,
            "n_participants": n_participants,
            "n_levels": k_levels,
            "levels": "|".join(map(str, levels)),
            "test": "friedman",
            "statistic": np.nan,
            "p_value": np.nan,
            "kendalls_w": np.nan,
            "note": "Need at least 2 participants and at least 3 complete levels for Friedman.",
        }

    arrays = [wide_complete[level].to_numpy() for level in levels]
    stat, p = friedmanchisquare(*arrays)

    # Kendall's W effect size for Friedman.
    kendalls_w = stat / (n_participants * (k_levels - 1))

    return {
        "factor": factor_col,
        "outcome": outcome,
        "n_participants": n_participants,
        "n_levels": k_levels,
        "levels": "|".join(map(str, levels)),
        "test": "friedman",
        "statistic": stat,
        "p_value": p,
        "kendalls_w": kendalls_w,
        "note": "",
    }


def wilcoxon_pairwise_for_factor(
    agg: pd.DataFrame,
    factor_col: str,
    outcome: str,
) -> list[dict]:
    wide = agg.pivot(index="participant_id", columns=factor_col, values=outcome)
    levels = list(wide.columns)

    rows = []

    for level_a, level_b in combinations(levels, 2):
        pair = wide[[level_a, level_b]].dropna(axis=0, how="any")

        n = pair.shape[0]

        if n < 2:
            rows.append(
                {
                    "factor": factor_col,
                    "outcome": outcome,
                    "level_a": level_a,
                    "level_b": level_b,
                    "n_participants": n,
                    "test": "wilcoxon_signed_rank",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "mean_a": np.nan,
                    "mean_b": np.nan,
                    "median_a": np.nan,
                    "median_b": np.nan,
                    "mean_diff_b_minus_a": np.nan,
                    "median_diff_b_minus_a": np.nan,
                    "note": "Not enough paired observations.",
                }
            )
            continue

        a = pair[level_a]
        b = pair[level_b]
        diff = b - a

        if np.allclose(diff.to_numpy(), 0):
            stat, p = 0.0, 1.0
            note = "All paired differences are zero."
        else:
            stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
            note = ""

        rows.append(
            {
                "factor": factor_col,
                "outcome": outcome,
                "level_a": level_a,
                "level_b": level_b,
                "n_participants": n,
                "test": "wilcoxon_signed_rank",
                "statistic": stat,
                "p_value": p,
                "mean_a": a.mean(),
                "mean_b": b.mean(),
                "median_a": a.median(),
                "median_b": b.median(),
                "mean_diff_b_minus_a": diff.mean(),
                "median_diff_b_minus_a": diff.median(),
                "note": note,
            }
        )

    return rows


def summarize_levels(
    agg: pd.DataFrame,
    factor_col: str,
    outcomes: list[str],
) -> list[dict]:
    rows = []

    for outcome in outcomes:
        if outcome not in agg.columns:
            continue

        for level, g in agg.groupby(factor_col):
            vals = g[outcome].dropna()

            rows.append(
                {
                    "factor": factor_col,
                    "level": level,
                    "outcome": outcome,
                    "n_participants": vals.shape[0],
                    "mean": vals.mean(),
                    "median": vals.median(),
                    "std": vals.std(ddof=1),
                    "min": vals.min(),
                    "max": vals.max(),
                }
            )

    return rows


def main():
    merged = load_and_merge()
    merged = make_level_columns(merged)

    outcomes = TE_METRICS + [m for m in PERFORMANCE_METRICS if m in merged.columns]

    factor_cols = [
        "tracking_level",
        "label_bias_magnitude_mean_deg_level",
        "label_bias_az_mean_deg_level",
        "label_bias_el_mean_deg_level",
        "label_jitter_2d_rms_deg_level",
    ]

    all_agg = []
    global_rows = []
    pairwise_rows = []
    summary_rows = []

    for factor_col in factor_cols:
        print(f"\nAnalyzing factor: {factor_col}")

        agg = participant_level_aggregate(merged, factor_col, outcomes)
        agg["factor_analyzed"] = factor_col
        all_agg.append(agg)

        print(
            f"  levels: {sorted(agg[factor_col].dropna().unique().tolist())} | "
            f"participants: {agg['participant_id'].nunique()}"
        )

        for outcome in outcomes:
            global_rows.append(friedman_for_factor(agg, factor_col, outcome))
            pairwise_rows.extend(wilcoxon_pairwise_for_factor(agg, factor_col, outcome))

        summary_rows.extend(summarize_levels(agg, factor_col, outcomes))

    participant_level = pd.concat(all_agg, ignore_index=True)
    global_df = pd.DataFrame(global_rows)
    pairwise_df = pd.DataFrame(pairwise_rows)
    summary_df = pd.DataFrame(summary_rows)

    # FDR correction for pairwise tests separately for each factor/outcome.
    if len(pairwise_df) > 0:
        pairwise_df["p_fdr_bh"] = np.nan

        for (factor, outcome), idx in pairwise_df.groupby(["factor", "outcome"]).groups.items():
            idx = list(idx)
            valid_idx = pairwise_df.loc[idx].dropna(subset=["p_value"]).index

            if len(valid_idx) > 0:
                pairwise_df.loc[valid_idx, "p_fdr_bh"] = benjamini_hochberg(
                    pairwise_df.loc[valid_idx, "p_value"]
                ).to_numpy()

    merged.to_csv(OUT_MERGED, index=False)
    participant_level.to_csv(OUT_AGG, index=False)
    global_df.to_csv(OUT_GLOBAL, index=False)
    pairwise_df.to_csv(OUT_PAIRWISE, index=False)
    summary_df.to_csv(OUT_SUMMARY, index=False)

    print("\nSaved:")
    print(f"  {OUT_MERGED}")
    print(f"  {OUT_AGG}")
    print(f"  {OUT_GLOBAL}")
    print(f"  {OUT_PAIRWISE}")
    print(f"  {OUT_SUMMARY}")

    print("\nFriedman global tests:")
    print(global_df.to_string(index=False, float_format=lambda x: f"{x:.6g}"))

    print("\nSignificant pairwise Wilcoxon tests after FDR correction:")
    sig = pairwise_df[
        pairwise_df["p_fdr_bh"].notna()
        & pairwise_df["p_fdr_bh"].lt(0.05)
    ].copy()

    if len(sig) > 0:
        print(sig.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    else:
        print("No significant pairwise Wilcoxon tests after FDR correction.")


if __name__ == "__main__":
    main()