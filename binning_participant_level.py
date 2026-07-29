from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCRIPT_REPO = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_REPO.parent
OUTPUT_DIR = WORKSPACE_ROOT / "outputs"

BLOCK_RESULTS_PATH = (
    OUTPUT_DIR
    / "histogram_te_block_results_diagnostic_pairwise_2d_3x3_quantile_lag1_shuffle100.csv"
)

PARTICIPANT_RESULTS_PATH = (
    OUTPUT_DIR
    / "histogram_te_participant_results_diagnostic_pairwise_2d_3x3_quantile_lag1_shuffle100.csv"
)

PARTICIPANT_CORRELATIONS_PATH = (
    OUTPUT_DIR
    / "histogram_te_participant_correlations_diagnostic_pairwise_2d_3x3_quantile_lag1_shuffle100.csv"
)


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

TE_METRICS = [
    "eye_to_head_te",
    "head_to_eye_te",
    "eye_to_head_corrected_te",
    "head_to_eye_corrected_te",
]

GROUP_COL = "participant_id"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def log(message: str) -> None:
    print(message, flush=True)


def safe_corr(x: pd.Series, y: pd.Series) -> tuple[float, float, float, float, int]:
    df = pd.DataFrame({"x": x, "y": y}).dropna()

    n = len(df)

    if n < 3:
        return np.nan, np.nan, np.nan, np.nan, n

    if df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return np.nan, np.nan, np.nan, np.nan, n

    pearson_r, pearson_p = pearsonr(df["x"], df["y"])
    spearman_rho, spearman_p = spearmanr(df["x"], df["y"])

    return pearson_r, pearson_p, spearman_rho, spearman_p, n


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna()

    if valid.sum() == 0:
        return np.nan

    v = values[valid].astype(float)
    w = weights[valid].astype(float)

    if w.sum() == 0:
        return np.nan

    return float(np.average(v, weights=w))


def aggregate_participant_level(block_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates block-level TE results to participant level.

    Current block-level TE run only used:
        Bias and BiasJitter

    Each participant has:
        3 layouts × 2 tracking modes = 6 analyzed blocks
        6 blocks × 12 trials = 72 analyzed degraded-mode trials

    So participant-level degraded accuracy is:
        total correct_count across Bias/BiasJitter blocks / total n_trials across those blocks
    """
    rows = []

    for participant_id, g in block_df.groupby(GROUP_COL):
        g = g.copy()

        total_correct = g["correct_count"].sum()
        total_trials = g["n_trials_expected"].sum()

        participant_accuracy = total_correct / total_trials
        participant_error_rate = 1.0 - participant_accuracy

        row = {
            "participant_id": participant_id,
            "n_blocks": len(g),
            "n_bias_blocks": int((g["tracking"] == "Bias").sum()),
            "n_biasjitter_blocks": int((g["tracking"] == "BiasJitter").sum()),
            "total_correct": total_correct,
            "total_trials": total_trials,
            "participant_accuracy": participant_accuracy,
            "participant_error_rate": participant_error_rate,
            "mean_block_accuracy": g["accuracy"].mean(),
            "mean_block_error_rate": g["error_rate"].mean(),
            "min_block_accuracy": g["accuracy"].min(),
            "max_block_accuracy": g["accuracy"].max(),
        }

        # Unweighted TE means: each block contributes equally.
        for metric in TE_METRICS:
            row[f"{metric}_mean"] = g[metric].mean()
            row[f"{metric}_median"] = g[metric].median()
            row[f"{metric}_std"] = g[metric].std()

        # Weighted TE means: longer/more sampled blocks contribute more.
        # Use n_active_rows if available, otherwise fallback to n_trials_expected.
        if "n_active_rows" in g.columns:
            weights = g["n_active_rows"]
        else:
            weights = g["n_trials_expected"]

        for metric in TE_METRICS:
            row[f"{metric}_weighted_by_frames"] = weighted_mean(g[metric], weights)

        rows.append(row)

    return pd.DataFrame(rows).sort_values("participant_id").reset_index(drop=True)


def compute_participant_correlations(participant_df: pd.DataFrame) -> pd.DataFrame:
    """
    Correlate participant-level TE summaries with participant-level performance.
    """
    rows = []

    outcomes = [
        "participant_accuracy",
        "participant_error_rate",
    ]

    te_summary_cols = []

    for metric in TE_METRICS:
        te_summary_cols.append(f"{metric}_mean")
        te_summary_cols.append(f"{metric}_median")
        te_summary_cols.append(f"{metric}_weighted_by_frames")

    for outcome in outcomes:
        for metric_col in te_summary_cols:
            pearson_r, pearson_p, spearman_rho, spearman_p, n = safe_corr(
                participant_df[metric_col],
                participant_df[outcome],
            )

            rows.append(
                {
                    "metric": metric_col,
                    "outcome": outcome,
                    "scope": "participant_level_bias_and_biasjitter",
                    "n_participants": n,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_rho": spearman_rho,
                    "spearman_p": spearman_p,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    log("Starting participant-level TE aggregation")
    log(f"Reading block-level results from: {BLOCK_RESULTS_PATH}")

    if not BLOCK_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing block-level results file: {BLOCK_RESULTS_PATH}")

    block_df = pd.read_csv(BLOCK_RESULTS_PATH, dtype={"participant_id": str})

    if "status" in block_df.columns:
        block_df = block_df[block_df["status"] == "ok"].copy()

    log(f"Loaded valid block rows: {len(block_df)}")

    required_cols = [
        "participant_id",
        "tracking",
        "correct_count",
        "n_trials_expected",
        "accuracy",
        "error_rate",
        *TE_METRICS,
    ]

    missing = [col for col in required_cols if col not in block_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    participant_df = aggregate_participant_level(block_df)

    participant_df.to_csv(PARTICIPANT_RESULTS_PATH, index=False)
    log(f"Saved participant-level results to: {PARTICIPANT_RESULTS_PATH}")

    corr_df = compute_participant_correlations(participant_df)

    corr_df.to_csv(PARTICIPANT_CORRELATIONS_PATH, index=False)
    log(f"Saved participant-level correlations to: {PARTICIPANT_CORRELATIONS_PATH}")

    log("\nParticipant-level data preview:")
    preview_cols = [
        "participant_id",
        "n_blocks",
        "total_correct",
        "total_trials",
        "participant_accuracy",
        "participant_error_rate",
        "head_to_eye_te_mean",
        "head_to_eye_corrected_te_mean",
    ]
    log(participant_df[preview_cols].to_string(index=False))

    log("\nParticipant-level correlation preview:")
    key_metrics = [
        "head_to_eye_te_mean",
        "head_to_eye_corrected_te_mean",
        "head_to_eye_te_weighted_by_frames",
        "head_to_eye_corrected_te_weighted_by_frames",
        "eye_to_head_te_mean",
        "eye_to_head_corrected_te_mean",
    ]

    preview_corr = corr_df[
        corr_df["metric"].isin(key_metrics)
    ].copy()

    log(preview_corr.to_string(index=False))

    log("\nDone.")


if __name__ == "__main__":
    main()