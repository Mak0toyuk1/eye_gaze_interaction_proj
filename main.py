from __future__ import annotations

import json
import math
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np

import pandas as pd
from scipy.stats import pearsonr, spearmanr
if not hasattr(np, "math"):
    np.math = math
from idtxl.data import Data
from idtxl.bivariate_te import BivariateTE


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCRIPT_REPO = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_REPO.parent
DATASET_ROOT = WORKSPACE_ROOT / "exp1-gaze-interaction-dataset"
OUTPUT_DIR = WORKSPACE_ROOT / "outputs"

RAW_FRAMES_DIR = DATASET_ROOT / "data" / "raw" / "frames"
PROCESSED_DIR = DATASET_ROOT / "data" / "processed"

BLOCK_SUMMARY_PATH = PROCESSED_DIR / "balanced_subject_block_summary.csv"

OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------------------

TRACKING_MODES = ["Bias", "BiasJitter"]

# Start with one-dimensional azimuth signals. Later we can repeat with
# elevation or angular velocity.
SIGNAL_COMPONENT = "az"  # options: "az", "el"

MIN_SAMPLES_PER_TRIAL = 30

# Set to an integer like 5 for a smoke test. Set to None for full run.
MAX_BLOCKS = None

# IDTxl settings. These are deliberately conservative for a first pass.
IDTXL_SETTINGS = {
    "cmi_estimator": "JidtKraskovCMI",
    "max_lag_sources": 5,
    "min_lag_sources": 1,
    "tau_sources": 1,
    "max_lag_target": 5,
    "tau_target": 1,
    "n_perm_max_stat": 100,
    "n_perm_min_stat": 100,
    "n_perm_omnibus": 100,
    "n_perm_max_seq": 100,
    "alpha_max_stat": 0.05,
    "alpha_min_stat": 0.05,
    "alpha_omnibus": 0.05,
    "alpha_max_seq": 0.05,
}


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def log(message: str) -> None:
    print(message, flush=True)


def frame_path_from_session_id(source_session_id: str) -> Path:
    return RAW_FRAMES_DIR / f"{source_session_id}_frames.csv"


def parse_source_session_id(source_sessions: str) -> str:
    """
    balanced_subject_block_summary.csv has values like:
    036_20260529_122759
    """
    if pd.isna(source_sessions):
        raise ValueError("Missing source_sessions value")

    # Some cells may theoretically contain multiple sessions joined by a delimiter.
    # The balanced selected blocks seem to use a single session.
    return str(source_sessions).split(";")[0].strip()


def quaternion_to_forward_vector(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Unity-style quaternion to forward vector.

    Formula assumes quaternion components are ordered x, y, z, w.
    This gives the rotated forward vector corresponding roughly to local +Z.
    """
    fx = 2.0 * (x * z + w * y)
    fy = 2.0 * (y * z - w * x)
    fz = 1.0 - 2.0 * (x * x + y * y)
    return fx, fy, fz


def forward_vector_to_az_el_deg(
    fx: np.ndarray,
    fy: np.ndarray,
    fz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    az = np.degrees(np.arctan2(fx, fz))
    horizontal = np.sqrt(fx * fx + fz * fz)
    el = np.degrees(np.arctan2(fy, horizontal))
    return az, el


def add_head_az_el(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "head_rotation_x",
        "head_rotation_y",
        "head_rotation_z",
        "head_rotation_w",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing head rotation columns: {missing}")

    fx, fy, fz = quaternion_to_forward_vector(
        df["head_rotation_x"].to_numpy(dtype=float),
        df["head_rotation_y"].to_numpy(dtype=float),
        df["head_rotation_z"].to_numpy(dtype=float),
        df["head_rotation_w"].to_numpy(dtype=float),
    )

    head_az, head_el = forward_vector_to_az_el_deg(fx, fy, fz)

    df = df.copy()
    df["head_az_deg"] = head_az
    df["head_el_deg"] = head_el
    return df


def get_signal_columns(component: str) -> tuple[str, str]:
    """
    Return eye column and head column.

    For the first replication attempt:
    - eye signal: raw gaze panel azimuth/elevation
    - head signal: head azimuth/elevation computed from head quaternion
    """
    if component == "az":
        return "raw_panel_az_deg", "head_az_deg"
    if component == "el":
        return "raw_panel_el_deg", "head_el_deg"

    raise ValueError(f"Unsupported SIGNAL_COMPONENT: {component}")


def prepare_block_array(
    frames: pd.DataFrame,
    block_index: int,
    tracking: str,
    component: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """
    Build IDTxl array for one participant x layout x tracking block.

    Output shape:
        processes x samples x replications

    process 0 = eye
    process 1 = head
    replications = trials
    """

    eye_col, head_col = get_signal_columns(component)

    block = frames[
        (frames["block_index"] == block_index)
        & (frames["tracking"] == tracking)
        & (frames["state"] == "TrialActive")
    ].copy()

    info: dict[str, Any] = {
        "n_active_rows": int(len(block)),
        "n_trials_found": 0,
        "n_trials_used": 0,
        "min_trial_len_before_trim": None,
        "n_samples_used": None,
        "skip_reason": None,
    }

    if block.empty:
        info["skip_reason"] = "no_trialactive_rows"
        return None, info

    block = add_head_az_el(block)

    needed = ["trial_index_in_block", "timestamp_utc", eye_col, head_col]
    missing = [col for col in needed if col not in block.columns]
    if missing:
        info["skip_reason"] = f"missing_columns: {missing}"
        return None, info

    block = block[needed].dropna(subset=[eye_col, head_col]).copy()

    if block.empty:
        info["skip_reason"] = "all_signal_rows_missing"
        return None, info

    trial_series: list[np.ndarray] = []

    for trial_idx, trial_df in block.groupby("trial_index_in_block"):
        trial_df = trial_df.sort_values("timestamp_utc")

        eye = trial_df[eye_col].to_numpy(dtype=float)
        head = trial_df[head_col].to_numpy(dtype=float)

        valid = np.isfinite(eye) & np.isfinite(head)
        eye = eye[valid]
        head = head[valid]

        if len(eye) >= MIN_SAMPLES_PER_TRIAL:
            trial_arr = np.vstack([eye, head])
            trial_series.append(trial_arr)

    info["n_trials_found"] = int(block["trial_index_in_block"].nunique())
    info["n_trials_used"] = len(trial_series)

    if len(trial_series) < 2:
        info["skip_reason"] = "fewer_than_two_usable_trials"
        return None, info

    min_len = min(arr.shape[1] for arr in trial_series)
    info["min_trial_len_before_trim"] = int(min_len)

    if min_len < MIN_SAMPLES_PER_TRIAL:
        info["skip_reason"] = "min_trial_length_too_short"
        return None, info

    # Trim all trials to the same length from the start.
    # Later we can try event-aligned windows before selection.
    trimmed = [arr[:, :min_len] for arr in trial_series]

    # shape before stack: each arr is processes x samples
    # after stack axis=2: processes x samples x replications
    data_array = np.stack(trimmed, axis=2)

    info["n_samples_used"] = int(min_len)

    return data_array, info


def extract_idtxl_diagnostics(results: Any, target: int) -> dict[str, Any]:
    """
    Extract useful diagnostics from an IDTxl result object.

    Important:
    IDTxl's BivariateTE pipeline performs statistical source selection.
    If no source is selected, selected_sources_te may be empty and the
    extracted TE will be 0.0 even though the analysis ran successfully.
    """

    out = {
        "te_value": np.nan,
        "selected_source_count": np.nan,
        "selected_sources": None,
        "omnibus_te": np.nan,
        "omnibus_pval": np.nan,
        "result_keys": None,
    }

    try:
        single = results.get_single_target(target, fdr=False)
    except Exception:
        try:
            single = results.get_single_target(target)
        except Exception:
            return out

    if not isinstance(single, dict):
        return out

    out["result_keys"] = ";".join(str(k) for k in single.keys())

    if "selected_vars_sources" in single:
        selected = single["selected_vars_sources"]
        out["selected_sources"] = str(selected)
        try:
            out["selected_source_count"] = len(selected)
        except Exception:
            out["selected_source_count"] = np.nan

    if "selected_sources_te" in single:
        try:
            vals = np.asarray(single["selected_sources_te"], dtype=float)
            if vals.size > 0:
                out["te_value"] = float(np.nansum(vals))
            else:
                out["te_value"] = 0.0
        except Exception:
            pass

    if "omnibus_te" in single:
        try:
            out["omnibus_te"] = float(single["omnibus_te"])
        except Exception:
            pass

    if "omnibus_pval" in single:
        try:
            out["omnibus_pval"] = float(single["omnibus_pval"])
        except Exception:
            pass

    return out


def run_bivariate_te(data_array: np.ndarray) -> dict[str, Any]:
    """
    Estimate bivariate TE in both directions.

    process 0 = eye
    process 1 = head
    """

    data = Data(data_array, dim_order="psr", normalise=True)

    out: dict[str, Any] = {
        "eye_to_head_te": np.nan,
        "head_to_eye_te": np.nan,
        "eye_to_head_selected_source_count": np.nan,
        "head_to_eye_selected_source_count": np.nan,
        "eye_to_head_selected_sources": None,
        "head_to_eye_selected_sources": None,
        "eye_to_head_omnibus_te": np.nan,
        "head_to_eye_omnibus_te": np.nan,
        "eye_to_head_omnibus_pval": np.nan,
        "head_to_eye_omnibus_pval": np.nan,
        "eye_to_head_result_keys": None,
        "head_to_eye_result_keys": None,
        "eye_to_head_result": None,
        "head_to_eye_result": None,
        "error": None,
    }

    try:
        analysis = BivariateTE()
        eye_to_head_result = analysis.analyse_single_target(
            settings=IDTXL_SETTINGS,
            data=data,
            target=1,
            sources=[0],
        )

        out["eye_to_head_result"] = eye_to_head_result
        diag = extract_idtxl_diagnostics(eye_to_head_result, target=1)

        out["eye_to_head_te"] = diag["te_value"]
        out["eye_to_head_selected_source_count"] = diag["selected_source_count"]
        out["eye_to_head_selected_sources"] = diag["selected_sources"]
        out["eye_to_head_omnibus_te"] = diag["omnibus_te"]
        out["eye_to_head_omnibus_pval"] = diag["omnibus_pval"]
        out["eye_to_head_result_keys"] = diag["result_keys"]

    except Exception as e:
        out["error"] = f"eye_to_head_failed: {repr(e)}"
        return out

    try:
        analysis = BivariateTE()
        head_to_eye_result = analysis.analyse_single_target(
            settings=IDTXL_SETTINGS,
            data=data,
            target=0,
            sources=[1],
        )

        out["head_to_eye_result"] = head_to_eye_result
        diag = extract_idtxl_diagnostics(head_to_eye_result, target=0)

        out["head_to_eye_te"] = diag["te_value"]
        out["head_to_eye_selected_source_count"] = diag["selected_source_count"]
        out["head_to_eye_selected_sources"] = diag["selected_sources"]
        out["head_to_eye_omnibus_te"] = diag["omnibus_te"]
        out["head_to_eye_omnibus_pval"] = diag["omnibus_pval"]
        out["head_to_eye_result_keys"] = diag["result_keys"]

    except Exception as e:
        prev = out["error"]
        msg = f"head_to_eye_failed: {repr(e)}"
        out["error"] = msg if prev is None else f"{prev}; {msg}"

    return out


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for metric in ["eye_to_head_te", "head_to_eye_te"]:
        subset = df[["tracking", "layout", "accuracy", metric]].dropna()

        if len(subset) < 3:
            continue

        pearson_r, pearson_p = pearsonr(subset[metric], subset["accuracy"])
        spearman_rho, spearman_p = spearmanr(subset[metric], subset["accuracy"])

        rows.append(
            {
                "metric": metric,
                "scope": "bias_modes",
                "n_blocks": len(subset),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
            }
        )

        for tracking, tracking_df in subset.groupby("tracking"):
            if len(tracking_df) < 3:
                continue

            pearson_r, pearson_p = pearsonr(
                tracking_df[metric],
                tracking_df["accuracy"],
            )
            spearman_rho, spearman_p = spearmanr(
                tracking_df[metric],
                tracking_df["accuracy"],
            )

            rows.append(
                {
                    "metric": metric,
                    "scope": tracking,
                    "n_blocks": len(tracking_df),
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_rho": spearman_rho,
                    "spearman_p": spearman_p,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    log("Starting IDTxl TE replication script")
    log(f"Dataset root: {DATASET_ROOT}")
    log(f"Raw frames dir: {RAW_FRAMES_DIR}")
    log(f"Block summary: {BLOCK_SUMMARY_PATH}")
    log(f"Output dir: {OUTPUT_DIR}")

    if not BLOCK_SUMMARY_PATH.exists():
        raise FileNotFoundError(BLOCK_SUMMARY_PATH)

    block_summary = pd.read_csv(BLOCK_SUMMARY_PATH)

    blocks = block_summary[
        block_summary["tracking"].isin(TRACKING_MODES)
    ].copy()

    blocks = blocks.sort_values(
        ["participant_id", "layout", "tracking"]
    ).reset_index(drop=True)

    if MAX_BLOCKS is not None:
        blocks = blocks.head(MAX_BLOCKS).copy()
        log(f"Running smoke test with MAX_BLOCKS={MAX_BLOCKS}")
    else:
        log("Running full block set")

    log(f"Blocks to process: {len(blocks)}")

    result_rows: list[dict[str, Any]] = []
    raw_result_objects: dict[str, Any] = {}

    for i, row in blocks.iterrows():
        participant_id = row["participant_id"]
        layout = row["layout"]
        tracking = row["tracking"]
        block_index = int(row["condition_block_key"].split("#block=")[1].split("#")[0])
        source_session_id = parse_source_session_id(row["source_sessions"])
        frame_path = frame_path_from_session_id(source_session_id)

        label = f"{participant_id} | {layout} | {tracking} | block {block_index}"
        log(f"\n[{i + 1}/{len(blocks)}] {label}")

        base_row = {
            "participant_id": participant_id,
            "layout": layout,
            "tracking": tracking,
            "condition_block_key": row["condition_block_key"],
            "source_session_id": source_session_id,
            "block_index": block_index,
            "accuracy": row["accuracy"],
            "correct_count": row["correct_count"],
            "n_trials_expected": row["n_trials"],
            "frame_path": str(frame_path),
            "signal_component": SIGNAL_COMPONENT,
        }

        if not frame_path.exists():
            log(f"  SKIP: frame file missing: {frame_path}")
            base_row.update(
                {
                    "status": "skipped",
                    "skip_reason": "frame_file_missing",
                    "eye_to_head_te": np.nan,
                    "head_to_eye_te": np.nan,
                    "idtxl_error": None,
                }
            )
            result_rows.append(base_row)
            continue

        frames = pd.read_csv(frame_path)

        data_array, info = prepare_block_array(
            frames=frames,
            block_index=block_index,
            tracking=tracking,
            component=SIGNAL_COMPONENT,
        )

        base_row.update(info)

        if data_array is None:
            log(f"  SKIP: {info.get('skip_reason')}")
            base_row.update(
                {
                    "status": "skipped",
                    "eye_to_head_te": np.nan,
                    "head_to_eye_te": np.nan,
                    "idtxl_error": None,
                }
            )
            result_rows.append(base_row)
            continue

        log(
            "  IDTxl data shape "
            f"processes x samples x replications = {data_array.shape}"
        )

        te_out = run_bivariate_te(data_array)

        base_row.update(
            {
                "status": "ok" if te_out["error"] is None else "idtxl_partial_or_failed",
                "eye_to_head_te": te_out["eye_to_head_te"],
                "head_to_eye_te": te_out["head_to_eye_te"],
                "eye_to_head_selected_source_count": te_out["eye_to_head_selected_source_count"],
                "head_to_eye_selected_source_count": te_out["head_to_eye_selected_source_count"],
                "eye_to_head_selected_sources": te_out["eye_to_head_selected_sources"],
                "head_to_eye_selected_sources": te_out["head_to_eye_selected_sources"],
                "eye_to_head_omnibus_te": te_out["eye_to_head_omnibus_te"],
                "head_to_eye_omnibus_te": te_out["head_to_eye_omnibus_te"],
                "eye_to_head_omnibus_pval": te_out["eye_to_head_omnibus_pval"],
                "head_to_eye_omnibus_pval": te_out["head_to_eye_omnibus_pval"],
                "eye_to_head_result_keys": te_out["eye_to_head_result_keys"],
                "head_to_eye_result_keys": te_out["head_to_eye_result_keys"],
                "idtxl_error": te_out["error"],
            }
        )

        result_rows.append(base_row)

        raw_key = f"{participant_id}_{layout}_{tracking}_block{block_index}"
        raw_result_objects[raw_key] = {
            "metadata": base_row,
            "eye_to_head_result": te_out["eye_to_head_result"],
            "head_to_eye_result": te_out["head_to_eye_result"],
        }

        log(
            "  TE results: "
            f"eye_to_head={base_row['eye_to_head_te']}, "
            f"head_to_eye={base_row['head_to_eye_te']}"
        )

    results_df = pd.DataFrame(result_rows)

    results_path = OUTPUT_DIR / "idtxl_te_block_results.csv"
    results_df.to_csv(results_path, index=False)

    log(f"\nSaved block TE results to: {results_path}")

    pickle_path = OUTPUT_DIR / "idtxl_raw_result_objects.pkl"
    with pickle_path.open("wb") as f:
        pickle.dump(raw_result_objects, f)

    log(f"Saved raw IDTxl result objects to: {pickle_path}")

    settings_path = OUTPUT_DIR / "idtxl_te_settings.json"
    with settings_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "tracking_modes": TRACKING_MODES,
                "signal_component": SIGNAL_COMPONENT,
                "min_samples_per_trial": MIN_SAMPLES_PER_TRIAL,
                "max_blocks": MAX_BLOCKS,
                "idtxl_settings": IDTXL_SETTINGS,
            },
            f,
            indent=2,
        )

    log(f"Saved settings to: {settings_path}")

    ok_df = results_df[results_df["status"].isin(["ok", "idtxl_partial_or_failed"])].copy()
    corr_df = compute_correlations(ok_df)

    corr_path = OUTPUT_DIR / "idtxl_te_accuracy_correlations.csv"
    corr_df.to_csv(corr_path, index=False)

    log(f"Saved TE-accuracy correlations to: {corr_path}")

    log("\nCorrelation preview:")
    if corr_df.empty:
        log("No valid correlations yet.")
    else:
        log(corr_df.to_string(index=False))

    log("\nDone.")


if __name__ == "__main__":
    main()