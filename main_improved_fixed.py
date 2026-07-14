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

from idtxl.bivariate_te import BivariateTE
from idtxl.data import Data


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

# The eye signal is reconstructed as eye-in-head angle by rotating the raw
# world-space gaze direction into the head-local coordinate frame.
# The head signal is head-in-world angle from the headset quaternion.
SIGNAL_AXIS = "az"                 # "az" or "el"
SIGNAL_REPRESENTATION = "velocity" # "position" or "velocity"

# IDTxl assumes equally spaced samples. Each trial is therefore resampled to
# one common time grid before it becomes a replication.
RESAMPLE_HZ = 60.0
WINDOW_SECONDS = 0.75
WINDOW_ALIGNMENT = "trial_start"  # "trial_start" or "pre_trigger"
MAX_INTERPOLATION_GAP_SECONDS = 0.050
MIN_VALID_FRACTION = 0.90
MIN_TRIALS_PER_BLOCK = 6

# Search lags in physical time, then convert them to frame lags only after the
# sampling rate has been fixed. At 60 Hz these defaults correspond to source
# lags 1..18 and target lags 1..12.
MIN_SOURCE_LAG_MS = 1000.0 / RESAMPLE_HZ
MAX_SOURCE_LAG_MS = 300.0
MAX_TARGET_LAG_MS = 200.0
N_PERMUTATIONS = 200  # use 500-1000 for final inference

# Set to an integer such as 5 for a smoke test. Set to None for a full run.
MAX_BLOCKS: int | None = None


def ms_to_samples(milliseconds: float, sample_rate_hz: float) -> int:
    return max(1, int(round(milliseconds * sample_rate_hz / 1000.0)))


IDTXL_SETTINGS = {
    "cmi_estimator": "JidtKraskovCMI",
    "min_lag_sources": ms_to_samples(MIN_SOURCE_LAG_MS, RESAMPLE_HZ),
    "max_lag_sources": ms_to_samples(MAX_SOURCE_LAG_MS, RESAMPLE_HZ),
    "tau_sources": 1,
    "max_lag_target": ms_to_samples(MAX_TARGET_LAG_MS, RESAMPLE_HZ),
    "tau_target": 1,
    "n_perm_max_stat": N_PERMUTATIONS,
    "n_perm_min_stat": N_PERMUTATIONS,
    "n_perm_omnibus": N_PERMUTATIONS,
    "n_perm_max_seq": N_PERMUTATIONS,
    "alpha_max_stat": 0.05,
    "alpha_min_stat": 0.05,
    "alpha_omnibus": 0.05,
    "alpha_max_seq": 0.05,
    "verbose": False,
}


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------


def log(message: str) -> None:
    print(message, flush=True)


def frame_path_from_session_id(source_session_id: str) -> Path:
    return RAW_FRAMES_DIR / f"{source_session_id}_frames.csv"


def parse_source_session_id(source_sessions: str) -> str:
    if pd.isna(source_sessions):
        raise ValueError("Missing source_sessions value")
    return str(source_sessions).split(";")[0].strip()


def normalize_rows(values: np.ndarray) -> np.ndarray:
    """Normalize row vectors, returning NaN for zero/invalid vectors."""
    values = np.asarray(values, dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    valid = np.isfinite(norms[:, 0]) & (norms[:, 0] > 1e-12)

    out = np.full_like(values, np.nan, dtype=float)
    out[valid] = values[valid] / norms[valid]
    return out


def normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    """Normalize x,y,z,w quaternions row-wise."""
    return normalize_rows(quaternions)


def rotate_vectors_by_quaternion(
    vectors: np.ndarray,
    quaternions_xyzw: np.ndarray,
) -> np.ndarray:
    """
    Rotate vectors by normalized x,y,z,w quaternions.

    Uses v' = v + 2*w*(q_xyz x v) + 2*q_xyz x (q_xyz x v).
    """
    vectors = np.asarray(vectors, dtype=float)
    quaternions_xyzw = np.asarray(quaternions_xyzw, dtype=float)

    q_xyz = quaternions_xyzw[:, :3]
    q_w = quaternions_xyzw[:, 3:4]
    t = 2.0 * np.cross(q_xyz, vectors)
    return vectors + q_w * t + np.cross(q_xyz, t)


def inverse_rotate_vectors_by_quaternion(
    vectors: np.ndarray,
    quaternions_xyzw: np.ndarray,
) -> np.ndarray:
    """Rotate world-space vectors into the quaternion's local frame."""
    inverse_q = quaternions_xyzw.copy()
    inverse_q[:, :3] *= -1.0
    return rotate_vectors_by_quaternion(vectors, inverse_q)


def vectors_to_az_el_deg(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert +Z-forward vectors to azimuth and elevation in degrees."""
    x = vectors[:, 0]
    y = vectors[:, 1]
    z = vectors[:, 2]

    az = np.degrees(np.arctan2(x, z))
    horizontal = np.sqrt(x * x + z * z)
    el = np.degrees(np.arctan2(y, horizontal))
    return az, el


def add_head_and_eye_angles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the two signals required for head-eye coordination analysis.

    eye_in_head_*:
        raw gaze direction expressed in the head-local frame.
    head_world_*:
        headset forward direction expressed in the world frame.

    This deliberately does not use raw_panel_* because those columns describe
    gaze relative to the task panel rather than ocular rotation relative to the
    head.
    """
    required = [
        "raw_direction_x",
        "raw_direction_y",
        "raw_direction_z",
        "head_rotation_x",
        "head_rotation_y",
        "head_rotation_z",
        "head_rotation_w",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing direction/quaternion columns: {missing}")

    gaze_world = normalize_rows(
        df[["raw_direction_x", "raw_direction_y", "raw_direction_z"]]
        .to_numpy(dtype=float)
    )
    head_q = normalize_quaternions(
        df[
            [
                "head_rotation_x",
                "head_rotation_y",
                "head_rotation_z",
                "head_rotation_w",
            ]
        ].to_numpy(dtype=float)
    )

    local_forward = np.zeros((len(df), 3), dtype=float)
    local_forward[:, 2] = 1.0

    head_forward_world = rotate_vectors_by_quaternion(local_forward, head_q)
    eye_direction_head = inverse_rotate_vectors_by_quaternion(gaze_world, head_q)

    head_az, head_el = vectors_to_az_el_deg(head_forward_world)
    eye_az, eye_el = vectors_to_az_el_deg(eye_direction_head)

    out = df.copy()
    out["head_world_az_deg"] = head_az
    out["head_world_el_deg"] = head_el
    out["eye_in_head_az_deg"] = eye_az
    out["eye_in_head_el_deg"] = eye_el
    out["raw_direction_norm"] = np.linalg.norm(
        df[["raw_direction_x", "raw_direction_y", "raw_direction_z"]]
        .to_numpy(dtype=float),
        axis=1,
    )
    out["head_quaternion_norm"] = np.linalg.norm(
        df[
            [
                "head_rotation_x",
                "head_rotation_y",
                "head_rotation_z",
                "head_rotation_w",
            ]
        ].to_numpy(dtype=float),
        axis=1,
    )
    return out


def signal_columns(axis: str) -> tuple[str, str]:
    if axis == "az":
        return "eye_in_head_az_deg", "head_world_az_deg"
    if axis == "el":
        return "eye_in_head_el_deg", "head_world_el_deg"
    raise ValueError(f"Unsupported SIGNAL_AXIS: {axis}")


def unwrap_degrees(values: np.ndarray) -> np.ndarray:
    return np.degrees(np.unwrap(np.radians(values)))


def interpolate_angle_with_gap_limit(
    time_seconds: np.ndarray,
    angle_degrees: np.ndarray,
    grid_seconds: np.ndarray,
    max_gap_seconds: float,
) -> np.ndarray:
    """
    Interpolate an angular signal onto a regular grid without bridging long gaps.

    Angles are unwrapped before interpolation. A grid point is accepted only if
    it is exactly observed or is bracketed by two observations no farther apart
    than max_gap_seconds.
    """
    valid = np.isfinite(time_seconds) & np.isfinite(angle_degrees)
    t = np.asarray(time_seconds[valid], dtype=float)
    a = np.asarray(angle_degrees[valid], dtype=float)

    if t.size < 2:
        return np.full(grid_seconds.shape, np.nan, dtype=float)

    order = np.argsort(t)
    t = t[order]
    a = a[order]

    # Collapse duplicate timestamps by averaging their circularly unwrapped values.
    unique_t, inverse = np.unique(t, return_inverse=True)
    if unique_t.size != t.size:
        sums = np.zeros(unique_t.size, dtype=float)
        counts = np.zeros(unique_t.size, dtype=int)
        unwrapped = unwrap_degrees(a)
        np.add.at(sums, inverse, unwrapped)
        np.add.at(counts, inverse, 1)
        t = unique_t
        a = sums / counts
    else:
        a = unwrap_degrees(a)

    if t.size < 2:
        return np.full(grid_seconds.shape, np.nan, dtype=float)

    result = np.full(grid_seconds.shape, np.nan, dtype=float)
    right = np.searchsorted(t, grid_seconds, side="left")

    exact_mask = right < t.size
    exact_indices = np.flatnonzero(exact_mask)
    exact_matches = np.isclose(
        t[right[exact_mask]],
        grid_seconds[exact_mask],
        rtol=0.0,
        atol=1e-9,
    )
    if np.any(exact_matches):
        output_indices = exact_indices[exact_matches]
        source_indices = right[exact_mask][exact_matches]
        result[output_indices] = a[source_indices]

    bracketed = (right > 0) & (right < t.size) & ~np.isfinite(result)
    if np.any(bracketed):
        output_indices = np.flatnonzero(bracketed)
        right_indices = right[bracketed]
        left_indices = right_indices - 1

        gaps = t[right_indices] - t[left_indices]
        acceptable = gaps <= max_gap_seconds

        if np.any(acceptable):
            out_idx = output_indices[acceptable]
            li = left_indices[acceptable]
            ri = right_indices[acceptable]
            weight = (grid_seconds[out_idx] - t[li]) / (t[ri] - t[li])
            result[out_idx] = a[li] + weight * (a[ri] - a[li])

    return result


def make_trial_grid(trial: pd.DataFrame) -> np.ndarray:
    timestamps = trial["time_seconds"].to_numpy(dtype=float)
    trial_start = float(np.nanmin(timestamps))
    trial_end = float(np.nanmax(timestamps))

    n_intervals = int(round(WINDOW_SECONDS * RESAMPLE_HZ))
    relative_grid = np.arange(n_intervals + 1, dtype=float) / RESAMPLE_HZ

    if WINDOW_ALIGNMENT == "trial_start":
        window_start = trial_start
        window_end = window_start + WINDOW_SECONDS
        if window_end > trial_end + 1e-9:
            return np.array([], dtype=float)
        return window_start + relative_grid

    if WINDOW_ALIGNMENT == "pre_trigger":
        if "trigger_down" in trial.columns:
            trigger_times = trial.loc[
                pd.to_numeric(trial["trigger_down"], errors="coerce").fillna(0) > 0,
                "time_seconds",
            ]
            anchor = float(trigger_times.max()) if not trigger_times.empty else trial_end
        else:
            anchor = trial_end

        window_start = anchor - WINDOW_SECONDS
        if window_start < trial_start - 1e-9:
            return np.array([], dtype=float)
        return window_start + relative_grid

    raise ValueError(f"Unsupported WINDOW_ALIGNMENT: {WINDOW_ALIGNMENT}")


def encode_trial(
    trial_df: pd.DataFrame,
    eye_col: str,
    head_col: str,
) -> tuple[np.ndarray | None, dict[str, float | str | None]]:
    trial = trial_df.copy()
    trial["timestamp_parsed"] = pd.to_datetime(
        trial["timestamp_utc"], errors="coerce", utc=True
    )
    trial = trial.dropna(subset=["timestamp_parsed"]).sort_values("timestamp_parsed")

    diagnostics: dict[str, float | str | None] = {
        "valid_fraction": 0.0,
        "median_dt_seconds": np.nan,
        "eye_abs_p99_deg": np.nan,
        "raw_direction_norm_median": np.nan,
        "head_quaternion_norm_median": np.nan,
        "skip_reason": None,
    }

    if len(trial) < 3:
        diagnostics["skip_reason"] = "too_few_timestamped_rows"
        return None, diagnostics

    trial["time_seconds"] = (
        trial["timestamp_parsed"] - trial["timestamp_parsed"].iloc[0]
    ).dt.total_seconds()

    dt = np.diff(trial["time_seconds"].to_numpy(dtype=float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size:
        diagnostics["median_dt_seconds"] = float(np.median(dt))

    valid = np.isfinite(trial[eye_col]) & np.isfinite(trial[head_col])
    if "validity_left" in trial.columns and "validity_right" in trial.columns:
        left_valid = pd.to_numeric(trial["validity_left"], errors="coerce").fillna(0) > 0
        right_valid = pd.to_numeric(trial["validity_right"], errors="coerce").fillna(0) > 0
        valid &= left_valid | right_valid

    diagnostics["valid_fraction"] = float(valid.mean())
    diagnostics["raw_direction_norm_median"] = float(
        np.nanmedian(trial["raw_direction_norm"])
    )
    diagnostics["head_quaternion_norm_median"] = float(
        np.nanmedian(trial["head_quaternion_norm"])
    )

    eye_valid_values = trial.loc[valid, eye_col].to_numpy(dtype=float)
    if eye_valid_values.size:
        diagnostics["eye_abs_p99_deg"] = float(
            np.nanpercentile(np.abs(eye_valid_values), 99)
        )

    if diagnostics["valid_fraction"] < MIN_VALID_FRACTION:
        diagnostics["skip_reason"] = "valid_fraction_below_threshold"
        return None, diagnostics

    grid = make_trial_grid(trial)
    if grid.size == 0:
        diagnostics["skip_reason"] = "trial_shorter_than_fixed_window"
        return None, diagnostics

    t = trial["time_seconds"].to_numpy(dtype=float, copy=True)
    eye = trial[eye_col].to_numpy(dtype=float, copy=True)
    head = trial[head_col].to_numpy(dtype=float, copy=True)
    valid_mask = valid.to_numpy(dtype=bool, copy=True)

    # Mark invalid samples without deleting their timestamps. Explicit copies
    # are required because recent pandas/NumPy combinations may return a
    # read-only view from Series.to_numpy().
    eye[~valid_mask] = np.nan
    head[~valid_mask] = np.nan

    eye_regular = interpolate_angle_with_gap_limit(
        t, eye, grid, MAX_INTERPOLATION_GAP_SECONDS
    )
    head_regular = interpolate_angle_with_gap_limit(
        t, head, grid, MAX_INTERPOLATION_GAP_SECONDS
    )

    if not np.all(np.isfinite(eye_regular)) or not np.all(np.isfinite(head_regular)):
        diagnostics["skip_reason"] = "unfilled_gap_in_analysis_window"
        return None, diagnostics

    if SIGNAL_REPRESENTATION == "velocity":
        # Because the grid is uniform, this is a true angular velocity in deg/s,
        # not an unscaled frame-to-frame difference.
        eye_signal = np.diff(eye_regular) * RESAMPLE_HZ
        head_signal = np.diff(head_regular) * RESAMPLE_HZ
    elif SIGNAL_REPRESENTATION == "position":
        # Remove arbitrary between-trial orientation offsets while retaining the
        # within-window trajectory.
        eye_signal = eye_regular - eye_regular[0]
        head_signal = head_regular - head_regular[0]
    else:
        raise ValueError(
            f"Unsupported SIGNAL_REPRESENTATION: {SIGNAL_REPRESENTATION}"
        )

    if np.std(eye_signal) < 1e-10 or np.std(head_signal) < 1e-10:
        diagnostics["skip_reason"] = "constant_signal"
        return None, diagnostics

    return np.vstack([eye_signal, head_signal]), diagnostics


def prepare_block_array(
    frames: pd.DataFrame,
    block_index: int,
    tracking: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """
    Build a processes x samples x replications array.

    process 0 = eye-in-head angle/velocity
    process 1 = head-in-world angle/velocity
    replication = one fixed-duration, uniformly sampled trial window
    """
    block = frames[
        (frames["block_index"] == block_index)
        & (frames["tracking"] == tracking)
        & (frames["state"] == "TrialActive")
    ].copy()

    info: dict[str, Any] = {
        "n_active_rows": int(len(block)),
        "n_trials_found": 0,
        "n_trials_used": 0,
        "n_samples_used": None,
        "effective_realisations": None,
        "median_original_dt_seconds": np.nan,
        "median_valid_fraction": np.nan,
        "median_eye_abs_p99_deg": np.nan,
        "median_raw_direction_norm": np.nan,
        "median_head_quaternion_norm": np.nan,
        "trial_skip_reasons": None,
        "skip_reason": None,
    }

    if block.empty:
        info["skip_reason"] = "no_trialactive_rows"
        return None, info

    try:
        block = add_head_and_eye_angles(block)
    except ValueError as exc:
        info["skip_reason"] = str(exc)
        return None, info

    eye_col, head_col = signal_columns(SIGNAL_AXIS)
    required = ["trial_index_in_block", "timestamp_utc", eye_col, head_col]
    missing = [column for column in required if column not in block.columns]
    if missing:
        info["skip_reason"] = f"missing_columns: {missing}"
        return None, info

    trial_series: list[np.ndarray] = []
    trial_diagnostics: list[dict[str, float | str | None]] = []

    info["n_trials_found"] = int(block["trial_index_in_block"].nunique())

    for _, trial_df in block.groupby("trial_index_in_block", sort=True):
        encoded, diagnostics = encode_trial(trial_df, eye_col, head_col)
        trial_diagnostics.append(diagnostics)
        if encoded is not None:
            trial_series.append(encoded)

    info["n_trials_used"] = len(trial_series)

    numeric_diagnostics = {
        "median_original_dt_seconds": "median_dt_seconds",
        "median_valid_fraction": "valid_fraction",
        "median_eye_abs_p99_deg": "eye_abs_p99_deg",
        "median_raw_direction_norm": "raw_direction_norm_median",
        "median_head_quaternion_norm": "head_quaternion_norm_median",
    }
    for output_name, diagnostic_name in numeric_diagnostics.items():
        values = np.asarray(
            [d[diagnostic_name] for d in trial_diagnostics], dtype=float
        )
        if np.any(np.isfinite(values)):
            info[output_name] = float(np.nanmedian(values))

    skip_counts: dict[str, int] = {}
    for diagnostic in trial_diagnostics:
        reason = diagnostic.get("skip_reason")
        if reason:
            skip_counts[str(reason)] = skip_counts.get(str(reason), 0) + 1
    info["trial_skip_reasons"] = json.dumps(skip_counts, sort_keys=True)

    if len(trial_series) < MIN_TRIALS_PER_BLOCK:
        info["skip_reason"] = "fewer_than_minimum_usable_trials"
        return None, info

    expected_shape = trial_series[0].shape
    if any(array.shape != expected_shape for array in trial_series):
        info["skip_reason"] = "inconsistent_encoded_trial_shapes"
        return None, info

    data_array = np.stack(trial_series, axis=2)
    info["n_samples_used"] = int(data_array.shape[1])
    info["effective_realisations"] = int(data_array.shape[1] * data_array.shape[2])

    return data_array, info


# ---------------------------------------------------------------------
# IDTxl analysis and diagnostics
# ---------------------------------------------------------------------


def first_present(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def extract_idtxl_diagnostics(results: Any, target: int) -> dict[str, Any]:
    """
    Extract the joint/omnibus TE and its significance.

    The omnibus statistic is the joint TE from all selected source-lag variables.
    It should not be replaced by a sum of lag-wise conditional contributions.
    """
    out = {
        "te_value": np.nan,
        "te_significant_only": np.nan,
        "selected_lag_count": 0,
        "selected_sources": None,
        "omnibus_pval": np.nan,
        "omnibus_significant": False,
        "individual_source_te": None,
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

    out["result_keys"] = ";".join(str(key) for key in single.keys())

    selected = first_present(single, ["selected_vars_sources", "selected_sources"])
    if selected is None:
        selected = []
    out["selected_sources"] = json.dumps(selected, default=str)
    try:
        out["selected_lag_count"] = int(len(selected))
    except TypeError:
        out["selected_lag_count"] = 0

    statistic = first_present(
        single,
        ["statistic_omnibus", "omnibus_te", "te_omnibus"],
    )
    if statistic is not None:
        try:
            out["te_value"] = float(statistic)
        except (TypeError, ValueError):
            pass
    elif out["selected_lag_count"] == 0:
        out["te_value"] = 0.0

    pvalue = first_present(
        single,
        ["pvalue_omnibus", "omnibus_pval", "pval_omnibus"],
    )
    if pvalue is not None:
        try:
            out["omnibus_pval"] = float(pvalue)
        except (TypeError, ValueError):
            pass

    significance = first_present(
        single,
        ["sign_omnibus", "sign_ominbus", "omnibus_significant"],
    )
    if significance is None and np.isfinite(out["omnibus_pval"]):
        significance = out["omnibus_pval"] < IDTXL_SETTINGS["alpha_omnibus"]
    out["omnibus_significant"] = bool(significance) if significance is not None else False

    individual = first_present(
        single,
        ["statistic_sign_sources", "selected_sources_te"],
    )
    if individual is not None:
        try:
            out["individual_source_te"] = json.dumps(
                np.asarray(individual, dtype=float).tolist()
            )
        except Exception:
            out["individual_source_te"] = str(individual)

    if np.isfinite(out["te_value"]):
        out["te_significant_only"] = (
            float(out["te_value"]) if out["omnibus_significant"] else 0.0
        )

    return out


def run_bivariate_te(data_array: np.ndarray) -> dict[str, Any]:
    data = Data(data_array, dim_order="psr", normalise=True)

    out: dict[str, Any] = {
        "eye_to_head_te": np.nan,
        "head_to_eye_te": np.nan,
        "eye_to_head_te_significant_only": np.nan,
        "head_to_eye_te_significant_only": np.nan,
        "eye_to_head_selected_lag_count": 0,
        "head_to_eye_selected_lag_count": 0,
        "eye_to_head_selected_sources": None,
        "head_to_eye_selected_sources": None,
        "eye_to_head_omnibus_pval": np.nan,
        "head_to_eye_omnibus_pval": np.nan,
        "eye_to_head_omnibus_significant": False,
        "head_to_eye_omnibus_significant": False,
        "eye_to_head_individual_source_te": None,
        "head_to_eye_individual_source_te": None,
        "eye_to_head_result_keys": None,
        "head_to_eye_result_keys": None,
        "eye_to_head_result": None,
        "head_to_eye_result": None,
        "error": None,
    }

    directions = [
        ("eye_to_head", 1, [0]),
        ("head_to_eye", 0, [1]),
    ]

    for prefix, target, sources in directions:
        try:
            result = BivariateTE().analyse_single_target(
                settings=IDTXL_SETTINGS,
                data=data,
                target=target,
                sources=sources,
            )
            out[f"{prefix}_result"] = result
            diagnostics = extract_idtxl_diagnostics(result, target=target)

            out[f"{prefix}_te"] = diagnostics["te_value"]
            out[f"{prefix}_te_significant_only"] = diagnostics[
                "te_significant_only"
            ]
            out[f"{prefix}_selected_lag_count"] = diagnostics[
                "selected_lag_count"
            ]
            out[f"{prefix}_selected_sources"] = diagnostics["selected_sources"]
            out[f"{prefix}_omnibus_pval"] = diagnostics["omnibus_pval"]
            out[f"{prefix}_omnibus_significant"] = diagnostics[
                "omnibus_significant"
            ]
            out[f"{prefix}_individual_source_te"] = diagnostics[
                "individual_source_te"
            ]
            out[f"{prefix}_result_keys"] = diagnostics["result_keys"]

        except Exception as exc:
            message = f"{prefix}_failed: {exc!r}"
            out["error"] = (
                message if out["error"] is None else f"{out['error']}; {message}"
            )

    eye_to_head = out["eye_to_head_te"]
    head_to_eye = out["head_to_eye_te"]
    if np.isfinite(eye_to_head) and np.isfinite(head_to_eye):
        total = eye_to_head + head_to_eye
        out["te_difference_head_minus_eye"] = head_to_eye - eye_to_head
        out["te_normalized_asymmetry"] = (
            (head_to_eye - eye_to_head) / total if abs(total) > 1e-12 else 0.0
        )
    else:
        out["te_difference_head_minus_eye"] = np.nan
        out["te_normalized_asymmetry"] = np.nan

    return out


def safe_correlation(x: pd.Series, y: pd.Series) -> dict[str, float] | None:
    valid = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
    x_valid = x.to_numpy(dtype=float)[valid]
    y_valid = y.to_numpy(dtype=float)[valid]

    if len(x_valid) < 3 or np.unique(x_valid).size < 2 or np.unique(y_valid).size < 2:
        return None

    pearson_r, pearson_p = pearsonr(x_valid, y_valid)
    spearman_rho, spearman_p = spearmanr(x_valid, y_valid)
    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
    }


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Exploratory correlations only; repeated-measures modelling should follow."""
    rows: list[dict[str, Any]] = []
    metrics = [
        "eye_to_head_te",
        "head_to_eye_te",
        "te_difference_head_minus_eye",
        "te_normalized_asymmetry",
    ]

    for metric in metrics:
        columns = ["tracking", "accuracy", metric]
        subset = df[columns].dropna().copy()
        if len(subset) < 3:
            continue

        overall = safe_correlation(subset[metric], subset["accuracy"])
        if overall is not None:
            rows.append(
                {
                    "metric": metric,
                    "scope": "bias_modes",
                    "n_blocks": len(subset),
                    **overall,
                }
            )

        for tracking, tracking_df in subset.groupby("tracking"):
            result = safe_correlation(tracking_df[metric], tracking_df["accuracy"])
            if result is not None:
                rows.append(
                    {
                        "metric": metric,
                        "scope": tracking,
                        "n_blocks": len(tracking_df),
                        **result,
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    log("Starting corrected IDTxl head-eye TE analysis")
    log(f"Dataset root: {DATASET_ROOT}")
    log(f"Signal: eye-in-head {SIGNAL_AXIS} {SIGNAL_REPRESENTATION}")
    log(f"Head: head-in-world {SIGNAL_AXIS} {SIGNAL_REPRESENTATION}")
    log(
        f"Window: {WINDOW_SECONDS:.3f}s aligned to {WINDOW_ALIGNMENT}; "
        f"resampled at {RESAMPLE_HZ:.1f} Hz"
    )
    log(
        "Source lag search: "
        f"{IDTXL_SETTINGS['min_lag_sources']}.."
        f"{IDTXL_SETTINGS['max_lag_sources']} samples "
        f"({MIN_SOURCE_LAG_MS:.1f}..{MAX_SOURCE_LAG_MS:.1f} ms)"
    )

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
        block_index = int(
            row["condition_block_key"].split("#block=")[1].split("#")[0]
        )
        source_session_id = parse_source_session_id(row["source_sessions"])
        frame_path = frame_path_from_session_id(source_session_id)

        label = f"{participant_id} | {layout} | {tracking} | block {block_index}"
        log(f"\n[{i + 1}/{len(blocks)}] {label}")

        base_row: dict[str, Any] = {
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
            "eye_signal": f"eye_in_head_{SIGNAL_AXIS}_deg",
            "head_signal": f"head_world_{SIGNAL_AXIS}_deg",
            "signal_representation": SIGNAL_REPRESENTATION,
            "resample_hz": RESAMPLE_HZ,
            "window_seconds": WINDOW_SECONDS,
            "window_alignment": WINDOW_ALIGNMENT,
        }

        if not frame_path.exists():
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
            log(f"  SKIP: frame file missing: {frame_path}")
            continue

        frames = pd.read_csv(frame_path)
        data_array, preprocessing_info = prepare_block_array(
            frames=frames,
            block_index=block_index,
            tracking=tracking,
        )
        base_row.update(preprocessing_info)

        if data_array is None:
            base_row.update(
                {
                    "status": "skipped",
                    "eye_to_head_te": np.nan,
                    "head_to_eye_te": np.nan,
                    "idtxl_error": None,
                }
            )
            result_rows.append(base_row)
            log(f"  SKIP: {preprocessing_info.get('skip_reason')}")
            continue

        log(
            "  IDTxl shape processes x samples x replications = "
            f"{data_array.shape}"
        )

        te_output = run_bivariate_te(data_array)
        base_row.update(
            {
                "status": "ok" if te_output["error"] is None else "idtxl_partial_or_failed",
                "idtxl_error": te_output["error"],
                **{
                    key: value
                    for key, value in te_output.items()
                    if not key.endswith("_result") and key != "error"
                },
            }
        )
        result_rows.append(base_row)

        raw_key = f"{participant_id}_{layout}_{tracking}_block{block_index}"
        raw_result_objects[raw_key] = {
            "metadata": base_row,
            "encoded_data_array": data_array,
            "eye_to_head_result": te_output["eye_to_head_result"],
            "head_to_eye_result": te_output["head_to_eye_result"],
        }

        log(
            "  Omnibus TE: "
            f"eye_to_head={base_row['eye_to_head_te']}, "
            f"head_to_eye={base_row['head_to_eye_te']}"
        )

    results_df = pd.DataFrame(result_rows)

    suffix = (
        f"eyeinhead_headworld_{SIGNAL_AXIS}_{SIGNAL_REPRESENTATION}_"
        f"{WINDOW_ALIGNMENT}_{int(round(WINDOW_SECONDS * 1000))}ms_"
        f"{int(round(RESAMPLE_HZ))}hz"
    )

    results_path = OUTPUT_DIR / f"idtxl_te_block_results_{suffix}.csv"
    results_df.to_csv(results_path, index=False)
    log(f"\nSaved block results to: {results_path}")

    pickle_path = OUTPUT_DIR / f"idtxl_raw_result_objects_{suffix}.pkl"
    with pickle_path.open("wb") as file:
        pickle.dump(raw_result_objects, file)
    log(f"Saved IDTxl objects and encoded arrays to: {pickle_path}")

    settings_path = OUTPUT_DIR / f"idtxl_te_settings_{suffix}.json"
    with settings_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "tracking_modes": TRACKING_MODES,
                "eye_signal": f"eye_in_head_{SIGNAL_AXIS}_deg",
                "head_signal": f"head_world_{SIGNAL_AXIS}_deg",
                "signal_representation": SIGNAL_REPRESENTATION,
                "resample_hz": RESAMPLE_HZ,
                "window_seconds": WINDOW_SECONDS,
                "window_alignment": WINDOW_ALIGNMENT,
                "max_interpolation_gap_seconds": MAX_INTERPOLATION_GAP_SECONDS,
                "min_valid_fraction": MIN_VALID_FRACTION,
                "min_trials_per_block": MIN_TRIALS_PER_BLOCK,
                "max_blocks": MAX_BLOCKS,
                "idtxl_settings": IDTXL_SETTINGS,
            },
            file,
            indent=2,
        )
    log(f"Saved settings to: {settings_path}")

    ok_df = results_df[
        results_df["status"].isin(["ok", "idtxl_partial_or_failed"])
    ].copy()
    correlation_df = compute_correlations(ok_df)
    correlation_path = OUTPUT_DIR / f"idtxl_te_accuracy_correlations_{suffix}.csv"
    correlation_df.to_csv(correlation_path, index=False)
    log(f"Saved exploratory correlations to: {correlation_path}")

    log("\nCorrelation preview (not repeated-measures inference):")
    if correlation_df.empty:
        log("No valid correlations yet.")
    else:
        log(correlation_df.to_string(index=False))

    log("\nDone.")


if __name__ == "__main__":
    main()
