from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


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

N_BINS_PER_DIM = 3
N_STATES = N_BINS_PER_DIM * N_BINS_PER_DIM

BINNING_METHOD = "quantile"

SOURCE_LAG = 1
TARGET_LAG = 1

MIN_SAMPLES_PER_TRIAL = 30

N_SHUFFLES = 100
RANDOM_SEED = 12345

MAX_BLOCKS = None

# Important diagnostic setting:
# If True, global bins are built only from the exact retained balanced blocks.
# This avoids contaminating bin edges with excluded/repeated/incomplete blocks.
STRICT_BALANCED_BINNING = True


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


def parse_block_index(condition_block_key: str) -> int:
    if "#block=" not in condition_block_key:
        raise ValueError(f"Cannot parse block index from: {condition_block_key}")
    return int(condition_block_key.split("#block=")[1].split("#")[0])


def quaternion_to_forward_vector(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Unity-style quaternion to forward vector.

    Assumes quaternion components are ordered x, y, z, w.
    This gives the rotated local +Z direction.
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

    out = df.copy()
    out["head_az_deg"] = head_az
    out["head_el_deg"] = head_el
    return out


def make_bin_edges(values: np.ndarray, n_bins: int, method: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("Cannot make bin edges from empty values")

    if method == "quantile":
        qs = np.linspace(0, 100, n_bins + 1)
        edges = np.nanpercentile(values, qs)
    elif method == "equal_width":
        edges = np.linspace(np.nanmin(values), np.nanmax(values), n_bins + 1)
    else:
        raise ValueError(f"Unknown binning method: {method}")

    # Guard against duplicate edges from low-variance data.
    if np.unique(edges).size < edges.size:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        if vmin == vmax:
            eps = 1e-9
            edges = np.linspace(vmin - eps, vmax + eps, n_bins + 1)
        else:
            edges = np.linspace(vmin, vmax, n_bins + 1)

    eps = 1e-9
    edges[0] -= eps
    edges[-1] += eps

    return edges


def encode_2d_state(
    yaw: np.ndarray,
    pitch: np.ndarray,
    yaw_edges: np.ndarray,
    pitch_edges: np.ndarray,
    n_bins_per_dim: int,
) -> np.ndarray:
    yaw_bin = np.digitize(yaw, yaw_edges[1:-1], right=False)
    pitch_bin = np.digitize(pitch, pitch_edges[1:-1], right=False)

    yaw_bin = np.clip(yaw_bin, 0, n_bins_per_dim - 1)
    pitch_bin = np.clip(pitch_bin, 0, n_bins_per_dim - 1)

    state = yaw_bin * n_bins_per_dim + pitch_bin
    return state.astype(int)


def validate_accuracy_column(blocks: pd.DataFrame) -> pd.DataFrame:
    out = blocks.copy()
    out["computed_accuracy"] = out["correct_count"] / out["n_trials"]
    out["accuracy_abs_diff"] = (out["accuracy"] - out["computed_accuracy"]).abs()
    return out


# ---------------------------------------------------------------------
# Data-loading diagnostics
# ---------------------------------------------------------------------

def diagnose_frame_match(
    frames: pd.DataFrame,
    block_index: int,
    layout: str,
    tracking: str,
    expected_n_trials: int,
) -> dict[str, Any]:
    """
    Verify that the block-level row actually matches frame-level rows.
    """
    info: dict[str, Any] = {}

    active = frames[frames["state"] == "TrialActive"].copy()

    available_blocks = (
        active.groupby(["block_index", "layout", "tracking"])["trial_index_in_block"]
        .nunique()
        .reset_index(name="n_trials_found")
        .sort_values(["block_index", "layout", "tracking"])
    )

    expected_match = available_blocks[
        (available_blocks["block_index"] == block_index)
        & (available_blocks["layout"] == layout)
        & (available_blocks["tracking"] == tracking)
    ]

    loaded_block = active[
        (active["block_index"] == block_index)
        & (active["layout"] == layout)
        & (active["tracking"] == tracking)
    ]

    trial_values = sorted(
        loaded_block["trial_index_in_block"].dropna().unique().tolist()
    )

    info["diagnostic_expected_match_found"] = not expected_match.empty
    info["diagnostic_loaded_trialactive_rows"] = int(len(loaded_block))
    info["diagnostic_loaded_n_trials"] = int(
        loaded_block["trial_index_in_block"].nunique()
    )
    info["diagnostic_expected_n_trials"] = int(expected_n_trials)
    info["diagnostic_trial_count_matches"] = (
        info["diagnostic_loaded_n_trials"] == int(expected_n_trials)
    )
    info["diagnostic_loaded_trials"] = ";".join(str(x) for x in trial_values)

    # This is only for console inspection.
    if expected_match.empty:
        log("  WARNING: expected block/layout/tracking not found in frame file")
        log(f"  Expected block_index={block_index}, layout={layout}, tracking={tracking}")
        log("  Available TrialActive blocks in this frame file:")
        log(available_blocks.to_string(index=False))
    else:
        log("  Matched frame block:")
        log(expected_match.to_string(index=False))

    log(f"  Loaded TrialActive rows: {info['diagnostic_loaded_trialactive_rows']}")
    log(f"  Loaded trials: {trial_values}")

    if not info["diagnostic_trial_count_matches"]:
        log(
            "  WARNING: trial count mismatch: "
            f"expected {expected_n_trials}, "
            f"got {info['diagnostic_loaded_n_trials']}"
        )

    return info


# ---------------------------------------------------------------------
# Bin edge construction
# ---------------------------------------------------------------------

def collect_values_from_exact_balanced_blocks(
    blocks: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """
    Strict version:
    collect values only from exact participant/layout/tracking/block rows
    in balanced_subject_block_summary.csv.
    """
    eye_az_values = []
    eye_el_values = []
    head_az_values = []
    head_el_values = []

    log("Collecting values for global bins from exact balanced block rows")

    # Cache frame files so we do not repeatedly read the same CSV.
    frame_cache: dict[str, pd.DataFrame] = {}

    for i, row in blocks.iterrows():
        source_session_id = parse_source_session_id(row["source_sessions"])
        frame_path = frame_path_from_session_id(source_session_id)

        block_index = parse_block_index(row["condition_block_key"])
        layout = row["layout"]
        tracking = row["tracking"]

        if not frame_path.exists():
            log(f"  Missing frame file during bin collection, skipping: {frame_path}")
            continue

        if source_session_id not in frame_cache:
            if len(frame_cache) % 10 == 0:
                log(f"  Reading frame file for bins: {frame_path.name}")
            frame_cache[source_session_id] = pd.read_csv(frame_path)

        frames = frame_cache[source_session_id]

        active = frames[
            (frames["state"] == "TrialActive")
            & (frames["block_index"] == block_index)
            & (frames["layout"] == layout)
            & (frames["tracking"] == tracking)
        ].copy()

        if active.empty:
            continue

        active = add_head_az_el(active)

        needed = [
            "raw_panel_az_deg",
            "raw_panel_el_deg",
            "head_az_deg",
            "head_el_deg",
        ]

        active = active[needed].dropna()

        if active.empty:
            continue

        eye_az_values.append(active["raw_panel_az_deg"].to_numpy(dtype=float))
        eye_el_values.append(active["raw_panel_el_deg"].to_numpy(dtype=float))
        head_az_values.append(active["head_az_deg"].to_numpy(dtype=float))
        head_el_values.append(active["head_el_deg"].to_numpy(dtype=float))

    if not eye_az_values:
        raise RuntimeError("No values collected for global bins")

    return {
        "eye_az": np.concatenate(eye_az_values),
        "eye_el": np.concatenate(eye_el_values),
        "head_az": np.concatenate(head_az_values),
        "head_el": np.concatenate(head_el_values),
    }


def collect_values_from_sessions_loose(
    blocks: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """
    Loose version:
    collect all Bias/BiasJitter TrialActive frames from sessions appearing
    in the balanced block table.

    This matches the first script more closely but is less strict.
    """
    eye_az_values = []
    eye_el_values = []
    head_az_values = []
    head_el_values = []

    unique_sessions = sorted(
        {parse_source_session_id(x) for x in blocks["source_sessions"].dropna()}
    )

    log(f"Collecting values for global bins from {len(unique_sessions)} frame files")

    for i, session_id in enumerate(unique_sessions, start=1):
        frame_path = frame_path_from_session_id(session_id)

        if not frame_path.exists():
            log(f"  Missing frame file, skipping: {frame_path}")
            continue

        if i % 10 == 0 or i == 1:
            log(f"  Reading frame file {i}/{len(unique_sessions)}: {frame_path.name}")

        frames = pd.read_csv(frame_path)

        active = frames[
            (frames["state"] == "TrialActive")
            & (frames["tracking"].isin(TRACKING_MODES))
        ].copy()

        if active.empty:
            continue

        active = add_head_az_el(active)

        needed = [
            "raw_panel_az_deg",
            "raw_panel_el_deg",
            "head_az_deg",
            "head_el_deg",
        ]

        active = active[needed].dropna()

        if active.empty:
            continue

        eye_az_values.append(active["raw_panel_az_deg"].to_numpy(dtype=float))
        eye_el_values.append(active["raw_panel_el_deg"].to_numpy(dtype=float))
        head_az_values.append(active["head_az_deg"].to_numpy(dtype=float))
        head_el_values.append(active["head_el_deg"].to_numpy(dtype=float))

    if not eye_az_values:
        raise RuntimeError("No values collected for global bins")

    return {
        "eye_az": np.concatenate(eye_az_values),
        "eye_el": np.concatenate(eye_el_values),
        "head_az": np.concatenate(head_az_values),
        "head_el": np.concatenate(head_el_values),
    }


def make_global_bin_config(blocks: pd.DataFrame) -> dict[str, Any]:
    if STRICT_BALANCED_BINNING:
        values = collect_values_from_exact_balanced_blocks(blocks)
    else:
        values = collect_values_from_sessions_loose(blocks)

    config = {
        "eye_az_edges": make_bin_edges(
            values["eye_az"], N_BINS_PER_DIM, BINNING_METHOD
        ),
        "eye_el_edges": make_bin_edges(
            values["eye_el"], N_BINS_PER_DIM, BINNING_METHOD
        ),
        "head_az_edges": make_bin_edges(
            values["head_az"], N_BINS_PER_DIM, BINNING_METHOD
        ),
        "head_el_edges": make_bin_edges(
            values["head_el"], N_BINS_PER_DIM, BINNING_METHOD
        ),
    }

    return config


# ---------------------------------------------------------------------
# Histogram transfer entropy
# ---------------------------------------------------------------------

def transfer_entropy_discrete(
    source_trials: list[np.ndarray],
    target_trials: list[np.ndarray],
    n_states: int,
    source_lag: int = 1,
    target_lag: int = 1,
) -> float:
    """
    Discrete transfer entropy:

        TE source -> target =
        sum p(y_t, y_past, x_past)
            log2(
                p(y_t | y_past, x_past)
                /
                p(y_t | y_past)
            )

    Counts are aggregated across trials.
    Trial boundaries are respected.
    """
    if len(source_trials) != len(target_trials):
        raise ValueError("source_trials and target_trials must have same length")

    max_lag = max(source_lag, target_lag)

    joint_counts: dict[tuple[int, int, int], int] = {}
    ypast_xpast_counts: dict[tuple[int, int], int] = {}
    yt_ypast_counts: dict[tuple[int, int], int] = {}
    ypast_counts: dict[int, int] = {}

    total = 0

    for source, target in zip(source_trials, target_trials):
        source = np.asarray(source, dtype=int)
        target = np.asarray(target, dtype=int)

        if len(source) != len(target):
            raise ValueError("Source and target trial arrays must have same length")

        if len(source) <= max_lag:
            continue

        for t in range(max_lag, len(source)):
            y_t = int(target[t])
            y_past = int(target[t - target_lag])
            x_past = int(source[t - source_lag])

            if not (0 <= y_t < n_states):
                continue
            if not (0 <= y_past < n_states):
                continue
            if not (0 <= x_past < n_states):
                continue

            joint_counts[(y_t, y_past, x_past)] = (
                joint_counts.get((y_t, y_past, x_past), 0) + 1
            )
            ypast_xpast_counts[(y_past, x_past)] = (
                ypast_xpast_counts.get((y_past, x_past), 0) + 1
            )
            yt_ypast_counts[(y_t, y_past)] = (
                yt_ypast_counts.get((y_t, y_past), 0) + 1
            )
            ypast_counts[y_past] = ypast_counts.get(y_past, 0) + 1

            total += 1

    if total == 0:
        return float("nan")

    te = 0.0

    for (y_t, y_past, x_past), c_joint in joint_counts.items():
        c_ypast_xpast = ypast_xpast_counts[(y_past, x_past)]
        c_yt_ypast = yt_ypast_counts[(y_t, y_past)]
        c_ypast = ypast_counts[y_past]

        p_joint = c_joint / total

        p_y_given_ypast_xpast = c_joint / c_ypast_xpast
        p_y_given_ypast = c_yt_ypast / c_ypast

        ratio = p_y_given_ypast_xpast / p_y_given_ypast

        if ratio > 0:
            te += p_joint * np.log2(ratio)

    return float(te)


def shuffled_te_baseline(
    source_trials: list[np.ndarray],
    target_trials: list[np.ndarray],
    n_states: int,
    rng: np.random.Generator,
    n_shuffles: int,
    source_lag: int,
    target_lag: int,
) -> np.ndarray:
    """
    Shuffle source states within each trial.

    This preserves source state frequencies within each trial, but breaks
    temporal alignment between source and target.
    """
    vals = []

    for _ in range(n_shuffles):
        shuffled_source_trials = []

        for source in source_trials:
            source = np.asarray(source, dtype=int)
            shuffled = source.copy()
            rng.shuffle(shuffled)
            shuffled_source_trials.append(shuffled)

        te = transfer_entropy_discrete(
            source_trials=shuffled_source_trials,
            target_trials=target_trials,
            n_states=n_states,
            source_lag=source_lag,
            target_lag=target_lag,
        )
        vals.append(te)

    return np.asarray(vals, dtype=float)


# ---------------------------------------------------------------------
# Block preparation
# ---------------------------------------------------------------------

def prepare_pairwise_state_trials(
    frames: pd.DataFrame,
    block_index: int,
    layout: str,
    tracking: str,
    bin_config: dict[str, Any],
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    """
    Return eye/head 9-state yaw/pitch trial arrays for one exact block.

    Important correction from the earlier script:
    This filters by block_index AND layout AND tracking.
    """
    block = frames[
        (frames["block_index"] == block_index)
        & (frames["layout"] == layout)
        & (frames["tracking"] == tracking)
        & (frames["state"] == "TrialActive")
    ].copy()

    info: dict[str, Any] = {
        "n_active_rows": int(len(block)),
        "n_trials_found": 0,
        "n_trials_used": 0,
        "min_trial_len": None,
        "max_trial_len": None,
        "mean_trial_len": None,
        "skip_reason": None,
    }

    if block.empty:
        info["skip_reason"] = "no_trialactive_rows_for_exact_block_layout_tracking"
        return [], [], info

    block = add_head_az_el(block)

    needed = [
        "trial_index_in_block",
        "timestamp_utc",
        "raw_panel_az_deg",
        "raw_panel_el_deg",
        "head_az_deg",
        "head_el_deg",
    ]

    block = block[needed].dropna().copy()

    if block.empty:
        info["skip_reason"] = "all_signal_rows_missing"
        return [], [], info

    eye_state_trials: list[np.ndarray] = []
    head_state_trials: list[np.ndarray] = []
    trial_lengths: list[int] = []

    info["n_trials_found"] = int(block["trial_index_in_block"].nunique())

    for _, trial_df in block.groupby("trial_index_in_block"):
        trial_df = trial_df.sort_values("timestamp_utc").copy()

        eye_state = encode_2d_state(
            yaw=trial_df["raw_panel_az_deg"].to_numpy(dtype=float),
            pitch=trial_df["raw_panel_el_deg"].to_numpy(dtype=float),
            yaw_edges=bin_config["eye_az_edges"],
            pitch_edges=bin_config["eye_el_edges"],
            n_bins_per_dim=N_BINS_PER_DIM,
        )

        head_state = encode_2d_state(
            yaw=trial_df["head_az_deg"].to_numpy(dtype=float),
            pitch=trial_df["head_el_deg"].to_numpy(dtype=float),
            yaw_edges=bin_config["head_az_edges"],
            pitch_edges=bin_config["head_el_edges"],
            n_bins_per_dim=N_BINS_PER_DIM,
        )

        if len(eye_state) >= MIN_SAMPLES_PER_TRIAL:
            eye_state_trials.append(eye_state)
            head_state_trials.append(head_state)
            trial_lengths.append(len(eye_state))

    info["n_trials_used"] = len(eye_state_trials)

    if trial_lengths:
        info["min_trial_len"] = int(np.min(trial_lengths))
        info["max_trial_len"] = int(np.max(trial_lengths))
        info["mean_trial_len"] = float(np.mean(trial_lengths))

    if len(eye_state_trials) < 2:
        info["skip_reason"] = "fewer_than_two_usable_trials"
        return [], [], info

    return eye_state_trials, head_state_trials, info


def run_te_for_block(
    eye_state_trials: list[np.ndarray],
    head_state_trials: list[np.ndarray],
    rng: np.random.Generator,
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    eye_to_head = transfer_entropy_discrete(
        source_trials=eye_state_trials,
        target_trials=head_state_trials,
        n_states=N_STATES,
        source_lag=SOURCE_LAG,
        target_lag=TARGET_LAG,
    )

    eye_to_head_shuffled = shuffled_te_baseline(
        source_trials=eye_state_trials,
        target_trials=head_state_trials,
        n_states=N_STATES,
        rng=rng,
        n_shuffles=N_SHUFFLES,
        source_lag=SOURCE_LAG,
        target_lag=TARGET_LAG,
    )

    head_to_eye = transfer_entropy_discrete(
        source_trials=head_state_trials,
        target_trials=eye_state_trials,
        n_states=N_STATES,
        source_lag=SOURCE_LAG,
        target_lag=TARGET_LAG,
    )

    head_to_eye_shuffled = shuffled_te_baseline(
        source_trials=head_state_trials,
        target_trials=eye_state_trials,
        n_states=N_STATES,
        rng=rng,
        n_shuffles=N_SHUFFLES,
        source_lag=SOURCE_LAG,
        target_lag=TARGET_LAG,
    )

    out["eye_to_head_te"] = eye_to_head
    out["eye_to_head_shuffled_mean"] = float(np.nanmean(eye_to_head_shuffled))
    out["eye_to_head_shuffled_std"] = float(np.nanstd(eye_to_head_shuffled))
    out["eye_to_head_corrected_te"] = (
        eye_to_head - out["eye_to_head_shuffled_mean"]
    )
    out["eye_to_head_perm_p"] = float(
        (np.sum(eye_to_head_shuffled >= eye_to_head) + 1) / (N_SHUFFLES + 1)
    )

    out["head_to_eye_te"] = head_to_eye
    out["head_to_eye_shuffled_mean"] = float(np.nanmean(head_to_eye_shuffled))
    out["head_to_eye_shuffled_std"] = float(np.nanstd(head_to_eye_shuffled))
    out["head_to_eye_corrected_te"] = (
        head_to_eye - out["head_to_eye_shuffled_mean"]
    )
    out["head_to_eye_perm_p"] = float(
        (np.sum(head_to_eye_shuffled >= head_to_eye) + 1) / (N_SHUFFLES + 1)
    )

    return out


# ---------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------

def safe_corr(x: pd.Series, y: pd.Series) -> tuple[float, float, float, float]:
    df = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(df) < 3:
        return np.nan, np.nan, np.nan, np.nan

    if df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return np.nan, np.nan, np.nan, np.nan

    pearson_r, pearson_p = pearsonr(df["x"], df["y"])
    spearman_rho, spearman_p = spearmanr(df["x"], df["y"])

    return pearson_r, pearson_p, spearman_rho, spearman_p


def compute_correlations(results_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "eye_to_head_te",
        "head_to_eye_te",
        "eye_to_head_corrected_te",
        "head_to_eye_corrected_te",
    ]

    outcomes = ["accuracy", "error_rate"]

    rows = []

    for outcome in outcomes:
        for metric in metrics:
            subset = results_df[["tracking", "layout", outcome, metric]].dropna()

            pearson_r, pearson_p, spearman_rho, spearman_p = safe_corr(
                subset[metric],
                subset[outcome],
            )

            rows.append(
                {
                    "metric": metric,
                    "outcome": outcome,
                    "scope": "bias_modes",
                    "n_blocks": len(subset),
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_rho": spearman_rho,
                    "spearman_p": spearman_p,
                }
            )

            for tracking, tracking_df in subset.groupby("tracking"):
                pearson_r, pearson_p, spearman_rho, spearman_p = safe_corr(
                    tracking_df[metric],
                    tracking_df[outcome],
                )

                rows.append(
                    {
                        "metric": metric,
                        "outcome": outcome,
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
# Main
# ---------------------------------------------------------------------

def main() -> None:
    log("Starting diagnostic histogram-based pairwise yaw/pitch TE script")
    log(f"Dataset root: {DATASET_ROOT}")
    log(f"Raw frames dir: {RAW_FRAMES_DIR}")
    log(f"Block summary: {BLOCK_SUMMARY_PATH}")
    log(f"Output dir: {OUTPUT_DIR}")

    rng = np.random.default_rng(RANDOM_SEED)

    block_summary = pd.read_csv(BLOCK_SUMMARY_PATH, dtype={"participant_id": str})

    blocks = block_summary[
        block_summary["tracking"].isin(TRACKING_MODES)
    ].copy()

    blocks = validate_accuracy_column(blocks)

    max_accuracy_diff = blocks["accuracy_abs_diff"].max()
    log(f"Max |accuracy - correct_count / n_trials|: {max_accuracy_diff:.12f}")

    if max_accuracy_diff > 1e-9:
        log("WARNING: accuracy column does not perfectly match correct_count / n_trials")

    blocks = blocks.sort_values(
        ["participant_id", "layout", "tracking"]
    ).reset_index(drop=True)

    if MAX_BLOCKS is not None:
        blocks = blocks.head(MAX_BLOCKS).copy()
        log(f"Running smoke test with MAX_BLOCKS={MAX_BLOCKS}")
    else:
        log("Running full block set")

    log(f"Blocks to process: {len(blocks)}")

    bin_config = make_global_bin_config(blocks)

    log("Global bin edges:")
    for key, edges in bin_config.items():
        log(f"  {key}: {edges}")

    result_rows: list[dict[str, Any]] = []

    # Cache frame files for the main loop as well.
    frame_cache: dict[str, pd.DataFrame] = {}

    for i, row in blocks.iterrows():
        participant_id = row["participant_id"]
        layout = row["layout"]
        tracking = row["tracking"]

        block_index = parse_block_index(row["condition_block_key"])
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
            "computed_accuracy": row["computed_accuracy"],
            "accuracy_abs_diff": row["accuracy_abs_diff"],
            "error_rate": 1.0 - row["accuracy"],
            "correct_count": row["correct_count"],
            "n_trials_expected": row["n_trials"],
            "frame_path": str(frame_path),
            "n_bins_per_dim": N_BINS_PER_DIM,
            "n_states": N_STATES,
            "binning_method": BINNING_METHOD,
            "strict_balanced_binning": STRICT_BALANCED_BINNING,
            "source_lag": SOURCE_LAG,
            "target_lag": TARGET_LAG,
        }

        if not frame_path.exists():
            log(f"  SKIP: frame file missing: {frame_path}")
            base_row.update(
                {
                    "status": "skipped",
                    "skip_reason": "frame_file_missing",
                }
            )
            result_rows.append(base_row)
            continue

        if source_session_id not in frame_cache:
            frame_cache[source_session_id] = pd.read_csv(frame_path)

        frames = frame_cache[source_session_id]

        diagnostic_info = diagnose_frame_match(
            frames=frames,
            block_index=block_index,
            layout=layout,
            tracking=tracking,
            expected_n_trials=int(row["n_trials"]),
        )
        base_row.update(diagnostic_info)

        eye_state_trials, head_state_trials, info = prepare_pairwise_state_trials(
            frames=frames,
            block_index=block_index,
            layout=layout,
            tracking=tracking,
            bin_config=bin_config,
        )

        base_row.update(info)

        if len(eye_state_trials) == 0:
            log(f"  SKIP: {info.get('skip_reason')}")
            base_row.update({"status": "skipped"})
            result_rows.append(base_row)
            continue

        log(
            f"  Trials used: {info['n_trials_used']}, "
            f"trial length range: {info['min_trial_len']} to {info['max_trial_len']}"
        )

        te_out = run_te_for_block(
            eye_state_trials=eye_state_trials,
            head_state_trials=head_state_trials,
            rng=rng,
        )

        base_row.update(te_out)
        base_row["status"] = "ok"

        result_rows.append(base_row)

        log(
            "  TE results: "
            f"eye_to_head={te_out['eye_to_head_te']:.6f}, "
            f"head_to_eye={te_out['head_to_eye_te']:.6f}, "
            f"eye_to_head_corrected={te_out['eye_to_head_corrected_te']:.6f}, "
            f"head_to_eye_corrected={te_out['head_to_eye_corrected_te']:.6f}"
        )

    results_df = pd.DataFrame(result_rows)

    suffix = (
        f"diagnostic_pairwise_2d_{N_BINS_PER_DIM}x{N_BINS_PER_DIM}_"
        f"{BINNING_METHOD}_lag{SOURCE_LAG}_shuffle{N_SHUFFLES}"
    )

    results_path = OUTPUT_DIR / f"histogram_te_block_results_{suffix}.csv"
    results_df.to_csv(results_path, index=False)
    log(f"\nSaved block TE results to: {results_path}")

    ok_df = results_df[results_df["status"] == "ok"].copy()

    corr_df = compute_correlations(ok_df)

    corr_path = OUTPUT_DIR / f"histogram_te_accuracy_correlations_{suffix}.csv"
    corr_df.to_csv(corr_path, index=False)
    log(f"Saved TE correlations to: {corr_path}")

    diagnostic_summary = {
        "n_rows": int(len(results_df)),
        "n_ok": int((results_df["status"] == "ok").sum()),
        "n_skipped": int((results_df["status"] != "ok").sum()),
        "n_expected_match_missing": int(
            (~results_df["diagnostic_expected_match_found"].fillna(False)).sum()
        ),
        "n_trial_count_mismatch": int(
            (~results_df["diagnostic_trial_count_matches"].fillna(False)).sum()
        ),
        "max_accuracy_abs_diff": float(results_df["accuracy_abs_diff"].max()),
    }

    diagnostic_path = OUTPUT_DIR / f"histogram_te_diagnostic_summary_{suffix}.json"
    with diagnostic_path.open("w", encoding="utf-8") as f:
        json.dump(diagnostic_summary, f, indent=2)

    log(f"Saved diagnostic summary to: {diagnostic_path}")

    settings_path = OUTPUT_DIR / f"histogram_te_settings_{suffix}.json"
    with settings_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "tracking_modes": TRACKING_MODES,
                "n_bins_per_dim": N_BINS_PER_DIM,
                "n_states": N_STATES,
                "binning_method": BINNING_METHOD,
                "strict_balanced_binning": STRICT_BALANCED_BINNING,
                "source_lag": SOURCE_LAG,
                "target_lag": TARGET_LAG,
                "min_samples_per_trial": MIN_SAMPLES_PER_TRIAL,
                "n_shuffles": N_SHUFFLES,
                "random_seed": RANDOM_SEED,
                "max_blocks": MAX_BLOCKS,
                "bin_edges": {
                    key: value.tolist() for key, value in bin_config.items()
                },
            },
            f,
            indent=2,
        )

    log(f"Saved settings to: {settings_path}")

    log("\nDiagnostic summary:")
    log(json.dumps(diagnostic_summary, indent=2))

    log("\nCorrelation preview:")
    if corr_df.empty:
        log("No valid correlations.")
    else:
        log(corr_df.to_string(index=False))

    log("\nDone.")


if __name__ == "__main__":
    main()