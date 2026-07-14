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

# Pairwise yaw/pitch binning:
# yaw has 3 bins, pitch has 3 bins, so each signal has 3 x 3 = 9 states.
N_BINS_PER_DIM = 3
N_STATES = N_BINS_PER_DIM * N_BINS_PER_DIM

# Binning method:
# "quantile" gives roughly balanced bins.
# "equal_width" gives geometrically equal bins.
BINNING_METHOD = "quantile"

# Transfer entropy lag definition:
# source_lag = 1 means source[t-1] -> target[t]
# target_lag = 1 means target[t-1] controls for target's own past.
SOURCE_LAG = 1
TARGET_LAG = 1

MIN_SAMPLES_PER_TRIAL = 30

# Shuffled baseline for corrected TE and an approximate permutation p-value.
N_SHUFFLES = 100
RANDOM_SEED = 12345

# Set to an integer for quick smoke test. Set to None for full run.
MAX_BLOCKS = None


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

    # Expand the outer edges slightly so boundary values are safely included.
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
    """
    Convert continuous yaw/pitch values into one of 9 pairwise states.

    If n_bins_per_dim = 3:

        yaw_bin in {0, 1, 2}
        pitch_bin in {0, 1, 2}

    state = yaw_bin * 3 + pitch_bin

    Therefore state is in {0, ..., 8}.
    """
    yaw_bin = np.digitize(yaw, yaw_edges[1:-1], right=False)
    pitch_bin = np.digitize(pitch, pitch_edges[1:-1], right=False)

    yaw_bin = np.clip(yaw_bin, 0, n_bins_per_dim - 1)
    pitch_bin = np.clip(pitch_bin, 0, n_bins_per_dim - 1)

    state = yaw_bin * n_bins_per_dim + pitch_bin
    return state.astype(int)


# ---------------------------------------------------------------------
# Bin edge construction
# ---------------------------------------------------------------------

def collect_values_for_global_bins(blocks: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Collect active-frame eye/head yaw/pitch values from the selected blocks.

    This creates global bin edges so that state 0-8 has the same meaning
    across all blocks.
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

    return {
        "eye_az": np.concatenate(eye_az_values),
        "eye_el": np.concatenate(eye_el_values),
        "head_az": np.concatenate(head_az_values),
        "head_el": np.concatenate(head_el_values),
    }


def make_global_bin_config(blocks: pd.DataFrame) -> dict[str, Any]:
    values = collect_values_for_global_bins(blocks)

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
    Histogram/discrete transfer entropy:

        TE source -> target =
        sum p(y_t, y_past, x_past)
            log2(
                p(y_t | y_past, x_past)
                /
                p(y_t | y_past)
            )

    This function aggregates transition counts across trials while avoiding
    artificial transitions between the end of one trial and the start of the next.
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
    Shuffle source states within each trial to break temporal source-target
    coupling while preserving the source state's marginal distribution.
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
    tracking: str,
    bin_config: dict[str, Any],
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    """
    Return:
        eye_state_trials: list of arrays, one per trial
        head_state_trials: list of arrays, one per trial
        info: diagnostic metadata
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
        "min_trial_len": None,
        "max_trial_len": None,
        "mean_trial_len": None,
        "skip_reason": None,
    }

    if block.empty:
        info["skip_reason"] = "no_trialactive_rows"
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

    if len(trial_lengths) > 0:
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
    """
    Compute raw TE, shuffled baseline, corrected TE, and approximate p-values
    in both directions.
    """
    out: dict[str, Any] = {}

    # eye -> head
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

    # head -> eye
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

    rows = []

    for metric in metrics:
        subset = results_df[["tracking", "layout", "accuracy", metric]].dropna()

        pearson_r, pearson_p, spearman_rho, spearman_p = safe_corr(
            subset[metric],
            subset["accuracy"],
        )

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
            pearson_r, pearson_p, spearman_rho, spearman_p = safe_corr(
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
# Main
# ---------------------------------------------------------------------

def main() -> None:
    log("Starting histogram-based pairwise yaw/pitch TE script")
    log(f"Dataset root: {DATASET_ROOT}")
    log(f"Raw frames dir: {RAW_FRAMES_DIR}")
    log(f"Block summary: {BLOCK_SUMMARY_PATH}")
    log(f"Output dir: {OUTPUT_DIR}")

    rng = np.random.default_rng(RANDOM_SEED)

    block_summary = pd.read_csv(BLOCK_SUMMARY_PATH, dtype={"participant_id": str})

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

    bin_config = make_global_bin_config(blocks)

    log("Global bin edges:")
    for key, edges in bin_config.items():
        log(f"  {key}: {edges}")

    result_rows: list[dict[str, Any]] = []

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
            "n_bins_per_dim": N_BINS_PER_DIM,
            "n_states": N_STATES,
            "binning_method": BINNING_METHOD,
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

        frames = pd.read_csv(frame_path)

        eye_state_trials, head_state_trials, info = prepare_pairwise_state_trials(
            frames=frames,
            block_index=block_index,
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
        f"pairwise_2d_{N_BINS_PER_DIM}x{N_BINS_PER_DIM}_"
        f"{BINNING_METHOD}_lag{SOURCE_LAG}_shuffle{N_SHUFFLES}"
    )

    results_path = OUTPUT_DIR / f"histogram_te_block_results_{suffix}.csv"
    results_df.to_csv(results_path, index=False)
    log(f"\nSaved block TE results to: {results_path}")

    corr_df = compute_correlations(results_df[results_df["status"] == "ok"].copy())

    corr_path = OUTPUT_DIR / f"histogram_te_accuracy_correlations_{suffix}.csv"
    corr_df.to_csv(corr_path, index=False)
    log(f"Saved TE-accuracy correlations to: {corr_path}")

    settings_path = OUTPUT_DIR / f"histogram_te_settings_{suffix}.json"
    with settings_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "tracking_modes": TRACKING_MODES,
                "n_bins_per_dim": N_BINS_PER_DIM,
                "n_states": N_STATES,
                "binning_method": BINNING_METHOD,
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

    log("\nCorrelation preview:")
    if corr_df.empty:
        log("No valid correlations.")
    else:
        log(corr_df.to_string(index=False))

    log("\nDone.")


if __name__ == "__main__":
    main()