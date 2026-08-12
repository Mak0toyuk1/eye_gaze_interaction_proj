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

RAW_FRAMES_DIR = DATASET_ROOT / "data" / "raw" / "frames"
BLOCK_SUMMARY_PATH = DATASET_ROOT / "data" / "processed" / "balanced_subject_block_summary.csv"

OUT_BLOCK = OUTPUT_DIR / "pid_head_eye_block_results_williams_beer_imin.csv"
OUT_PARTICIPANT_CONDITION = OUTPUT_DIR / "pid_head_eye_participant_condition_results_williams_beer_imin.csv"
OUT_GLOBAL = OUTPUT_DIR / "pid_head_eye_friedman_global_tests_williams_beer_imin.csv"
OUT_PAIRWISE = OUTPUT_DIR / "pid_head_eye_pairwise_wilcoxon_tests_williams_beer_imin.csv"
OUT_SUMMARY = OUTPUT_DIR / "pid_head_eye_condition_summary_williams_beer_imin.csv"


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

N_BINS = 3
LAG = 1

TRACKING_ORDER = ["Clean", "Jitter", "Bias", "BiasJitter"]

PID_COMPONENTS = [
    # Quantities retained for participant-level summaries and inference.
    #
    # The TE-specific names are preferred for hypothesis tests:
    #   state_independent_te == unique_source_1
    #   state_dependent_te   == synergistic_info
    #
    # We therefore do not also test the exact duplicate aliases
    # unique_source_1 and synergistic_info, avoiding duplicate p-values.
    "transfer_entropy",
    "state_independent_te",
    "state_dependent_te",

    # Remaining Williams-Beer PID / MI terms.
    "redundant_info",
    "unique_source_2",
    "total_mi",
    "source_1_mi",
    "source_2_mi",
]

# IMPORTANT:
# For every analysis, source_1 is the putative DRIVER past and source_2 is
# the TARGET'S OWN past. This orientation makes the Williams-Beer transfer
# decomposition directly interpretable:
#
#   transfer_entropy = unique_source_1 + synergistic_info
#   SITE              = unique_source_1
#   SDTE              = synergistic_info
#
# Williams & Beer:
#   - Nonnegative Decomposition of Multivariate Information (2010)
#   - Generalized Measures of Information Transfer (2011)
PID_ANALYSES = [
    {
        "analysis": "head_to_eye_yaw",
        "direction": "head_to_eye",
        "source_1": "head_yaw_state_past",
        "source_2": "eye_yaw_state_past",
        "target": "eye_yaw_state_now",
        "source_1_label": "past_head_yaw_driver",
        "source_2_label": "past_eye_yaw_target_history",
        "target_label": "current_eye_yaw",
    },
    {
        "analysis": "head_to_eye_pitch",
        "direction": "head_to_eye",
        "source_1": "head_pitch_state_past",
        "source_2": "eye_pitch_state_past",
        "target": "eye_pitch_state_now",
        "source_1_label": "past_head_pitch_driver",
        "source_2_label": "past_eye_pitch_target_history",
        "target_label": "current_eye_pitch",
    },
    {
        "analysis": "eye_to_head_yaw",
        "direction": "eye_to_head",
        "source_1": "eye_yaw_state_past",
        "source_2": "head_yaw_state_past",
        "target": "head_yaw_state_now",
        "source_1_label": "past_eye_yaw_driver",
        "source_2_label": "past_head_yaw_target_history",
        "target_label": "current_head_yaw",
    },
    {
        "analysis": "eye_to_head_pitch",
        "direction": "eye_to_head",
        "source_1": "eye_pitch_state_past",
        "source_2": "head_pitch_state_past",
        "target": "head_pitch_state_now",
        "source_1_label": "past_eye_pitch_driver",
        "source_2_label": "past_head_pitch_target_history",
        "target_label": "current_head_pitch",
    },
]


# ------------------------------------------------------------
# Column helpers
# ------------------------------------------------------------

def find_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def compute_head_yaw_pitch_from_quaternion(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    x_col = find_first_existing_column(df, ["head_rotation_x", "head_rot_x", "camera_rot_x"])
    y_col = find_first_existing_column(df, ["head_rotation_y", "head_rot_y", "camera_rot_y"])
    z_col = find_first_existing_column(df, ["head_rotation_z", "head_rot_z", "camera_rot_z"])
    w_col = find_first_existing_column(df, ["head_rotation_w", "head_rot_w", "camera_rot_w"])

    if not all([x_col, y_col, z_col, w_col]):
        rotation_like_cols = [c for c in df.columns if "rot" in c.lower() or "quat" in c.lower()]
        raise KeyError(
            "Could not find quaternion columns for head rotation.\n"
            f"Available rotation-like columns: {rotation_like_cols}"
        )

    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    z = pd.to_numeric(df[z_col], errors="coerce")
    w = pd.to_numeric(df[w_col], errors="coerce")

    # Quaternion to approximate Euler yaw/pitch.
    siny_cosp = 2.0 * (w * y + x * z)
    cosy_cosp = 1.0 - 2.0 * (y * y + x * x)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    sinp = 2.0 * (w * x - z * y)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    df["head_yaw_deg"] = np.degrees(yaw)
    df["head_pitch_deg"] = np.degrees(pitch)

    return df


# ------------------------------------------------------------
# Information theory helpers
# ------------------------------------------------------------

def make_joint_state(*arrays: np.ndarray) -> np.ndarray:
    """
    Converts multiple discrete arrays into a single joint-state array.

    Example:
        source_1 = [0, 1, 0]
        source_2 = [2, 2, 1]

    becomes joint states for:
        (0, 2), (1, 2), (0, 1)
    """
    arrays = [np.asarray(a) for a in arrays]

    if len(arrays) == 0:
        return np.array([])

    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("All arrays passed to make_joint_state() must have the same length.")

    tuples = pd.Series(list(zip(*arrays)))
    codes, _ = pd.factorize(tuples, sort=True)

    return codes

def mutual_information_discrete(x: np.ndarray, y: np.ndarray) -> float:
    """Plug-in mutual information estimator for discrete variables, in bits."""
    x = np.asarray(x)
    y = np.asarray(y)

    valid = ~(pd.isna(x) | pd.isna(y))
    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return np.nan

    xy = pd.DataFrame({"x": x, "y": y})

    p_xy = xy.value_counts(normalize=True)
    p_x = xy["x"].value_counts(normalize=True)
    p_y = xy["y"].value_counts(normalize=True)

    mi = 0.0

    for (x_val, y_val), pxy in p_xy.items():
        px = p_x.loc[x_val]
        py = p_y.loc[y_val]
        mi += pxy * np.log2(pxy / (px * py))

    return float(mi)


def specific_information_discrete(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[dict, dict]:
    """
    Williams-Beer specific information I(T=t ; S), estimated from empirical PMFs.

    For each target outcome t,

        I(T=t ; S)
            = sum_s p(s | t) log2[p(t | s) / p(t)]
            = D_KL(p(S | t) || p(S)).

    Returns
    -------
    specific_info : dict
        Mapping target state t -> I(T=t ; S), in bits.
    target_prob : dict
        Mapping target state t -> p(t).

    Notes
    -----
    Williams & Beer define I_min redundancy by first taking the minimum
    specific information across sources separately for EACH target outcome,
    and only then averaging over p(t). This is different from simply taking
    min(I(S1;T), I(S2;T)).
    """
    source = np.asarray(source)
    target = np.asarray(target)

    valid = ~(pd.isna(source) | pd.isna(target))
    source = source[valid]
    target = target[valid]

    if len(target) == 0:
        return {}, {}

    df = pd.DataFrame({"source": source, "target": target})
    n = float(len(df))

    source_counts = df["source"].value_counts()
    target_counts = df["target"].value_counts()
    joint_counts = df.groupby(["target", "source"]).size()

    p_source = source_counts / n
    p_target = target_counts / n

    specific_info = {}

    for t, n_t in target_counts.items():
        info_t = 0.0

        # Only source states observed jointly with this target outcome contribute.
        try:
            counts_given_t = joint_counts.loc[t]
        except KeyError:
            specific_info[t] = 0.0
            continue

        for s, n_ts in counts_given_t.items():
            p_s_given_t = float(n_ts) / float(n_t)
            p_s = float(p_source.loc[s])

            # Equivalent to log2[p(t|s)/p(t)] by Bayes' rule.
            info_t += p_s_given_t * np.log2(p_s_given_t / p_s)

        specific_info[t] = float(info_t)

    return specific_info, {t: float(p) for t, p in p_target.items()}


def imin_redundancy_two_sources(
    source_1: np.ndarray,
    source_2: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Williams-Beer I_min redundancy for two sources:

        I_min(T; S1, S2)
          = sum_t p(t) min{ I(T=t;S1), I(T=t;S2) }.

    This is the original Williams & Beer redundancy measure, NOT
    min(I(S1;T), I(S2;T)).
    """
    source_1 = np.asarray(source_1)
    source_2 = np.asarray(source_2)
    target = np.asarray(target)

    valid = ~(pd.isna(source_1) | pd.isna(source_2) | pd.isna(target))
    source_1 = source_1[valid]
    source_2 = source_2[valid]
    target = target[valid]

    if len(target) == 0:
        return np.nan

    spec_1, p_target_1 = specific_information_discrete(source_1, target)
    spec_2, p_target_2 = specific_information_discrete(source_2, target)

    # The same complete-case rows are used for both sources, so the target
    # distributions should be identical. Average using that common p(t).
    target_states = set(p_target_1) | set(p_target_2)

    redundancy = 0.0
    for t in target_states:
        p_t = p_target_1.get(t, p_target_2.get(t, 0.0))
        redundancy += p_t * min(spec_1.get(t, 0.0), spec_2.get(t, 0.0))

    return float(redundancy)


def _zero_if_tiny(value: float, tol: float = 1e-12) -> float:
    """Remove floating-point noise without masking substantive negative values."""
    if np.isfinite(value) and abs(value) < tol:
        return 0.0
    return float(value)


def pid_two_sources_williams_beer_imin(
    source_1: np.ndarray,
    source_2: np.ndarray,
    target: np.ndarray,
) -> dict:
    """
    Two-source Williams-Beer PID using the original I_min redundancy.

    The full mutual information decomposes as:

        I(T ; S1,S2) = R + U1 + U2 + Syn

    where
        R   = I_min(T; S1,S2)
        U1  = I(T;S1) - R
        U2  = I(T;S2) - R
        Syn = I(T;S1,S2) - R - U1 - U2

    In this script, the analysis specifications are deliberately oriented so:
        S1 = DRIVER past
        S2 = TARGET past
        T  = TARGET present/future

    Therefore Schreiber transfer entropy from the driver to the target is

        TE = I(T ; S1 | S2)
           = I(T ; S1,S2) - I(T ; S2)
           = U1 + Syn.

    Following Williams & Beer (2011):
        U1  is state-independent transfer entropy (SITE)
        Syn is state-dependent transfer entropy (SDTE).

    This is a plug-in / histogram estimator after discretization.
    It is not BROJA PID and it does not apply finite-sample bias correction.
    """
    source_1 = np.asarray(source_1)
    source_2 = np.asarray(source_2)
    target = np.asarray(target)

    valid = ~(pd.isna(source_1) | pd.isna(source_2) | pd.isna(target))
    source_1 = source_1[valid]
    source_2 = source_2[valid]
    target = target[valid]

    empty = {
        "n_pid_samples": 0,
        "total_mi": np.nan,
        "source_1_mi": np.nan,
        "source_2_mi": np.nan,
        "redundant_info": np.nan,
        "unique_source_1": np.nan,
        "unique_source_2": np.nan,
        "synergistic_info": np.nan,
        "transfer_entropy": np.nan,
        "state_independent_te": np.nan,
        "state_dependent_te": np.nan,
        "te_reconstruction_error": np.nan,
    }

    if len(target) == 0:
        return empty

    joint_sources = make_joint_state(source_1, source_2)

    total_mi = mutual_information_discrete(joint_sources, target)
    source_1_mi = mutual_information_discrete(source_1, target)
    source_2_mi = mutual_information_discrete(source_2, target)

    redundancy = imin_redundancy_two_sources(source_1, source_2, target)

    unique_1 = source_1_mi - redundancy
    unique_2 = source_2_mi - redundancy
    synergy = total_mi - redundancy - unique_1 - unique_2

    # Since source_2 is the target's own past, this is TE(driver -> target).
    transfer_entropy = total_mi - source_2_mi

    # Williams-Beer transfer decomposition.
    state_independent_te = unique_1
    state_dependent_te = synergy
    te_reconstruction_error = transfer_entropy - (
        state_independent_te + state_dependent_te
    )

    values = {
        "n_pid_samples": int(len(target)),
        "total_mi": total_mi,
        "source_1_mi": source_1_mi,
        "source_2_mi": source_2_mi,
        "redundant_info": redundancy,
        "unique_source_1": unique_1,
        "unique_source_2": unique_2,
        "synergistic_info": synergy,
        "transfer_entropy": transfer_entropy,
        "state_independent_te": state_independent_te,
        "state_dependent_te": state_dependent_te,
        "te_reconstruction_error": te_reconstruction_error,
    }

    # Williams-Beer atoms are nonnegative theoretically. Only remove tiny
    # floating-point deviations from zero; retain larger negatives so any
    # estimator/data problem remains visible.
    for key in [
        "total_mi",
        "source_1_mi",
        "source_2_mi",
        "redundant_info",
        "unique_source_1",
        "unique_source_2",
        "synergistic_info",
        "transfer_entropy",
        "state_independent_te",
        "state_dependent_te",
        "te_reconstruction_error",
    ]:
        values[key] = _zero_if_tiny(values[key])

    return values


def assign_quantile_bins(series: pd.Series, n_bins: int) -> pd.Series:
    try:
        return pd.qcut(series, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(np.nan, index=series.index)


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
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


# ------------------------------------------------------------
# Data loading and standardization
# ------------------------------------------------------------

def load_raw_frames() -> pd.DataFrame:
    files = sorted(RAW_FRAMES_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No raw frame CSV files found in:\n{RAW_FRAMES_DIR}")

    chunks = []

    for path in files:
        print(f"Reading {path.name}")
        df = pd.read_csv(path, low_memory=False)
        df["source_frame_file"] = path.name
        chunks.append(df)

    frames = pd.concat(chunks, ignore_index=True)

    print(f"Loaded frame rows: {len(frames):,}")
    print(f"Loaded frame columns: {len(frames.columns):,}")

    return frames


def standardize_frame_columns(frames: pd.DataFrame) -> pd.DataFrame:
    frames = frames.copy()

    candidate_map = {
        "participant_id": ["participant_id", "participant", "subject", "subject_id"],
        "layout": ["layout", "layout_name", "difficulty"],
        "tracking": ["tracking", "tracking_mode", "mode"],
        "block_index": ["block_index", "block", "block_id"],

        # Actual raw frame files use trial_index_in_block.
        "trial_index": ["trial_index", "trial_index_in_block", "trial", "trial_id"],

        "timestamp": ["timestamp_utc", "timestamp", "time", "frame_time"],
        "state": ["state", "trial_state"],

        "eye_yaw_deg": ["raw_panel_az_deg", "eye_az_deg", "eye_yaw_deg"],
        "eye_pitch_deg": ["raw_panel_el_deg", "eye_el_deg", "eye_pitch_deg"],
    }

    rename_map = {}

    for standard_name, candidates in candidate_map.items():
        found = find_first_existing_column(frames, candidates)
        if found is not None and found != standard_name:
            rename_map[found] = standard_name

    frames = frames.rename(columns=rename_map)

    required = [
        "participant_id",
        "layout",
        "tracking",
        "trial_index",
        "timestamp",
        "eye_yaw_deg",
        "eye_pitch_deg",
    ]

    missing = [c for c in required if c not in frames.columns]
    if missing:
        raise KeyError(
            f"Missing required columns after standardization: {missing}\n\n"
            f"Available columns:\n{frames.columns.tolist()}"
        )

    if "head_yaw_deg" not in frames.columns or "head_pitch_deg" not in frames.columns:
        frames = compute_head_yaw_pitch_from_quaternion(frames)

    print("State values before filtering:")
    if "state" in frames.columns:
        print(frames["state"].value_counts(dropna=False).head(20).to_string())

    if "state" in frames.columns:
        before = len(frames)
        frames = frames[frames["state"].astype(str).eq("TrialActive")].copy()
        print(f"TrialActive filter: {before:,} -> {len(frames):,} rows")

    frames["participant_id"] = frames["participant_id"].astype(str)
    frames["layout"] = frames["layout"].astype(str)
    frames["tracking"] = frames["tracking"].astype(str)

    if "block_index" in frames.columns:
        frames["block_index"] = pd.to_numeric(frames["block_index"], errors="coerce")

    frames["trial_index"] = pd.to_numeric(frames["trial_index"], errors="coerce")

    # Do not force timestamp to numeric. Keep the original sortable value.
    # Create a helper sort column instead.
    timestamp_numeric = pd.to_numeric(frames["timestamp"], errors="coerce")

    if timestamp_numeric.notna().mean() > 0.95:
        frames["timestamp_sort"] = timestamp_numeric
        print("Using numeric timestamp_sort.")
    else:
        timestamp_datetime = pd.to_datetime(frames["timestamp"], errors="coerce")
        if timestamp_datetime.notna().mean() > 0.95:
            frames["timestamp_sort"] = timestamp_datetime
            print("Using datetime timestamp_sort.")
        else:
            frames["timestamp_sort"] = frames.groupby(
                ["participant_id", "layout", "tracking", "trial_index"]
            ).cumcount()
            print("Using row-order timestamp_sort fallback.")

    for col in ["eye_yaw_deg", "eye_pitch_deg", "head_yaw_deg", "head_pitch_deg"]:
        frames[col] = pd.to_numeric(frames[col], errors="coerce")

    before_drop = len(frames)
    frames = frames.dropna(
        subset=[
            "participant_id",
            "layout",
            "tracking",
            "trial_index",
            "eye_yaw_deg",
            "eye_pitch_deg",
            "head_yaw_deg",
            "head_pitch_deg",
            "timestamp_sort",
        ]
    ).copy()

    print(f"Drop missing required frame fields: {before_drop:,} -> {len(frames):,} rows")

    print("Rows by tracking after standardization:")
    print(frames["tracking"].value_counts(dropna=False).to_string())

    print("Unique participants after standardization:", frames["participant_id"].nunique())

    return frames


def load_block_summary() -> pd.DataFrame:
    if not BLOCK_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Could not find block summary:\n{BLOCK_SUMMARY_PATH}")

    block_df = pd.read_csv(BLOCK_SUMMARY_PATH, low_memory=False)

    for col in ["participant_id", "layout", "tracking"]:
        if col in block_df.columns:
            block_df[col] = block_df[col].astype(str)

    if "block_index" in block_df.columns:
        block_df["block_index"] = pd.to_numeric(block_df["block_index"], errors="coerce")

    if "accuracy" not in block_df.columns:
        if "correct_count" in block_df.columns:
            block_df["accuracy"] = pd.to_numeric(block_df["correct_count"], errors="coerce") / 12.0

    if "error_rate" not in block_df.columns and "accuracy" in block_df.columns:
        block_df["error_rate"] = 1.0 - block_df["accuracy"]

    print(f"Loaded block summary rows: {len(block_df):,}")
    print(f"Block summary columns: {block_df.columns.tolist()}")

    return block_df


# ------------------------------------------------------------
# State construction
# ------------------------------------------------------------

def add_discrete_states(frames: pd.DataFrame) -> pd.DataFrame:
    frames = frames.copy()

    for col in ["eye_yaw_deg", "eye_pitch_deg", "head_yaw_deg", "head_pitch_deg"]:
        state_col = col.replace("_deg", "_state")
        frames[state_col] = assign_quantile_bins(frames[col], N_BINS)

    before = len(frames)
    frames = frames.dropna(
        subset=[
            "eye_yaw_state",
            "eye_pitch_state",
            "head_yaw_state",
            "head_pitch_state",
        ]
    ).copy()

    print(f"Drop missing discrete states: {before:,} -> {len(frames):,} rows")

    return frames


def add_lagged_states(frames: pd.DataFrame) -> pd.DataFrame:
    frames = frames.sort_values(
        ["participant_id", "layout", "tracking", "trial_index", "timestamp_sort"]
    ).copy()

    grouping_cols = ["participant_id", "layout", "tracking", "trial_index"]

    for state_col in [
        "eye_yaw_state",
        "eye_pitch_state",
        "head_yaw_state",
        "head_pitch_state",
    ]:
        past_col = state_col.replace("_state", "_state_past")
        now_col = state_col.replace("_state", "_state_now")

        frames[now_col] = frames[state_col]
        frames[past_col] = frames.groupby(grouping_cols)[state_col].shift(LAG)

    before = len(frames)
    frames = frames.dropna(
        subset=[
            "eye_yaw_state_past",
            "eye_pitch_state_past",
            "head_yaw_state_past",
            "head_pitch_state_past",
            "eye_yaw_state_now",
            "eye_pitch_state_now",
            "head_yaw_state_now",
            "head_pitch_state_now",
        ]
    ).copy()

    print(f"Drop missing lagged states: {before:,} -> {len(frames):,} rows")

    return frames


def infer_block_group_columns(frames: pd.DataFrame) -> list[str]:
    if "block_index" in frames.columns and frames["block_index"].notna().any():
        return ["participant_id", "layout", "tracking", "block_index"]

    return ["participant_id", "layout", "tracking"]


# ------------------------------------------------------------
# PID computation
# ------------------------------------------------------------

def normalize_key_types_for_merge(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]):
    left = left.copy()
    right = right.copy()

    for key in keys:
        if key == "block_index":
            left[key] = pd.to_numeric(left[key], errors="coerce")
            right[key] = pd.to_numeric(right[key], errors="coerce")
        else:
            left[key] = left[key].astype(str)
            right[key] = right[key].astype(str)

    return left, right


def compute_pid_blocks(frames: pd.DataFrame, block_df: pd.DataFrame) -> pd.DataFrame:
    block_group_cols = infer_block_group_columns(frames)
    print(f"Using block grouping columns: {block_group_cols}")

    rows = []

    grouped = frames.groupby(block_group_cols, dropna=False)

    for block_key, g in grouped:
        if not isinstance(block_key, tuple):
            block_key = (block_key,)

        block_info = {col: value for col, value in zip(block_group_cols, block_key)}

        for analysis_spec in PID_ANALYSES:
            result = pid_two_sources_williams_beer_imin(
                source_1=g[analysis_spec["source_1"]].to_numpy(),
                source_2=g[analysis_spec["source_2"]].to_numpy(),
                target=g[analysis_spec["target"]].to_numpy(),
            )

            row = {
                **block_info,
                "analysis": analysis_spec["analysis"],
                "direction": analysis_spec["direction"],
                "source_1_role": "driver_past",
                "source_2_role": "target_past",
                "source_1_label": analysis_spec["source_1_label"],
                "source_2_label": analysis_spec["source_2_label"],
                "target_label": analysis_spec["target_label"],
                **result,
            }

            rows.append(row)

    pid_df = pd.DataFrame(rows)

    print("PID block result columns:")
    print(pid_df.columns.tolist())
    print(f"PID block rows: {len(pid_df):,}")

    # Numerical identity check:
    # TE(driver -> target) must equal SITE + SDTE.
    if "te_reconstruction_error" in pid_df.columns and len(pid_df) > 0:
        max_te_error = pid_df["te_reconstruction_error"].abs().max()
        print(f"Max |TE - (SITE + SDTE)|: {max_te_error:.3e}")

    # Williams-Beer I_min atoms are nonnegative in theory. Larger negative
    # values should therefore be treated as a diagnostic warning rather than
    # silently clipped away.
    atom_cols = [
        "redundant_info",
        "unique_source_1",
        "unique_source_2",
        "synergistic_info",
    ]
    negative_counts = {
        col: int((pid_df[col] < -1e-10).sum())
        for col in atom_cols
        if col in pid_df.columns
    }
    if any(negative_counts.values()):
        print(f"Warning: substantive negative PID atoms detected: {negative_counts}")

    if len(pid_df) == 0:
        raise ValueError(
            "No PID block rows were created. Check TrialActive filtering, binning, and lagging."
        )

    required_cols = ["participant_id", "layout", "tracking", "analysis"]
    missing = [c for c in required_cols if c not in pid_df.columns]

    if missing:
        raise KeyError(
            f"PID block result is missing required columns: {missing}\n"
            f"Available columns:\n{pid_df.columns.tolist()}"
        )

    # ------------------------------------------------------------
    # Merge accuracy/error_rate from block summary
    # ------------------------------------------------------------

    merge_candidates = [
        ["participant_id", "layout", "tracking", "block_index"],
        ["participant_id", "layout", "tracking"],
    ]

    merge_keys = None
    for keys in merge_candidates:
        if all(k in pid_df.columns for k in keys) and all(k in block_df.columns for k in keys):
            merge_keys = keys
            break

    if merge_keys is None:
        print("Warning: could not merge block summary accuracy/error_rate.")
        return pid_df

    print(f"Merging block summary using keys: {merge_keys}")

    pid_df_merge, block_df_merge = normalize_key_types_for_merge(pid_df, block_df, merge_keys)

    keep_cols = merge_keys.copy()
    for col in ["condition_block_key", "accuracy", "error_rate", "correct_count", "n_trials"]:
        if col in block_df_merge.columns and col not in keep_cols:
            keep_cols.append(col)

    block_small = block_df_merge[keep_cols].drop_duplicates(subset=merge_keys)

    pid_df_merge = pid_df_merge.merge(
        block_small,
        on=merge_keys,
        how="left",
        validate="many_to_one",
    )

    if "accuracy" in pid_df_merge.columns:
        missing_acc = pid_df_merge["accuracy"].isna().sum()
        print(f"Rows missing accuracy after merge: {missing_acc:,}")

    return pid_df_merge


# ------------------------------------------------------------
# Repeated-measures aggregation and tests
# ------------------------------------------------------------

def participant_condition_aggregate(pid_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["participant_id", "tracking", "analysis"]
    missing = [c for c in required_cols if c not in pid_df.columns]

    if missing:
        raise KeyError(
            f"Cannot aggregate because pid_df is missing: {missing}\n"
            f"Available columns:\n{pid_df.columns.tolist()}"
        )

    metrics = PID_COMPONENTS.copy()

    for col in ["accuracy", "error_rate"]:
        if col in pid_df.columns:
            metrics.append(col)

    existing_metrics = [m for m in metrics if m in pid_df.columns]

    agg = (
        pid_df.groupby(["participant_id", "tracking", "analysis"], as_index=False)
        .agg({metric: "mean" for metric in existing_metrics})
    )

    print("Participant-condition result columns:")
    print(agg.columns.tolist())
    print(f"Participant-condition rows: {len(agg):,}")

    return agg


def friedman_test_for_metric(agg: pd.DataFrame, analysis: str, metric: str) -> dict:
    temp = agg[agg["analysis"].eq(analysis)].copy()

    wide = temp.pivot(index="participant_id", columns="tracking", values=metric)

    ordered_cols = [c for c in TRACKING_ORDER if c in wide.columns]
    wide = wide[ordered_cols]

    wide_complete = wide.dropna(axis=0, how="any")

    n_participants = wide_complete.shape[0]
    n_conditions = wide_complete.shape[1]

    if n_participants < 2 or n_conditions < 3:
        return {
            "analysis": analysis,
            "metric": metric,
            "test": "friedman",
            "n_participants": n_participants,
            "n_conditions": n_conditions,
            "conditions": "|".join(wide_complete.columns.astype(str)),
            "statistic": np.nan,
            "p_value": np.nan,
            "kendalls_w": np.nan,
            "note": "Need at least 2 participants and 3 complete conditions.",
        }

    arrays = [wide_complete[col].to_numpy() for col in wide_complete.columns]
    stat, p = friedmanchisquare(*arrays)

    kendalls_w = stat / (n_participants * (n_conditions - 1))

    return {
        "analysis": analysis,
        "metric": metric,
        "test": "friedman",
        "n_participants": n_participants,
        "n_conditions": n_conditions,
        "conditions": "|".join(wide_complete.columns.astype(str)),
        "statistic": stat,
        "p_value": p,
        "kendalls_w": kendalls_w,
        "note": "",
    }


def wilcoxon_pairwise_for_metric(agg: pd.DataFrame, analysis: str, metric: str) -> list[dict]:
    temp = agg[agg["analysis"].eq(analysis)].copy()

    wide = temp.pivot(index="participant_id", columns="tracking", values=metric)

    ordered_cols = [c for c in TRACKING_ORDER if c in wide.columns]
    wide = wide[ordered_cols]

    rows = []

    for cond_a, cond_b in combinations(ordered_cols, 2):
        pair = wide[[cond_a, cond_b]].dropna(axis=0, how="any")

        n = pair.shape[0]

        if n < 2:
            rows.append(
                {
                    "analysis": analysis,
                    "metric": metric,
                    "condition_a": cond_a,
                    "condition_b": cond_b,
                    "n_participants": n,
                    "test": "wilcoxon_signed_rank",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "p_fdr_bh": np.nan,
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

        a = pair[cond_a]
        b = pair[cond_b]
        diff = b - a

        if np.allclose(diff.to_numpy(), 0):
            stat, p = 0.0, 1.0
            note = "All paired differences are zero."
        else:
            stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
            note = ""

        rows.append(
            {
                "analysis": analysis,
                "metric": metric,
                "condition_a": cond_a,
                "condition_b": cond_b,
                "n_participants": n,
                "test": "wilcoxon_signed_rank",
                "statistic": stat,
                "p_value": p,
                "p_fdr_bh": np.nan,
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


def summarize_conditions(agg: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in PID_COMPONENTS + ["accuracy", "error_rate"] if c in agg.columns]

    rows = []

    for analysis, g_analysis in agg.groupby("analysis"):
        for tracking, g in g_analysis.groupby("tracking"):
            for metric in metrics:
                vals = g[metric].dropna()

                rows.append(
                    {
                        "analysis": analysis,
                        "tracking": tracking,
                        "metric": metric,
                        "n_participants": vals.shape[0],
                        "mean": vals.mean(),
                        "median": vals.median(),
                        "std": vals.std(ddof=1),
                        "min": vals.min(),
                        "max": vals.max(),
                    }
                )

    return pd.DataFrame(rows)


def run_repeated_measures_tests(agg: pd.DataFrame):
    metrics = [c for c in PID_COMPONENTS + ["accuracy", "error_rate"] if c in agg.columns]
    analyses = sorted(agg["analysis"].dropna().unique().tolist())

    global_rows = []
    pairwise_rows = []

    for analysis in analyses:
        for metric in metrics:
            global_rows.append(friedman_test_for_metric(agg, analysis, metric))
            pairwise_rows.extend(wilcoxon_pairwise_for_metric(agg, analysis, metric))

    global_df = pd.DataFrame(global_rows)
    pairwise_df = pd.DataFrame(pairwise_rows)

    if len(pairwise_df) > 0:
        for (analysis, metric), idx in pairwise_df.groupby(["analysis", "metric"]).groups.items():
            idx = list(idx)
            valid_idx = pairwise_df.loc[idx].dropna(subset=["p_value"]).index

            if len(valid_idx) > 0:
                pairwise_df.loc[valid_idx, "p_fdr_bh"] = benjamini_hochberg(
                    pairwise_df.loc[valid_idx, "p_value"]
                ).to_numpy()

    return global_df, pairwise_df


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("PID method: Williams-Beer I_min")
    print("TE decomposition: transfer_entropy = state_independent_te + state_dependent_te")
    print("  state_independent_te = unique information from driver past")
    print("  state_dependent_te   = synergy(driver past, target past -> target now)")

    print("Loading raw frames...")
    frames = load_raw_frames()

    print("Standardizing frame columns...")
    frames = standardize_frame_columns(frames)

    print("Adding discrete states...")
    frames = add_discrete_states(frames)

    print("Adding lagged states...")
    frames = add_lagged_states(frames)

    print("Loading block summary...")
    block_df = load_block_summary()

    print("Computing block-level PID...")
    pid_block = compute_pid_blocks(frames, block_df)

    print("Aggregating to participant x condition...")
    participant_condition = participant_condition_aggregate(pid_block)

    print("Running Friedman and Wilcoxon tests...")
    global_df, pairwise_df = run_repeated_measures_tests(participant_condition)

    print("Summarizing conditions...")
    summary_df = summarize_conditions(participant_condition)

    pid_block.to_csv(OUT_BLOCK, index=False)
    participant_condition.to_csv(OUT_PARTICIPANT_CONDITION, index=False)
    global_df.to_csv(OUT_GLOBAL, index=False)
    pairwise_df.to_csv(OUT_PAIRWISE, index=False)
    summary_df.to_csv(OUT_SUMMARY, index=False)

    print("\nSaved:")
    print(f"  {OUT_BLOCK}")
    print(f"  {OUT_PARTICIPANT_CONDITION}")
    print(f"  {OUT_GLOBAL}")
    print(f"  {OUT_PAIRWISE}")
    print(f"  {OUT_SUMMARY}")

    print("\nFriedman tests with p < 0.05:")
    sig_global = global_df[global_df["p_value"].lt(0.05, fill_value=False)].copy()

    if len(sig_global) > 0:
        print(sig_global.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    else:
        print("No significant Friedman tests at p < 0.05.")

    print("\nPairwise Wilcoxon tests with FDR p < 0.05:")
    sig_pairwise = pairwise_df[pairwise_df["p_fdr_bh"].lt(0.05, fill_value=False)].copy()

    if len(sig_pairwise) > 0:
        print(sig_pairwise.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    else:
        print("No significant pairwise Wilcoxon tests after FDR correction.")


if __name__ == "__main__":
    main()