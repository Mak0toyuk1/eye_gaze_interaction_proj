from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.formula.api as smf


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

BLOCK_SUMMARY_PATH = (
    DATASET_ROOT / "data" / "processed" / "balanced_subject_block_summary.csv"
)

OUT_MERGED = OUTPUT_DIR / "te_accuracy_direct_merged.csv"
OUT_SINGLE = OUTPUT_DIR / "te_accuracy_single_direction_models.csv"
OUT_JOINT = OUTPUT_DIR / "te_accuracy_joint_direction_models.csv"
OUT_CONTRAST = OUTPUT_DIR / "te_accuracy_directional_contrasts.csv"
OUT_SUMMARY = OUTPUT_DIR / "te_accuracy_model_summaries.txt"


# ------------------------------------------------------------
# Analysis configuration
# ------------------------------------------------------------

TE_PAIRS = {
    "raw": {
        "head_to_eye": "head_to_eye_te",
        "eye_to_head": "eye_to_head_te",
    },
    "corrected": {
        "head_to_eye": "head_to_eye_corrected_te",
        "eye_to_head": "eye_to_head_corrected_te",
    },
}

MERGE_CANDIDATES = [
    ["condition_block_key"],
    ["participant_id", "layout", "tracking", "block_index"],
    ["participant_id", "layout", "tracking"],
]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def choose_merge_keys(te_df, block_df):
    for keys in MERGE_CANDIDATES:
        if all(k in te_df.columns for k in keys) and all(k in block_df.columns for k in keys):
            return keys
    raise KeyError("Could not find compatible merge keys.")


def coalesce(df, name):
    x = f"{name}_x"
    y = f"{name}_y"

    if x in df.columns and y in df.columns:
        df[name] = df[x].combine_first(df[y])
        df = df.drop(columns=[x, y])
    elif x in df.columns:
        df = df.rename(columns={x: name})
    elif y in df.columns:
        df = df.rename(columns={y: name})

    return df


def zscore(series):
    series = pd.to_numeric(series, errors="coerce")
    sd = series.std(ddof=1)

    if pd.isna(sd) or np.isclose(sd, 0):
        return pd.Series(np.nan, index=series.index)

    return (series - series.mean()) / sd


def load_and_merge():
    te_df = pd.read_csv(TE_BLOCK_PATH, low_memory=False)
    block_df = pd.read_csv(BLOCK_SUMMARY_PATH, low_memory=False)

    if "status" in te_df.columns:
        te_df = te_df[te_df["status"].eq("ok")].copy()

    required_te = sorted(
        {v for pair in TE_PAIRS.values() for v in pair.values()}
    )

    missing = [c for c in required_te if c not in te_df.columns]
    if missing:
        raise KeyError(f"Missing TE columns: {missing}")

    if "accuracy" not in block_df.columns:
        if "correct_count" not in block_df.columns:
            raise KeyError("Need accuracy or correct_count in block summary.")

        if "n_trials" in block_df.columns:
            denom = pd.to_numeric(block_df["n_trials"], errors="coerce")
        else:
            denom = 12.0

        block_df["accuracy"] = (
            pd.to_numeric(block_df["correct_count"], errors="coerce") / denom
        )

    merge_keys = choose_merge_keys(te_df, block_df)
    print("Using merge keys:", merge_keys)

    keep = merge_keys.copy()
    for c in [
        "participant_id",
        "tracking",
        "layout",
        "block_index",
        "accuracy",
        "correct_count",
        "n_trials",
    ]:
        if c in block_df.columns and c not in keep:
            keep.append(c)

    block_small = block_df[keep].drop_duplicates()

    df = te_df.merge(
        block_small,
        on=merge_keys,
        how="left",
        validate="many_to_one",
    )

    for c in [
        "participant_id",
        "tracking",
        "layout",
        "block_index",
        "accuracy",
        "correct_count",
        "n_trials",
    ]:
        df = coalesce(df, c)

    df["participant_id"] = df["participant_id"].astype(str)
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")

    for metric in required_te:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    if "tracking" in df.columns:
        df["tracking"] = df["tracking"].astype(str)

    if "layout" in df.columns:
        df["layout"] = df["layout"].astype(str)

    return df


def add_within_between(df, metric):
    """
    Separate the TE-performance relationship into:

    WITHIN participant:
        Is a participant more/less accurate on blocks where TE is
        above/below that participant's own usual TE?

    BETWEEN participants:
        Do participants with higher mean TE tend to be more/less accurate?
    """
    df = df.copy()

    mean_col = f"{metric}_participant_mean"
    within_col = f"{metric}_within"
    between_col = f"{metric}_between"

    df[mean_col] = df.groupby("participant_id")[metric].transform("mean")
    df[within_col] = df[metric] - df[mean_col]

    participant_means = (
        df[["participant_id", mean_col]]
        .drop_duplicates("participant_id")[mean_col]
    )

    grand_mean = participant_means.mean()
    between_sd = participant_means.std(ddof=1)

    df[between_col] = df[mean_col] - grand_mean

    within_z = f"{within_col}_z"
    between_z = f"{between_col}_z"

    df[within_z] = zscore(df[within_col])

    if pd.isna(between_sd) or np.isclose(between_sd, 0):
        df[between_z] = np.nan
    else:
        df[between_z] = df[between_col] / between_sd

    return df, within_z, between_z


def adjustment_rhs(df):
    terms = []

    if "tracking" in df.columns and df["tracking"].nunique(dropna=True) > 1:
        terms.append("C(tracking)")

    if "layout" in df.columns and df["layout"].nunique(dropna=True) > 1:
        terms.append("C(layout)")

    return " + ".join(terms)


def fit_model(df, formula):
    model = smf.mixedlm(
        formula,
        data=df,
        groups=df["participant_id"],
        re_formula="1",
    )

    try:
        return model.fit(
            reml=False,
            method="lbfgs",
            maxiter=3000,
            disp=False,
        )
    except Exception:
        return model.fit(
            reml=False,
            method="powell",
            maxiter=3000,
            disp=False,
        )


def extract_effect(result, term, metadata):
    ci = result.conf_int()

    return {
        **metadata,
        "term": term,
        "coefficient": result.params.get(term, np.nan),
        "std_error": result.bse.get(term, np.nan),
        "p_value": result.pvalues.get(term, np.nan),
        "ci95_low": ci.loc[term, 0] if term in ci.index else np.nan,
        "ci95_high": ci.loc[term, 1] if term in ci.index else np.nan,
        "converged": bool(getattr(result, "converged", True)),
    }


def formal_directional_contrast(result, h2e_term, e2h_term, family, effect_type, n, npart):
    """
    Tests beta(H->E) - beta(E->H) = 0.

    This is the correct way to test whether the two directional
    TE-performance relationships differ.
    """
    b = result.params
    cov = result.cov_params()

    estimate = b[h2e_term] - b[e2h_term]

    variance = (
        cov.loc[h2e_term, h2e_term]
        + cov.loc[e2h_term, e2h_term]
        - 2 * cov.loc[h2e_term, e2h_term]
    )

    if variance <= 0:
        se = z = p = np.nan
    else:
        se = np.sqrt(variance)
        z = estimate / se
        p = 2 * norm.sf(abs(z))

    return {
        "te_family": family,
        "effect_type": effect_type,
        "contrast": "head_to_eye_minus_eye_to_head",
        "estimate": estimate,
        "std_error": se,
        "z_value": z,
        "p_value": p,
        "n_blocks": n,
        "n_participants": npart,
    }


# ------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_merge()

    single_rows = []
    joint_rows = []
    contrast_rows = []
    text_summaries = []

    adjust = adjustment_rhs(df)

    # --------------------------------------------------------
    # 1. Single-direction models
    # --------------------------------------------------------
    #
    # accuracy ~ TE_within + TE_between + tracking + layout
    #            + participant random intercept
    #
    # These answer whether each direction is directly related
    # to performance.
    # --------------------------------------------------------

    for family, pair in TE_PAIRS.items():
        for direction, metric in pair.items():
            temp, within_z, between_z = add_within_between(df, metric)

            needed = [
                "participant_id",
                "accuracy",
                within_z,
                between_z,
            ]

            if "tracking" in temp.columns:
                needed.append("tracking")
            if "layout" in temp.columns:
                needed.append("layout")

            model_df = temp[needed].dropna().copy()

            rhs = f"{within_z} + {between_z}"
            if adjust:
                rhs += f" + {adjust}"

            formula = f"accuracy ~ {rhs}"

            print("\nFitting:", family, direction)
            print(formula)

            result = fit_model(model_df, formula)

            meta = {
                "model": "single_direction",
                "te_family": family,
                "direction": direction,
                "n_blocks": len(model_df),
                "n_participants": model_df["participant_id"].nunique(),
            }

            row = extract_effect(result, within_z, meta)
            row["effect"] = "within_participant_TE"
            single_rows.append(row)

            row = extract_effect(result, between_z, meta)
            row["effect"] = "between_participant_TE"
            single_rows.append(row)

            text_summaries.append(
                "\n".join(
                    [
                        "=" * 80,
                        f"SINGLE DIRECTION | {family} | {direction}",
                        f"Formula: {formula}",
                        "",
                        result.summary().as_text(),
                    ]
                )
            )

    # --------------------------------------------------------
    # 2. Joint-direction models
    # --------------------------------------------------------
    #
    # Put H->E and E->H in the SAME model.
    #
    # This asks whether H->E predicts accuracy while controlling
    # for E->H, and vice versa.
    #
    # It also permits the formal H->E vs E->H contrast.
    # --------------------------------------------------------

    for family, pair in TE_PAIRS.items():
        temp, h2e_w, h2e_b = add_within_between(
            df, pair["head_to_eye"]
        )
        temp, e2h_w, e2h_b = add_within_between(
            temp, pair["eye_to_head"]
        )

        needed = [
            "participant_id",
            "accuracy",
            h2e_w,
            h2e_b,
            e2h_w,
            e2h_b,
        ]

        if "tracking" in temp.columns:
            needed.append("tracking")
        if "layout" in temp.columns:
            needed.append("layout")

        model_df = temp[needed].dropna().copy()

        rhs = f"{h2e_w} + {e2h_w} + {h2e_b} + {e2h_b}"
        if adjust:
            rhs += f" + {adjust}"

        formula = f"accuracy ~ {rhs}"

        print("\nFitting joint model:", family)
        print(formula)

        result = fit_model(model_df, formula)

        n = len(model_df)
        npart = model_df["participant_id"].nunique()

        terms = {
            "H2E_within_participant_TE": h2e_w,
            "E2H_within_participant_TE": e2h_w,
            "H2E_between_participant_TE": h2e_b,
            "E2H_between_participant_TE": e2h_b,
        }

        for effect, term in terms.items():
            meta = {
                "model": "joint_direction",
                "te_family": family,
                "direction": "both",
                "effect": effect,
                "n_blocks": n,
                "n_participants": npart,
            }

            joint_rows.append(
                extract_effect(result, term, meta)
            )

        contrast_rows.append(
            formal_directional_contrast(
                result,
                h2e_w,
                e2h_w,
                family,
                "within_participant",
                n,
                npart,
            )
        )

        contrast_rows.append(
            formal_directional_contrast(
                result,
                h2e_b,
                e2h_b,
                family,
                "between_participant",
                n,
                npart,
            )
        )

        text_summaries.append(
            "\n".join(
                [
                    "=" * 80,
                    f"JOINT DIRECTIONS | {family}",
                    f"Formula: {formula}",
                    "",
                    result.summary().as_text(),
                ]
            )
        )

    single_df = pd.DataFrame(single_rows)
    joint_df = pd.DataFrame(joint_rows)
    contrast_df = pd.DataFrame(contrast_rows)

    df.to_csv(OUT_MERGED, index=False)
    single_df.to_csv(OUT_SINGLE, index=False)
    joint_df.to_csv(OUT_JOINT, index=False)
    contrast_df.to_csv(OUT_CONTRAST, index=False)

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(
            "DIRECT TE-ACCURACY ANALYSIS\n"
            "===========================\n\n"
            "Primary interpretation:\n"
            "1. within_participant_TE tests whether a participant is more or less\n"
            "   accurate on blocks where their TE is higher than their own normal TE.\n"
            "2. between_participant_TE tests whether participants with higher mean TE\n"
            "   tend to have higher or lower overall accuracy.\n"
            "3. Tracking and layout are included as fixed effects when available.\n"
            "4. Participant is included as a random intercept.\n"
            "5. The joint-direction model tests H->E and E->H simultaneously.\n"
            "6. The directional contrast formally tests whether the H->E coefficient\n"
            "   differs from the E->H coefficient.\n\n"
        )

        for summary in text_summaries:
            f.write(summary)
            f.write("\n\n")

    print("\nSaved:")
    print(OUT_MERGED)
    print(OUT_SINGLE)
    print(OUT_JOINT)
    print(OUT_CONTRAST)
    print(OUT_SUMMARY)

    print("\nPRIMARY RESULTS: single-direction within-participant TE")
    print(
        single_df[
            single_df["effect"].eq("within_participant_TE")
        ][
            [
                "te_family",
                "direction",
                "coefficient",
                "std_error",
                "p_value",
                "ci95_low",
                "ci95_high",
                "n_blocks",
                "n_participants",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6g}")
    )

    print("\nJOINT MODEL: within-participant directional effects")
    print(
        joint_df[
            joint_df["effect"].isin(
                [
                    "H2E_within_participant_TE",
                    "E2H_within_participant_TE",
                ]
            )
        ][
            [
                "te_family",
                "effect",
                "coefficient",
                "std_error",
                "p_value",
                "ci95_low",
                "ci95_high",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6g}")
    )

    print("\nFORMAL H->E VS E->H CONTRASTS")
    print(
        contrast_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}",
        )
    )


if __name__ == "__main__":
    main()
