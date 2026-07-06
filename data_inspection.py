from pathlib import Path
import pandas as pd
from contextlib import redirect_stdout


# This file is inside:
# eye_gaze_proj/scripts_eye_gaze_interaction_proj/data_inspection.py
SCRIPT_REPO = Path(__file__).resolve().parent

# Parent folder:
# eye_gaze_proj/
WORKSPACE_ROOT = SCRIPT_REPO.parent

# Dataset folder:
DATASET_ROOT = WORKSPACE_ROOT / "exp1-gaze-interaction-dataset"

# Output folder:
OUTPUT_DIR = WORKSPACE_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_DIR = DATASET_ROOT / "data"
RAW_FRAMES_DIR = DATA_DIR / "raw" / "frames"
RAW_TRIAL_SUMMARIES_DIR = DATA_DIR / "raw" / "trial_summaries"
PROCESSED_DIR = DATA_DIR / "processed"
DERIVED_DIR = DATA_DIR / "derived"
METADATA_DIR = DATASET_ROOT / "metadata"
DOCS_DIR = DATASET_ROOT / "docs"


def list_csvs(folder: Path, max_files: int = 20):
    print(f"\n=== {folder} ===")

    if not folder.exists():
        print("Folder does not exist.")
        return []

    files = sorted(folder.glob("*.csv"))
    print(f"Found {len(files)} CSV files.")

    for file in files[:max_files]:
        print(f" - {file.name}")

    if len(files) > max_files:
        print(f" ... {len(files) - max_files} more")

    return files


def inspect_csv(path: Path, nrows_preview: int = 5):
    print(f"\n\n--- Inspecting: {path} ---")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Could not read {path}: {e}")
        return None

    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

    print("\nColumns:")
    for col in df.columns:
        print(f" - {col}")

    print("\nDtypes:")
    print(df.dtypes.to_string())

    print("\nMissing values, top 20:")
    print(df.isna().sum().sort_values(ascending=False).head(20).to_string())

    print("\nPreview:")
    print(df.head(nrows_preview).to_string())

    return df


def write_inspection_report(output_path: Path):
    with output_path.open("w", encoding="utf-8") as f:
        with redirect_stdout(f):
            print("Eye Gaze Dataset Inspection Report")
            print("=" * 40)

            print(f"Script repo: {SCRIPT_REPO}")
            print(f"Workspace root: {WORKSPACE_ROOT}")
            print(f"Dataset root: {DATASET_ROOT}")
            print(f"Output dir: {OUTPUT_DIR}")

            frame_files = list_csvs(RAW_FRAMES_DIR)
            trial_summary_files = list_csvs(RAW_TRIAL_SUMMARIES_DIR)
            processed_files = list_csvs(PROCESSED_DIR)
            derived_files = list_csvs(DERIVED_DIR)
            metadata_files = list_csvs(METADATA_DIR)

            # Inspect one raw frame file first, since these may be large.
            if frame_files:
                inspect_csv(frame_files[0])

            # Inspect one raw trial summary file.
            if trial_summary_files:
                inspect_csv(trial_summary_files[0])

            # Inspect processed, derived, and metadata tables.
            for file in processed_files:
                inspect_csv(file)

            for file in derived_files:
                inspect_csv(file)

            for file in metadata_files:
                inspect_csv(file)

            # Field dictionary.
            field_dict = DOCS_DIR / "field_dictionary.csv"
            if field_dict.exists():
                inspect_csv(field_dict)
            else:
                print(f"\nNo field dictionary found at: {field_dict}")


def write_file_list(output_path: Path):
    with output_path.open("w", encoding="utf-8") as f:
        with redirect_stdout(f):
            print("CSV File List")
            print("=" * 40)

            folders = [
                RAW_FRAMES_DIR,
                RAW_TRIAL_SUMMARIES_DIR,
                PROCESSED_DIR,
                DERIVED_DIR,
                METADATA_DIR,
                DOCS_DIR,
            ]

            for folder in folders:
                print(f"\n=== {folder} ===")
                if not folder.exists():
                    print("Folder does not exist.")
                    continue

                files = sorted(folder.glob("*.csv"))
                print(f"Found {len(files)} CSV files.")
                for file in files:
                    print(file)


def main():
    full_report_path = OUTPUT_DIR / "data_inspection_report.txt"
    file_list_path = OUTPUT_DIR / "csv_file_list.txt"

    write_inspection_report(full_report_path)
    write_file_list(file_list_path)

    print(f"Saved full inspection report to: {full_report_path}")
    print(f"Saved CSV file list to: {file_list_path}")


if __name__ == "__main__":
    main()