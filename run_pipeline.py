import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse

from phishing_pipeline.analysis import (
    classify_columns,
    load_dataset,
    print_header,
    summarize_duplicates,
    summarize_missing_values,
    summarize_shape_and_types,
    summarize_target,
)
from phishing_pipeline.cleaning import clean_dataset, extract_hostname
from phishing_pipeline.features import FEATURE_ORDER, build_feature_frame, print_feature_documentation
from phishing_pipeline.preprocessing import split_and_preprocess


DEFAULT_CSV = Path(__file__).parent / "dataset" / "PhiUSIIL_Phishing_URL_Dataset.csv"
DEFAULT_OUTPUT = Path(__file__).parent / "outputs"


def save_matrix(matrix, path):
    if sparse.issparse(matrix):
        sparse.save_npz(str(path.with_suffix(".npz")), matrix)
        return path.with_suffix(".npz")
    dense_path = path.with_suffix(".npy")
    np.save(str(dense_path), matrix)
    return dense_path


def save_artifacts(result, metadata, output_dir):
    print_header("STAGE 9 - SAVED ARTIFACTS")
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor_path = output_dir / "preprocessor.joblib"
    joblib.dump(result["preprocessor"], preprocessor_path)
    print(f"  {preprocessor_path.name:<28} fitted preprocessing pipeline (train only)")

    train_path = save_matrix(result["x_train"], output_dir / "X_train")
    test_path = save_matrix(result["x_test"], output_dir / "X_test")
    print(f"  {train_path.name:<28} processed training feature matrix")
    print(f"  {test_path.name:<28} processed test feature matrix")

    result["y_train"].to_csv(output_dir / "y_train.csv", index=False)
    result["y_test"].to_csv(output_dir / "y_test.csv", index=False)
    print("  y_train.csv / y_test.csv         labels aligned with the matrices")

    indices_path = output_dir / "split_indices.npz"
    np.savez_compressed(indices_path, train=result["train_index"], test=result["test_index"])
    print(f"  {indices_path.name:<28} original row indices for both splits")

    names_path = output_dir / "processed_feature_names.json"
    names_path.write_text(
        json.dumps(result["preprocessor"].get_feature_names_out().tolist(), indent=2),
        encoding="utf-8",
    )
    print(f"  {names_path.name:<28} names of the processed output columns")

    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value)),
        encoding="utf-8",
    )
    print(f"  {metadata_path.name:<28} full analysis, cleaning and split report")
    print("\n  Model training is intentionally not performed in this script.")


def run(csv_path, output_dir, test_size, seed):
    df = load_dataset(csv_path)
    shape = summarize_shape_and_types(df)
    missing = summarize_missing_values(df)
    duplicates = summarize_duplicates(df)
    target_counts = summarize_target(df)
    column_types = classify_columns(df)

    cleaned, cleaning_report = clean_dataset(df)

    features = build_feature_frame(cleaned["URL"])
    print_feature_documentation()
    print(f"\n  Feature matrix shape from raw URLs: {features.shape}")
    print(f"  Features with any missing value   : {int(features.isna().sum().sum())}")
    print("  Feature count per URL is deterministic, so all rows have the same schema.")

    groups = cleaned["URL"].map(extract_hostname)
    empty_hosts = int((groups == "").sum())
    print(f"  URLs with unparseable hostname    : {empty_hosts}")

    result = split_and_preprocess(features, cleaned["label"], groups, test_size, seed)

    metadata = {
        "source_csv": str(csv_path),
        "dataset_shape": shape,
        "missing_values": missing,
        "duplicates": duplicates,
        "target_counts": target_counts,
        "column_classification": column_types,
        "cleaning": cleaning_report,
        "url_feature_columns": list(FEATURE_ORDER),
        "extracted_feature_shape": list(features.shape),
        "processed_shapes": {
            "train": list(result["x_train"].shape),
            "test": list(result["x_test"].shape),
        },
        "split": {
            "method": "StratifiedGroupKFold first fold",
            "groups": "hostname",
            "test_size_requested": test_size,
            "seed": seed,
            "hostname_overlap": result["hostname_overlap"],
        },
        "model_trained": False,
    }
    save_artifacts(result, metadata, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Clean and prepare URL-only phishing data; no model training.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to the phishing URL CSV dataset")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Directory for processed outputs")
    parser.add_argument("--test-size", type=float, default=0.2, help="Requested test-set fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the grouped stratified split")
    args = parser.parse_args()
    run(args.csv, args.output_dir, args.test_size, args.seed)


if __name__ == "__main__":
    main()
