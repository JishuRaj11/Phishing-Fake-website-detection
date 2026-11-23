from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from .analysis import print_header
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def stratified_group_split(features, labels, groups, test_size=0.2, seed=42):
    n_splits = int(round(1.0 / test_size))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_index, test_index = next(splitter.split(features, labels, groups))
    return train_index, test_index


def build_preprocessor():
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=True)),
    ])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def split_and_preprocess(features, labels, groups, test_size=0.2, seed=42):
    print_header("STAGE 8 - LEAKAGE-SAFE SPLIT AND PREPROCESSING")
    train_index, test_index = stratified_group_split(features, labels, groups, test_size, seed)

    train_groups = set(groups.iloc[train_index])
    test_groups = set(groups.iloc[test_index])
    overlap = len(train_groups & test_groups)

    print(f"  Split method                     : StratifiedGroupKFold, first fold")
    print(f"  Grouping key                     : hostname (same site cannot appear in both sets)")
    print(f"  Random seed                      : {seed}")
    print(f"  Train rows                       : {len(train_index):,}")
    print(f"  Test rows                        : {len(test_index):,}")
    print(f"  Train share                      : {len(train_index) / len(features):.2%}")
    print(f"  Hostnames shared train/test      : {overlap} (must be 0)")
    print(f"  Train label 0 / 1                : "
          f"{(labels.iloc[train_index] == 0).sum():,} / {(labels.iloc[train_index] == 1).sum():,}")
    print(f"  Test  label 0 / 1                : "
          f"{(labels.iloc[test_index] == 0).sum():,} / {(labels.iloc[test_index] == 1).sum():,}")
    print("\n  Preprocessing fitted on TRAIN ONLY:")
    print("    numeric   -> median imputation + RobustScaler (resistant to extreme URL lengths)")
    print("    tld/scheme -> constant imputation + one-hot encoding (rare TLDs grouped, unknown ignored)")
    print("  The test set is only transformed, never used for fitting scalers or encoders.")

    preprocessor = build_preprocessor()
    x_train = features.iloc[train_index]
    x_test = features.iloc[test_index]
    y_train = labels.iloc[train_index]
    y_test = labels.iloc[test_index]

    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)

    print(f"\n  Processed train matrix            : {x_train_processed.shape}")
    print(f"  Processed test matrix             : {x_test_processed.shape}")
    print(f"  Output feature count              : {len(preprocessor.get_feature_names_out())}")
    print("  No machine-learning model has been fitted.")

    return {
        "preprocessor": preprocessor,
        "train_index": train_index,
        "test_index": test_index,
        "x_train": x_train_processed,
        "x_test": x_test_processed,
        "y_train": y_train,
        "y_test": y_test,
        "hostname_overlap": overlap,
    }
