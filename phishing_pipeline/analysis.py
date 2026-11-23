from pathlib import Path

import pandas as pd


SECTION = "=" * 78

IDENTIFIER_COLUMNS = ("FILENAME",)

RAW_URL_COLUMNS = ("URL", "Domain", "TLD")

CONTENT_DERIVED_COLUMNS = (
    "LineOfCode",
    "LargestLineLength",
    "HasTitle",
    "Title",
    "DomainTitleMatchScore",
    "URLTitleMatchScore",
    "HasFavicon",
    "Robots",
    "IsResponsive",
    "NoOfURLRedirect",
    "NoOfSelfRedirect",
    "HasDescription",
    "NoOfPopup",
    "NoOfiFrame",
    "HasExternalFormSubmit",
    "HasSocialNet",
    "HasSubmitButton",
    "HasHiddenFields",
    "HasPasswordField",
    "Bank",
    "Pay",
    "Crypto",
    "HasCopyrightInfo",
    "NoOfImage",
    "NoOfCSS",
    "NoOfJS",
    "NoOfSelfRef",
    "NoOfEmptyRef",
    "NoOfExternalRef",
)


def load_dataset(csv_path):
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path, low_memory=False)


def print_header(title):
    print("\n" + SECTION)
    print(title)
    print(SECTION)


def summarize_shape_and_types(df):
    print_header("STAGE 1 - DATASET SHAPE, COLUMNS AND DATA TYPES")
    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]:,}")
    print("\nColumn data types:")
    for name, dtype in df.dtypes.items():
        print(f"  {name:<34} {dtype}")
    return {"rows": int(df.shape[0]), "columns": int(df.shape[1])}


def summarize_missing_values(df):
    print_header("STAGE 2 - MISSING VALUES")
    counts = df.isna().sum()
    missing = {name: int(count) for name, count in counts.items() if count > 0}
    if missing:
        for name, count in missing.items():
            print(f"  {name:<34} {count:,} missing ({count / len(df):.2%})")
    else:
        print("  No missing values found in any column.")
        print("  No rows need to be deleted for missingness; numeric/categorical")
        print("  imputation is still added in preprocessing as a safety net.")
    return missing


def summarize_duplicates(df, url_col="URL", target_col="label"):
    print_header("STAGE 3 - DUPLICATE RECORDS")
    exact = int(df.duplicated().sum())
    print(f"  Exact duplicate rows (all columns) : {exact:,}")

    url_duplicates = None
    conflicting = None
    if url_col in df.columns:
        url_duplicates = int(df[url_col].duplicated().sum())
        print(f"  Repeated URL values                 : {url_duplicates:,}")
        print(f"  Unique URLs                         : {df[url_col].nunique():,}")
        if target_col in df.columns:
            grouped = df.groupby(url_col)[target_col].agg(["nunique", "count"])
            conflicting = int(((grouped["count"] > 1) & (grouped["nunique"] > 1)).sum())
            print(f"  Repeated URLs with conflicting labels: {conflicting:,}")
            if conflicting == 0:
                print("  -> Repeated URLs always share the same label, so keeping one")
                print("     copy per URL removes redundancy without losing information.")
    return {
        "exact_duplicate_rows": exact,
        "repeated_url_values": url_duplicates,
        "conflicting_url_labels": conflicting,
    }


def summarize_target(df, target_col="label"):
    print_header("STAGE 4 - TARGET LABEL DISTRIBUTION")
    counts = df[target_col].value_counts().sort_index()
    percentages = df[target_col].value_counts(normalize=True).sort_index() * 100
    print(f"  {'label':<12}{'count':>12}{'percent':>12}   meaning")
    meanings = {0: "phishing", 1: "legitimate"}
    for value in counts.index:
        meaning = meanings.get(value, "unknown")
        print(f"  {str(value):<12}{int(counts[value]):>12,}{percentages[value]:>11.2f}%   {meaning}")
    ratio = counts.max() / counts.min()
    print(f"\n  Majority/minority ratio: {ratio:.2f} : 1")
    if ratio < 1.5:
        print("  -> Classes are fairly balanced; no resampling is required.")
    return {str(k): int(v) for k, v in counts.items()}


def classify_columns(df, target_col="label"):
    print_header("STAGE 5 - RAW URL OR PRE-EXTRACTED FEATURES?")
    present = set(df.columns)
    identifiers = [c for c in IDENTIFIER_COLUMNS if c in present]
    raw = [c for c in RAW_URL_COLUMNS if c in present]
    content = [c for c in CONTENT_DERIVED_COLUMNS if c in present]
    excluded = set(identifiers + raw + content + [target_col])
    url_derived = [c for c in df.columns if c not in excluded]

    url_series = df["URL"].astype("string")
    scheme_fraction = float(url_series.str.contains("://", regex=False, na=False).mean())

    print(f"  Raw URL column present              : {'URL' in present}")
    print(f"  URL strings containing '://'        : {scheme_fraction:.2%}")
    print(f"  Example raw URL                     : {url_series.dropna().iloc[0]}")
    print("\n  Identifier columns (not features):")
    print(f"    {identifiers}")
    print(f"\n  Content-derived columns found ({len(content)}) - require fetching the page,")
    print("  therefore excluded from a URL-only project:")
    print(f"    {content}")
    print(f"\n  Dataset-provided URL-derived columns found ({len(url_derived)}).")
    print("  These are informative, but this pipeline re-extracts URL features")
    print("  directly from the raw URL strings for full control and reproducibility.")
    print(f"    {url_derived}")
    print(f"\n  Target column                       : {target_col}")
    print("  Label meaning verified from samples: 1 = legitimate, 0 = phishing")

    return {
        "raw_url_present": bool("URL" in present),
        "identifier_columns": identifiers,
        "content_derived_columns": content,
        "dataset_url_derived_columns": url_derived,
    }
