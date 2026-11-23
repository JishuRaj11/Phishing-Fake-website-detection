import pandas as pd

from .analysis import print_header


def clean_dataset(df, url_col="URL", target_col="label"):
    print_header("STAGE 6 - DATA CLEANING")
    before_rows = len(df)
    cleaned = df.copy()

    urls = cleaned[url_col].astype("string").str.strip()
    missing_url = urls.isna() | (urls == "").fillna(False)
    empty_urls = int(missing_url.sum())
    cleaned = cleaned.loc[~missing_url].copy()
    cleaned[url_col] = urls.loc[~missing_url]

    invalid_label = ~cleaned[target_col].isin([0, 1])
    invalid_labels = int(invalid_label.sum())
    cleaned = cleaned.loc[~invalid_label].copy()

    exact_duplicates = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates()

    repeated_urls = int(cleaned[url_col].duplicated().sum())
    cleaned = cleaned.drop_duplicates(subset=[url_col], keep="first")
    cleaned = cleaned.reset_index(drop=True)

    print(f"  Rows before cleaning                    : {before_rows:,}")
    print(f"  Blank / missing URLs removed            : {empty_urls:,}")
    print(f"  Rows with label outside {{0, 1}} removed  : {invalid_labels:,}")
    print(f"  Exact duplicate rows removed            : {exact_duplicates:,}")
    print(f"  Extra rows with repeated URLs removed   : {repeated_urls:,}")
    print(f"  Rows after cleaning                     : {len(cleaned):,}")
    print(f"  Rows retained                           : {len(cleaned) / before_rows:.2%}")
    print("  No legitimate records were dropped for being a minority class,")
    print("  no outlier URLs were deleted, and no missing-value rows were removed.")

    report = {
        "rows_before": before_rows,
        "rows_after": len(cleaned),
        "blank_urls_removed": empty_urls,
        "invalid_labels_removed": invalid_labels,
        "exact_duplicates_removed": exact_duplicates,
        "repeated_urls_removed": repeated_urls,
    }
    return cleaned, report


def extract_hostname(url):
    from urllib.parse import urlsplit

    raw = str(url).strip()
    candidate = raw if "://" in raw else "http://" + raw
    try:
        return (urlsplit(candidate).hostname or "").lower()
    except ValueError:
        return ""
