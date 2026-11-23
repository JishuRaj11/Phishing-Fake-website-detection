import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlsplit

import pandas as pd

from .analysis import print_header


SUSPICIOUS_KEYWORDS = (
    "login", "logon", "signin", "sign-in", "verify", "verification",
    "account", "secure", "security", "update", "confirm", "password",
    "passwd", "bank", "banking", "paypal", "payment", "pay", "invoice",
    "crypto", "wallet", "free", "gift", "prize", "reward", "urgent",
    "click", "bonus",
)

URL_SHORTENERS = (
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "is.gd", "ow.ly",
    "buff.ly", "cutt.ly", "rebrand.ly", "adf.ly", "shorte.st", "s.id",
    "tiny.cc", "cli.gs", "j.mp", "rb.gy", "lnkd.in",
)

HEX_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")

FEATURE_DESCRIPTIONS = {
    "url_length": "Total number of characters in the raw URL string; very long URLs are often used to hide the real destination.",
    "hostname_length": "Number of characters in the parsed hostname (domain); long hostnames can indicate randomly generated phishing domains.",
    "path_length": "Number of characters in the URL path; long paths can hide keywords or tracking tokens.",
    "query_length": "Number of characters in the query string; phishing links often carry long tracking or credential parameters.",
    "fragment_length": "Number of characters after '#'; fragments can be used to disguise the visible link target.",
    "path_depth": "Count of non-empty path segments separated by '/', a measure of how deeply nested the URL is.",
    "num_dots": "Number of '.' characters in the whole URL; many dots can signal stacked or deceptive subdomains.",
    "num_hyphens": "Number of '-' characters in the whole URL; hyphen-heavy domains are common in look-alike phishing sites.",
    "num_underscores": "Number of '_' characters in the whole URL; unusual in normal domains and often used in generated URLs.",
    "num_slashes": "Number of '/' characters in the whole URL; extreme values are associated with redirects or deep paths.",
    "num_question_marks": "Number of '?' characters; more than one is abnormal and may indicate URL parsing tricks.",
    "num_ampersands": "Number of '&' characters, which separate query parameters and can reveal excessive tracking data.",
    "num_equals": "Number of '=' characters, another indicator of parameter-rich query strings.",
    "num_at": "Number of '@' characters anywhere in the URL; '@' in the authority can hide the real host from users.",
    "num_percent": "Number of '%' characters, a sign of percent-encoded or obfuscated characters.",
    "num_hash": "Number of '#' characters in the URL; repeated fragments are uncommon in legitimate links.",
    "num_letters": "Total alphabetic characters in the URL.",
    "num_digits": "Total numeric characters in the URL; many digits often appear in generated or IP-based URLs.",
    "letter_ratio": "Fraction of URL characters that are letters (num_letters / url_length).",
    "digit_ratio": "Fraction of URL characters that are digits (num_digits / url_length).",
    "num_special_chars": "Count of characters that are neither letters nor digits (punctuation and symbols).",
    "special_char_ratio": "Fraction of URL characters that are non-alphanumeric (num_special_chars / url_length).",
    "num_query_params": "Number of query parameters, estimated as '&' count plus one when a query string exists.",
    "is_https": "1 if the URL scheme is 'https', else 0; HTTPS is a weak but useful trust signal.",
    "is_http": "1 if the URL scheme is 'http', else 0; plain HTTP is common among phishing pages.",
    "has_port": "1 if an explicit network port is present (for example ':8080'), which is unusual in normal links.",
    "is_ip_address": "1 if the hostname is a literal IPv4 or IPv6 address; direct-IP hosting is a strong phishing indicator.",
    "has_at_in_netloc": "1 if '@' appears in the authority section (userinfo trick that obscures the true hostname).",
    "has_double_slash_in_path": "1 if '//' appears inside the path, a known trick for injecting an apparent redirect.",
    "subdomain_count": "Number of hostname labels minus the registered-name and TLD labels; many subdomains can be deceptive.",
    "has_www": "1 if the hostname starts with 'www', a common pattern for conventional legitimate sites.",
    "hostname_digit_count": "Number of digits inside the hostname; numeric domains are frequently auto-generated.",
    "hostname_hyphen_count": "Number of hyphens inside the hostname; used to imitate trusted brand names.",
    "max_label_length": "Length of the longest single hostname label, capturing unusually long generated subdomains.",
    "tld_length": "Number of characters in the top-level domain label (for example, 3 for 'com').",
    "hostname_entropy": "Shannon entropy of hostname characters; high entropy suggests random or machine-generated names.",
    "url_entropy": "Shannon entropy of all URL characters; high entropy suggests encoded or randomized links.",
    "percent_encoded_count": "Number of percent-encoded byte sequences such as '%2F'; commonly used to obfuscate URLs.",
    "has_punycode": "1 if the hostname contains 'xn--', indicating an internationalized domain that can look like a brand.",
    "suspicious_keyword_count": "Count of phishing-related words found in the URL (login, verify, bank, password, free, and similar).",
    "has_shortener": "1 if the hostname matches a list of well-known URL shorteners, which mask the final destination.",
    "has_fragment": "1 if the URL contains a non-empty fragment after '#'.",
    "tld": "Top-level domain string (for example 'com'); kept as a categorical feature and one-hot encoded.",
    "scheme": "URL scheme string (http, https, and so on); kept as a categorical feature and one-hot encoded.",
}

FEATURE_ORDER = tuple(FEATURE_DESCRIPTIONS.keys())
CATEGORICAL_FEATURES = ("tld", "scheme")
NUMERIC_FEATURES = tuple(name for name in FEATURE_ORDER if name not in CATEGORICAL_FEATURES)


def shannon_entropy(text):
    if not text:
        return 0.0
    length = len(text)
    frequencies = Counter(text)
    return -sum((count / length) * math.log2(count / length) for count in frequencies.values())


def _safe_port(parts):
    try:
        return parts.port
    except ValueError:
        return None


def extract_url_features(url):
    raw = str(url).strip()
    candidate = raw if "://" in raw else "http://" + raw
    try:
        parts = urlsplit(candidate)
        hostname = (parts.hostname or "").lower()
        path = parts.path or ""
        query = parts.query or ""
        fragment = parts.fragment or ""
        netloc = parts.netloc or ""
        scheme = (parts.scheme or "").lower()
        port = _safe_port(parts)
    except ValueError:
        hostname, path, query, fragment, netloc, scheme, port = "", "", "", "", "", "", None

    lowered = raw.lower()
    labels = [label for label in hostname.split(".") if label]
    tld = labels[-1] if labels else "unknown"

    num_letters = sum(character.isalpha() for character in raw)
    num_digits = sum(character.isdigit() for character in raw)
    num_special = sum(not character.isalnum() for character in raw)
    url_length = len(raw)

    try:
        is_ip_address = int(ipaddress.ip_address(hostname) is not None)
    except ValueError:
        is_ip_address = 0

    return {
        "url_length": url_length,
        "hostname_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),
        "fragment_length": len(fragment),
        "path_depth": len([segment for segment in path.split("/") if segment]),
        "num_dots": raw.count("."),
        "num_hyphens": raw.count("-"),
        "num_underscores": raw.count("_"),
        "num_slashes": raw.count("/"),
        "num_question_marks": raw.count("?"),
        "num_ampersands": raw.count("&"),
        "num_equals": raw.count("="),
        "num_at": raw.count("@"),
        "num_percent": raw.count("%"),
        "num_hash": raw.count("#"),
        "num_letters": num_letters,
        "num_digits": num_digits,
        "letter_ratio": num_letters / url_length if url_length else 0.0,
        "digit_ratio": num_digits / url_length if url_length else 0.0,
        "num_special_chars": num_special,
        "special_char_ratio": num_special / url_length if url_length else 0.0,
        "num_query_params": (query.count("&") + 1) if query else 0,
        "is_https": int(scheme == "https"),
        "is_http": int(scheme == "http"),
        "has_port": int(port is not None),
        "is_ip_address": is_ip_address,
        "has_at_in_netloc": int("@" in netloc),
        "has_double_slash_in_path": int("//" in path),
        "subdomain_count": max(len(labels) - 2, 0),
        "has_www": int(bool(labels) and labels[0] == "www"),
        "hostname_digit_count": sum(character.isdigit() for character in hostname),
        "hostname_hyphen_count": hostname.count("-"),
        "max_label_length": max((len(label) for label in labels), default=0),
        "tld_length": len(tld),
        "hostname_entropy": shannon_entropy(hostname),
        "url_entropy": shannon_entropy(raw),
        "percent_encoded_count": len(HEX_ESCAPE.findall(raw)),
        "has_punycode": int("xn--" in hostname),
        "suspicious_keyword_count": sum(keyword in lowered for keyword in SUSPICIOUS_KEYWORDS),
        "has_shortener": int(any(hostname == name or hostname.endswith("." + name) for name in URL_SHORTENERS)),
        "has_fragment": int(bool(fragment)),
        "tld": tld,
        "scheme": scheme,
    }


def build_feature_frame(urls):
    records = [extract_url_features(url) for url in urls]
    frame = pd.DataFrame.from_records(records, columns=FEATURE_ORDER)
    return frame


def print_feature_documentation():
    print_header("STAGE 7 - URL FEATURES EXTRACTED (no network calls, URL text only)")
    print(f"  Total features: {len(FEATURE_ORDER)} "
          f"({len(NUMERIC_FEATURES)} numeric, {len(CATEGORICAL_FEATURES)} categorical)")
    for index, (name, description) in enumerate(FEATURE_DESCRIPTIONS.items(), start=1):
        print(f"\n  {index:>2}. {name}")
        print(f"      {description}")
