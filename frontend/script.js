const API_URL = "https://phishing-guard-ai-backend.onrender.com/predict";

const FEATURE_META = [
  { key: "url_length", label: "URL length", note: "Total characters in the submitted URL string." },
  { key: "hostname_length", label: "Hostname length", note: "Number of characters in the parsed hostname." },
  { key: "path_length", label: "Path length", note: "Number of characters in the URL path." },
  { key: "query_length", label: "Query length", note: "Number of characters in the query string." },
  { key: "fragment_length", label: "Fragment length", note: "Number of characters after '#'." },
  { key: "path_depth", label: "Path depth", note: "Count of non-empty path segments separated by '/'." },
  { key: "num_dots", label: "Number of dots", note: "Count of '.' characters in the whole URL." },
  { key: "num_hyphens", label: "Number of hyphens", note: "Count of '-' characters in the whole URL." },
  { key: "num_underscores", label: "Number of underscores", note: "Count of '_' characters in the whole URL." },
  { key: "num_slashes", label: "Number of slashes", note: "Count of '/' characters in the whole URL." },
  { key: "num_question_marks", label: "Number of question marks", note: "Count of '?' characters in the URL." },
  { key: "num_ampersands", label: "Number of ampersands", note: "Count of '&' characters separating query parameters." },
  { key: "num_equals", label: "Number of equals signs", note: "Count of '=' characters in the URL." },
  { key: "num_at", label: "Number of @ symbols", note: "Count of '@' characters in the URL." },
  { key: "num_percent", label: "Number of percent signs", note: "Count of '%' characters in the URL." },
  { key: "num_hash", label: "Number of hash symbols", note: "Count of '#' characters in the URL." },
  { key: "num_letters", label: "Number of letters", note: "Count of alphabetic characters in the URL." },
  { key: "num_digits", label: "Number of digits", note: "Count of numeric characters in the URL." },
  { key: "letter_ratio", label: "Letter ratio", note: "Fraction of URL characters that are letters." },
  { key: "digit_ratio", label: "Digit ratio", note: "Fraction of URL characters that are digits." },
  { key: "num_special_chars", label: "Special character count", note: "Count of characters that are not letters or digits." },
  { key: "special_char_ratio", label: "Special character ratio", note: "Fraction of URL characters that are not letters or digits." },
  { key: "num_query_params", label: "Number of query parameters", note: "Number of query parameters estimated from the query string." },
  { key: "is_https", label: "HTTPS usage", note: "1 = yes, 0 = no. Whether the parsed scheme is https." },
  { key: "is_http", label: "HTTP usage", note: "1 = yes, 0 = no. Whether the parsed scheme is http." },
  { key: "has_port", label: "Explicit port", note: "1 = yes, 0 = no. Whether an explicit port is present." },
  { key: "is_ip_address", label: "IP address usage", note: "1 = yes, 0 = no. Whether the hostname is a literal IP address." },
  { key: "has_at_in_netloc", label: "@ in authority", note: "1 = yes, 0 = no. Whether '@' appears in the authority section." },
  { key: "has_double_slash_in_path", label: "Double slash in path", note: "1 = yes, 0 = no. Whether '//' appears in the path." },
  { key: "subdomain_count", label: "Number of subdomains", note: "Number of hostname labels beyond the registered name and TLD." },
  { key: "has_www", label: "www prefix", note: "1 = yes, 0 = no. Whether the hostname starts with 'www'." },
  { key: "hostname_digit_count", label: "Digits in hostname", note: "Number of digits inside the hostname." },
  { key: "hostname_hyphen_count", label: "Hyphens in hostname", note: "Number of hyphens inside the hostname." },
  { key: "max_label_length", label: "Longest hostname label", note: "Length of the longest hostname label." },
  { key: "tld_length", label: "TLD length", note: "Number of characters in the top-level domain label." },
  { key: "hostname_entropy", label: "Hostname entropy", note: "Shannon entropy of the hostname characters." },
  { key: "url_entropy", label: "URL entropy", note: "Shannon entropy of the URL characters." },
  { key: "percent_encoded_count", label: "Percent-encoded sequences", note: "Count of percent-encoded sequences such as '%2F'." },
  { key: "has_punycode", label: "Punycode detection", note: "1 = yes, 0 = no. Whether the hostname contains 'xn--'." },
  { key: "suspicious_keyword_count", label: "Suspicious keywords", note: "Count of words from the configured keyword list found in the URL text." },
  { key: "has_shortener", label: "URL shortener detection", note: "1 = yes, 0 = no. Whether the hostname matches the known shortener list." },
  { key: "has_fragment", label: "Fragment present", note: "1 = yes, 0 = no. Whether a non-empty fragment is present." },
  { key: "tld", label: "Top-level domain", note: "Top-level domain string extracted from the hostname." },
  { key: "scheme", label: "URL scheme", note: "URL scheme string extracted from the URL." }
];

const urlForm = document.getElementById("urlForm");
const urlInput = document.getElementById("urlInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const analyzeBtnText = document.getElementById("analyzeBtnText");
const clearBtn = document.getElementById("clearBtn");
const formError = document.getElementById("formError");
const loadingStatus = document.getElementById("loadingStatus");
const result = document.getElementById("result");
const resultBanner = document.getElementById("resultBanner");
const resultTitle = document.getElementById("resultTitle");
const resultUrl = document.getElementById("resultUrl");
const predictionPill = document.getElementById("predictionPill");
const confidenceValue = document.getElementById("confidenceValue");
const featureCount = document.getElementById("featureCount");
const featuresGrid = document.getElementById("featuresGrid");

function showError(message) {
  formError.textContent = message;
  formError.hidden = false;
}

function clearError() {
  formError.textContent = "";
  formError.hidden = true;
}

function validateUrl(value) {
  if (!value) {
    return "Please enter a website URL.";
  }
  if (value.length > 2048) {
    return "URL must be at most 2048 characters.";
  }
  if (/[\u0000-\u001f\u007f]/.test(value)) {
    return "URL must not contain control characters.";
  }
  if (/\s/.test(value)) {
    return "URL must not contain spaces or line breaks.";
  }

  const schemeMatch = value.match(/^([A-Za-z][A-Za-z0-9+.-]*):\/\//);
  if (schemeMatch && !["http", "https"].includes(schemeMatch[1].toLowerCase())) {
    return "Only http and https URLs are supported.";
  }

  return "";
}

function setLoading(isLoading) {
  analyzeBtn.disabled = isLoading;
  clearBtn.disabled = isLoading;
  urlInput.disabled = isLoading;
  analyzeBtn.classList.toggle("is-loading", isLoading);
  analyzeBtnText.textContent = isLoading ? "Analyzing..." : "Analyze URL";
  loadingStatus.hidden = !isLoading;
  urlForm.setAttribute("aria-busy", String(isLoading));
}

function formatApiError(payload, status) {
  const detail = payload && payload.detail;

  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (item && item.message) {
          return item.field ? `${item.field}: ${item.message}` : item.message;
        }
        return String(item);
      })
      .join("; ");
  }

  if (typeof detail === "string") {
    return detail;
  }

  if (payload && typeof payload.message === "string") {
    return payload.message;
  }

  return `Backend request failed (status ${status}).`;
}

function makeFeatureCard(label, value, note) {
  const card = document.createElement("article");
  card.className = "feature-card";

  const labelEl = document.createElement("span");
  labelEl.className = "feature-label";
  labelEl.textContent = label;

  const valueEl = document.createElement("span");
  valueEl.className = "feature-value";
  valueEl.textContent = String(value);

  const noteEl = document.createElement("span");
  noteEl.className = "feature-note";
  noteEl.textContent = note;

  card.append(labelEl, valueEl, noteEl);
  return card;
}

function renderFeatures(features) {
  featuresGrid.replaceChildren();

  const renderedKeys = new Set();

  FEATURE_META.forEach((meta) => {
    if (!Object.prototype.hasOwnProperty.call(features, meta.key)) {
      return;
    }
    renderedKeys.add(meta.key);
    featuresGrid.appendChild(makeFeatureCard(meta.label, features[meta.key], meta.note));
  });

  Object.keys(features).forEach((key) => {
    if (!renderedKeys.has(key)) {
      featuresGrid.appendChild(makeFeatureCard(key, features[key], "Value returned by the API."));
    }
  });
}

function renderResult(data) {
  const isPhishing = Boolean(data.is_phishing);
  const features = data.features && typeof data.features === "object" ? data.features : {};

  resultBanner.classList.toggle("phishing", isPhishing);
  resultBanner.classList.toggle("legitimate", !isPhishing);
  resultTitle.textContent = isPhishing
    ? "Phishing Website Detected!"
    : "No Phishing Indicators Detected";
  resultUrl.textContent = data.url;
  predictionPill.textContent = data.prediction;
  confidenceValue.textContent = `${data.confidence}%`;
  featureCount.textContent = `(${Object.keys(features).length})`;

  renderFeatures(features);
  result.hidden = false;
  result.scrollIntoView({ block: "nearest" });
}

function resetResult() {
  result.hidden = true;
  resultBanner.classList.remove("phishing", "legitimate");
  resultTitle.textContent = "";
  resultUrl.textContent = "";
  predictionPill.textContent = "";
  confidenceValue.textContent = "";
  featureCount.textContent = "";
  featuresGrid.replaceChildren();
}

urlForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();

  const url = urlInput.value.trim();

  const validationError = validateUrl(url);
  if (validationError) {
    showError(validationError);
    result.hidden = true;
    return;
  }

  setLoading(true);

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url })
    });

    const payload = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(formatApiError(payload, response.status));
    }

    if (!payload || typeof payload.prediction !== "string" || typeof payload.features !== "object") {
      throw new Error("Backend returned an unexpected response format.");
    }

    renderResult(payload);
  } catch (error) {
    if (error instanceof TypeError && error.message.includes("fetch")) {
      showError(`Could not connect to the backend at ${API_URL}. Make sure the API is running.`);
    } else {
      showError(error.message || "Something went wrong while analyzing the URL.");
    }
    result.hidden = true;
  } finally {
    setLoading(false);
  }
});

clearBtn.addEventListener("click", () => {
  urlInput.value = "";
  clearError();
  resetResult();
  urlInput.focus();
});

urlInput.addEventListener("input", clearError);
