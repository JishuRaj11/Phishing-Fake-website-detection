import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "xgboost.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "outputs" / "preprocessor.joblib"

MAX_URL_LENGTH = 2048

PHISHING_LABEL = 0
LEGITIMATE_LABEL = 1
PREDICTION_LABELS = {
    PHISHING_LABEL: "Phishing",
    LEGITIMATE_LABEL: "Legitimate",
}
