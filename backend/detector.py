from pathlib import Path
from typing import Any

import joblib

from .config import LEGITIMATE_LABEL, PHISHING_LABEL, PREDICTION_LABELS
from phishing_pipeline.features import FEATURE_ORDER, build_feature_frame


class PhishDetector:
    def __init__(self, model_path: Path, preprocessor_path: Path) -> None:
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None
        self.preprocessor = None

    def load(self) -> None:
        for path in (self.model_path, self.preprocessor_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required artifact not found: {path}")

        self.preprocessor = joblib.load(self.preprocessor_path)
        self.model = joblib.load(self.model_path)

        classes = set(self.model.classes_.tolist())
        if classes != {PHISHING_LABEL, LEGITIMATE_LABEL}:
            raise RuntimeError(f"Unexpected model classes: {sorted(classes)}")

        raw_feature_count = len(FEATURE_ORDER)
        if len(self.preprocessor.feature_names_in_) != raw_feature_count:
            raise RuntimeError("Preprocessor does not expect 44 raw URL features.")

        processed_feature_count = len(self.preprocessor.get_feature_names_out())
        if self.model.n_features_in_ != processed_feature_count:
            raise RuntimeError("Loaded model and preprocessor feature counts do not match.")

    def predict(self, url: str) -> dict[str, Any]:
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model and preprocessor are not loaded.")

        features = build_feature_frame([url])

        if features.shape != (1, len(FEATURE_ORDER)):
            raise RuntimeError("Feature extraction did not produce exactly 44 features.")
        if list(features.columns) != list(FEATURE_ORDER):
            raise RuntimeError("Extracted feature order does not match the training order.")

        processed = self.preprocessor.transform(features)

        if processed.shape[1] != self.model.n_features_in_:
            raise RuntimeError("Processed feature count does not match the model.")

        label = int(self.model.predict(processed)[0])
        if label not in PREDICTION_LABELS:
            raise RuntimeError(f"Unexpected prediction label: {label}")

        feature_values = {
            name: value.item() if hasattr(value, "item") else value
            for name, value in features.iloc[0].items()
        }

        probabilities = self.model.predict_proba(processed)[0]
        class_index = list(self.model.classes_).index(label)
        probability = float(probabilities[class_index])

        if not 0.0 <= probability <= 1.0:
            raise RuntimeError("Model returned an invalid class probability.")

        return {
            "url": url,
            "prediction": PREDICTION_LABELS[label],
            "is_phishing": label == PHISHING_LABEL,
            "confidence": round(probability * 100, 2),
            "features": feature_values,
        }
