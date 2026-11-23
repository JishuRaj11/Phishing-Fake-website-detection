import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

DEFAULT_OUTPUT = Path(__file__).parent / "outputs"
RANDOM_STATE = 42
LABEL_NAMES = {0: "phishing", 1: "legitimate"}
LEGITIMATE_LABEL = 1
PHISHING_LABEL = 0
DISPLAY_LABELS = [LABEL_NAMES[PHISHING_LABEL], LABEL_NAMES[LEGITIMATE_LABEL]]
SECTION = "=" * 78


def print_header(title):
    print("\n" + SECTION)
    print(title)
    print(SECTION)


def load_matrix(path):
    if path.suffix == ".npz":
        matrix = sparse.load_npz(str(path))
    else:
        matrix = np.load(str(path))
    return matrix


def load_labels(path):
    frame = pd.read_csv(path)
    if "label" not in frame.columns:
        raise ValueError(f"Expected a 'label' column in {path}, found {frame.columns.tolist()}")
    return frame["label"].astype(int).to_numpy()


def load_processed_data(output_dir):
    x_train = load_matrix(output_dir / "X_train.npz")
    x_test = load_matrix(output_dir / "X_test.npz")
    y_train = load_labels(output_dir / "y_train.csv")
    y_test = load_labels(output_dir / "y_test.csv")

    if x_train.shape[0] != len(y_train):
        raise ValueError(f"X_train rows {x_train.shape[0]} != y_train rows {len(y_train)}")
    if x_test.shape[0] != len(y_test):
        raise ValueError(f"X_test rows {x_test.shape[0]} != y_test rows {len(y_test)}")
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("X_train and X_test have different feature counts")

    feature_names = json.loads((output_dir / "processed_feature_names.json").read_text(encoding="utf-8"))
    if len(feature_names) != x_train.shape[1]:
        raise ValueError("processed_feature_names.json does not match the processed matrix width")

    for name, labels in (("y_train", y_train), ("y_test", y_test)):
        values = set(np.unique(labels).tolist())
        if values != {PHISHING_LABEL, LEGITIMATE_LABEL}:
            raise ValueError(f"{name} must contain only labels 0 and 1, found {sorted(values)}")

    print_header("MODEL TRAINING - LOADING PROCESSED DATA")
    print("  No feature extraction or preprocessing is performed in this script.")
    print(f"  X_train shape                : {x_train.shape}")
    print(f"  X_test shape                 : {x_test.shape}")
    print(f"  Processed feature count      : {len(feature_names)}")
    print(f"  y_train 0/1                  : {(y_train == 0).sum():,} / {(y_train == 1).sum():,}")
    print(f"  y_test  0/1                  : {(y_test == 0).sum():,} / {(y_test == 1).sum():,}")
    print("  Label semantics verified     : 0 = phishing, 1 = legitimate")
    print("  Test set is used only for final evaluation, never for tuning.")
    return x_train, x_test, y_train, y_test, feature_names


def build_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
    }


def class_probability(model, probabilities, label):
    index = list(model.classes_).index(label)
    return probabilities[:, index]


def calculate_metrics(y_true, y_pred, model, probabilities):
    score_legit = class_probability(model, probabilities, LEGITIMATE_LABEL)
    score_phishing = class_probability(model, probabilities, PHISHING_LABEL)
    phishing_true = (y_true == PHISHING_LABEL).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[PHISHING_LABEL, LEGITIMATE_LABEL])

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_legitimate": float(precision_score(y_true, y_pred, pos_label=LEGITIMATE_LABEL, zero_division=0)),
        "recall_legitimate": float(recall_score(y_true, y_pred, pos_label=LEGITIMATE_LABEL, zero_division=0)),
        "f1_legitimate": float(f1_score(y_true, y_pred, pos_label=LEGITIMATE_LABEL, zero_division=0)),
        "roc_auc_legitimate": float(roc_auc_score(y_true, score_legit)),
        "precision_phishing": float(precision_score(y_true, y_pred, pos_label=PHISHING_LABEL, zero_division=0)),
        "recall_phishing": float(recall_score(y_true, y_pred, pos_label=PHISHING_LABEL, zero_division=0)),
        "f1_phishing": float(f1_score(y_true, y_pred, pos_label=PHISHING_LABEL, zero_division=0)),
        "roc_auc_phishing": float(roc_auc_score(phishing_true, score_phishing)),
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=[PHISHING_LABEL, LEGITIMATE_LABEL],
        target_names=DISPLAY_LABELS,
        digits=6,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=[PHISHING_LABEL, LEGITIMATE_LABEL],
        target_names=DISPLAY_LABELS,
        output_dict=True,
        zero_division=0,
    )
    fpr_legit, tpr_legit, _ = roc_curve(y_true, score_legit)
    fpr_phishing, tpr_phishing, _ = roc_curve(phishing_true, score_phishing)
    return {
        "metrics": metrics,
        "confusion_matrix": matrix,
        "classification_report_text": report,
        "classification_report": report_dict,
        "fpr_legitimate": fpr_legit,
        "tpr_legitimate": tpr_legit,
        "fpr_phishing": fpr_phishing,
        "tpr_phishing": tpr_phishing,
    }


def slugify(name):
    return name.lower().replace(" ", "_")


def save_confusion_matrix_plot(matrix, name, path):
    figure, axes = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=DISPLAY_LABELS)
    display.plot(ax=axes, cmap="Blues", values_format="d", colorbar=False)
    axes.set_title(f"{name} - Confusion Matrix")
    axes.set_xlabel("Predicted class")
    axes.set_ylabel("True class")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_roc_plot(results, class_name, fpr_key, tpr_key, auc_key, output_dir, suffix):
    figure, axes = plt.subplots(figsize=(7, 6))
    axes.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    for name, result in results.items():
        axes.plot(
            result[fpr_key],
            result[tpr_key],
            label=f"{name} (AUC = {result['metrics'][auc_key]:.4f})",
        )
        single_figure, single_axes = plt.subplots(figsize=(6, 5))
        single_axes.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        single_axes.plot(result[fpr_key], result[tpr_key], color="tab:blue")
        single_axes.set_title(f"{name} - ROC Curve ({class_name} class)")
        single_axes.set_xlabel("False positive rate")
        single_axes.set_ylabel("True positive rate")
        single_axes.legend(loc="lower right")
        single_figure.tight_layout()
        single_figure.savefig(output_dir / "plots" / f"roc_{slugify(name)}_{suffix}.png", dpi=150)
        plt.close(single_figure)
    axes.set_title(f"ROC Curves - Positive class: {class_name}")
    axes.set_xlabel("False positive rate")
    axes.set_ylabel("True positive rate")
    axes.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_dir / "plots" / f"roc_comparison_{suffix}.png", dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate phishing detection models on processed data.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Directory containing processed outputs")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed for model reproducibility")
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = args.output_dir
    models_dir = output_dir / "models"
    plots_dir = output_dir / "plots"
    reports_dir = output_dir / "classification_reports"
    for directory in (models_dir, plots_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test, feature_names = load_processed_data(output_dir)
    models = build_models()
    results = {}
    comparison_rows = []

    print_header("MODEL TRAINING AND EVALUATION")
    for name, model in models.items():
        start = time.perf_counter()
        print(f"\n  Training {name} ...")
        model.fit(x_train, y_train)
        training_time = time.perf_counter() - start

        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)
        result = calculate_metrics(y_test, predictions, model, probabilities)
        result["training_seconds"] = float(training_time)
        result["model_parameters"] = {key: str(value) for key, value in model.get_params().items()}
        results[name] = result

        slug = slugify(name)
        joblib.dump(model, models_dir / f"{slug}.joblib")
        save_confusion_matrix_plot(
            result["confusion_matrix"],
            name,
            plots_dir / f"confusion_matrix_{slug}.png",
        )
        (reports_dir / f"{slug}.txt").write_text(
            f"{name}\n\n{result['classification_report_text']}",
            encoding="utf-8",
        )

        metrics = result["metrics"]
        comparison_rows.append({"model": name, **metrics})
        print(f"  Finished {name} in {training_time:.1f}s")
        print(f"  Accuracy = {metrics['accuracy']:.4f} | "
              f"ROC-AUC (legitimate) = {metrics['roc_auc_legitimate']:.4f} | "
              f"Phishing precision/recall = {metrics['precision_phishing']:.4f}/{metrics['recall_phishing']:.4f}")

    save_roc_plot(results, "legitimate", "fpr_legitimate", "tpr_legitimate", "roc_auc_legitimate", output_dir, "legitimate")
    save_roc_plot(results, "phishing", "fpr_phishing", "tpr_phishing", "roc_auc_phishing", output_dir, "phishing")
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)

    best_name = max(results, key=lambda key: results[key]["metrics"]["roc_auc_legitimate"])
    serializable_results = {}
    for name, result in results.items():
        matrix = result["confusion_matrix"]
        serializable_results[name] = {
            "metrics": result["metrics"],
            "confusion_matrix": {
                "labels": [PHISHING_LABEL, LEGITIMATE_LABEL],
                "display_labels": DISPLAY_LABELS,
                "rows_are_true_columns_are_predicted": matrix.tolist(),
                "true_phishing_predicted_phishing": int(matrix[0, 0]),
                "true_phishing_predicted_legitimate": int(matrix[0, 1]),
                "true_legitimate_predicted_phishing": int(matrix[1, 0]),
                "true_legitimate_predicted_legitimate": int(matrix[1, 1]),
            },
            "classification_report": result["classification_report"],
            "classification_report_text": result["classification_report_text"],
            "training_seconds": result["training_seconds"],
            "model_parameters": result["model_parameters"],
        }

    evaluation = {
        "random_state": args.seed,
        "label_semantics": LABEL_NAMES,
        "positive_class_for_core_metrics": LEGITIMATE_LABEL,
        "phishing_class": PHISHING_LABEL,
        "train_rows": int(x_train.shape[0]),
        "test_rows": int(x_test.shape[0]),
        "processed_feature_count": len(feature_names),
        "preprocessing_reused_without_refitting": True,
        "results": serializable_results,
        "best_model_by_roc_auc_legitimate": best_name,
    }
    (output_dir / "evaluation_results.json").write_text(
        json.dumps(evaluation, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value)),
        encoding="utf-8",
    )
    (reports_dir / "all_classification_reports.txt").write_text(
        "\n\n".join(f"{name}\n\n{result['classification_report_text']}" for name, result in results.items()),
        encoding="utf-8",
    )

    print_header("MODEL COMPARISON (all models evaluated on the same untouched test set)")
    print(comparison.to_string(index=False))
    print("\n  Confusion matrices use rows = true class, columns = predicted class.")
    print("  Class order: 0 = phishing, 1 = legitimate.")
    print(f"\n  Highest ROC-AUC for legitimate class: {best_name}")
    print("\n  Saved artifacts:")
    print(f"    {models_dir / '*.joblib'}")
    print(f"    {output_dir / 'model_comparison.csv'}")
    print(f"    {output_dir / 'evaluation_results.json'}")
    print(f"    {plots_dir}")
    print(f"    {reports_dir}")


if __name__ == "__main__":
    main()
