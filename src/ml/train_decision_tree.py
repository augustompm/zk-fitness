"""
train_decision_tree.py

Trains a Decision Tree classifier for exercise intensity using MHEALTH data.
Uses 10-fold stratified cross-validation following HAR literature standards.

References:
    [1] WHO Physical Activity Guidelines 2020
    [12] ACSM Guidelines 10th ed
    [19] Fan et al. 2014 - Decision Tree for HAR
    [20] Sukor et al. 2018 - Activity recognition with ML
    [21] Breiman et al. 1984 - CART algorithm
    [22] Brehler et al. 2023 - DT for wearables
"""

import numpy as np
import json
import csv
import platform
import sys
from pathlib import Path
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


MHEALTH_DIR = Path(__file__).parent.parent.parent.parent / "datasets" / "MHEALTH_extracted" / "MHEALTHDATASET"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"

MHEALTH_SAMPLING_RATE_HZ = 50
WINDOW_DURATION_SECONDS = 1.0
WINDOW_OVERLAP_RATIO = 0.5
WINDOW_SIZE_SAMPLES = int(MHEALTH_SAMPLING_RATE_HZ * WINDOW_DURATION_SECONDS)
WINDOW_STEP_SAMPLES = int(WINDOW_SIZE_SAMPLES * (1 - WINDOW_OVERLAP_RATIO))

MAX_TREE_DEPTH = 5
MIN_SAMPLES_LEAF = 10
SCALE_FACTOR_FOR_CIRCOM = 100

N_CV_FOLDS = 10
RANDOM_STATE = 42
TEST_SET_RATIO = 0.2

N_BENCHMARK_RUNS = 5

CHEST_ACCEL_COLS = [0, 1, 2]
ANKLE_ACCEL_COLS = [5, 6, 7]
ANKLE_GYRO_COLS = [8, 9, 10]
WRIST_ACCEL_COLS = [14, 15, 16]
WRIST_GYRO_COLS = [17, 18, 19]
LABEL_COL = 23

MHEALTH_ACTIVITY_TO_INTENSITY = {
    0: -1,
    1: 0,
    2: 0,
    3: 0,
    4: 1,
    5: 2,
    6: 1,
    7: 1,
    8: 1,
    9: 1,
    10: 2,
    11: 2,
    12: 2,
}

INTENSITY_CLASS_NAMES = {0: "rest", 1: "moderate", 2: "vigorous"}

FEATURE_NAMES = [
    "chest_accel_mean",
    "chest_accel_std",
    "ankle_accel_mean",
    "ankle_accel_std",
    "ankle_gyro_mean",
    "wrist_accel_mean",
    "wrist_accel_std",
    "wrist_gyro_mean",
]


class TrainingLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.start_time = datetime.now()

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")

    def elapsed_seconds(self):
        return (datetime.now() - self.start_time).total_seconds()


def compute_vector_magnitude(xyz_data):
    return np.sqrt(np.sum(xyz_data ** 2, axis=1))


def extract_window_features(window_data):
    chest_accel_mag = compute_vector_magnitude(window_data[:, CHEST_ACCEL_COLS])
    ankle_accel_mag = compute_vector_magnitude(window_data[:, ANKLE_ACCEL_COLS])
    ankle_gyro_mag = compute_vector_magnitude(window_data[:, ANKLE_GYRO_COLS])
    wrist_accel_mag = compute_vector_magnitude(window_data[:, WRIST_ACCEL_COLS])
    wrist_gyro_mag = compute_vector_magnitude(window_data[:, WRIST_GYRO_COLS])

    features = [
        np.mean(chest_accel_mag),
        np.std(chest_accel_mag),
        np.mean(ankle_accel_mag),
        np.std(ankle_accel_mag),
        np.mean(ankle_gyro_mag),
        np.mean(wrist_accel_mag),
        np.std(wrist_accel_mag),
        np.mean(wrist_gyro_mag),
    ]

    return features


def load_mhealth_subject(subject_id):
    file_path = MHEALTH_DIR / f"mHealth_subject{subject_id}.log"
    return np.loadtxt(file_path)


def create_windowed_dataset(logger):
    all_features = []
    all_labels = []
    samples_per_subject = []

    for subject_id in range(1, 11):
        logger.log(f"Processing subject {subject_id}/10")
        subject_data = load_mhealth_subject(subject_id)

        n_windows = (len(subject_data) - WINDOW_SIZE_SAMPLES) // WINDOW_STEP_SAMPLES + 1
        subject_samples = 0

        for window_idx in range(n_windows):
            start = window_idx * WINDOW_STEP_SAMPLES
            end = start + WINDOW_SIZE_SAMPLES
            window = subject_data[start:end]

            window_labels = window[:, LABEL_COL].astype(int)
            majority_activity = int(np.bincount(window_labels).argmax())
            intensity = MHEALTH_ACTIVITY_TO_INTENSITY.get(majority_activity, -1)

            if intensity >= 0:
                features = extract_window_features(window)
                all_features.append(features)
                all_labels.append(intensity)
                subject_samples += 1

        samples_per_subject.append(subject_samples)
        logger.log(f"  Subject {subject_id}: {subject_samples} valid windows")

    return np.array(all_features), np.array(all_labels), samples_per_subject


def perform_stratified_kfold_cv(X, y, logger):
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_results = []
    all_y_true = []
    all_y_pred = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        clf = DecisionTreeClassifier(
            criterion="gini",
            max_depth=MAX_TREE_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE
        )
        clf.fit(X_train_fold, y_train_fold)

        y_pred_fold = clf.predict(X_val_fold)
        fold_accuracy = accuracy_score(y_val_fold, y_pred_fold)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val_fold, y_pred_fold, average="weighted", zero_division=0
        )

        fold_results.append({
            "fold": fold_idx + 1,
            "accuracy": fold_accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "n_train": len(train_idx),
            "n_val": len(val_idx)
        })

        all_y_true.extend(y_val_fold)
        all_y_pred.extend(y_pred_fold)

        logger.log(f"  Fold {fold_idx + 1}/{N_CV_FOLDS}: accuracy={fold_accuracy:.4f}, F1={f1:.4f}")

    accuracies = [r["accuracy"] for r in fold_results]
    cv_mean = np.mean(accuracies)
    cv_std = np.std(accuracies)

    logger.log(f"10-Fold CV Result: {cv_mean:.4f} (+/- {cv_std:.4f})")

    return fold_results, np.array(all_y_true), np.array(all_y_pred), cv_mean, cv_std


def train_final_model(X_train, y_train, logger):
    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=MAX_TREE_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE
    )
    clf.fit(X_train, y_train)

    logger.log(f"Final model: depth={clf.get_depth()}, leaves={clf.get_n_leaves()}")

    return clf


def compute_per_class_metrics(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    per_class = {}
    for class_idx in range(len(precision)):
        class_name = INTENSITY_CLASS_NAMES[class_idx]
        per_class[class_name] = {
            "precision": float(precision[class_idx]),
            "recall": float(recall[class_idx]),
            "f1_score": float(f1[class_idx]),
            "support": int(support[class_idx])
        }

    return per_class


def export_tree_for_circom(tree, output_path):
    tree_structure = {
        "n_features": int(tree.n_features_in_),
        "n_classes": int(tree.n_classes_),
        "max_depth": int(tree.get_depth()),
        "n_leaves": int(tree.get_n_leaves()),
        "feature_names": FEATURE_NAMES,
        "class_names": list(INTENSITY_CLASS_NAMES.values()),
        "scale_factor": SCALE_FACTOR_FOR_CIRCOM,
        "nodes": []
    }

    tree_internal = tree.tree_
    n_nodes = tree_internal.node_count
    children_left = tree_internal.children_left
    children_right = tree_internal.children_right
    feature_indices = tree_internal.feature
    thresholds = tree_internal.threshold
    values = tree_internal.value

    for node_id in range(n_nodes):
        is_leaf = children_left[node_id] == children_right[node_id]

        node_info = {
            "node_id": int(node_id),
            "is_leaf": bool(is_leaf),
        }

        if is_leaf:
            class_counts = values[node_id][0]
            node_info["predicted_class"] = int(np.argmax(class_counts))
            node_info["class_counts"] = [int(c) for c in class_counts]
        else:
            feat_idx = int(feature_indices[node_id])
            node_info["feature_index"] = feat_idx
            node_info["feature_name"] = FEATURE_NAMES[feat_idx]
            node_info["threshold"] = int(thresholds[node_id] * SCALE_FACTOR_FOR_CIRCOM)
            node_info["threshold_original"] = float(thresholds[node_id])
            node_info["left_child"] = int(children_left[node_id])
            node_info["right_child"] = int(children_right[node_id])

        tree_structure["nodes"].append(node_info)

    with open(output_path, "w") as f:
        json.dump(tree_structure, f, indent=2)

    return tree_structure


def save_cv_results_csv(fold_results, output_path):
    fieldnames = ["fold", "accuracy", "precision", "recall", "f1_score", "n_train", "n_val"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fold_results)


def save_confusion_matrix_csv(cm, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + list(INTENSITY_CLASS_NAMES.values()))
        for i, row in enumerate(cm):
            writer.writerow([INTENSITY_CLASS_NAMES[i]] + list(row))


def save_feature_importance_csv(importances, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "importance"])
        for name, imp in zip(FEATURE_NAMES, importances):
            writer.writerow([name, f"{imp:.6f}"])


def run_benchmark_iterations(X, y, logger):
    benchmark_results = []

    for run_idx in range(N_BENCHMARK_RUNS):
        run_seed = RANDOM_STATE + run_idx

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SET_RATIO, random_state=run_seed, stratify=y
        )

        clf = DecisionTreeClassifier(
            criterion="gini",
            max_depth=MAX_TREE_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=run_seed
        )
        clf.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))

        benchmark_results.append({
            "run_id": run_idx + 1,
            "seed": run_seed,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "tree_depth": clf.get_depth(),
            "n_leaves": clf.get_n_leaves()
        })

        logger.log(f"  Benchmark run {run_idx + 1}/{N_BENCHMARK_RUNS}: test_acc={test_acc:.4f}")

    return benchmark_results


def save_benchmark_csv(benchmark_results, output_path):
    fieldnames = ["run_id", "seed", "train_accuracy", "test_accuracy", "tree_depth", "n_leaves"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(benchmark_results)


def create_experiment_metadata(
    X, y, samples_per_subject, cv_mean, cv_std,
    test_accuracy, clf, benchmark_results, elapsed_time
):
    class_counts = np.bincount(y)

    benchmark_test_accs = [r["test_accuracy"] for r in benchmark_results]

    metadata = {
        "experiment_name": "decision_tree_har_classification",
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": elapsed_time,
        "status": "completed",

        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.system(),
            "platform_version": platform.version(),
        },

        "parameters": {
            "window_size_samples": WINDOW_SIZE_SAMPLES,
            "window_step_samples": WINDOW_STEP_SAMPLES,
            "window_duration_seconds": WINDOW_DURATION_SECONDS,
            "window_overlap_ratio": WINDOW_OVERLAP_RATIO,
            "sampling_rate_hz": MHEALTH_SAMPLING_RATE_HZ,
            "max_tree_depth": MAX_TREE_DEPTH,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "scale_factor": SCALE_FACTOR_FOR_CIRCOM,
            "n_cv_folds": N_CV_FOLDS,
            "n_benchmark_runs": N_BENCHMARK_RUNS,
            "test_set_ratio": TEST_SET_RATIO,
            "random_state": RANDOM_STATE,
            "split_criterion": "gini",
        },

        "dataset": {
            "name": "MHEALTH",
            "source": "UCI Machine Learning Repository",
            "n_subjects": 10,
            "samples_per_subject": samples_per_subject,
            "total_samples": int(len(X)),
            "n_features": len(FEATURE_NAMES),
            "n_classes": len(INTENSITY_CLASS_NAMES),
            "class_distribution": {
                INTENSITY_CLASS_NAMES[i]: int(class_counts[i])
                for i in range(len(class_counts))
            },
        },

        "results": {
            "cv_accuracy_mean": float(cv_mean),
            "cv_accuracy_std": float(cv_std),
            "cv_accuracy_95ci": float(cv_std * 1.96),
            "holdout_test_accuracy": float(test_accuracy),
            "benchmark_mean_accuracy": float(np.mean(benchmark_test_accs)),
            "benchmark_std_accuracy": float(np.std(benchmark_test_accs)),
            "final_tree_depth": int(clf.get_depth()),
            "final_n_leaves": int(clf.get_n_leaves()),
        },

        "feature_names": FEATURE_NAMES,
        "feature_importances": {
            name: float(imp) for name, imp in zip(FEATURE_NAMES, clf.feature_importances_)
        },

        "references": {
            "methodology": "[19] Fan et al. 2014, [20] Sukor et al. 2018",
            "split_criterion": "[21] Breiman et al. 1984 CART",
            "intensity_guidelines": "[1] WHO 2020, [12] ACSM Guidelines",
        }
    }

    return metadata


def train_and_evaluate():
    date_str = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%H%M%S")

    log_dir = LOGS_DIR / date_str
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_decision_tree_{timestamp}.log"

    results_dir = RESULTS_DIR / date_str
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = TrainingLogger(log_file)

    logger.log("=" * 60)
    logger.log("Decision Tree Training for zkML Fitness Classification")
    logger.log("=" * 60)
    logger.log(f"Window: {WINDOW_SIZE_SAMPLES} samples ({WINDOW_DURATION_SECONDS}s @ {MHEALTH_SAMPLING_RATE_HZ}Hz)")
    logger.log(f"Overlap: {WINDOW_OVERLAP_RATIO * 100}%")
    logger.log(f"Max depth: {MAX_TREE_DEPTH}, Min samples leaf: {MIN_SAMPLES_LEAF}")
    logger.log(f"Cross-validation: {N_CV_FOLDS}-fold stratified")
    logger.log(f"Benchmark runs: {N_BENCHMARK_RUNS}")
    logger.log("")

    logger.log("Creating dataset from MHEALTH...")
    X, y, samples_per_subject = create_windowed_dataset(logger)
    logger.log(f"Dataset: {len(X)} samples, {len(FEATURE_NAMES)} features, {len(INTENSITY_CLASS_NAMES)} classes")

    class_counts = np.bincount(y)
    for class_idx, count in enumerate(class_counts):
        logger.log(f"  {INTENSITY_CLASS_NAMES[class_idx]}: {count} samples ({count/len(y)*100:.1f}%)")
    logger.log("")

    logger.log(f"Performing {N_CV_FOLDS}-fold stratified cross-validation...")
    fold_results, cv_y_true, cv_y_pred, cv_mean, cv_std = perform_stratified_kfold_cv(X, y, logger)
    logger.log("")

    logger.log("Training final model on train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SET_RATIO, random_state=RANDOM_STATE, stratify=y
    )
    logger.log(f"Train: {len(X_train)}, Test: {len(X_test)}")

    final_clf = train_final_model(X_train, y_train, logger)

    y_pred_train = final_clf.predict(X_train)
    y_pred_test = final_clf.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    logger.log(f"Train accuracy: {train_accuracy:.4f}")
    logger.log(f"Test accuracy: {test_accuracy:.4f}")
    logger.log("")

    logger.log("Per-class metrics on test set:")
    per_class_metrics = compute_per_class_metrics(y_test, y_pred_test)
    for class_name, metrics in per_class_metrics.items():
        logger.log(f"  {class_name}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")
    logger.log("")

    logger.log("Confusion matrix (test set):")
    cm = confusion_matrix(y_test, y_pred_test)
    for i, row in enumerate(cm):
        logger.log(f"  {INTENSITY_CLASS_NAMES[i]}: {list(row)}")
    logger.log("")

    logger.log("Feature importances:")
    sorted_indices = np.argsort(final_clf.feature_importances_)[::-1]
    for idx in sorted_indices:
        logger.log(f"  {FEATURE_NAMES[idx]}: {final_clf.feature_importances_[idx]:.4f}")
    logger.log("")

    logger.log(f"Running {N_BENCHMARK_RUNS} benchmark iterations...")
    benchmark_results = run_benchmark_iterations(X, y, logger)
    benchmark_accs = [r["test_accuracy"] for r in benchmark_results]
    logger.log(f"Benchmark: {np.mean(benchmark_accs):.4f} (+/- {np.std(benchmark_accs):.4f})")
    logger.log("")

    logger.log("Exporting results...")

    tree_json_path = results_dir / f"decision_tree_{timestamp}.json"
    export_tree_for_circom(final_clf, tree_json_path)
    logger.log(f"  Tree structure: {tree_json_path}")

    model_path = Path(__file__).parent / "trained_tree.json"
    export_tree_for_circom(final_clf, model_path)
    logger.log(f"  Model exported: {model_path}")

    cv_csv_path = results_dir / f"cv_results_{timestamp}.csv"
    save_cv_results_csv(fold_results, cv_csv_path)
    logger.log(f"  CV results: {cv_csv_path}")

    cm_csv_path = results_dir / f"confusion_matrix_{timestamp}.csv"
    save_confusion_matrix_csv(cm, cm_csv_path)
    logger.log(f"  Confusion matrix: {cm_csv_path}")

    fi_csv_path = results_dir / f"feature_importance_{timestamp}.csv"
    save_feature_importance_csv(final_clf.feature_importances_, fi_csv_path)
    logger.log(f"  Feature importance: {fi_csv_path}")

    benchmark_csv_path = results_dir / f"benchmark_runs_{timestamp}.csv"
    save_benchmark_csv(benchmark_results, benchmark_csv_path)
    logger.log(f"  Benchmark runs: {benchmark_csv_path}")

    elapsed = logger.elapsed_seconds()
    metadata = create_experiment_metadata(
        X, y, samples_per_subject, cv_mean, cv_std,
        test_accuracy, final_clf, benchmark_results, elapsed
    )

    meta_path = results_dir / f"experiment_meta_{timestamp}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.log(f"  Metadata: {meta_path}")

    logger.log("")
    logger.log("=" * 60)
    logger.log(f"Training completed in {elapsed:.1f} seconds")
    logger.log(f"10-Fold CV Accuracy: {cv_mean*100:.2f}% (+/- {cv_std*100:.2f}%)")
    logger.log(f"Holdout Test Accuracy: {test_accuracy*100:.2f}%")
    logger.log("=" * 60)

    return final_clf, metadata


if __name__ == "__main__":
    train_and_evaluate()
