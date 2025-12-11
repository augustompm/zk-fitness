"""
classifiers.py

Classifiers for physical activity intensity detection.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import time


class IntensityClassifier:

    def __init__(self, model_type="random_forest", random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.training_time = None
        self.feature_names = None

        self._create_model()

    def _create_model(self):
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == "decision_tree":
            self.model = DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state
            )
        elif self.model_type == "svm":
            self.model = SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                random_state=self.random_state
            )
        elif self.model_type == "knn":
            self.model = KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                n_jobs=-1
            )
        elif self.model_type == "logistic":
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, X, y):
        if hasattr(X, "columns"):
            self.feature_names = list(X.columns)
            X = X.values

        X_scaled = self.scaler.fit_transform(X)

        start_time = time.time()
        self.model.fit(X_scaled, y)
        self.training_time = time.time() - start_time

        self.is_fitted = True

        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        if hasattr(X, "values"):
            X = X.values

        X_scaled = self.scaler.transform(X)

        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        if not hasattr(self.model, "predict_proba"):
            raise RuntimeError(f"{self.model_type} does not support predict_proba")

        if hasattr(X, "values"):
            X = X.values

        X_scaled = self.scaler.transform(X)

        return self.model.predict_proba(X_scaled)

    def evaluate(self, X, y):
        y_pred = self.predict(X)

        results = {
            "accuracy": accuracy_score(y, y_pred),
            "f1_weighted": f1_score(y, y_pred, average="weighted"),
            "f1_macro": f1_score(y, y_pred, average="macro"),
            "confusion_matrix": confusion_matrix(y, y_pred),
            "classification_report": classification_report(y, y_pred, output_dict=True)
        }

        return results

    def cross_validate(self, X, y, cv=5):
        if hasattr(X, "values"):
            X = X.values

        X_scaled = self.scaler.fit_transform(X)

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        accuracy_scores = cross_val_score(self.model, X_scaled, y, cv=skf, scoring="accuracy")
        f1_scores = cross_val_score(self.model, X_scaled, y, cv=skf, scoring="f1_weighted")

        results = {
            "accuracy_mean": np.mean(accuracy_scores),
            "accuracy_std": np.std(accuracy_scores),
            "f1_mean": np.mean(f1_scores),
            "f1_std": np.std(f1_scores),
            "cv_folds": cv
        }

        return results

    def get_feature_importance(self):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            if self.feature_names:
                return dict(zip(self.feature_names, importances))
            return importances

        return None

    def get_model_size_bytes(self):
        import pickle
        return len(pickle.dumps(self.model))


def compare_classifiers(X, y, model_types=None, cv=5):
    if model_types is None:
        model_types = ["random_forest", "decision_tree", "svm", "knn", "logistic"]

    results = {}

    for model_type in model_types:
        classifier = IntensityClassifier(model_type=model_type)

        start_time = time.time()
        cv_results = classifier.cross_validate(X, y, cv=cv)
        eval_time = time.time() - start_time

        classifier.fit(X, y)
        model_size = classifier.get_model_size_bytes()

        results[model_type] = {
            "accuracy_mean": cv_results["accuracy_mean"],
            "accuracy_std": cv_results["accuracy_std"],
            "f1_mean": cv_results["f1_mean"],
            "f1_std": cv_results["f1_std"],
            "training_time": classifier.training_time,
            "evaluation_time": eval_time,
            "model_size_bytes": model_size
        }

    return results
