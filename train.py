"""
Train Decision Tree, Random Forest and SVM classifiers to predict Pass/Fail.

"Fail" is treated as the positive class, because the goal is to catch
at-risk students: recall = share of actual failing students the model flags.

Usage:  python train.py [path/to/students.csv]
"""
import json
import sys
import warnings

import os

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"  # also silence the parallel worker processes

# Keep parallelism small so training fits in low-memory hosts (e.g. Streamlit Cloud, 1 GB).
# Set TRAIN_N_JOBS=-1 to use every CPU core on a powerful machine.
N_JOBS = int(os.environ.get("TRAIN_N_JOBS", "2"))
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"

NUMERIC = ["study_hours", "attendance", "previous_grade", "sleep_hours", "tutoring_sessions"]
ORDINAL = {
    "socioeconomic_status": ["Low", "Medium", "High"],
    "parental_education": ["No Formal", "High School", "Bachelor", "Master+"],
}
BINARY = ["internet_access", "extracurricular"]
TARGET = "result"
POSITIVE = "Fail"  # at-risk


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("ord", OrdinalEncoder(categories=list(ORDINAL.values())), list(ORDINAL.keys())),
        ("bin", OneHotEncoder(drop="if_binary"), BINARY),
    ])


MODELS = {
    "Decision Tree": (
        DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        {"clf__max_depth": [3, 5, 7, 10], "clf__min_samples_leaf": [1, 5, 10]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=1),
        {"clf__n_estimators": [100, 200], "clf__max_depth": [8, 12, 16]},
    ),
    "SVM": (
        SVC(probability=True, random_state=42, class_weight="balanced"),
        {"clf__C": [0.5, 1, 5, 10], "clf__kernel": ["rbf", "linear"]},
    ),
}


def feature_names(pre: ColumnTransformer) -> list[str]:
    return [n.split("__", 1)[1] for n in pre.get_feature_names_out()]


def main(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    # only the predictive features are used; identifiers like name/class are ignored
    X = df[NUMERIC + list(ORDINAL) + BINARY]
    y = (df[TARGET] == POSITIVE).astype(int)  # 1 = Fail / at-risk

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    MODELS_DIR.mkdir(exist_ok=True)

    report = {"n_train": len(X_train), "n_test": len(X_test),
              "positive_class": POSITIVE, "models": {}}

    for name, (clf, grid) in MODELS.items():
        pipe = Pipeline([("pre", build_preprocessor()), ("clf", clf)])
        search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=N_JOBS)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        pred = best.predict(X_test)
        proba = best.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba),
            "cv_f1": search.best_score_,
        }
        cm = confusion_matrix(y_test, pred, labels=[0, 1]).tolist()

        # feature importance
        names = feature_names(best.named_steps["pre"])
        model = best.named_steps["clf"]
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif getattr(model, "kernel", None) == "linear":
            imp = np.abs(model.coef_).ravel()
            imp = imp / imp.sum()
        else:
            from sklearn.inspection import permutation_importance
            Xt = best.named_steps["pre"].transform(X_test)
            imp = permutation_importance(model, Xt, y_test, scoring="f1",
                                         n_repeats=5, random_state=42).importances_mean
            imp = np.clip(imp, 0, None)
            imp = imp / imp.sum() if imp.sum() else imp

        report["models"][name] = {
            "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
            "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            "confusion_matrix": cm,  # rows=actual [Pass, Fail], cols=predicted
            "feature_importance": dict(sorted(
                {n: round(float(i), 4) for n, i in zip(names, imp)}.items(),
                key=lambda kv: -kv[1])),
        }
        joblib.dump(best, MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib")
        print(f"{name:14s}  " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

    best_name = max(report["models"], key=lambda n: report["models"][n]["metrics"]["f1"])
    report["best_model"] = best_name
    (MODELS_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\nBest model by F1: {best_name}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "students.csv"
    main(path)
