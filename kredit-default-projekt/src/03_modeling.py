"""
03_modeling.py
---------------
Trainiert drei Modelle (Logistische Regression als Baseline, Random Forest,
Gradient Boosting) jeweils mit drei Strategien im Umgang mit dem
Klassenungleichgewicht:
  (a) unveraendert + klassengewichteter Loss (class_weight / sample_weight)
  (b) manuelles Oversampling der Minderheitsklasse per SMOTE-Implementierung
      (k-NN-Interpolation, da das imbalanced-learn-Paket in dieser
      Umgebung nicht offline installierbar war)
  (c) Random Undersampling der Mehrheitsklasse

Alle Modelle werden auf demselben, unveraenderten Testset evaluiert.
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, brier_score_loss, confusion_matrix,
    roc_curve, precision_recall_curve,
)

DATA_DIR = "data"
MODEL_DIR = "models"
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)


def load_data():
    X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv")["default"]
    y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv")["default"]
    return X_train, X_test, y_train, y_test


def manual_smote(X, y, minority_label=1, k=5, random_state=RANDOM_STATE):
    """Einfache SMOTE-Implementierung: erzeugt synthetische Minderheits-
    beispiele durch lineare Interpolation zwischen einem Minderheitspunkt
    und einem seiner k naechsten Nachbarn (ebenfalls Minderheitsklasse)."""
    rng = np.random.RandomState(random_state)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    X_min = X[y == minority_label]
    n_min = len(X_min)
    n_maj = (y != minority_label).sum()
    n_to_generate = n_maj - n_min
    if n_to_generate <= 0:
        return X, y

    k_eff = min(k, n_min - 1) if n_min > 1 else 1
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(X_min)
    _, neighbor_idx = nn.kneighbors(X_min)

    synthetic = []
    for _ in range(n_to_generate):
        i = rng.randint(0, n_min)
        neighbor_choices = neighbor_idx[i][1:]  # ohne sich selbst
        j = neighbor_choices[rng.randint(0, len(neighbor_choices))]
        gap = rng.rand()
        new_point = X_min[i] + gap * (X_min[j] - X_min[i])
        synthetic.append(new_point)

    X_syn = np.vstack(synthetic)
    y_syn = np.full(len(X_syn), minority_label)

    X_res = np.vstack([X, X_syn])
    y_res = np.concatenate([y, y_syn])
    return X_res, y_res


def random_undersample(X, y, majority_label=0, random_state=RANDOM_STATE):
    rng = np.random.RandomState(random_state)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    idx_min = np.where(y != majority_label)[0]
    idx_maj = np.where(y == majority_label)[0]
    idx_maj_sample = rng.choice(idx_maj, size=len(idx_min), replace=False)
    idx_final = np.concatenate([idx_min, idx_maj_sample])
    rng.shuffle(idx_final)
    return X[idx_final], y[idx_final]


def evaluate(model, X_test, y_test, name):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "model": name,
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "pr_auc": round(average_precision_score(y_test, proba), 4),
        "precision_class1": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall_class1": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1_class1": round(f1_score(y_test, pred, zero_division=0), 4),
        "brier_score": round(brier_score_loss(y_test, proba), 4),
    }
    return metrics, proba, pred


def main():
    X_train, X_test, y_train, y_test = load_data()
    feature_names = X_train.columns.tolist()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.joblib")

    all_metrics = []
    roc_curves = {}
    pr_curves = {}
    probas = {}

    # ---------------------------------------------------------------
    # 1) Baseline: Logistische Regression (class_weight='balanced')
    # ---------------------------------------------------------------
    lr = LogisticRegression(class_weight="balanced", max_iter=2000,
                             random_state=RANDOM_STATE)
    lr.fit(X_train_scaled, y_train)
    m, proba, pred = evaluate(lr, X_test_scaled, y_test, "LogisticRegression_baseline")
    all_metrics.append(m)
    probas["LogisticRegression_baseline"] = proba
    joblib.dump(lr, f"{MODEL_DIR}/logreg_baseline.joblib")

    # ---------------------------------------------------------------
    # 2) Random Forest Varianten
    # ---------------------------------------------------------------
    rf_balanced = RandomForestClassifier(
        n_estimators=400, max_depth=6, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_balanced.fit(X_train, y_train)
    m, proba, pred = evaluate(rf_balanced, X_test, y_test, "RandomForest_class_weight")
    all_metrics.append(m)
    probas["RandomForest_class_weight"] = proba
    joblib.dump(rf_balanced, f"{MODEL_DIR}/random_forest_class_weight.joblib")

    X_train_smote, y_train_smote = manual_smote(X_train.values, y_train.values)
    rf_smote = RandomForestClassifier(
        n_estimators=400, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_smote.fit(X_train_smote, y_train_smote)
    m, proba, pred = evaluate(rf_smote, X_test.values, y_test, "RandomForest_SMOTE")
    all_metrics.append(m)
    probas["RandomForest_SMOTE"] = proba
    joblib.dump(rf_smote, f"{MODEL_DIR}/random_forest_smote.joblib")

    X_train_under, y_train_under = random_undersample(X_train.values, y_train.values)
    rf_under = RandomForestClassifier(
        n_estimators=400, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_under.fit(X_train_under, y_train_under)
    m, proba, pred = evaluate(rf_under, X_test.values, y_test, "RandomForest_undersampled")
    all_metrics.append(m)
    probas["RandomForest_undersampled"] = proba

    # ---------------------------------------------------------------
    # 3) Gradient Boosting Varianten
    # ---------------------------------------------------------------
    sample_weight = np.where(y_train == 1, (y_train == 0).sum() / (y_train == 1).sum(), 1.0)
    gb_weighted = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        random_state=RANDOM_STATE,
    )
    gb_weighted.fit(X_train, y_train, sample_weight=sample_weight)
    m, proba, pred = evaluate(gb_weighted, X_test, y_test, "GradientBoosting_class_weight")
    all_metrics.append(m)
    probas["GradientBoosting_class_weight"] = proba
    joblib.dump(gb_weighted, f"{MODEL_DIR}/gradient_boosting_class_weight.joblib")

    gb_smote = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        random_state=RANDOM_STATE,
    )
    gb_smote.fit(X_train_smote, y_train_smote)
    m, proba, pred = evaluate(gb_smote, X_test.values, y_test, "GradientBoosting_SMOTE")
    all_metrics.append(m)
    probas["GradientBoosting_SMOTE"] = proba
    joblib.dump(gb_smote, f"{MODEL_DIR}/gradient_boosting_smote.joblib")

    # ---------------------------------------------------------------
    # Ergebnisse speichern
    # ---------------------------------------------------------------
    results_df = pd.DataFrame(all_metrics).sort_values("roc_auc", ascending=False)
    results_df.to_csv("results_model_comparison.csv", index=False)
    print(results_df.to_string(index=False))

    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        prec, rec, _ = precision_recall_curve(y_test, proba)
        roc_curves[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        pr_curves[name] = {"precision": prec.tolist(), "recall": rec.tolist()}

    with open("results_curves.json", "w") as f:
        json.dump({"roc": roc_curves, "pr": pr_curves,
                   "y_test": y_test.tolist()}, f)

    with open("results_model_comparison.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    best_name = results_df.iloc[0]["model"]
    print(f"\nBestes Modell nach ROC-AUC: {best_name}")

    with open("results_best_model.json", "w") as f:
        json.dump({"best_model": best_name,
                   "feature_names": feature_names}, f, indent=2)


if __name__ == "__main__":
    main()
