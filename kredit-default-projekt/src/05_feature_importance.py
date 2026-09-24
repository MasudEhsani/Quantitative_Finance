"""
05_feature_importance.py
--------------------------
Feature-Importance-Analyse fuer das Random-Forest-Modell (class_weight-
Variante, da auf den unveraenderten Originaldaten trainiert):
  1) impurity-basierte Feature Importance (RF-eigene Metrik)
  2) Permutation Importance (modellagnostisch, robuster gegenueber
     verzerrten impurity-basierten Werten bei Merkmalen mit vielen
     Kategorien)

Hinweis: SHAP-Werte konnten in dieser Umgebung nicht berechnet werden,
da das shap-Paket offline nicht installierbar war (kein allgemeiner
Internetzugriff fuer pip in dieser Session). Permutation Importance
liefert methodisch einen vergleichbaren, modellagnostischen Blick auf
den Einfluss einzelner Merkmale und wird hier als Ersatz verwendet;
fuer eine produktive Umsetzung wird SHAP zur granularen, lokalen
Erklaerung einzelner Vorhersagen empfohlen (siehe Bericht).
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

FIG_DIR = "figures"

FEATURE_LABELS = {
    "loan_amnt": "Kredithoehe",
    "term": "Laufzeit",
    "int_rate": "Zinssatz",
    "installment": "Monatliche Rate",
    "grade": "LendingClub-Rating",
    "sub_grade": "LendingClub-Subrating",
    "emp_length": "Beschaeftigungsdauer",
    "home_ownership": "Wohnverhaeltnis",
    "annual_inc": "Jahreseinkommen",
    "verification_status": "Verifizierungsstatus",
    "purpose": "Verwendungszweck",
    "dti": "Schuldendienstquote (DTI)",
    "delinq_2yrs": "Zahlungsverzuege (2J)",
    "inq_last_6mths": "Kreditanfragen (6M)",
    "open_acc": "Offene Kreditlinien",
    "pub_rec": "Negative Eintraege",
    "revol_bal": "Revolv. Saldo",
    "revol_util": "Revolv. Auslastung",
    "total_acc": "Kreditlinien gesamt",
    "emp_length_missing": "Beschaeftigungsdauer fehlt (Flag)",
}


def main():
    model = joblib.load("models/random_forest_class_weight.joblib")
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv")["default"]

    # 1) Impurity-basierte Importance
    importances = pd.Series(model.feature_importances_, index=X_test.columns)
    importances = importances.sort_values(ascending=False)

    # 2) Permutation Importance (auf Testdaten, ROC-AUC als Scoring)
    perm = permutation_importance(
        model, X_test, y_test, n_repeats=30, random_state=42,
        scoring="roc_auc", n_jobs=-1,
    )
    perm_importances = pd.Series(perm.importances_mean, index=X_test.columns)
    perm_std = pd.Series(perm.importances_std, index=X_test.columns)
    perm_importances = perm_importances.sort_values(ascending=False)

    results = {
        "impurity_importance": {
            FEATURE_LABELS.get(k, k): round(v, 4)
            for k, v in importances.items()
        },
        "permutation_importance_mean": {
            FEATURE_LABELS.get(k, k): round(v, 4)
            for k, v in perm_importances.items()
        },
    }
    with open("results_feature_importance.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # --- Plot: Top 10 impurity-based ---
    top10 = importances.head(10)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    labels = [FEATURE_LABELS.get(k, k) for k in top10.index]
    ax.barh(labels[::-1], top10.values[::-1], color="#4C72B0")
    ax.set_xlabel("Feature Importance (Gini-basiert)")
    ax.set_title("Top-10-Merkmale nach Random-Forest-Importance")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/09_feature_importance_rf.png", dpi=150)
    plt.close()

    # --- Plot: Top 10 permutation importance mit Fehlerbalken ---
    top10_perm = perm_importances.head(10)
    top10_perm_std = perm_std.loc[top10_perm.index]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    labels = [FEATURE_LABELS.get(k, k) for k in top10_perm.index]
    ax.barh(labels[::-1], top10_perm.values[::-1],
            xerr=top10_perm_std.values[::-1], color="#55A868", capsize=3)
    ax.set_xlabel("Permutation Importance (Abfall ROC-AUC)")
    ax.set_title("Top-10-Merkmale nach Permutation Importance\n(modellagnostisch, SHAP-Ersatz)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/10_permutation_importance.png", dpi=150)
    plt.close()

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
