"""
04_evaluation.py
------------------
Erstellt Vergleichsgrafiken (ROC-Kurven, Precision-Recall-Kurven,
Kalibrierungskurve, Konfusionsmatrix) fuer die trainierten Modelle.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_auc_score

FIG_DIR = "figures"

with open("results_curves.json") as f:
    curves = json.load(f)
y_test = np.array(curves["y_test"])

results_df = pd.read_csv("results_model_comparison.csv")

MODEL_LABELS_DE = {
    "LogisticRegression_baseline": "Logistische Regression (Baseline)",
    "RandomForest_class_weight": "Random Forest (class_weight)",
    "RandomForest_SMOTE": "Random Forest (SMOTE)",
    "RandomForest_undersampled": "Random Forest (Undersampling)",
    "GradientBoosting_class_weight": "Gradient Boosting (class_weight)",
    "GradientBoosting_SMOTE": "Gradient Boosting (SMOTE)",
}

COLORS = plt.cm.tab10.colors


def plot_roc():
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (name, data) in enumerate(curves["roc"].items()):
        auc = results_df.loc[results_df["model"] == name, "roc_auc"].values[0]
        ax.plot(data["fpr"], data["tpr"], label=f"{MODEL_LABELS_DE.get(name, name)} (AUC={auc:.3f})",
                color=COLORS[i % 10], linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Zufallsmodell (AUC=0.5)")
    ax.set_xlabel("Falsch-Positiv-Rate")
    ax.set_ylabel("Richtig-Positiv-Rate")
    ax.set_title("ROC-Kurven im Modellvergleich")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/05_roc_kurven.png", dpi=150)
    plt.close()


def plot_pr():
    fig, ax = plt.subplots(figsize=(7, 6))
    baseline_rate = y_test.mean()
    for i, (name, data) in enumerate(curves["pr"].items()):
        pr_auc = results_df.loc[results_df["model"] == name, "pr_auc"].values[0]
        ax.plot(data["recall"], data["precision"],
                label=f"{MODEL_LABELS_DE.get(name, name)} (PR-AUC={pr_auc:.3f})",
                color=COLORS[i % 10], linewidth=2)
    ax.axhline(baseline_rate, linestyle="--", color="gray",
               label=f"Zufallsmodell (Praevalenz={baseline_rate:.3f})")
    ax.set_xlabel("Recall (Trefferquote)")
    ax.set_ylabel("Precision (Praezision)")
    ax.set_title("Precision-Recall-Kurven im Modellvergleich\n"
                 "(besonders relevant bei Klassenungleichgewicht)")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/06_pr_kurven.png", dpi=150)
    plt.close()


def plot_calibration_and_confusion(best_model_name):
    import joblib
    name_to_file = {
        "LogisticRegression_baseline": ("models/logreg_baseline.joblib", True),
        "RandomForest_class_weight": ("models/random_forest_class_weight.joblib", False),
        "RandomForest_SMOTE": ("models/random_forest_smote.joblib", False),
        "GradientBoosting_class_weight": ("models/gradient_boosting_class_weight.joblib", False),
        "GradientBoosting_SMOTE": ("models/gradient_boosting_smote.joblib", False),
    }
    if best_model_name not in name_to_file:
        best_model_name = "RandomForest_class_weight"
    path, needs_scaler = name_to_file[best_model_name]
    model = joblib.load(path)

    X_test = pd.read_csv("data/X_test.csv")
    if needs_scaler:
        scaler = joblib.load("models/scaler.joblib")
        X_eval = scaler.transform(X_test)
    else:
        X_eval = X_test.values

    proba = model.predict_proba(X_eval)[:, 1]
    pred = (proba >= 0.5).astype(int)

    # Kalibrierungskurve
    fig, ax = plt.subplots(figsize=(6, 5.5))
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=8, strategy="quantile")
    ax.plot(mean_pred, frac_pos, marker="o", label=MODEL_LABELS_DE.get(best_model_name, best_model_name))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfekte Kalibrierung")
    ax.set_xlabel("Mittlere vorhergesagte Ausfallwahrscheinlichkeit")
    ax.set_ylabel("Beobachtete Ausfallrate")
    ax.set_title(f"Kalibrierungskurve: {MODEL_LABELS_DE.get(best_model_name, best_model_name)}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/07_kalibrierung.png", dpi=150)
    plt.close()

    # Konfusionsmatrix
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Kein Ausfall", "Ausfall"],
                yticklabels=["Kein Ausfall", "Ausfall"])
    ax.set_xlabel("Vorhergesagt")
    ax.set_ylabel("Tatsaechlich")
    ax.set_title(f"Konfusionsmatrix: {MODEL_LABELS_DE.get(best_model_name, best_model_name)}\n"
                 f"(Schwellenwert 0.5)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/08_konfusionsmatrix.png", dpi=150)
    plt.close()

    return {"confusion_matrix": cm.tolist()}


def main():
    plot_roc()
    plot_pr()
    with open("results_best_model.json") as f:
        best = json.load(f)
    cm_result = plot_calibration_and_confusion(best["best_model"])
    with open("results_evaluation.json", "w") as f:
        json.dump(cm_result, f, indent=2)
    print("Evaluation abgeschlossen.")
    print(cm_result)


if __name__ == "__main__":
    main()
