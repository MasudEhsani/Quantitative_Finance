"""
02_preprocessing.py
--------------------
Datenaufbereitung: fehlende Werte, Train/Test-Split, Skalierung.
Ergebnis wird als .npz / .csv fuer die folgenden Schritte gespeichert.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json

DATA_PATH = "data/lending_club_sample.csv"
OUT_DIR = "data"

FEATURES = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "purpose", "dti", "delinq_2yrs", "inq_last_6mths", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc",
]
TARGET = "default"


def main():
    df = pd.read_csv(DATA_PATH)

    # --- Fehlende Werte: emp_length ---
    # Fehlende Beschaeftigungsdauer wird durch den Median ersetzt; zusaetzlich
    # wird ein Missing-Indikator angelegt, da das Fehlen selbst informativ
    # sein kann (z. B. Selbststaendige / nicht angegeben).
    df["emp_length_missing"] = df["emp_length"].isnull().astype(int)
    median_emp_length = df["emp_length"].median()
    df["emp_length"] = df["emp_length"].fillna(median_emp_length)

    feature_cols = FEATURES + ["emp_length_missing"]

    X = df[feature_cols].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    X_train.to_csv(f"{OUT_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{OUT_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{OUT_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{OUT_DIR}/y_test.csv", index=False)

    info = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train_default_rate_pct": round(y_train.mean() * 100, 2),
        "test_default_rate_pct": round(y_test.mean() * 100, 2),
        "n_train_defaults": int(y_train.sum()),
        "n_test_defaults": int(y_test.sum()),
        "median_emp_length_imputed": float(median_emp_length),
        "feature_columns": feature_cols,
    }
    with open("results_preprocessing.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
