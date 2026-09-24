"""
01_eda.py
---------
Explorative Datenanalyse (EDA) fuer das Kreditausfall-Datenset.

Datenquelle: Ein realer, oeffentlich zugaenglicher Datensatz auf Basis von
LendingClub-Kreditvergabedaten (Peer-to-Peer-Konsumentenkredite, USA),
bezogen ueber die oeffentliche Hugging-Face-Datasets-API
(RPD123-byte/credit-risk-datasets, Datei lending_club_clean.csv,
150.000 Zeilen im Original). Aus Gruenden der Umgebungsbeschraenkung
(kein allgemeiner Internetzugang fuer Massendownloads in dieser Session)
wurde eine Stichprobe von 500 Zeilen ueber die paginierte, oeffentliche
Datasets-Server-API gezogen (5 Bloecke von je 100 Zeilen, verteilt ueber
den gesamten Indexbereich 0-149.999). Alle Werte sind reale, unveraenderte
Beobachtungen aus dem Originaldatensatz - es wurden keine Werte simuliert
oder synthetisiert.

Zielvariable: `default` (1 = Kredit notleidend / Charge-Off, 0 = vollstaendig
zurueckgezahlt).
"""
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")

DATA_PATH = "data/lending_club_sample.csv"
FIG_DIR = "figures"
RESULTS_PATH = "results_eda.json"

FEATURE_LABELS = {
    "loan_amnt": "Kredithoehe (USD)",
    "term": "Laufzeit (Monate)",
    "int_rate": "Zinssatz (%)",
    "installment": "Monatliche Rate (USD)",
    "grade": "LendingClub-Rating (0=A ... 6=G)",
    "sub_grade": "LendingClub-Subrating (0-34)",
    "emp_length": "Beschaeftigungsdauer (Jahre)",
    "home_ownership": "Wohnverhaeltnis (kodiert)",
    "annual_inc": "Jahreseinkommen (USD)",
    "verification_status": "Einkommens-Verifizierungsstatus (kodiert)",
    "purpose": "Verwendungszweck (kodiert)",
    "dti": "Schuldendienstquote / DTI (%)",
    "delinq_2yrs": "Zahlungsverzuege letzte 2 Jahre (Anzahl)",
    "inq_last_6mths": "Kreditanfragen letzte 6 Monate (Anzahl)",
    "open_acc": "Offene Kreditlinien (Anzahl)",
    "pub_rec": "Negative oeffentliche Eintraege (Anzahl)",
    "revol_bal": "Revolvierender Saldo (USD)",
    "revol_util": "Auslastung revolvierender Kredit (%)",
    "total_acc": "Kreditlinien insgesamt (Anzahl)",
}


def main():
    df = pd.read_csv(DATA_PATH)

    n_rows, n_cols = df.shape
    n_features = n_cols - 1  # ohne Zielvariable

    target_counts = df["default"].value_counts().to_dict()
    target_rate = df["default"].mean()

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    numeric_summary = df.describe().T

    results = {
        "n_rows": int(n_rows),
        "n_features": int(n_features),
        "target_counts": {str(k): int(v) for k, v in target_counts.items()},
        "target_default_rate_pct": round(target_rate * 100, 2),
        "imbalance_ratio_nondefault_to_default": round(
            target_counts[0] / target_counts[1], 2
        ),
        "missing_values": {k: int(v) for k, v in missing.to_dict().items()},
        "missing_pct_emp_length": round(
            df["emp_length"].isnull().mean() * 100, 2
        ),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(json.dumps(results, indent=2, ensure_ascii=False))

    # --- Plot 1: Klassenverteilung ---
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["default"].value_counts().sort_index()
    labels = ["Kein Ausfall (0)", "Ausfall (1)"]
    colors = ["#4C72B0", "#C44E52"]
    bars = ax.bar(labels, counts.values, color=colors)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 3, str(v),
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Anzahl Kredite")
    ax.set_title(f"Klassenverteilung der Zielvariable (n={n_rows})\n"
                 f"Ausfallquote: {target_rate*100:.1f}%")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/01_klassenverteilung.png", dpi=150)
    plt.close()

    # --- Plot 2: Kredithoehe & Einkommen nach Klasse ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col, label in zip(
        axes, ["loan_amnt", "annual_inc"],
        ["Kredithoehe (USD)", "Jahreseinkommen (USD, gekappt bei 250k)"]
    ):
        plot_df = df.copy()
        if col == "annual_inc":
            plot_df[col] = plot_df[col].clip(upper=250000)
        sns.boxplot(data=plot_df, x="default", y=col, ax=ax,
                    hue="default", palette=colors, legend=False)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Kein Ausfall", "Ausfall"])
        ax.set_xlabel("")
        ax.set_ylabel(label)
    plt.suptitle("Kredithoehe und Einkommen nach Ausfallstatus")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/02_loan_income_by_class.png", dpi=150)
    plt.close()

    # --- Plot 3: Korrelationsmatrix (numerische Merkmale) ---
    num_cols = list(FEATURE_LABELS.keys()) + ["default"]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="RdBu_r", center=0, ax=ax, annot=False,
                cbar_kws={"label": "Korrelationskoeffizient"})
    ax.set_title("Korrelationsmatrix der Merkmale (inkl. Zielvariable)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/03_korrelationsmatrix.png", dpi=150)
    plt.close()

    # --- Plot 4: Zahlungshistorie (delinq_2yrs) vs Ausfall ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ct = pd.crosstab(df["delinq_2yrs"].clip(upper=3), df["default"],
                      normalize="index") * 100
    ct.plot(kind="bar", stacked=True, color=colors, ax=ax)
    ax.set_xlabel("Zahlungsverzuege letzte 2 Jahre (>=3 zusammengefasst)")
    ax.set_ylabel("Anteil (%)")
    ax.set_title("Ausfallquote nach Zahlungshistorie")
    ax.legend(["Kein Ausfall", "Ausfall"], title="Status")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/04_zahlungshistorie.png", dpi=150)
    plt.close()

    print("\nEDA abgeschlossen. Diagramme gespeichert in", FIG_DIR)


if __name__ == "__main__":
    main()
