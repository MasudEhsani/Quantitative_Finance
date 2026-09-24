const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow,
  TableCell, WidthType, ShadingType, ImageRun, AlignmentType, BorderStyle,
  PageBreak, Header, Footer, PageNumber, Numbering, LevelFormat,
  convertInchesToTwip, VerticalAlign,
} = require("docx");

const FIG = (name) => path.join(__dirname, "..", "figures", name);
const PAGE_W = 12240, PAGE_H = 15840; // US Letter, DXA

// ---------- Ergebnisse laden ----------
const root = path.join(__dirname, "..");
const eda = JSON.parse(fs.readFileSync(path.join(root, "results_eda.json")));
const pre = JSON.parse(fs.readFileSync(path.join(root, "results_preprocessing.json")));
const modelCmp = JSON.parse(fs.readFileSync(path.join(root, "results_model_comparison.json")));
const bestModel = JSON.parse(fs.readFileSync(path.join(root, "results_best_model.json")));
const featImp = JSON.parse(fs.readFileSync(path.join(root, "results_feature_importance.json")));
const evalRes = JSON.parse(fs.readFileSync(path.join(root, "results_evaluation.json")));

const best = modelCmp.find((m) => m.model === bestModel.best_model);
const lr = modelCmp.find((m) => m.model === "LogisticRegression_baseline");
const rfSmote = modelCmp.find((m) => m.model === "RandomForest_SMOTE");
const rfWeighted = modelCmp.find((m) => m.model === "RandomForest_class_weight");
const cm = evalRes.confusion_matrix; // [[TN,FP],[FN,TP]]

const topImpurity = Object.entries(featImp.impurity_importance).slice(0, 5);
const topPerm = Object.entries(featImp.permutation_importance_mean).slice(0, 5);

// ---------- Hilfsfunktionen ----------
const COLOR = { primary: "1F4E79", accent: "C0392B", grey: "595959" };

function h1(text) {
  return new Paragraph({
    text, heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 180 },
  });
}
function h2(text) {
  return new Paragraph({
    text, heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 140 },
  });
}
function frage(text) {
  return new Paragraph({
    spacing: { before: 260, after: 60 },
    children: [new TextRun({ text: "F: " + text, bold: true, color: COLOR.primary })],
  });
}
function antwortStart() {
  return { spacing: { after: 60 } };
}
function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120 },
    children: [new TextRun({ text, ...opts })],
  });
}
function antwort(text) {
  return new Paragraph({
    spacing: { after: 160 },
    children: [new TextRun({ text: "A: " + text })],
  });
}
function bild(name, w, hgt, caption) {
  const buf = fs.readFileSync(FIG(name));
  const els = [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 60 },
      children: [
        new ImageRun({ data: buf, transformation: { width: w, height: hgt }, type: "png" }),
      ],
    }),
  ];
  if (caption) {
    els.push(new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
      children: [new TextRun({ text: caption, italics: true, size: 18, color: COLOR.grey })],
    }));
  }
  return els;
}
function cell(text, opts = {}) {
  return new TableCell({
    width: opts.width ? { size: opts.width, type: WidthType.DXA } : undefined,
    shading: opts.header ? { fill: "1F4E79", type: ShadingType.CLEAR } : undefined,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({
      alignment: opts.center ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: [new TextRun({
        text, bold: !!opts.header, color: opts.header ? "FFFFFF" : "000000", size: 20,
      })],
    })],
  });
}

// ---------- Modellvergleichstabelle ----------
const colW = [2600, 1400, 1400, 1600, 1400, 1200, 1400];
const headerRow = new TableRow({
  tableHeader: true,
  children: [
    cell("Modell", { header: true, width: colW[0] }),
    cell("ROC-AUC", { header: true, width: colW[1], center: true }),
    cell("PR-AUC", { header: true, width: colW[2], center: true }),
    cell("Precision", { header: true, width: colW[3], center: true }),
    cell("Recall", { header: true, width: colW[4], center: true }),
    cell("F1", { header: true, width: colW[5], center: true }),
    cell("Brier", { header: true, width: colW[6], center: true }),
  ],
});
const labelMap = {
  LogisticRegression_baseline: "Log. Regression (Baseline)",
  RandomForest_class_weight: "Random Forest (class_weight)",
  RandomForest_SMOTE: "Random Forest (SMOTE)",
  RandomForest_undersampled: "Random Forest (Undersampling)",
  GradientBoosting_class_weight: "Gradient Boosting (class_weight)",
  GradientBoosting_SMOTE: "Gradient Boosting (SMOTE)",
};
const sorted = [...modelCmp].sort((a, b) => b.roc_auc - a.roc_auc);
const bodyRows = sorted.map((m) => new TableRow({
  children: [
    cell(labelMap[m.model] || m.model, { width: colW[0] }),
    cell(m.roc_auc.toFixed(3), { width: colW[1], center: true }),
    cell(m.pr_auc.toFixed(3), { width: colW[2], center: true }),
    cell(m.precision_class1.toFixed(3), { width: colW[3], center: true }),
    cell(m.recall_class1.toFixed(3), { width: colW[4], center: true }),
    cell(m.f1_class1.toFixed(3), { width: colW[5], center: true }),
    cell(m.brier_score.toFixed(3), { width: colW[6], center: true }),
  ],
}));
const modelTable = new Table({
  width: { size: colW.reduce((a, b) => a + b, 0), type: WidthType.DXA },
  columnWidths: colW,
  rows: [headerRow, ...bodyRows],
});

// =====================================================================
// DOKUMENT
// =====================================================================
const doc = new Document({
  numbering: {
    config: [
      {
        reference: "bullet-list",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 480, hanging: 260 } } } }],
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: PAGE_W, height: PAGE_H },
          margin: { top: 1080, bottom: 1080, left: 1200, right: 1200 },
        },
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            alignment: AlignmentType.RIGHT,
            children: [new TextRun({ text: "Kreditrisikomodellierung – Projektbericht", size: 16, color: COLOR.grey })],
          })],
        }),
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [
              new TextRun({ text: "Seite ", size: 16, color: COLOR.grey }),
              new TextRun({ children: [PageNumber.CURRENT], size: 16, color: COLOR.grey }),
              new TextRun({ text: " von ", size: 16, color: COLOR.grey }),
              new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 16, color: COLOR.grey }),
            ],
          })],
        }),
      },
      children: [
        // ---------------- Titel ----------------
        new Paragraph({
          spacing: { before: 800, after: 100 },
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Kreditrisikomodellierung mit maschinellem Lernen", bold: true, size: 40, color: COLOR.primary })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 60 },
          children: [new TextRun({ text: "End-to-End-Projekt: Datenaufbereitung, Modellierung, Evaluation und Erklärbarkeit", size: 24, color: COLOR.grey })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 40 },
          children: [new TextRun({ text: "Datengrundlage: reale LendingClub-Kreditvergabedaten (öffentlich verfügbar)", size: 20, italics: true, color: COLOR.grey })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 400 },
          children: [new TextRun({ text: "Masud — September 2026", size: 20, color: COLOR.grey })],
        }),

        h2("Kurzübersicht"),
        p("Dieses Projekt implementiert eine vollständige Pipeline zur Kreditausfallprognose – von der Datenbeschaffung über Preprocessing und Modelltraining bis zur Evaluation und Erklärbarkeit. Es dient als praktische Grundlage für die untenstehenden Interview-Fragen zum Thema Kreditrisikomodellierung; alle Kennzahlen in den Antworten stammen aus dem tatsächlich ausgeführten Code (siehe Ordner src/), nicht aus Schätzungen."),
        p(`Datensatz: ${eda.n_rows} Kredite, ${eda.n_features} Merkmale, Ausfallquote ${eda.target_default_rate_pct}% – Bestes Modell: ${labelMap[best.model]} mit ROC-AUC ${best.roc_auc.toFixed(3)}.`, { bold: true }),

        new Paragraph({ children: [new PageBreak()] }),

        // ================= A. KREDITRISIKOMODELLIERUNG =================
        h1("A. Kreditrisikomodellierung"),

        frage("Wie sah der Datensatz aus (Größe, Merkmale, Zielvariable, Klassenungleichgewicht)?"),
        antwort(
          `Der Datensatz basiert auf realen LendingClub-Konsumentenkrediten (Peer-to-Peer-Kreditvergabe, USA) und enthält typische Merkmale wie Jahreseinkommen (annual_inc), Kredithöhe (loan_amnt), Laufzeit (term, 36 oder 60 Monate), Zinssatz, Rating/Subrating, Schuldendienstquote (DTI) sowie mehrere Zahlungshistorie-Merkmale (Zahlungsverzüge der letzten 2 Jahre, Kreditanfragen der letzten 6 Monate, Anzahl offener und gesamter Kreditlinien, negative öffentliche Einträge, revolvierender Saldo und dessen Auslastung). Zielvariable war der binäre Kreditausfall (default: 1 = notleidend/Charge-off, 0 = vollständig zurückgezahlt).\n\n` +
          `Der vollständige Originaldatensatz umfasst 150.000 Kredite. Da diese Projektumgebung keinen allgemeinen Internetzugang für einen Massendownload erlaubte, wurde eine reale Stichprobe von ${eda.n_rows} Zeilen über die öffentliche, paginierte Datasets-API bezogen (5 Blöcke à 100 Zeilen, gleichmäßig über den gesamten Indexbereich verteilt) – alle Werte sind unveränderte Originalbeobachtungen, keine synthetischen Daten. In einer produktiven Umgebung würde selbstverständlich der vollständige Datensatz verwendet; die hier beschriebene Methodik (Preprocessing, Modellauswahl, Evaluationsmetriken) skaliert unverändert.\n\n` +
          `Wie bei Kreditrisikodaten üblich war die Zielvariable unausgeglichen: ${eda.target_counts["0"]} Nicht-Ausfälle stehen ${eda.target_counts["1"]} Ausfällen gegenüber (Ausfallquote ${eda.target_default_rate_pct}%, Verhältnis ca. ${eda.imbalance_ratio_nondefault_to_default}:1). Das ist moderater als bei manchen Retail-Scoring-Datensätzen (z. B. dem populären „Give Me Some Credit“-Datensatz mit ca. 7% Ausfallquote), aber immer noch deutlich genug, um Resampling- und Gewichtungsstrategien zu rechtfertigen. Bei der Beschaeftigungsdauer (emp_length) fehlten ${eda.missing_values.emp_length} von ${eda.n_rows} Werten (${eda.missing_pct_emp_length}%); diese wurden durch den Median (${pre.median_emp_length_imputed} Jahre) ersetzt, ergänzt um ein binäres „Fehlt“-Flag, da das Fehlen selbst informativ sein kann.`
        ),

        frage("Warum Random Forest und Gradient Boosting – welche Alternativen hast du erwogen?"),
        antwort(
          "Ich habe mich für Random Forest und Gradient Boosting entschieden, weil beide gut mit nichtlinearen Zusammenhängen und gemischten Merkmalstypen umgehen und gleichzeitig Feature-Importance-Werte liefern, die sich gut kommunizieren lassen. Als interpretierbare Baseline habe ich eine logistische Regression herangezogen, um den zusätzlichen Nutzen der komplexeren Modelle zu quantifizieren. In diesem konkreten Projekt zeigte sich – bei allerdings kleiner Stichprobe – interessanterweise, dass die einfache logistische Regression im Recall für die Ausfallklasse konkurrenzfähig war; das unterstreicht, wie wichtig ein sauberer Baseline-Vergleich ist, bevor man automatisch zum komplexeren Modell greift. Für eine Weiterentwicklung würde ich das Setup um XGBoost oder LightGBM ergänzen, die in dieser Offline-Umgebung nicht installierbar waren, deren Boosting-Logik hier aber bereits mit sklearns GradientBoostingClassifier abgebildet wurde."
        ),

        frage("Wie hast du die Modelle evaluiert?"),
        antwort(
          `Da bei Kreditausfällen die Kosten falscher Entscheidungen asymmetrisch sind, habe ich neben AUC/ROC vor allem die Precision-Recall-Kurve (PR-AUC) sowie Precision und Recall für die Minderheitsklasse betrachtet und zusätzlich die Kalibrierung der vorhergesagten Ausfallwahrscheinlichkeiten (Brier Score, Reliability-Diagramm) geprüft. Auf dem Testset (n=${pre.n_test}, davon ${pre.n_test_defaults} tatsächliche Ausfälle) erzielte das beste Modell – ${labelMap[best.model]} – einen ROC-AUC von ${best.roc_auc.toFixed(3)} und einen PR-AUC von ${best.pr_auc.toFixed(3)} (zum Vergleich: eine zufällige Einstufung läge bei PR-AUC ≈ ${(pre.test_default_rate_pct/100).toFixed(3)}, der Prävalenz der Ausfallklasse). Die logistische Regression erreichte einen ROC-AUC von ${lr.roc_auc.toFixed(3)} bei deutlich höherem Recall (${(lr.recall_class1*100).toFixed(1)}% vs. ${(best.recall_class1*100).toFixed(1)}% beim besten Random-Forest-Modell) – ein klassischer Precision-Recall-Trade-off, den man in der Praxis über die Wahl des Entscheidungsschwellenwerts und die Kosten falscher Positiv-/Negativ-Entscheidungen steuert, nicht allein über die Modellwahl. Bei einem Schwellenwert von 0.5 ergab die Konfusionsmatrix des besten Modells: ${cm[0][0]} korrekt als „kein Ausfall“, ${cm[0][1]} fälschlich als „Ausfall“, ${cm[1][0]} übersehene tatsächliche Ausfälle und ${cm[1][1]} korrekt erkannte Ausfälle – die vergleichsweise niedrige Trefferquote bei den tatsächlichen Ausfällen zeigt exemplarisch, warum der Schwellenwert in der Praxis anhand der Kostenmatrix (Kosten eines übersehenen Ausfalls vs. Kosten einer abgelehnten guten Bonität) und nicht per Standardwert 0.5 festgelegt werden sollte. Alle sechs trainierten Modellvarianten sind in Tabelle 1 gegenübergestellt.`
        ),
        ...bild("05_roc_kurven.png", 400, 343, "Abbildung 1: ROC-Kurven aller sechs Modellvarianten im Vergleich."),
        ...bild("06_pr_kurven.png", 400, 343, "Abbildung 2: Precision-Recall-Kurven – aussagekräftiger als ROC bei ausgeprägtem Klassenungleichgewicht."),
        p("Tabelle 1: Modellvergleich auf dem Testset (sortiert nach ROC-AUC).", { italics: true, color: COLOR.grey, size: 18 }),
        modelTable,
        p(""),
        ...bild("07_kalibrierung.png", 380, 349, `Abbildung 3: Kalibrierungskurve des besten Modells (${labelMap[best.model]}) – Abweichungen von der Diagonalen zeigen, wo vorhergesagte Wahrscheinlichkeiten systematisch zu hoch oder zu niedrig liegen.`),
        ...bild("08_konfusionsmatrix.png", 360, 324, `Abbildung 4: Konfusionsmatrix des besten Modells bei Schwellenwert 0{,}5.`),

        frage("Wie bist du mit Klassenungleichgewicht umgegangen?"),
        antwort(
          "Um mit dem Ungleichgewicht umzugehen, habe ich drei Strategien parallel implementiert und verglichen: (1) klassengewichtete Verlustfunktionen (class_weight='balanced' bei logistischer Regression und Random Forest, entsprechend gewichtete sample_weight bei Gradient Boosting), (2) Oversampling der Minderheitsklasse und (3) Undersampling der Mehrheitsklasse. Da das imbalanced-learn-Paket in dieser Offline-Umgebung nicht nachinstallierbar war, habe ich SMOTE (Synthetic Minority Oversampling Technique) selbst nach dem Originalalgorithmus implementiert: Für jedes synthetische Beispiel wird ein Punkt der Minderheitsklasse mit einem seiner k nächsten Nachbarn (ebenfalls Minderheitsklasse) linear interpoliert. Alle Modelle wurden konsistent auf demselben, unveränderten Testset evaluiert und anhand von Metriken bewertet, die robust gegenüber Ungleichgewicht sind (ROC-AUC, PR-AUC statt reiner Accuracy). In diesem Projekt erzielte die SMOTE-Variante des Random Forest den besten ROC-AUC, während die klassengewichtete Variante ohne Resampling den schwächsten Recall zeigte – ein Hinweis darauf, dass bei kleineren Datenmengen synthetisches Oversampling der reinen Gewichtsanpassung überlegen sein kann, was sich mit größeren Trainingsdaten aber durchaus ändern kann und in Produktion erneut getestet werden sollte."
        ),

        frage("Wie würdest du Feature Importance einem Fachbereich erklären?"),
        antwort(
          `Über Feature-Importance-Werte oder Ansätze wie SHAP lässt sich zeigen, welche Faktoren den größten Einfluss auf die Risikoeinschätzung haben. In diesem Projekt konnte SHAP nicht berechnet werden, da das shap-Paket in der Offline-Umgebung nicht installierbar war; als methodisch verwandte, modellagnostische Alternative habe ich Permutation Importance eingesetzt (mit ROC-AUC als Scoring-Metrik, 30 Wiederholungen). Nach der klassischen (impurity-basierten) Random-Forest-Importance dominieren revolvierende Kreditauslastung, Zinssatz, revolvierender Saldo, Gesamtzahl der Kreditlinien und das LendingClub-Subrating (siehe Abbildung 5). Die Permutation Importance bestätigt Zinssatz und Anzahl der Kreditlinien als zentrale Treiber, zeigt aber bei einzelnen Merkmalen (insb. der revolvierenden Auslastung) aufgrund des kleinen Testsets (n=${pre.n_test}) instabile bzw. sogar leicht negative Werte – ein Artefakt der geringen Stichprobengröße, das bei Verwendung des vollständigen 150.000-Zeilen-Datensatzes verschwinden sollte. Fachlich ließe sich das etwa so zusammenfassen: „Der Zinssatz, den ein Kreditnehmer bereits erhält, sowie wie stark er seine bestehenden Kreditlinien ausreizt, sagen mehr über das Ausfallrisiko aus als die reine Kredithöhe.“ Aus meiner Lehrtätigkeit und der Betreuung von Studierenden habe ich gelernt, solche technischen Ergebnisse in einfache, bildhafte Sprache zu übersetzen, ohne die fachliche Genauigkeit zu verlieren – zum Beispiel durch den Vergleich mit einem Kontostand, der regelmäßig bis an die Grenze ausgereizt wird.`
        ),
        ...bild("09_feature_importance_rf.png", 400, 315, "Abbildung 5: Top-10-Merkmale nach Random-Forest-Importance (Gini-basiert)."),
        ...bild("10_permutation_importance.png", 400, 315, "Abbildung 6: Top-10-Merkmale nach Permutation Importance (SHAP-Ersatz, modellagnostisch)."),

        frage("Wie würdest du dieses Modell produktiv nehmen?"),
        antwort(
          "Ich würde das Modell in eine modulare Pipeline mit automatisierten Tests und Qualitätskontrollen für neue Daten überführen (Schema-Validierung, Wertebereichsprüfungen, Drift-Erkennung bei Eingabemerkmalen), die Vorhersagegüte laufend überwachen (z. B. Drift in AUC oder Kalibrierung gegen ein rollierendes Referenzfenster) und einen klar definierten Retraining-Zyklus festlegen, sobald sich Datenverteilung oder Geschäftsbedingungen ändern – analog zu den produktionsnahen Python-Anwendungen, die ich in Bochum entwickelt habe. Konkret würde ich außerdem: den Entscheidungsschwellenwert anhand einer Kostenmatrix statt 0,5 festlegen, die Modellversionen und Trainingsdaten versionieren (z. B. via MLflow oder DVC), und A/B-Tests bzw. Champion-Challenger-Vergleiche vor jedem Rollout eines neuen Modells durchführen."
        ),

        frage("Was war die größte Herausforderung im Projekt?"),
        antwort(
          "Die größte Herausforderung war paradoxerweise nicht die Modellierung selbst, sondern der Datenzugriff: Die Ausführungsumgebung erlaubte keinen allgemeinen Internetzugriff für Paketinstallationen (imbalanced-learn, shap, xgboost) oder Massendownloads, sodass ich sowohl SMOTE als auch eine Permutation-Importance-Alternative zu SHAP von Grund auf selbst implementieren und eine repräsentative Stichprobe über eine öffentliche, paginierte API statt eines direkten Bulk-Downloads beziehen musste. Das hat mich noch einmal dafür sensibilisiert, wie wichtig es ist, Kernalgorithmen tatsächlich zu verstehen und nicht nur als Blackbox-Bibliotheksaufruf zu kennen. Inhaltlich lag die zweite Herausforderung in der kleinen Stichprobe (500 statt 150.000 Zeilen): Sie führte zu spürbar breiteren Konfidenzintervallen bei den Metriken und instabilen Permutation-Importance-Werten – ein guter Anlass, im Bericht transparent auf die Grenzen der Aussagekraft hinzuweisen, statt die Ergebnisse überzuinterpretieren."
        ),

        frage("Wie ließe sich das Konzept auf Handels-Use-Cases übertragen?"),
        antwort(
          "Das gleiche methodische Vorgehen – Klassifikation und Scoring auf Basis historischer Daten, Umgang mit Klassenungleichgewicht, Kalibrierung und Erklärbarkeit – lässt sich direkt auf das Ausfallrisiko von Lieferanten, Bonitätsprüfungen bei Ratenzahlungsmodellen oder Betrugs-Scoring bei Zahlungsvorgängen übertragen. In allen Fällen handelt es sich um seltene, kostenintensive Ereignisse, bei denen Precision-Recall-Analysen und eine sorgfältige Schwellenwertwahl wichtiger sind als eine hohe Gesamttrefferquote."
        ),

        new Paragraph({ children: [new PageBreak()] }),

        // ================= ANHANG =================
        h1("Anhang"),
        h2("Projektstruktur"),
        p("Das Projekt ist als reproduzierbare Python-Pipeline aufgebaut:"),
        ...["data/ – Rohdaten-Batches, kombinierte CSV, Train/Test-Splits",
            "src/01_eda.py – Explorative Datenanalyse und Diagramme",
            "src/02_preprocessing.py – Imputation, Feature-Auswahl, Train/Test-Split",
            "src/03_modeling.py – Training von 6 Modellvarianten (LR, RF, GB × 3 Imbalance-Strategien) inkl. eigener SMOTE-Implementierung",
            "src/04_evaluation.py – ROC-/PR-Kurven, Kalibrierung, Konfusionsmatrix",
            "src/05_feature_importance.py – Impurity- und Permutation-Importance",
            "figures/ – alle Diagramme (PNG)",
            "models/ – gespeicherte, trainierte Modelle (joblib)",
            "reports/ – dieser Bericht"]
          .map((t) => new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, spacing: { after: 60 }, children: [new TextRun(t)] })),

        h2("Datenherkunft und Transparenzhinweis"),
        p("Datenquelle: öffentlicher Datensatz „RPD123-byte/credit-risk-datasets“ (Datei lending_club_clean.csv, 150.000 Zeilen, 20 Spalten), bezogen über die öffentliche Hugging-Face-Datasets-Server-API (https://datasets-server.huggingface.co). Es handelt sich um bereinigte, reale LendingClub-Kreditvergabedaten (US-amerikanische Peer-to-Peer-Kredite). Kategoriale Merkmale (grade, sub_grade, home_ownership, verification_status, purpose) liegen im Quelldatensatz bereits ordinal/nominal kodiert vor; die exakte Zuordnung der Kategoriewerte zu Klartextbezeichnungen war in der verfügbaren Dateiversion nicht dokumentiert, weshalb im Bericht bewusst auf spekulative Klartext-Labels (z. B. konkrete Wohnverhältnis-Kategorien) verzichtet wurde."),
        p("Aus den oben genannten Umgebungsgründen (kein allgemeiner Bulk-Internetzugriff) wurde eine Stichprobe von 500 Zeilen statt des vollständigen Datensatzes verwendet. Alle Zahlen in diesem Bericht sind real berechnete Ergebnisse dieser Stichprobe – keine Simulationen. Für ein produktives oder prüfungsrelevantes Projekt wird empfohlen, den vollständigen 150.000-Zeilen-Datensatz zu laden (z. B. lokal per Kaggle-/HuggingFace-CLI) und die identische Pipeline (src/01–05) erneut auszuführen – der Code ist dafür ohne Änderungen einsatzbereit, sofern data/lending_club_sample.csv durch den vollständigen Datensatz ersetzt wird."),

        h2("Grenzen dieser Analyse"),
        ...["Stichprobengröße (n=500, davon nur 150 im Testset) führt zu breiten Konfidenzintervallen; Metrikunterschiede zwischen Modellen sind mit Vorsicht zu interpretieren.",
            "Kategoriale Merkmale wurden als bereits kodierte Ganzzahlen übernommen, ohne die ursprünglichen Kategorienamen zu kennen – für Baumverfahren unproblematisch, schränkt aber die inhaltliche Interpretierbarkeit einzelner Kategorien ein.",
            "SHAP-Werte konnten nicht berechnet werden (Paketinstallation in der Umgebung nicht möglich); Permutation Importance wurde als methodisch verwandter Ersatz verwendet.",
            "SMOTE wurde selbst implementiert statt über imbalanced-learn bezogen; die Kernlogik (k-NN-Interpolation) entspricht dem Originalverfahren, ist aber nicht auf Randfälle (z. B. kategoriale Merkmale) spezialisiert wie manche Bibliotheksvarianten."]
          .map((t) => new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, spacing: { after: 60 }, children: [new TextRun(t)] })),

        h2("Explorative Datenanalyse (ergänzende Abbildungen)"),
        ...bild("01_klassenverteilung.png", 320, 256, "Abbildung 7: Klassenverteilung der Zielvariable."),
        ...bild("02_loan_income_by_class.png", 480, 196, "Abbildung 8: Kredithöhe und Jahreseinkommen nach Ausfallstatus."),
        ...bild("04_zahlungshistorie.png", 380, 285, "Abbildung 9: Ausfallquote nach Anzahl der Zahlungsverzüge der letzten 2 Jahre."),
        ...bild("03_korrelationsmatrix.png", 480, 384, "Abbildung 10: Korrelationsmatrix aller numerischen Merkmale inkl. Zielvariable."),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  const outPath = path.join(__dirname, "Kreditrisikomodellierung_Projektbericht.docx");
  fs.writeFileSync(outPath, buf);
  console.log("Bericht gespeichert:", outPath);
});
