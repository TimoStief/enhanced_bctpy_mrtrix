# -------------------------------
# Automatische Paketprüfung + Installation
# -------------------------------
import subprocess
import sys
import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import bct
from scipy.stats import f_oneway
from statsmodels.stats.anova import AnovaRM

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = ["pandas", "numpy", "bctpy", "openpyxl", "scipy", "statsmodels"]
for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        print(f"{pkg} fehlt, wird installiert...")
        install(pkg)

# -------------------------------
# Pfad zu Testmatrizen
# -------------------------------
# Get project root (one level up from scripts/)
project_root = Path(__file__).resolve().parent.parent
root = project_root / "Test_matrizen"
sessions = ["ses-1", "ses-2", "ses-3", "ses-4"]

results_dir = project_root / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------
# XLSX → NPY Konvertierung (falls nötig)
# -------------------------------
for ses in sessions:
    ses_folder = root / ses
    if not ses_folder.is_dir():
        continue

    xlsx_files = list(ses_folder.glob("*.xlsx"))
    npy_files = list(ses_folder.glob("*.npy"))

    if xlsx_files and not npy_files:
        print(f"Konvertiere XLSX → NPY in {ses_folder}")

        for file in xlsx_files:
            name = file.stem

            df_mat = pd.read_excel(file, header=None)
            df_mat = df_mat.dropna(axis=0, how="all").dropna(axis=1, how="all")
            A = df_mat.to_numpy(dtype=float)

            if A.shape[0] != A.shape[1]:
                raise ValueError(f"Matrix nicht quadratisch: {file}")

            npy_path = ses_folder / f"{name}.npy"
            np.save(npy_path, A)
            print(f"Gespeichert: {npy_path}")

# -------------------------------
# Daten laden und Metriken berechnen
# -------------------------------
all_data = []

for ses in sessions:
    ses_folder = root / ses
    files = list(ses_folder.glob("*.npy"))
    if len(files) == 0:
        print(f"Warnung: Keine Dateien gefunden in {ses_folder}")
        continue

    for f in files:
        subject = f.stem.split("_")[0]
        A = np.load(f)
        A_bin = (A > 0).astype(int)

        # Globale Werte
        avg_degree = np.mean(bct.degrees_und(A_bin))
        avg_strength = np.mean(bct.strengths_und(A))

        # Nodalwerte
        degree_vec = bct.degrees_und(A_bin)
        strength_vec = bct.strengths_und(A)

        data_row = {
            "subject": subject,
            "session": ses,
            "avg_degree": avg_degree,
            "avg_strength": avg_strength
        }

        for i, val in enumerate(degree_vec):
            data_row[f"deg_region_{i + 1}"] = val
        for i, val in enumerate(strength_vec):
            data_row[f"str_region_{i + 1}"] = val

        all_data.append(data_row)

# -------------------------------
# DataFrame erstellen
# -------------------------------
df = pd.DataFrame(all_data)

if df.empty:
    raise RuntimeError("Keine Daten gefunden! Prüfe den Pfad zu Testmatrizen und Session-Ordnern.")

xlsx_file = results_dir / "graph_metrics_all_sessions.xlsx"
df.to_excel(xlsx_file, index=False)
print(f"Datei gespeichert: {xlsx_file}")
print("Spalten in df:", df.columns.tolist())
print(df.head())

# -------------------------------
# ANOVA global
# -------------------------------
deg_ses = [df[df["session"]==s]["avg_degree"] for s in sessions if s in df["session"].values]
f_val, p_val = f_oneway(*deg_ses)
print(f"Global Degree ANOVA: F={f_val:.3f}, p={p_val:.3f}")

# -------------------------------
# Nodal ANOVAs
# -------------------------------
anova_txt = os.path.join(results_dir, "anova_results.txt")
excel_rows = []

with open(anova_txt, "w") as f:
    f.write(f"Global Degree ANOVA: F={f_val:.3f}, p={p_val:.3f}\n\n")
    excel_rows.append({"type":"global","region":"all","F_value":f_val,"p_value":p_val})

    # Nodal ANOVAs basierend auf allen Spalten deg_region_*
    deg_cols = [c for c in df.columns if c.startswith("deg_region_")]
    for col in deg_cols:
        aov = AnovaRM(df, depvar=col, subject="subject", within=["session"]).fit()
        header = f"{col} ANOVA:\n"
        f.write(header)
        f.write(aov.summary().as_text() + "\n\n")
        print(header)
        print(aov.summary())

        aov_table = aov.anova_table
        F_value = aov_table.loc["session", "F Value"]
        p_value = aov_table.loc["session", "Pr > F"]
        excel_rows.append({"type":"nodal","region":col,"F_value":F_value,"p_value":p_value})

# Excel-Tabelle speichern
anova_xlsx = os.path.join(results_dir, "anova_results.xlsx")
pd.DataFrame(excel_rows).to_excel(anova_xlsx, index=False)
print(f"ANOVA-Ergebnisse gespeichert als TXT: {anova_txt}")
print(f"ANOVA-Ergebnisse gespeichert als XLSX: {anova_xlsx}")
