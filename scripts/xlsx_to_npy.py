import os
from pathlib import Path
import numpy as np
import pandas as pd

# -------------------------------
# Pfade
# -------------------------------
# Get project root (one level up from scripts/)
project_root = Path(__file__).resolve().parent.parent
input_file = project_root / "results" / "bct_all_metrics.xlsx"
output_dir = project_root / "Test_matrizen" / "converted_npy"
output_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Excel laden
# -------------------------------
df = pd.read_excel(input_file)

# -------------------------------
# Schleife über alle Subjects/Sessions
# -------------------------------
for idx, row in df.iterrows():
    # Filter auf die Spalten, die Vektoren darstellen (z.B. degree_vec_region_1, ...)
    vector_cols = [c for c in df.columns if '_region_' in c]
    vectors = row[vector_cols].values.astype(float)

    # Beispiel: eine NPY-Datei pro Subject_Session
    subject = row['subject']
    session = row['session']
    out_file = output_dir / f"{subject}_{session}_metrics.npy"

    np.save(out_file, vectors)
    print(f"Gespeichert: {out_file}")
