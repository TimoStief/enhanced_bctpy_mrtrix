import os
import numpy as np
import pandas as pd

# -------------------------------
# Pfade
# -------------------------------
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(script_dir, "results", "bct_all_metrics.xlsx")
output_dir = os.path.join(script_dir, "Test_matrizen", "converted_npy")
os.makedirs(output_dir, exist_ok=True)

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
    out_file = os.path.join(output_dir, f"{subject}_{session}_metrics.npy")

    np.save(out_file, vectors)
    print(f"Gespeichert: {out_file}")
