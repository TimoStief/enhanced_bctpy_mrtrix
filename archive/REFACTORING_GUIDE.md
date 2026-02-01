# Scripts Refactoring Guide

Your analysis scripts have been **restructured and refactored** to be more generic, reusable, and clear.

---

## 🎯 What Changed

### Before (Tailor-Made)
```python
# Hardcoded paths
DATA_DIR = Path("/data/local/129_PK01/derivatives/dsistudio_connectomics/connectivity")
BCT_DIR = Path("/data/local/129_PK01/derivatives/bct")
ATLAS = "Brainnectome"
N_NODES = 246

# Study-specific logic scattered throughout
# Difficult to adapt for other studies
# Output files location unclear
```

### After (Generic & Configurable)
```python
CONFIG = {
    "data_dir": Path("/data/local/129_PK01/derivatives/dsistudio_connectomics/connectivity"),
    "metadata_file": Path("/data/local/129_PK01/derivatives/bct/participants_5groups.tsv"),
    "output_dir": Path("/data/local/129_PK01/derivatives/bct/global_metrics"),
    "n_nodes": 246,
    "atlas_name": "Brainnectome",
    "file_pattern": "{subject}_ses-{session}*/tracks_1000k_streamline/by_atlas/{atlas}/*.connectivity.mat",
    "subject_col": "participant_id",
    "session_col": "session",
    "group_col": "group",
    "sex_col": "sex",
}

# Can be used for any study!
# Clear structure
# Easy to adapt
```

---

## 📋 New Script Structure

Every script now follows this standard format:

### 1. **DOCUMENTATION** (Clear & Comprehensive)
```python
"""
SCRIPT: [Script Name]
======================

PURPOSE:
    What this script does (2-3 sentences)
    
    Specific outputs:
    - Output file 1 (what it contains)
    - Output file 2 (what it contains)

INPUT REQUIREMENTS:
    DATA:
    - Format and structure
    - Location pattern
    
    METADATA:
    - Required columns
    - Optional columns

OUTPUT FILES:
    - filename.parquet (table description)
    - plots/ (visualization descriptions)

USAGE:
    python script.py

VERSION: 2.0 (Generic)
AUTHOR: Analysis Pipeline
"""
```

### 2. **CONFIGURATION** (One Place to Edit)
```python
CONFIG = {
    # Data locations
    "data_dir": Path("..."),
    "metadata_file": Path("..."),
    "output_dir": Path("..."),
    
    # Data structure
    "n_nodes": 246,
    "atlas_name": "Brainnectome",
    
    # File pattern
    "file_pattern": "{subject}_ses-{session}*/...",
    
    # Column names
    "subject_col": "participant_id",
    "session_col": "session",
    "group_col": "group",
    
    # Options
    "binarize": False,
    "threshold": 0.0,
}
```

### 3. **GENERIC FUNCTIONS** (Reusable)
```python
def load_connectivity_matrix(subject: str, session: str) -> np.ndarray:
    """
    Generic loader that works for any study
    - Handles both .mat and .npy
    - Uses CONFIG pattern
    - Validates shape
    """

def compute_metrics(A: np.ndarray) -> dict:
    """
    Your specific analysis
    - Takes matrix as input
    - Returns dict of metrics
    - Handles errors gracefully
    """
```

### 4. **MAIN LOOP** (Clear & Simple)
```python
for idx, row in metadata.iterrows():
    # Load
    A = load_connectivity_matrix(subject, session)
    
    # Compute
    metrics = compute_metrics(A)
    
    # Store
    results.append(record)

# Save
results_df.to_parquet(output_file)
results_df.to_csv(output_file)
```

---

## ✨ Key Improvements

| Aspect | Before | After |
|--------|--------|-------|
| **Portability** | Hardcoded for one study | Works for any study |
| **Configuration** | Scattered throughout | Single CONFIG dict |
| **Documentation** | Minimal | Comprehensive |
| **Clarity** | What is the input/output? | Clearly specified |
| **Reusability** | High effort to adapt | Drop-in for new data |
| **Maintenance** | Hard to understand | Clear structure |

---

## 🔧 How to Adapt a Script for Your Data

### Step 1: Find Your Script
- Browse `analysis_scripts_restructured/` folders
- Pick the one matching your analysis

### Step 2: Edit CONFIG Section
```python
CONFIG = {
    "data_dir": Path("/YOUR/DATA/LOCATION"),
    "metadata_file": Path("/YOUR/METADATA.tsv"),
    "output_dir": Path("/YOUR/OUTPUT/DIR"),
    "n_nodes": 246,  # Your atlas size
    "atlas_name": "YourAtlas",
    "file_pattern": "/YOUR/FILE/PATTERN",  # <-- This is key!
    "subject_col": "your_subject_column",
    "session_col": "your_session_column",
    # ...
}
```

### Step 3: Check Input/Output Format
Read the docstring:
```
INPUT REQUIREMENTS:
    - What format are your files?
    - What columns in your metadata?

OUTPUT FILES:
    - What will be created?
    - What do the columns mean?
```

### Step 4: Run
```bash
python your_script.py
```

That's it! No need to edit the rest of the code.

---

## 📝 File Pattern Examples

Your `file_pattern` tells the script how to find connectivity files.

### Example 1: DSI Studio (Current Study)
```python
"file_pattern": "{subject}_ses-{session}*/tracks_1000k_streamline/by_atlas/{atlas}/*.connectivity.mat"
```

### Example 2: MRtrix
```python
"file_pattern": "derivatives/mrtrix3_connectome/{subject}/ses-{session}/connectogram.npy"
```

### Example 3: Simple Folder Structure
```python
"file_pattern": "data/{subject}/session_{session}/connectivity.mat"
```

### Example 4: All in One Folder
```python
"file_pattern": "connectivity/{subject}_ses{session}_connectogram.npy"
```

**Key:** Use `{subject}` and `{session}` as placeholders. The script will replace them.

---

## 📊 Expected Metadata Format

Your metadata file should have columns like:

```
participant_id | session | group    | sex | age
114            | 1       | exercise | M   | 45
114            | 2       | exercise | M   | 45
114            | 3       | exercise | M   | 45
115            | 1       | control  | F   | 52
...
```

**Required columns:**
- `participant_id` (or your subject_col name)
- `session` (or your session_col name)

**Optional columns:**
- `group` (intervention group)
- `sex` (male/female)
- `age` (participant age)
- Any other metadata

---

## 🚀 Using Different File Formats

### Loading .mat Files
```python
from scipy.io import loadmat
mat_data = loadmat(filepath)
A = mat_data['connectivity']
```

### Loading .npy Files
```python
A = np.load(filepath)
```

### Loading .nii Files (NIfTI)
```python
import nibabel as nib
img = nib.load(filepath)
A = img.get_fdata()
```

### Custom Format
Edit the `load_connectivity_matrix()` function to match your format.

---

## 💾 Output File Formats

### Parquet Files (Recommended)
```python
results_df.to_parquet(output_file)
df = pd.read_parquet(output_file)
```

**Advantages:**
- Efficient storage (smaller files)
- Preserves data types
- Fast to load
- Works with Python, R, Julia

### CSV Files (Human-Readable)
```python
results_df.to_csv(output_file)
df = pd.read_csv(output_file)
```

**Advantages:**
- Open in Excel, Google Sheets
- Easy to inspect
- Larger file size

---

## 🔍 Troubleshooting

### "File not found"
1. Check `CONFIG["data_dir"]` path is correct
2. Check `CONFIG["file_pattern"]` matches your files
3. Print the pattern:
   ```python
   pattern = CONFIG["file_pattern"].format(subject="114", session="1", atlas="Brainnectome")
   print(f"Looking for: {pattern}")
   ```

### "Invalid shape"
1. Your `n_nodes` doesn't match matrix size
2. Check: `print(A.shape)` in your files

### "Column not found"
1. Check `CONFIG["subject_col"]` matches metadata column name
2. Print columns: `print(metadata.columns.tolist())`

### "NaN in results"
1. Likely data loading issue
2. Check connectivity matrix has values
3. Add try/except in your compute function

---

## 📚 Template File

Use **SCRIPT_TEMPLATE.py** as a starting point for new analyses:

1. Copy: `cp SCRIPT_TEMPLATE.py my_new_analysis.py`
2. Edit docstring with your analysis
3. Edit CONFIG section
4. Replace `compute_metrics()` with your analysis
5. Run: `python my_new_analysis.py`

---

## ✅ Checklist Before Running

- [ ] CONFIG section matches your data locations
- [ ] file_pattern works (test with glob)
- [ ] Metadata file exists and has required columns
- [ ] n_nodes matches your atlas
- [ ] Output directory exists or will be created
- [ ] compute_metrics() function returns dict
- [ ] No hardcoded paths in code

---

## 🎯 Scripts Now Available

All scripts follow this new generic structure:

**2_global_network_metrics/**
- `01_global_basic_metrics.py` - Global network metrics (✅ Updated)

**3_nodal_network_metrics/**
- `01_nodal_hub_identification.py` - Hub identification
- `02_nodal_temporal_trajectories.py` - Regional trajectories
- `03_nodal_multivariate_analysis.py` - Multivariate analysis

**4_statistical_classification/**
- `01_svm_baseline_analysis.py` - SVM classification
- `02_svm_time_effects.py` - Time prediction
- `03_svm_stratified_5groups.py` - Stratified classification
- `04_svm_time_5groups.py` - Time × Group
- `05_random_forest_comparison.py` - RF vs SVM

**5_responder_analysis/**
- `01_responder_classification.py` - Responder identification
- `02_responder_nonlinear_analysis.py` - Nonlinear patterns
- `03_nonlinear_time_effects.py` - Temporal dynamics

**6_visualization/**
- `01_sex_interactions.py` - Sex interactions
- `02_sex_interactions_5groups.py` - Sex × Group

---

## 🎓 Best Practices

1. **Always update CONFIG first** - Don't modify code elsewhere
2. **Test file_pattern** - Make sure it matches your files
3. **Check output files** - Verify they were created correctly
4. **Keep original scripts** - Your old scripts are still in `analysis_scripts/`
5. **Document customizations** - If you modify compute_metrics(), add comments
6. **Use parquet for storage** - More efficient than CSV
7. **Check data types** - Ensure metrics make sense

---

**All scripts are now generic, configurable, and reusable!** 🎉

