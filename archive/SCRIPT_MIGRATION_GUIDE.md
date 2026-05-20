# Script Migration Mapping Guide

Use this table to find where each of your 18 scripts goes in the new organization.

---

## Old Name → New Name & Location

| Old Script Name | New Location | New Name | Category | Purpose |
|---|---|---|---|---|
| `TEMPLATE_analysis.py` | `1_utilities/` | `TEMPLATE_analysis.py` | Utilities | Template for new analyses |
| `log_analysis.py` | `1_utilities/` | `log_analysis.py` | Utilities | Logging helper |
| `complete_metrics_umap_trajectories.py` | `2_global_network_metrics/` | `01_global_basic_metrics.py` | **Global metrics** | Path length, efficiency, UMAP |
| `node_level_hub_analysis.py` | `3_nodal_network_metrics/` | `01_nodal_hub_identification.py` | **Nodal metrics** | Hub identification, classification |
| `node_temporal_trajectory_analysis.py` | `3_nodal_network_metrics/` | `02_nodal_temporal_trajectories.py` | **Nodal metrics** | Regional temporal analysis |
| `node_comprehensive_multivariate_analysis.py` | `3_nodal_network_metrics/` | `03_nodal_multivariate_analysis.py` | **Nodal metrics** | PCA, multivariate tests |
| `svm_design_variants_analysis.py` | `4_statistical_classification/` | `01_svm_baseline_analysis.py` | **Classification** | SVM hyperparameter exploration |
| `svm_time_effect_analysis.py` | `4_statistical_classification/` | `02_svm_time_effects.py` | **Classification** | Time prediction via SVM |
| `svm_stratified_analysis_5groups.py` | `4_statistical_classification/` | `03_svm_stratified_5groups.py` | **Classification** | Group classification, stratified |
| `svm_time_effect_analysis_5groups.py` | `4_statistical_classification/` | `04_svm_time_5groups.py` | **Classification** | Time effects within 5 groups |
| `rf_vs_svm_design_variants.py` | `4_statistical_classification/` | `05_random_forest_comparison.py` | **Classification** | Random Forest vs SVM comparison |
| `responder_phenotyping_analysis.py` | `5_responder_analysis/` | `01_responder_classification.py` | **Responder analysis** | Identify responders vs non-responders |
| `nonlinear_responders_analysis.py` | `5_responder_analysis/` | `02_responder_nonlinear_analysis.py` | **Responder analysis** | Nonlinear patterns in responders |
| `nonlinear_time_effects_analysis.py` | `5_responder_analysis/` | `03_nonlinear_time_effects.py` | **Responder analysis** | Nonlinear temporal dynamics |
| `visualize_sex_interactions.py` | `6_visualization/` | `01_sex_interactions.py` | **Visualization** | Sex × Intervention interactions |
| `visualize_sex_interactions_5groups.py` | `6_visualization/` | `02_sex_interactions_5groups.py` | **Visualization** | Sex interactions by group |

---

## Migration Steps

### Option A: Manual Reorganization (Recommended)

1. **Create new folder structure:**
   ```bash
   mkdir -p analysis_scripts_restructured/{1_utilities,2_global_network_metrics,3_nodal_network_metrics,4_statistical_classification,5_responder_analysis,6_visualization}
   ```

2. **Copy and rename scripts:**
   ```bash
   # Utilities
   cp analysis_scripts/TEMPLATE_analysis.py analysis_scripts_restructured/1_utilities/
   cp analysis_scripts/log_analysis.py analysis_scripts_restructured/1_utilities/

   # Global metrics
   cp analysis_scripts/complete_metrics_umap_trajectories.py analysis_scripts_restructured/2_global_network_metrics/global_basic_metrics.py

   # Nodal metrics
   cp analysis_scripts/node_level_hub_analysis.py analysis_scripts_restructured/3_nodal_network_metrics/nodal_hub_identification.py
   cp analysis_scripts/node_temporal_trajectory_analysis.py analysis_scripts_restructured/3_nodal_network_metrics/02_nodal_temporal_trajectories.py
   cp analysis_scripts/node_comprehensive_multivariate_analysis.py analysis_scripts_restructured/3_nodal_network_metrics/03_nodal_multivariate_analysis.py

   # Classification
   cp analysis_scripts/svm_design_variants_analysis.py analysis_scripts_restructured/4_statistical_classification/01_svm_baseline_analysis.py
   cp analysis_scripts/svm_time_effect_analysis.py analysis_scripts_restructured/4_statistical_classification/02_svm_time_effects.py
   cp analysis_scripts/svm_stratified_analysis_5groups.py analysis_scripts_restructured/4_statistical_classification/03_svm_stratified_5groups.py
   cp analysis_scripts/svm_time_effect_analysis_5groups.py analysis_scripts_restructured/4_statistical_classification/stat_svm_time_5groups.py
   cp analysis_scripts/rf_vs_svm_design_variants.py analysis_scripts_restructured/4_statistical_classification/stat_random_forest_comparison.py

   # Responder analysis
   cp analysis_scripts/responder_phenotyping_analysis.py analysis_scripts_restructured/5_responder_analysis/01_responder_classification.py
   cp analysis_scripts/nonlinear_responders_analysis.py analysis_scripts_restructured/5_responder_analysis/02_responder_nonlinear_analysis.py
   cp analysis_scripts/nonlinear_time_effects_analysis.py analysis_scripts_restructured/5_responder_analysis/03_nonlinear_time_effects.py

   # Visualization
   cp analysis_scripts/visualize_sex_interactions.py analysis_scripts_restructured/6_visualization/01_sex_interactions.py
   cp analysis_scripts/visualize_sex_interactions_5groups.py analysis_scripts_restructured/6_visualization/02_sex_interactions_5groups.py
   ```

3. **Copy README files** (provided in ANALYSIS_SCRIPTS_QUICK_START.md)

4. **Verify** all scripts moved correctly:
   ```bash
   find analysis_scripts_restructured -name "*.py" | wc -l  # Should be 18
   ```

5. **Backup old folder:**
   ```bash
   mv analysis_scripts analysis_scripts_OLD_BACKUP
   mv analysis_scripts_restructured analysis_scripts
   ```

### Option B: Automated Bash Script

Save this as `reorganize_scripts.sh` and run:

```bash
#!/bin/bash
set -e

SOURCE_DIR="analysis_scripts"
TARGET_DIR="analysis_scripts_restructured"

# Create target directories
mkdir -p "$TARGET_DIR"/{1_utilities,2_global_network_metrics,3_nodal_network_metrics,4_statistical_classification,5_responder_analysis,6_visualization}

# Utilities
cp "$SOURCE_DIR/TEMPLATE_analysis.py" "$TARGET_DIR/1_utilities/"
cp "$SOURCE_DIR/log_analysis.py" "$TARGET_DIR/1_utilities/"

# Global metrics
cp "$SOURCE_DIR/complete_metrics_umap_trajectories.py" "$TARGET_DIR/2_global_network_metrics/01_global_basic_metrics.py"

# Nodal metrics
cp "$SOURCE_DIR/node_level_hub_analysis.py" "$TARGET_DIR/3_nodal_network_metrics/01_nodal_hub_identification.py"
cp "$SOURCE_DIR/node_temporal_trajectory_analysis.py" "$TARGET_DIR/3_nodal_network_metrics/02_nodal_temporal_trajectories.py"
cp "$SOURCE_DIR/node_comprehensive_multivariate_analysis.py" "$TARGET_DIR/3_nodal_network_metrics/03_nodal_multivariate_analysis.py"

# Classification
cp "$SOURCE_DIR/svm_design_variants_analysis.py" "$TARGET_DIR/4_statistical_classification/01_svm_baseline_analysis.py"
cp "$SOURCE_DIR/svm_time_effect_analysis.py" "$TARGET_DIR/4_statistical_classification/02_svm_time_effects.py"
cp "$SOURCE_DIR/svm_stratified_analysis_5groups.py" "$TARGET_DIR/4_statistical_classification/03_svm_stratified_5groups.py"
cp "$SOURCE_DIR/svm_time_effect_analysis_5groups.py" "$TARGET_DIR/4_statistical_classification/04_svm_time_5groups.py"
cp "$SOURCE_DIR/rf_vs_svm_design_variants.py" "$TARGET_DIR/4_statistical_classification/05_random_forest_comparison.py"

# Responder analysis
cp "$SOURCE_DIR/responder_phenotyping_analysis.py" "$TARGET_DIR/5_responder_analysis/01_responder_classification.py"
cp "$SOURCE_DIR/nonlinear_responders_analysis.py" "$TARGET_DIR/5_responder_analysis/02_responder_nonlinear_analysis.py"
cp "$SOURCE_DIR/nonlinear_time_effects_analysis.py" "$TARGET_DIR/5_responder_analysis/03_nonlinear_time_effects.py"

# Visualization
cp "$SOURCE_DIR/visualize_sex_interactions.py" "$TARGET_DIR/6_visualization/01_sex_interactions.py"
cp "$SOURCE_DIR/visualize_sex_interactions_5groups.py" "$TARGET_DIR/6_visualization/02_sex_interactions_5groups.py"

echo "✅ Migration complete!"
echo "Total scripts in new structure: $(find $TARGET_DIR -name '*.py' | wc -l)"
```

Run with:
```bash
chmod +x reorganize_scripts.sh
./reorganize_scripts.sh
```

---

## Checking for Broken Imports

After moving scripts, check if any imports are broken:

```bash
# Test all scripts for import errors
for f in analysis_scripts_restructured/**/*.py; do
    echo "Checking $f..."
    python3 -m py_compile "$f" 2>&1 | grep -i "error" && echo "  ⚠️  Problem found" || echo "  ✅ OK"
done
```

---

## Updating Documentation

After reorganization, update these files:

1. **REPRODUCIBLE_WORKFLOW.md** 
   - Update section on running analyses
   - Point to new folder structure

2. **README.md** in main project
   - Add link to ANALYSIS_SCRIPTS_QUICK_START.md
   - Add folder structure diagram

3. **Your own scripts**
   - If you have any orchestration scripts that import from analysis_scripts/
   - Update import paths to point to new locations

---

## Before/After Comparison

### Before (Confusing)
```
analysis_scripts/
├── complete_metrics_umap_trajectories.py    # What does this do?
├── node_level_hub_analysis.py               # Is this before or after the above?
├── svm_time_effect_analysis.py              # Same as svm_time_effect_analysis_5groups.py?
├── svm_time_effect_analysis_5groups.py      # What's the difference?
├── visualize_sex_interactions.py            # Should I run this first?
└── visualize_sex_interactions_5groups.py    # What does "5groups" add?
```

### After (Clear)
```
analysis_scripts/
├── 2_global_network_metrics/
│   └── 01_global_basic_metrics.py           # Start with this - whole network
├── 3_nodal_network_metrics/
│   ├── 01_nodal_hub_identification.py       # Then this - which regions?
│   └── 02_nodal_temporal_trajectories.py    # Then this - how do they evolve?
├── 4_statistical_classification/
│   ├── 01_svm_baseline_analysis.py          # Can we predict group?
│   └── 02_svm_time_effects.py               # Can we predict time?
└── 6_visualization/
    └── 01_sex_interactions.py               # Sex differences
```

**User can now:**
- ✅ Understand the folder structure at a glance
- ✅ Know which scripts to run in which order
- ✅ Find related analyses in the same folder
- ✅ Understand the distinction (01_, 02_ indicates order)

---

## Troubleshooting

**Q: I moved the scripts but they fail with import errors**  
A: Check that relative imports still work. You may need to update paths in scripts if they reference:
```python
from analysis_scripts.log_analysis import AnalysisLogger
```
Change to:
```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utilities.log_analysis import AnalysisLogger
```

**Q: I have a script that orchestrates all analyses**  
A: Update it to point to new locations:
```python
# Before:
from analysis_scripts import svm_design_variants_analysis

# After:
from analysis_scripts_restructured._statistical_classification import svm_baseline_analysis  # or import as module
```

**Q: Should I keep the old analysis_scripts folder?**  
A: Backup it first (`analysis_scripts_OLD`), then:
- Yes if: Other projects depend on the old paths
- No if: Only this project uses these scripts
  
Recommended: Keep backup for 1 month, then delete.

---

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Clarity** | 18 scripts, unclear relationships | 6 categories, clear purpose |
| **User Experience** | "Which script should I run?" | Clear guidance via README files |
| **Maintenance** | Hard to add new scripts | Easy - follows template |
| **Documentation** | Scattered docstrings | Centralized README per category |
| **Learning Curve** | High | Low |

