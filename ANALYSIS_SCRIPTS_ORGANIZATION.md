# Analysis Scripts Organization Guide

## Overview
The analysis scripts are currently scattered and confusing. This document proposes a clear organizational structure dividing analyses into **Global** (whole-brain metrics) and **Nodal** (region-specific metrics) categories, with clear naming conventions.

---

## Proposed Folder Structure

```
analysis_scripts/
├── 1_utilities/
│   ├── TEMPLATE_analysis.py              # Template for new analyses
│   └── log_analysis.py                   # Logging utility
│
├── 2_global_network_metrics/
│   ├── 01_global_basic_metrics.py        # [RENAME: complete_metrics_umap_trajectories.py]
│   │                                       # Computes: path length, global efficiency, 
│   │                                       # small-worldness, UMAP trajectories
│   │
│   ├── 02_global_richclub_analysis.py    # [NEW: Extract from node_level_hub_analysis.py]
│   │                                       # Rich-club coefficient analysis
│   └── README.md                          # Global analyses overview
│
├── 3_nodal_network_metrics/
│   ├── 01_nodal_hub_identification.py    # [RENAME: node_level_hub_analysis.py]
│   │                                       # Node strength, degree, betweenness,
│   │                                       # participation coefficient, within-module z-score,
│   │                                       # hub classification
│   │
│   ├── 02_nodal_temporal_trajectories.py # [RENAME: node_temporal_trajectory_analysis.py]
│   │                                       # Regional temporal change analysis
│   │
│   ├── 03_nodal_multivariate_analysis.py # [RENAME: node_comprehensive_multivariate_analysis.py]
│   │                                       # Multivariate node-level analysis
│   └── README.md                          # Nodal analyses overview
│
├── 4_statistical_classification/
│   ├── 01_svm_baseline_analysis.py       # [RENAME: svm_design_variants_analysis.py]
│   │                                       # SVM classification on connectivity metrics
│   │
│   ├── 02_svm_time_effects.py            # [RENAME: svm_time_effect_analysis.py]
│   │                                       # SVM: Time effects across 3 timepoints
│   │
│   ├── 03_svm_stratified_5groups.py      # [RENAME: svm_stratified_analysis_5groups.py]
│   │                                       # SVM: Stratified by 5 exercise groups
│   │
│   ├── 04_svm_time_5groups.py            # [RENAME: svm_time_effect_analysis_5groups.py]
│   │                                       # SVM: Time effects × 5 groups
│   │
│   ├── 05_random_forest_comparison.py    # [RENAME: rf_vs_svm_design_variants.py]
│   │                                       # Random Forest vs SVM comparison
│   └── README.md                          # Classification methods overview
│
├── 5_responder_analysis/
│   ├── 01_responder_classification.py    # [RENAME: responder_phenotyping_analysis.py]
│   │                                       # Identify responders vs non-responders
│   │
│   ├── 02_responder_nonlinear_analysis.py # [RENAME: nonlinear_responders_analysis.py]
│   │                                       # Nonlinear trajectory patterns
│   │
│   ├── 03_nonlinear_time_effects.py      # [RENAME: nonlinear_time_effects_analysis.py]
│   │                                       # Nonlinear temporal dynamics
│   └── README.md                          # Responder & nonlinear analyses overview
│
└── 6_visualization/
    ├── 01_sex_interactions.py            # [RENAME: visualize_sex_interactions.py]
    │                                       # Sex × intervention interactions
    │
    ├── 02_sex_interactions_5groups.py    # [RENAME: visualize_sex_interactions_5groups.py]
    │                                       # Sex interactions stratified by group
    │
    └── README.md                          # Visualization overview
```

---

## Script Categorization

### 🔧 **Utilities** (1_utilities)
- Foundation scripts that support other analyses
- Not meant to be run independently
- **Scripts:**
  - `TEMPLATE_analysis.py` - Template with logging
  - `log_analysis.py` - Shared logging utilities

---

### 🌐 **Global Network Metrics** (2_global_network_metrics)
**Whole-brain connectivity characteristics - single metric per subject per timepoint**

| Old Name | New Name | What It Does |
|----------|----------|--------------|
| `complete_metrics_umap_trajectories.py` | `01_global_basic_metrics.py` | Path length, global efficiency, small-worldness, UMAP dimensionality reduction, trajectory clustering |
| *(extracted)* | `02_global_richclub_analysis.py` | Rich-club coefficient analysis (can be extracted from node-level script) |

**Use when:** You want to characterize the overall network topology and how it changes over time.

---

### 🎯 **Nodal Network Metrics** (3_nodal_network_metrics)
**Region-by-region (246 Brainnectome regions) connectivity characteristics**

| Old Name | New Name | What It Does |
|----------|----------|--------------|
| `node_level_hub_analysis.py` | `01_nodal_hub_identification.py` | Node strength, degree, betweenness, participation coefficient, within-module z-score, hub classification, temporal trajectories |
| `node_temporal_trajectory_analysis.py` | `02_nodal_temporal_trajectories.py` | Regional temporal dynamics and interaction effects |
| `node_comprehensive_multivariate_analysis.py` | `03_nodal_multivariate_analysis.py` | Multivariate analysis including PCA, correlations of regional metrics |

**Use when:** You want to identify which brain regions change the most and characterize their role (hub vs non-hub).

---

### 📊 **Statistical Classification** (4_statistical_classification)
**Machine learning classification of subjects based on connectivity metrics**

| Old Name | New Name | What It Does |
|----------|----------|--------------|
| `svm_design_variants_analysis.py` | `01_svm_baseline_analysis.py` | SVM classification: various kernel and parameter configurations |
| `svm_time_effect_analysis.py` | `02_svm_time_effects.py` | SVM: Predict timepoint from connectivity metrics (3 timepoints) |
| `svm_stratified_analysis_5groups.py` | `03_svm_stratified_5groups.py` | SVM: Classify exercise groups, stratified approach |
| `svm_time_effect_analysis_5groups.py` | `04_svm_time_5groups.py` | SVM: Time effects nested within exercise groups |
| `rf_vs_svm_design_variants.py` | `05_random_forest_comparison.py` | Random Forest vs SVM: model comparison |

**Use when:** You want to test if machine learning can predict group membership or timepoint from connectivity data.

---

### 👥 **Responder Analysis** (5_responder_analysis)
**Identify intervention responders vs non-responders and characterize nonlinear patterns**

| Old Name | New Name | What It Does |
|----------|----------|--------------|
| `responder_phenotyping_analysis.py` | `01_responder_classification.py` | Stratify into responders (top 50%) vs non-responders (bottom 50%) based on response magnitude |
| `nonlinear_responders_analysis.py` | `02_responder_nonlinear_analysis.py` | Nonlinear trajectory patterns in responders vs non-responders |
| `nonlinear_time_effects_analysis.py` | `03_nonlinear_time_effects.py` | General nonlinear temporal dynamics and effect patterns |

**Use when:** You want to separate subjects by their response to intervention and identify who benefits most.

---

### 📈 **Visualization** (6_visualization)
**Scripts focused on visualization and interaction effects**

| Old Name | New Name | What It Does |
|----------|----------|--------------|
| `visualize_sex_interactions.py` | `01_sex_interactions.py` | Sex × Intervention interactions in network metrics |
| `visualize_sex_interactions_5groups.py` | `02_sex_interactions_5groups.py` | Sex × Group × Intervention interactions |

**Use when:** You want to examine if intervention effects differ by sex.

---

## Typical Analysis Pipeline

**Quick overview** of what to run in order:

1. **First: Global Metrics** 
   - `2_global_network_metrics/01_global_basic_metrics.py` - Understand overall network changes

2. **Then: Regional Changes**
   - `3_nodal_network_metrics/01_nodal_hub_identification.py` - Where do the biggest changes happen?
   - `3_nodal_network_metrics/02_nodal_temporal_trajectories.py` - How do regions evolve?

3. **Then: Responders**
   - `5_responder_analysis/01_responder_classification.py` - Who responds to intervention?
   - `5_responder_analysis/02_responder_nonlinear_analysis.py` - Do responders show different patterns?

4. **Then: Prediction**
   - `4_statistical_classification/01_svm_baseline_analysis.py` - Can we predict group from connectivity?
   - `4_statistical_classification/02_svm_time_effects.py` - Can we predict time?

5. **Finally: Interactions**
   - `6_visualization/01_sex_interactions.py` - Do sex differences matter?

---

## Key Benefits

✅ **Clear categorization** - Users know which script addresses which question  
✅ **Logical ordering** - Folder numbers suggest analysis sequence  
✅ **Descriptive names** - No more guessing what "svm_time_effect_analysis_5groups.py" does  
✅ **Grouped dependencies** - Related scripts live together  
✅ **Easy to scale** - Adding new analyses follows the same pattern  
✅ **README files** - Each category has guidance for that analysis type  

---

## Implementation Steps

1. Create new folder structure in `analysis_scripts/`
2. Rename and move scripts according to the mapping above
3. Create README.md files in each category folder
4. Update any imports/paths in moved scripts
5. Update REPRODUCIBLE_WORKFLOW.md to reference new structure
6. Consider extracting related functionality (e.g., rich-club analysis) into separate scripts

---

## Questions to Address

- Should the 5-group variants be in separate folders or alongside the standard versions?
  - **Recommendation:** Keep alongside (as `02_*` to `04_*` progression) to show the evolution
  
- Should we keep both SVM and RF in the same folder?
  - **Recommendation:** Yes - they're both classification methods with similar use cases
  
- Any scripts that should be archived/deprecated?
  - **Review needed:** Which variants represent final analysis vs. exploratory dead-ends?

