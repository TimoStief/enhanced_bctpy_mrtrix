# Quick Start: Understanding Your Analysis Scripts

> **TL;DR** → Jump to [Analysis Categories](#analysis-categories) below

---

## The Problem (Before Organization)

You have 18 analysis scripts with confusing names:
- What's the difference between `svm_time_effect_analysis.py` and `svm_time_effect_analysis_5groups.py`?
- Should I run `node_level_hub_analysis.py` or `node_comprehensive_multivariate_analysis.py`?
- Is `complete_metrics_umap_trajectories.py` the first step or the last?

**Result:** Users are confused. 😕

---

## The Solution: Organized Categories

All 18 scripts now fit into **6 logical categories**:

```
1. 🔧 Utilities (helpers, not run independently)
2. 🌐 Global metrics (whole-brain characteristics)
3. 🎯 Nodal metrics (region-by-region analysis)
4. 📊 Classification (machine learning prediction)
5. 👥 Responder analysis (who responds to intervention?)
6. 📈 Visualization (plots and interactions)
```

---

## Quick Answer: Which Script Should I Run?

### "I want to understand network changes"
→ Start here: **[2_global_network_metrics/01_global_basic_metrics.py](../../analysis_scripts_restructured/2_global_network_metrics/)**

**→ Then:** **[3_nodal_network_metrics/01_nodal_hub_identification.py](../../analysis_scripts_restructured/3_nodal_network_metrics/)**

### "I want to identify which brain regions change"
→ **[3_nodal_network_metrics/](../../analysis_scripts_restructured/3_nodal_network_metrics/)** (all 3 scripts)

### "I want to see if I can predict group from connectivity"
→ **[4_statistical_classification/01_svm_baseline_analysis.py](../../analysis_scripts_restructured/4_statistical_classification/)**

### "I want to understand who responds to the intervention"
→ **[5_responder_analysis/01_responder_classification.py](../../analysis_scripts_restructured/5_responder_analysis/)**

### "I want to see if males/females respond differently"
→ **[6_visualization/01_sex_interactions.py](../../analysis_scripts_restructured/6_visualization/)**

---

## Analysis Categories

### Category 1: 🔧 Utilities
**Purpose:** Helper scripts and templates  
**Run independently?** No - supporting other analyses  
**Key file:** `TEMPLATE_analysis.py` - copy this as template for new analyses  

[→ View category details](../../analysis_scripts_restructured/1_utilities/)

---

### Category 2: 🌐 Global Network Metrics  
**Purpose:** Whole-brain connectivity (one number per subject per timepoint)  
**Run this if:** You want to know "Is the whole network more efficient?"  
**Result:** Path length, global efficiency, small-worldness  

**Scripts:**
- `01_global_basic_metrics.py` - START HERE for global analysis
- `02_global_richclub_analysis.py` - Rich-club coefficient (optional)

**Typical question answered:** *"Does exercise make the network topology more efficient?"*

[→ Read detailed guide](../../analysis_scripts_restructured/2_global_network_metrics/README.md)

---

### Category 3: 🎯 Nodal Network Metrics
**Purpose:** Region-by-region (246 regions) connectivity analysis  
**Run this if:** You want to know "Which brain regions change the most?"  
**Result:** Node strength, degree, hub classification for each region  

**Scripts (run in order):**
1. `01_nodal_hub_identification.py` - Identify hub regions ← START HERE
2. `02_nodal_temporal_trajectories.py` - How regions evolve over time
3. `03_nodal_multivariate_analysis.py` - Correlations between node metrics

**Typical question answered:** *"Which 5 regions are hubs? Do they change more than other regions?"*

[→ Read detailed guide](../../analysis_scripts_restructured/3_nodal_network_metrics/README.md)

---

### Category 4: 📊 Statistical Classification (Machine Learning)
**Purpose:** Use ML to predict group/timepoint from connectivity metrics  
**Run this if:** You want to know "Can the algorithm predict who is in which group?"  
**Result:** Classification accuracy, feature importance rankings  

**Scripts:**
- `01_svm_baseline_analysis.py` - Basic classification ← START HERE
- `02_svm_time_effects.py` - Can we predict timepoint?
- `03_svm_stratified_5groups.py` - Classify by exercise group
- `04_svm_time_5groups.py` - Time effects within groups
- `05_random_forest_comparison.py` - Validate with Random Forest

**Typical question answered:** *"Can the algorithm tell which exercise group a person is in based on their connectivity?"*

[→ Read detailed guide](../../analysis_scripts_restructured/4_statistical_classification/README.md)

---

### Category 5: 👥 Responder Analysis
**Purpose:** Identify who responds to the intervention vs who doesn't  
**Run this if:** You want to know "Does everyone respond equally? Who benefits most?"  
**Result:** Responder classification, baseline predictors, trajectory types  

**Scripts (run in order):**
1. `01_responder_classification.py` - Who responds? ← START HERE
2. `02_responder_nonlinear_analysis.py` - How do trajectories differ?
3. `03_nonlinear_time_effects.py` - Are effects truly nonlinear?

**Key insight:** Response is heterogeneous - some subjects benefit more than others

**Typical question answered:** *"Top 50% of subjects show 2× connectivity increase. What makes them different?"*

[→ Read detailed guide](../../analysis_scripts_restructured/5_responder_analysis/README.md)

---

### Category 6: 📈 Visualization & Interactions
**Purpose:** Create plots and test if effects differ by sex  
**Run this if:** You want to know "Do males and females respond differently?"  
**Result:** Interaction plots, statistical significance tests  

**Scripts:**
- `01_sex_interactions.py` - Sex × Intervention interactions ← START HERE
- `02_sex_interactions_5groups.py` - Sex interactions by exercise group

**Typical question answered:** *"Do females show greater connectivity increase in frontal regions while males show greater motor region change?"*

[→ Read detailed guide](../../analysis_scripts_restructured/6_visualization/README.md)

---

## Typical Analysis Pipeline (Run in Order)

**Step 1: Get Oriented (Understanding)**
```
2_global_network_metrics/01_global_basic_metrics.py
   ↓ (Get overall picture)
3_nodal_network_metrics/01_nodal_hub_identification.py
   ↓ (Understand regional changes)
```

**Step 2: Understand Temporal Patterns (Time Course)**
```
3_nodal_network_metrics/02_nodal_temporal_trajectories.py
   ↓ (How do regions evolve?)
5_responder_analysis/01_responder_classification.py
   ↓ (Do all subjects respond the same?)
```

**Step 3: Test Prediction (Machine Learning)**
```
4_statistical_classification/01_svm_baseline_analysis.py
   ↓ (Can we predict group?)
4_statistical_classification/02_svm_time_effects.py
   ↓ (Can we predict time?)
```

**Step 4: Check Interactions (Precision Medicine)**
```
6_visualization/01_sex_interactions.py
   ↓ (Do males/females differ?)
```

---

## Common Questions → Quick Answers

| Question | Script | Category |
|----------|--------|----------|
| "Is the whole network more efficient?" | `2_global.../01_*` | Global metrics |
| "Which regions change most?" | `3_nodal.../01_*` | Nodal metrics |
| "Do regions change over time?" | `3_nodal.../02_*` | Nodal metrics |
| "Are the changes meaningful?" | `4_classification/01_*` | Classification |
| "Can we predict timepoint?" | `4_classification/02_*` | Classification |
| "Does everyone respond equally?" | `5_responder.../01_*` | Responder analysis |
| "Do males/females differ?" | `6_visualization/01_*` | Visualization |

---

## Key Concepts Explained

### Global vs Nodal
- **Global:** Summary of entire brain network (1 number per subject)
  - Example: "Global efficiency = 0.45"
  - Asks: "What's the overall network property?"

- **Nodal:** Individual regions within the network (1 number per region × subject)
  - Example: "Region 42 strength = 12.3" (across 246 regions)
  - Asks: "Which regions matter most?"

### Linear vs Nonlinear
- **Linear:** Consistent rate of change (straight line)
  - "Connectivity increases 5 units per month"
- **Nonlinear:** Rate changes over time (curved line)
  - "Connectivity increases fast initially, then plateaus"

### Interaction
- **No interaction:** Effect is the same for everyone
- **Interaction:** Effect depends on another variable
  - Example: "Females respond more to exercise than males"

---

## File Structure (After Organization)

```
analysis_scripts_restructured/
├── 1_utilities/
│   ├── TEMPLATE_analysis.py
│   ├── log_analysis.py
│   └── README.md
├── 2_global_network_metrics/
│   ├── 01_global_basic_metrics.py
│   ├── 02_global_richclub_analysis.py
│   └── README.md ← Read this first!
├── 3_nodal_network_metrics/
│   ├── 01_nodal_hub_identification.py
│   ├── 02_nodal_temporal_trajectories.py
│   ├── 03_nodal_multivariate_analysis.py
│   └── README.md ← Read this first!
├── 4_statistical_classification/
│   ├── 01_svm_baseline_analysis.py
│   ├── 02_svm_time_effects.py
│   ├── 03_svm_stratified_5groups.py
│   ├── 04_svm_time_5groups.py
│   ├── 05_random_forest_comparison.py
│   └── README.md ← Read this first!
├── 5_responder_analysis/
│   ├── 01_responder_classification.py
│   ├── 02_responder_nonlinear_analysis.py
│   ├── 03_nonlinear_time_effects.py
│   └── README.md ← Read this first!
└── 6_visualization/
    ├── 01_sex_interactions.py
    ├── 02_sex_interactions_5groups.py
    └── README.md ← Read this first!
```

---

## Implementation Notes

✅ **Numbers (01_, 02_, etc.)** suggest execution order within a category  
✅ **README.md files** in each folder provide detailed guidance  
✅ **Folder names** clearly state what type of analysis  
✅ **Descriptive script names** - no more guessing!  

---

## Questions?

For each category:
1. Read the **README.md** in that folder
2. Check the **script docstrings** at the top of each .py file
3. Review **comments** in the code for detailed explanations

---

## Next Steps

1. **Move scripts** to the new folder structure above
2. **Read the README.md** for your category of interest
3. **Run the numbered scripts in order** (01, 02, 03, etc.)
4. **Check outputs** - each script documents where results are saved

Good luck! 🚀

