# Analysis Scripts: Visual Overview

## The 6 Categories at a Glance

```
                        ┌─────────────────────────────────────┐
                        │   Your Brain Network Analysis       │
                        └──────────────┬──────────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │  1️⃣  Utilities (Tools & Templates)  │
                    │  TEMPLATE_analysis.py, log_analysis  │
                    └──────────────────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
                    ▼                                     ▼
        ┌──────────────────────┐          ┌──────────────────────┐
        │  2️⃣  GLOBAL METRICS   │          │  3️⃣  NODAL METRICS   │
        │  ─────────────────    │          │  ────────────────    │
        │ (Whole brain picture) │          │ (Region by region)   │
        │                      │          │                      │
        │ • Path length        │          │ • Hub identification │
        │ • Global efficiency  │          │ • Temporal change    │
        │ • Small-worldness    │          │ • Multivariate       │
        │ • UMAP trajectories  │          │                      │
        │                      │          │ Result: 246 values   │
        │ Result: 1 value      │          │ per subject          │
        │ per subject          │          │                      │
        └──────────────────────┘          └──────────────────────┘
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │  4️⃣  CLASSIFICATION (ML Prediction)  │
                    │  ──────────────────────────────────  │
                    │  Can we predict group from metrics?  │
                    │  • SVM baseline                      │
                    │  • Time prediction                   │
                    │  • Group stratified                  │
                    │  • Random Forest                     │
                    └──────────────────────────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │  5️⃣  RESPONDER ANALYSIS              │
                    │  ──────────────────────────────────  │
                    │  Who responds? Who doesn't?          │
                    │  • Responder classification          │
                    │  • Nonlinear trajectories            │
                    │  • Time dynamics                     │
                    └──────────────────────────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │  6️⃣  VISUALIZATION & INTERACTIONS    │
                    │  ──────────────────────────────────  │
                    │  Sex differences? Group differences? │
                    │  • Sex × Intervention interactions   │
                    │  • Sex × Group interactions          │
                    └──────────────────────────────────────┘
```

---

## What Each Category Answers

### 🌐 **Global Metrics** — The Big Picture
> **Question:** "How is the whole network organized?"

**Input:** Connectivity matrix (246×246)  
**Output:** 1 value per subject (e.g., global efficiency = 0.45)

```
Example Results:
┌─────────────┬──────────────┬──────────────┐
│ Subject     │ Baseline     │ Post-Ex      │
├─────────────┼──────────────┼──────────────┤
│ 101         │ 0.43         │ 0.48         │ ← More efficient!
│ 102         │ 0.42         │ 0.41         │ ← No change
│ 103         │ 0.44         │ 0.49         │ ← More efficient!
└─────────────┴──────────────┴──────────────┘

Average change: +0.04 → Exercise improves network efficiency
```

---

### 🎯 **Nodal Metrics** — The Details
> **Question:** "Which 5 brain regions are most important? Are they hubs?"

**Input:** Connectivity matrix (246×246)  
**Output:** 246 values per subject (one per region)

```
Example Results (Temporal Trajectories):
Hub Regions (top 5 by change magnitude):
┌─────────────────┬─────────┬──────────┬──────────┐
│ Region          │ T0      │ T2       │ Change   │
├─────────────────┼─────────┼──────────┼──────────┤
│ 42 (hub)        │ 10.2    │ 15.3     │ +5.1 ✅✅ │
│ 187 (hub)       │ 9.8     │ 14.1     │ +4.3 ✅  │
│ 65 (non-hub)    │ 5.3     │ 5.2      │ -0.1     │
└─────────────────┴─────────┴──────────┴──────────┘

Insight: Hub regions show larger changes than non-hubs
```

---

### 📊 **Classification** — Can We Predict?
> **Question:** "Does the algorithm know which group this person is in?"

**Input:** All metrics (global + nodal)  
**Output:** Classification accuracy + feature importance

```
Example Results (5-fold cross-validation):
┌───────────────────────┬──────────┐
│ Model                 │ Accuracy │
├───────────────────────┼──────────┤
│ SVM (baseline)        │ 78%      │ ✅ Good!
│ Random Forest         │82%       │ ✅ Better!
│ Random chance (5 grps)│ 20%      │ (baseline)
└───────────────────────┴──────────┘

Top important features:
  1. Hub region 42 strength
  2. Global efficiency
  3. Hub region 187 degree
  
Interpretation: If we know these 3 metrics, 
we can guess the exercise group 82% of the time
```

---

### 👥 **Responder Analysis** — Individual Differences
> **Question:** "Does everyone respond equally? Who benefits most?"

**Input:** Metrics + intervention status  
**Output:** Responder classification + baseline predictors

```
Example Results:
Stratified by response magnitude:
┌──────────────┬───────────────┬────────────────┐
│ Group        │ Change        │ Response Class │
├──────────────┼───────────────┼────────────────┤
│ Exercise 1   │ Global +0.10  │ RESPONDER      │
│ Exercise 1   │ Global +0.08  │ RESPONDER      │
│ Exercise 1   │ Global +0.02  │ NON-RESPONDER  │
│ Exercise 1   │ Global -0.01  │ NON-RESPONDER  │
└──────────────┴───────────────┴────────────────┘

Can we predict responder status from baseline?
  ✅ Yes! High baseline cognitive reserve → 
     More likely to be responder
```

---

### 📈 **Visualization** — Interactions
> **Question:** "Do males and females respond differently?"

**Input:** All metrics + sex labels  
**Output:** Interaction plots + p-values

```
Example Results (Sex × Intervention):

Global Efficiency Change:
                Females    Males
              /            /
            /            /        (lines are PARALLEL)
          /            /          
    ────────────────────────
    Before         After
    
    → NO interaction: Both respond equally, just 
      females start higher

OR:

                Females
              /        
            /          
          /   Males   
    ────────/──────────
    Before         After
    
    → INTERACTION: Females respond more than males!
```

---

## The "Flow" of Analysis

```
START
  │
  ├─→ 2_Global Metrics
  │     "Overall network efficient?"
  │     ↓
  ├─→ 3_Nodal Metrics  
  │     "Which regions change?"
  │     ↓
  ├─→ 4_Classification
  │     "Can ML predict group?"
  │     ↓
  ├─→ 5_Responder Analysis
  │     "Who benefits most?"
  │     ↓
  └─→ 6_Visualization
        "Sex differences?"
        ↓
      RESULTS & INTERPRETATION
```

---

## How to Read the Results

### For Global Metrics
✅ **Look for:** Consistent positive change in efficiency/clustering  
❌ **Bad sign:** No change, or decrease  
📊 **Report:** Mean change ± SD, p-value from t-test

### For Nodal Metrics
✅ **Look for:** Hub regions show larger changes than non-hubs  
❌ **Bad sign:** Random pattern, no regional specificity  
📊 **Report:** Top 5-10 changing regions + effect sizes

### For Classification
✅ **Look for:** Accuracy >> random chance (>40% for 5 groups)  
❌ **Bad sign:** Accuracy ≈ random chance  
📊 **Report:** Overall accuracy + confusion matrix + top 10 features

### For Responder Analysis
✅ **Look for:** Clear separation of responders vs non-responders  
❌ **Bad sign:** Overlapping distributions  
📊 **Report:** % Responders, baseline differences (Cohen's d)

### For Interactions
✅ **Look for:** Significant p-value for interaction term (p < 0.05)  
❌ **Bad sign:** Parallel lines (no interaction)  
📊 **Report:** Interaction p-value + plot + simple effects analysis

---

## Quick Troubleshooting

| Symptom | Possible Cause | Solution |
|---------|----------------|----------|
| No change in global metrics | Intervention too weak OR wrong timepoint comparison | Check data quality, try different metric |
| Nodal metrics show random pattern | Noise, not signal | Increase smoothing, check preprocessing |
| Classification accuracy = random chance | No group signal in connectivity | Reconsider hypothesis, check data |
| No responders (all respond equally) | Good! Everyone responds uniformly | Still informative for precision medicine |
| Interaction p-value borderline (p=0.08) | Underpowered study OR weak effect | Increase n, use larger effect size thresholds |

---

## File Organization Summary

```
🔧 Utilities (helpers only)
├─ TEMPLATE_analysis.py
└─ log_analysis.py

🌐 Global Metrics
└─ 01_global_basic_metrics.py

🎯 Nodal Metrics
├─ 01_nodal_hub_identification.py
├─ 02_nodal_temporal_trajectories.py
└─ 03_nodal_multivariate_analysis.py

📊 Classification (ML)
├─ 01_svm_baseline_analysis.py
├─ 02_svm_time_effects.py
├─ 03_svm_stratified_5groups.py
├─ 04_svm_time_5groups.py
└─ 05_random_forest_comparison.py

👥 Responder Analysis
├─ 01_responder_classification.py
├─ 02_responder_nonlinear_analysis.py
└─ 03_nonlinear_time_effects.py

📈 Visualization
├─ 01_sex_interactions.py
└─ 02_sex_interactions_5groups.py
```

---

## Pro Tips

1. **Always run in order (01, 02, 03...)**
   - Later scripts depend on earlier results

2. **Check README.md in each folder**
   - Specific guidance for that analysis type

3. **Keep outputs organized**
   - Each script saves to `/data/local/129_PK01/derivatives/bct/`
   - Check the script docstring for exact output paths

4. **Save important figures**
   - Figures are your best communication tool
   - Save high-res versions (.pdf, .png 300dpi)

5. **Document parameters**
   - If you change defaults, note them in a log file
   - Reproducibility!

