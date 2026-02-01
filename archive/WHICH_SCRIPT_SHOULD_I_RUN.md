# 🎯 Decision Tree: Which Analysis Script Should I Run?

Use this flowchart to find the right script for your question.

---

## Start Here: What Do You Want to Know?

```
                    START: YOUR RESEARCH QUESTION
                              │
                    ┌─────────┴─────────┐
                    │                   │
            "BASIC UNDERSTANDING"    "DETAILED ANALYSIS"
                    │                   │
         (What changed?)        (How/Why did it change?)
                    │                   │
                    ▼                   ▼
            ┌───────────────┐    ┌──────────────┐
            │ Global vs     │    │ Responders?  │
            │ Regional?     │    │ Nonlinear?   │
            └───────┬───────┘    │ Sex effects? │
                    │            └──────────────┘
         ┌──────────┴──────────┐
         │                     │
         ▼                     ▼
    "WHOLE BRAIN"          "REGIONS"
         │                     │
         ▼                     ▼
    2️⃣ GLOBAL            3️⃣ NODAL
    METRICS             METRICS
```

---

## Decision Tree (Full Version)

### Q1: "I want to understand the basic changes in my data"

#### YES → Q2: "Is it whole-brain or regional?"

**🌐 WHOLE-BRAIN (Global)**
- Question: "Is the network more efficient overall?"
- Answer: Run **2_global_network_metrics/01_global_basic_metrics.py**
- Output: Global efficiency, path length, small-worldness
- Time: ~5-10 minutes

**🎯 REGIONAL (Nodal)**
- Question: "Which brain regions change most?"
- Answer: Run **3_nodal_network_metrics/01_nodal_hub_identification.py**
- Output: Node strength, hub classification for all 246 regions
- Time: ~10-20 minutes

---

### Q2: "I want to predict groups using connectivity"

#### YES → **4_statistical_classification/01_svm_baseline_analysis.py**

**What it answers:**
- "Can an algorithm predict which exercise group someone is in?"
- "Which connectivity metrics are most important?"

**Suggested order:**
1. `01_svm_baseline_analysis.py` - Basic classification
2. `02_svm_time_effects.py` - Can we predict timepoint?
3. `05_random_forest_comparison.py` - Validate with RF

**Output:** Accuracy %, feature importance rankings

---

### Q3: "I want to understand temporal dynamics (time effects)"

#### YES → Which aspect?

**Time patterns:**
- "Do regions change over the 4 timepoints?"
- Answer: **3_nodal_network_metrics/02_nodal_temporal_trajectories.py**

**Time prediction:**
- "Can we predict which timepoint this is from?"
- Answer: **4_statistical_classification/02_svm_time_effects.py**

**Nonlinear patterns:**
- "Are changes smooth (linear) or curved (nonlinear)?"
- Answer: **5_responder_analysis/03_nonlinear_time_effects.py**

---

### Q4: "I want to understand responder heterogeneity"

#### YES → Which aspect?

**Basic responder classification:**
- "Who responds to the intervention? Who doesn't?"
- Answer: **5_responder_analysis/01_responder_classification.py**
- Output: Responder/non-responder labels + baseline predictors

**Responder patterns:**
- "Do responders show different trajectory shapes?"
- Answer: **5_responder_analysis/02_responder_nonlinear_analysis.py**
- Output: Trajectory classifications, response phenotypes

---

### Q5: "I want to check for sex differences"

#### YES → **6_visualization/01_sex_interactions.py**

**What it answers:**
- "Do males and females respond differently to exercise?"
- "Is there a significant Sex × Intervention interaction?"

**If yes, refine with:**
- **6_visualization/02_sex_interactions_5groups.py** - By exercise group

**Output:** Interaction p-values, effect sizes, plots

---

## Common Scenarios → Quick Answers

### Scenario 1: "I'm just starting - help me understand my data"
1. Run: `2_global_network_metrics/01_global_basic_metrics.py`
   - Takes 5-10 min
   - Shows overall network changes
   - Gives confidence that something changed

2. Run: `3_nodal_network_metrics/01_nodal_hub_identification.py`
   - Takes 10-20 min
   - Shows which regions drive the global changes
   - Identifies hub regions

3. Read: ANALYSIS_SCRIPTS_VISUAL_GUIDE.md
   - Understand what your outputs mean

✅ **Done!** You now understand your data.

---

### Scenario 2: "I want to know if our intervention 'worked'"
1. Run: `2_global_network_metrics/01_global_basic_metrics.py`
   - Check: Did global efficiency improve? (p < 0.05)

2. Run: `3_nodal_network_metrics/01_nodal_hub_identification.py`
   - Check: Did specific regions improve significantly?

3. Run: `4_statistical_classification/01_svm_baseline_analysis.py`
   - Check: Can we predict group from metrics? (>40% accuracy)

✅ **Done!** You can write your results section.

---

### Scenario 3: "Some people respond more than others"
1. Run: `5_responder_analysis/01_responder_classification.py`
   - Identifies top 50% responders vs bottom 50%

2. Run: `5_responder_analysis/02_responder_nonlinear_analysis.py`
   - Characterize responder trajectory types

3. Run: `6_visualization/01_sex_interactions.py`
   - Do responders differ by sex?

✅ **Done!** You have a precision medicine angle.

---

### Scenario 4: "Publication-ready - need all analyses"
1. **Global understanding:**
   - `2_global_network_metrics/01_global_basic_metrics.py`

2. **Regional details:**
   - `3_nodal_network_metrics/01_nodal_hub_identification.py`
   - `3_nodal_network_metrics/02_nodal_temporal_trajectories.py`

3. **Validation (ML):**
   - `4_statistical_classification/01_svm_baseline_analysis.py`
   - `4_statistical_classification/05_random_forest_comparison.py`

4. **Special populations:**
   - `5_responder_analysis/01_responder_classification.py`
   - `6_visualization/01_sex_interactions.py`

✅ **Done!** Multiple complementary findings for publication.

---

## Flowchart: Visual Decision Tree

```
                        START
                         │
                ┌────────┴────────┐
                │                 │
           UNDERSTAND        PREDICT/EXPLAIN
           THE DATA          THE DATA
                │                 │
                │                 │
         ┌──────┴──────┐      ┌────┴──────┐
         │             │      │           │
      GLOBAL       REGIONAL  ML        HETEROGENEITY
      (1 value)   (246 values) 
         │             │      │           │
         │             │      │           │
    Category 2    Category 3  Category 4  Category 5
         │             │      │           │
         ├─ Efficiency ├─ Hubs├─ SVM      ├─ Responders
         │  Small-     │ Traj │ Time      │ Nonlinear
         │  world      │      │ RF        │
         └─────────────┴──────┴──────┬────┴─────────
                                     │
                             Category 6 (optional)
                              SEX INTERACTIONS
                                   │
                         ┌─────────┴─────────┐
                         │                   │
                    Overall           By group
                    effects          effects
```

---

## Quick Reference: Which Script for Each Question?

| Your Question | Script | Category | Time |
|---|---|---|---|
| Is network topology improving? | 2/01 | Global | 5 min |
| Which regions change most? | 3/01 | Nodal | 10 min |
| Hub vs non-hub differences? | 3/01 | Nodal | 10 min |
| Regional temporal patterns? | 3/02 | Nodal | 10 min |
| Metric correlations? | 3/03 | Nodal | 15 min |
| Can we predict group? | 4/01 | ML | 20 min |
| Can we predict time? | 4/02 | ML | 20 min |
| Group differences (SVM)? | 4/03 | ML | 20 min |
| Time × Group effects? | 4/04 | ML | 20 min |
| RF vs SVM? | 4/05 | ML | 25 min |
| Who responds? | 5/01 | Responder | 15 min |
| Response patterns (nonlinear)? | 5/02 | Responder | 15 min |
| Temporal nonlinearity? | 5/03 | Responder | 15 min |
| Sex × Intervention? | 6/01 | Interaction | 20 min |
| Sex effects by group? | 6/02 | Interaction | 20 min |

---

## Recommended Execution Plans

### ⚡ Quick (30 min)
```
2/01_global_basic_metrics.py
3/01_nodal_hub_identification.py
→ You understand what changed
```

### 📊 Standard (1.5 hours)
```
2/01_global_basic_metrics.py
3/01_nodal_hub_identification.py
3/02_nodal_temporal_trajectories.py
4/01_svm_baseline_analysis.py
5/01_responder_classification.py
→ Complete understanding with validation
```

### 🔬 Comprehensive (3-4 hours)
```
ALL scripts in order: Category 2 → 3 → 4 → 5 → 6
→ Publication-ready with all analyses
```

---

## For Different User Types

### 👨‍🔬 Research Student (New)
```
START HERE:
ANALYSIS_SCRIPTS_QUICK_START.md
↓
Follow the "Typical Analysis Pipeline"
↓
Run: 2/01 → 3/01 → 3/02 → 4/01
```

### 🧬 PI / Lab Manager
```
OVERVIEW:
ANALYSIS_SCRIPTS_VISUAL_GUIDE.md
↓
Run complete pipeline: Categories 2-6
↓
Check ANALYSIS_SCRIPTS_ORGANIZATION.md for details
```

### 📈 Data Analyst (Experienced)
```
REFERENCE:
ANALYSIS_SCRIPTS_ORGANIZATION.md
↓
Script-specific README files
↓
Customize parameters as needed
```

### 👨‍💼 Clinical Collaborator
```
INTERPRETATION:
ANALYSIS_SCRIPTS_VISUAL_GUIDE.md (example results)
↓
Category-specific README for your analysis type
↓
Ask your analyst to run appropriate scripts
```

---

## Help! I'm Still Confused

**Try this:**

1. **What's your main research question?**
   - Answer: Check the category that addresses it

2. **Do you understand global vs nodal?**
   - Read: ANALYSIS_SCRIPTS_VISUAL_GUIDE.md → "Global vs Nodal"

3. **Should you run multiple scripts?**
   - Read: ANALYSIS_SCRIPTS_QUICK_START.md → "Typical Pipeline"

4. **Don't know which specific scripts to run?**
   - Follow: "Recommended Execution Plans" above

5. **Still confused?**
   - Start with: `2_global_network_metrics/01_global_basic_metrics.py`
   - This is always a safe first choice!

---

## Summary

✅ **Start with:** Global metrics (Category 2)  
✅ **Then move to:** Nodal metrics (Category 3)  
✅ **If predicting:** Classification (Category 4)  
✅ **If heterogeneous:** Responders (Category 5)  
✅ **If interactive:** Visualization (Category 6)  

**Remember:** Run scripts in numbered order (01, 02, 03...) within each category!

