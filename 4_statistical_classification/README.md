# Statistical Classification

These scripts use **machine learning** to test if we can predict group membership or timepoint from connectivity metrics.

## What are "Classification" Methods?

Instead of testing group differences with t-tests/ANOVA, we:
1. Train an algorithm on data from some subjects (training set)
2. Test if it can correctly predict group/time for held-out subjects (test set)
3. High accuracy = connectivity metrics contain strong group/time signal

**Result:** Classifier accuracy (%) + feature importance

## Scripts in This Category

### 01_svm_baseline_analysis.py
**Purpose:** Support Vector Machine (SVM) classification with hyperparameter exploration

**Input:**
- Global and nodal metrics

**Output:**
- SVM model performance (accuracy, precision, recall, AUC)
- Feature importance rankings
- Cross-validation results

**When to use:**
- Test if connectivity metrics distinguish between exercise groups
- Identify which metrics are most discriminative
- Compare different SVM kernels (linear, RBF, polynomial)

**Design question answered:**
- "Can we predict who is in which exercise group from their connectivity?"

---

### 02_svm_time_effects.py
**Purpose:** SVM classification for temporal patterns (3 post-baseline timepoints)

**Input:**
- Connectivity metrics across 4 timepoints

**Output:**
- Time-prediction accuracy
- Which metrics best capture temporal changes
- Time×group interactions

**When to use:**
- Test if intervention causes detectable network changes over time
- Compare which timepoint differences are largest

**Design question answered:**
- "Can we predict which timepoint a measurement is from?"

---

### 03_svm_stratified_5groups.py
**Purpose:** SVM classification stratified by 5 exercise groups

**Input:**
- Metrics organized by group

**Output:**
- Group classification accuracy
- Within-group performance metrics
- Group-specific important features

**When to use:**
- Test if exercise groups have distinct connectivity profiles
- Compare classification power across group types

---

### 04_svm_time_5groups.py
**Purpose:** SVM classification with nested group and time structure

**Input:**
- Metrics with group and timepoint labels

**Output:**
- Time prediction within each group
- Group prediction at each timepoint
- Interaction patterns

**When to use:**
- Understand if time effects differ by exercise group
- Test for group × time interactions in connectivity

---

### 05_random_forest_comparison.py
**Purpose:** Random Forest classification vs SVM comparison

**Input:**
- Same metrics as SVM analyses

**Output:**
- RF vs SVM performance comparison
- Feature importance from RF (different ranking than SVM)
- Which algorithm is more robust

**When to use:**
- Verify SVM findings with alternative algorithm
- Understand feature importance robustness
- Compare linear (SVM) vs nonlinear (RF) decision boundaries

---

## SVM vs Random Forest

| Aspect | SVM | Random Forest |
|--------|-----|---------------|
| **Decision boundary** | Linear/curved hyperplane | Nonlinear trees |
| **Feature scaling** | Required | Not required |
| **Interpretability** | Medium | High |
| **Robustness** | Good for small n | Good for medium n |
| **Speed** | Fast | Fast |
| **Use case** | Linear separability | Complex patterns |

## Typical Workflow

1. Run **01_svm_baseline_analysis.py** → Baseline classification performance
2. Run **02_svm_time_effects.py** → Temporal signal strength
3. Run **03_svm_stratified_5groups.py** → Group-specific patterns
4. Run **05_random_forest_comparison.py** → Validate with RF

## Next Steps

- **5_responder_analysis/** - Do responders vs non-responders differ in classification accuracy?
- **Interpretation** - Features ranked high in SVM → focus nodal analysis on those regions

---

## Technical Notes

- Cross-validation: 5-fold stratified
- Hyperparameters: Grid search over kernel, C, gamma
- Feature scaling: StandardScaler (zero-mean, unit-variance)
- Imbalanced classes: Handled via class_weight='balanced'
- Metrics: Accuracy, Precision, Recall, F1, AUC-ROC

