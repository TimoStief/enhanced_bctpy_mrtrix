# Responder Analysis

These scripts identify **intervention responders vs non-responders** and characterize nonlinear response patterns that differ from simple linear effects.

## What is a "Responder"?

A responder is a subject whose brain network connectivity **changes substantially in response to the intervention**.

**Approach:**
1. Calculate "response magnitude" = combined connectivity change across key metrics
2. Stratify within intervention groups: top 50% = responders, bottom 50% = non-responders
3. Compare responders vs non-responders on:
   - Baseline characteristics (Could we predict who would respond?)
   - Temporal trajectories (Do they change differently?)
   - Network features (Which regions discriminate?)

**Key finding:** Not everyone responds equally → Precision medicine perspective

## Scripts in This Category

### 01_responder_classification.py
**Purpose:** Identify responders vs non-responders and characterize their differences

**Input:**
- Connectivity metrics across all timepoints
- Intervention group assignment

**Output:**
- Responder classification (1 = responder, 0 = non-responder)
- Baseline comparison tables (responders vs non-responders at T0)
- Feature importance for predicting responder status

**When to use:**
- To understand intervention heterogeneity
- To identify predictive baseline characteristics
- To segment subjects for precision medicine

**Key outputs:**
- `responder_classification.parquet` - Responder status per subject
- `responder_vs_nonresponder_comparison.csv` - Statistical tests
- `baseline_prediction_importance.png` - Which baseline metrics predict response?

---

### 02_responder_nonlinear_analysis.py
**Purpose:** Analyze nonlinear trajectory patterns specific to responders

**Input:**
- Metrics classified by responder status

**Output:**
- Trajectory type classification (linear, accelerating, decelerating, plateauing)
- Visualization of typical responder vs non-responder trajectories
- Nonlinear fit parameters

**When to use:**
- To characterize *how* responders change differently
- Do responders show continuous improvement or rapid initial change?
- To identify distinct trajectory subtypes

---

### 03_nonlinear_time_effects.py
**Purpose:** General nonlinear temporal dynamics across all subjects

**Input:**
- Metrics with timepoint labels

**Output:**
- Nonlinear model fits (polynomial, spline, exponential)
- Comparison of linear vs nonlinear explanatory power
- Group-specific nonlinear effects

**When to use:**
- Test if linear models miss important temporal dynamics
- Identify acceleration/deceleration of effects
- Compare trajectory shapes across groups

---

## Responder vs Nonlinear Analysis

| Analysis | Question | Comparison |
|----------|----------|-----------|
| **Responder classification** | "Who responds?" | Responders vs non-responders |
| **Nonlinear time effects** | "What shape is the response?" | Linear vs curved trajectories |

You can combine them:
- **Question:** Do responders show nonlinear curves while non-responders stay flat?

---

## Typical Workflow

1. Run **01_responder_classification.py** → Identify who responds
2. Examine responder characteristics → What baseline features predict response?
3. Run **02_responder_nonlinear_analysis.py** → How do trajectories differ?
4. Run **03_nonlinear_time_effects.py** → Are effects truly nonlinear?

---

## Clinical/Practical Interpretation

**Example findings:**

- "Top 50% of responders show 2× connectivity change in hub regions"
  - → Precision medicine: Use these hub metrics to predict treatment response

- "Responders have higher baseline cognitive reserve"
  - → Screen candidates: prefer high-reserve subjects?

- "Non-responders show flat trajectories; responders show sigmoid (S-curve)"
  - → Some subjects need longer intervention duration

- "Sex interacts with response: females respond more in frontal regions"
  - → Personalized: tailor intervention by sex

---

## Next Steps

After responder analysis:
- **4_statistical_classification/** - Can we predict responder status from baseline metrics?
- **6_visualization/** - Visualize responder × sex interactions
- **Clinical interpretation** - Use to design precision medicine interventions

---

## Technical Notes

- Response magnitude: Standardized Euclidean distance in metric space
- Stratification: Median split within intervention groups (preserves group size)
- Prediction: Logistic regression + cross-validation
- Nonlinear fits: Polynomial (2nd, 3rd order) and natural cubic splines
- Comparison metric: R² (coefficient of determination)

## Key Hypothesis

**Intervention effects are heterogeneous**
- Not everyone responds the same way
- Understanding responders vs non-responders is clinically important
- Baseline network features may predict who will benefit

