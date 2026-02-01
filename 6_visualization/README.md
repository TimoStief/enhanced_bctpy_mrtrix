# Visualization & Interaction Analysis

These scripts focus on **creating visualizations** and testing **statistical interactions** between variables.

## What Are "Interactions"?

An interaction means the effect of one variable depends on another:

**Example: Sex × Intervention Interaction**
- Women: Intervention increases frontal connectivity by 15%
- Men: Intervention increases frontal connectivity by 5%
- **Interaction:** Sex changes HOW MUCH men/women respond

**Visual:** Different slopes for men vs women = interaction present

## Scripts in This Category

### 01_sex_interactions.py
**Purpose:** Test and visualize if intervention effects differ by sex

**Input:**
- All metrics (global and nodal)
- Sex labels for each subject
- Intervention group assignment

**Output:**
- Sex × Intervention ANOVA/regression results
- Visualization plots (males vs females)
- Effect size comparisons

**When to use:**
- Test if exercise intervention effects are equal across sexes
- Identify sex-specific network changes
- Understand precision medicine: does sex guide treatment?

**Typical findings:**
- "Females show greater hippocampal connectivity increase"
- "Males respond more in motor regions"
- "Sex interaction is significant at p<0.05"

---

### 02_sex_interactions_5groups.py
**Purpose:** Sex interactions stratified by 5 exercise groups

**Input:**
- Metrics organized by exercise group and sex

**Output:**
- Group-specific sex interaction tests
- Comparative visualization across all 5 groups
- Interaction × Group × Sex patterns

**When to use:**
- Test if sex differences vary by exercise type (e.g., aerobic vs strength)
- Compare which groups show strongest sex interactions
- Understand if certain groups are more effective for one sex

---

## Interaction Patterns: Understanding the Results

```
Scenario 1: NO INTERACTION (Parallel lines)
        Connectivity
              |     Female
              |    /
              |   /  Male
              |  /
              |___________ Time

Effect is the same for both sexes, just shifted up/down.

---

Scenario 2: INTERACTION (Crossing or diverging lines)
        Connectivity
              |     Female
              |    /
              |   /
              |  /  Male
              | /
              |___________ Time

Female response is LARGER than male response → Interaction!
```

## Typical Workflow

1. Run **01_sex_interactions.py** → Does intervention affect sexes differently?
2. If significant interaction found → Run **02_sex_interactions_5groups.py**
3. Interpret results → Does every group show the interaction?

---

## What to Look For

**Statistical significance:** 
- p-value < 0.05 for Sex × Intervention interaction term

**Effect size:**
- How much do effect sizes differ? (Cohen's d)
- Is the difference clinically meaningful?

**Region specificity:**
- Do all brain regions show the interaction?
- Only certain regions (e.g., limbic, motor)?

---

## Next Steps

After visualization/interaction analysis:
- Integrate findings into main results section
- Combine with responder analysis → "Do responders × sex show interactions?"
- Create publication-ready figures

---

## Technical Notes

- Statistical model: 2-way or 3-way ANOVA
- Follow-up tests: Simple effects analysis (test slopes separately)
- Visualization: Interaction plots, boxplots by group
- Correction: Multiple comparison correction (FDR) across regions
- Effect size: η² (eta-squared) for effect magnitude

## Key Questions Answered

1. "Do males and females respond differently to exercise?" ✓
2. "Which regions show sex-specific responses?" ✓
3. "Do certain exercise types benefit one sex more?" ✓
4. "Should we tailor interventions by sex?" ✓

