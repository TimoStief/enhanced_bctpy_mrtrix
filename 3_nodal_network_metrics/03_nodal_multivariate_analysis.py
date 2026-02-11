"""
Node-Level Comprehensive Analysis
==================================

Mirrors the full spectrum of analyses done on global metrics, but at node level:
1. 5-group intervention comparison (full spectrum)
2. Social effects (alone vs group)
3. Duration effects (2w vs 4w)
4. Intervention vs control (binary)
5. Gender effects
6. Age correlations
7. Responder phenotyping at node level

For each analysis, identifies which specific nodes show significant effects.

Output: Statistical results, node-specific effect maps, demographic analyses

Author: Analysis Pipeline
Date: January 2026
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import f_oneway, ttest_ind, pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ======================== CONFIGURATION ========================
# ======================== CONFIGURATION ========================

from pathlib import Path

DATA_DIR = Path("C:/Users/timo-/Desktop/Forschung/Test_matrizen")
OUTPUT_DIR = Path("C:/Users/timo-/Desktop/Forschung/enhanced_bctpy_mrtrix/outputs/nodal_change")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ATLAS = "Brodmann"
N_NODES = 78

SESSIONS = [1, 2, 3, 4]

METRICS = [
    "degree",
    "strength",
    "richness"
]

print("="*70)
print("NODAL LONGITUDINAL CHANGE ANALYSIS")
print("="*70)
print(f"Atlas: {ATLAS} ({N_NODES} nodes)")
print(f"Output: {OUTPUT_DIR}")
print()


# ======================== LOAD DATA ========================
print("Loading node-level data...")

node_df = pd.read_parquet(NODE_ANALYSIS_DIR / "node_level_metrics.parquet")
print(f"Loaded {len(node_df)} node-level records")

# Node data already has demographics merged, no need to load participants separately
print(f"Demographics already included in node data")

# Convert sex to numeric if needed (M=1, F=2)
if node_df['sex'].dtype == 'object':
    node_df['sex'] = node_df['sex'].map({'M': 1, 'F': 2}).fillna(node_df['sex'])

print(f"Ready for analysis: {len(node_df)} records")
print()

# Define metrics to analyze
METRICS = ['degree', 'strength', 'betweenness', 'clustering', 
           'participation_coef', 'within_module_zscore']

# Group definitions
GROUP_LABELS = {
    1: 'alone_2w',
    2: 'alone_4w', 
    3: 'group_2w',
    4: 'group_4w',
    5: 'control'
}

# ======================== ANALYSIS 1: 5-GROUP COMPARISON ========================
print("="*70)
print("ANALYSIS 1: 5-GROUP FULL SPECTRUM COMPARISON")
print("="*70)

five_group_results = []

for node in range(1, N_NODES + 1):
    for metric in METRICS:
        node_metric_data = node_df[
            (node_df['node'] == node) &
            (node_df[metric].notna())
        ]
        
        if len(node_metric_data) < 5:
            continue
        
        # Get values per group
        groups_data = []
        for grp in [1, 2, 3, 4, 5]:
            grp_data = node_metric_data[node_metric_data['group'] == grp][metric].values
            if len(grp_data) > 0:
                groups_data.append(grp_data)
        
        if len(groups_data) < 2:
            continue
        
        # ANOVA
        try:
            f_stat, p_val = f_oneway(*groups_data)
            
            # Calculate effect size (eta-squared)
            group_means = [np.mean(g) for g in groups_data]
            grand_mean = np.mean(np.concatenate(groups_data))
            ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups_data)
            ss_total = sum(np.sum((g - grand_mean)**2) for g in groups_data)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            five_group_results.append({
                'node': node,
                'metric': metric,
                'f_statistic': f_stat,
                'p_value': p_val,
                'eta_squared': eta_squared,
                'significant': p_val < 0.05,
                'mean_group1': group_means[0] if len(group_means) > 0 else np.nan,
                'mean_group2': group_means[1] if len(group_means) > 1 else np.nan,
                'mean_group3': group_means[2] if len(group_means) > 2 else np.nan,
                'mean_group4': group_means[3] if len(group_means) > 3 else np.nan,
                'mean_group5': group_means[4] if len(group_means) > 4 else np.nan,
            })
        except:
            continue

five_group_df = pd.DataFrame(five_group_results)
print(f"Analyzed {len(five_group_df)} node-metric combinations")
print(f"Significant effects (p<0.05): {five_group_df['significant'].sum()}")
print(f"Strong effects (η²>0.1): {(five_group_df['eta_squared'] > 0.1).sum()}")
print()

# Top nodes with group effects
print("Top 10 nodes with strongest 5-group effects:")
top_5group = five_group_df.nlargest(10, 'eta_squared')[['node', 'metric', 'eta_squared', 'p_value']]
print(top_5group.to_string(index=False))
print()

# Save
five_group_df.to_parquet(OUTPUT_DIR / "five_group_anova.parquet", index=False)
print(f"✅ Saved: five_group_anova.parquet")
print()

# ======================== ANALYSIS 2: SOCIAL EFFECTS (ALONE VS GROUP) ========================
print("="*70)
print("ANALYSIS 2: SOCIAL EFFECTS (ALONE VS GROUP)")
print("="*70)

social_results = []

for node in range(1, N_NODES + 1):
    for metric in METRICS:
        # Get alone (groups 1,2) vs group (groups 3,4)
        alone_data = node_df[
            (node_df['node'] == node) &
            (node_df['group'].isin([1, 2])) &
            (node_df[metric].notna())
        ][metric].values
        
        group_data = node_df[
            (node_df['node'] == node) &
            (node_df['group'].isin([3, 4])) &
            (node_df[metric].notna())
        ][metric].values
        
        if len(alone_data) < 3 or len(group_data) < 3:
            continue
        
        # T-test
        try:
            t_stat, p_val = ttest_ind(alone_data, group_data)
            
            # Cohen's d
            pooled_std = np.sqrt((np.std(alone_data)**2 + np.std(group_data)**2) / 2)
            cohens_d = (np.mean(alone_data) - np.mean(group_data)) / pooled_std if pooled_std > 0 else 0
            
            social_results.append({
                'node': node,
                'metric': metric,
                't_statistic': t_stat,
                'p_value': p_val,
                'cohens_d': cohens_d,
                'abs_cohens_d': np.abs(cohens_d),
                'mean_alone': np.mean(alone_data),
                'mean_group': np.mean(group_data),
                'direction': 'alone>group' if cohens_d > 0 else 'group>alone',
                'significant': p_val < 0.05,
                'n_alone': len(alone_data),
                'n_group': len(group_data)
            })
        except:
            continue

social_df = pd.DataFrame(social_results)
print(f"Analyzed {len(social_df)} node-metric combinations")
print(f"Significant social effects (p<0.05): {social_df['significant'].sum()}")
print(f"Large effects (|d|>0.8): {(social_df['abs_cohens_d'] > 0.8).sum()}")
print()

print("Top 10 nodes with strongest social effects:")
top_social = social_df.nlargest(10, 'abs_cohens_d')[['node', 'metric', 'cohens_d', 'direction', 'p_value']]
print(top_social.to_string(index=False))
print()

social_df.to_parquet(OUTPUT_DIR / "social_effects.parquet", index=False)
print(f"✅ Saved: social_effects.parquet")
print()

# ======================== ANALYSIS 3: DURATION EFFECTS (2W VS 4W) ========================
print("="*70)
print("ANALYSIS 3: DURATION EFFECTS (2W VS 4W)")
print("="*70)

duration_results = []

for node in range(1, N_NODES + 1):
    for metric in METRICS:
        # Get 2w (groups 1,3) vs 4w (groups 2,4)
        week2_data = node_df[
            (node_df['node'] == node) &
            (node_df['group'].isin([1, 3])) &
            (node_df[metric].notna())
        ][metric].values
        
        week4_data = node_df[
            (node_df['node'] == node) &
            (node_df['group'].isin([2, 4])) &
            (node_df[metric].notna())
        ][metric].values
        
        if len(week2_data) < 3 or len(week4_data) < 3:
            continue
        
        # T-test
        try:
            t_stat, p_val = ttest_ind(week2_data, week4_data)
            
            pooled_std = np.sqrt((np.std(week2_data)**2 + np.std(week4_data)**2) / 2)
            cohens_d = (np.mean(week2_data) - np.mean(week4_data)) / pooled_std if pooled_std > 0 else 0
            
            duration_results.append({
                'node': node,
                'metric': metric,
                't_statistic': t_stat,
                'p_value': p_val,
                'cohens_d': cohens_d,
                'abs_cohens_d': np.abs(cohens_d),
                'mean_2w': np.mean(week2_data),
                'mean_4w': np.mean(week4_data),
                'direction': '2w>4w' if cohens_d > 0 else '4w>2w',
                'significant': p_val < 0.05,
                'n_2w': len(week2_data),
                'n_4w': len(week4_data)
            })
        except:
            continue

duration_df = pd.DataFrame(duration_results)
print(f"Analyzed {len(duration_df)} node-metric combinations")
print(f"Significant duration effects (p<0.05): {duration_df['significant'].sum()}")
print(f"Large effects (|d|>0.8): {(duration_df['abs_cohens_d'] > 0.8).sum()}")
print()

print("Top 10 nodes with strongest duration effects:")
top_duration = duration_df.nlargest(10, 'abs_cohens_d')[['node', 'metric', 'cohens_d', 'direction', 'p_value']]
print(top_duration.to_string(index=False))
print()

duration_df.to_parquet(OUTPUT_DIR / "duration_effects.parquet", index=False)
print(f"✅ Saved: duration_effects.parquet")
print()

# ======================== ANALYSIS 4: INTERVENTION VS CONTROL ========================
print("="*70)
print("ANALYSIS 4: INTERVENTION VS CONTROL (BINARY)")
print("="*70)

binary_results = []

for node in range(1, N_NODES + 1):
    for metric in METRICS:
        intervention_data = node_df[
            (node_df['node'] == node) &
            (node_df['group'].isin([1, 2, 3, 4])) &
            (node_df[metric].notna())
        ][metric].values
        
        control_data = node_df[
            (node_df['node'] == node) &
            (node_df['group'] == 5) &
            (node_df[metric].notna())
        ][metric].values
        
        if len(intervention_data) < 5 or len(control_data) < 5:
            continue
        
        try:
            t_stat, p_val = ttest_ind(intervention_data, control_data)
            
            pooled_std = np.sqrt((np.std(intervention_data)**2 + np.std(control_data)**2) / 2)
            cohens_d = (np.mean(intervention_data) - np.mean(control_data)) / pooled_std if pooled_std > 0 else 0
            
            binary_results.append({
                'node': node,
                'metric': metric,
                't_statistic': t_stat,
                'p_value': p_val,
                'cohens_d': cohens_d,
                'abs_cohens_d': np.abs(cohens_d),
                'mean_intervention': np.mean(intervention_data),
                'mean_control': np.mean(control_data),
                'direction': 'intervention>control' if cohens_d > 0 else 'control>intervention',
                'significant': p_val < 0.05,
                'n_intervention': len(intervention_data),
                'n_control': len(control_data)
            })
        except:
            continue

binary_df = pd.DataFrame(binary_results)
print(f"Analyzed {len(binary_df)} node-metric combinations")
print(f"Significant intervention effects (p<0.05): {binary_df['significant'].sum()}")
print(f"Large effects (|d|>0.8): {(binary_df['abs_cohens_d'] > 0.8).sum()}")
print()

print("Top 10 nodes with strongest intervention effects:")
top_binary = binary_df.nlargest(10, 'abs_cohens_d')[['node', 'metric', 'cohens_d', 'direction', 'p_value']]
print(top_binary.to_string(index=False))
print()

binary_df.to_parquet(OUTPUT_DIR / "intervention_vs_control.parquet", index=False)
print(f"✅ Saved: intervention_vs_control.parquet")
print()

# ======================== ANALYSIS 5: GENDER EFFECTS ========================
print("="*70)
print("ANALYSIS 5: GENDER EFFECTS")
print("="*70)

gender_results = []

for node in range(1, N_NODES + 1):
    for metric in METRICS:
        male_data = node_df[
            (node_df['node'] == node) &
            ((node_df['sex'] == 1) | (node_df['sex'] == 'M')) &
            (node_df[metric].notna())
        ][metric].values
        
        female_data = node_df[
            (node_df['node'] == node) &
            ((node_df['sex'] == 2) | (node_df['sex'] == 'F')) &
            (node_df[metric].notna())
        ][metric].values
        
        if len(male_data) < 3 or len(female_data) < 3:
            continue
        
        try:
            t_stat, p_val = ttest_ind(male_data, female_data)
            
            pooled_std = np.sqrt((np.std(male_data)**2 + np.std(female_data)**2) / 2)
            cohens_d = (np.mean(male_data) - np.mean(female_data)) / pooled_std if pooled_std > 0 else 0
            
            gender_results.append({
                'node': node,
                'metric': metric,
                't_statistic': t_stat,
                'p_value': p_val,
                'cohens_d': cohens_d,
                'abs_cohens_d': np.abs(cohens_d),
                'mean_male': np.mean(male_data),
                'mean_female': np.mean(female_data),
                'direction': 'male>female' if cohens_d > 0 else 'female>male',
                'significant': p_val < 0.05,
                'n_male': len(male_data),
                'n_female': len(female_data)
            })
        except:
            continue

gender_df = pd.DataFrame(gender_results)
print(f"Analyzed {len(gender_df)} node-metric combinations")

if len(gender_df) > 0 and 'significant' in gender_df.columns:
    print(f"Significant gender effects (p<0.05): {gender_df['significant'].sum()}")
    print(f"Large effects (|d|>0.8): {(gender_df['abs_cohens_d'] > 0.8).sum()}")
    print()
    
    print("Top 10 nodes with strongest gender effects:")
    top_gender = gender_df.nlargest(10, 'abs_cohens_d')[['node', 'metric', 'cohens_d', 'direction', 'p_value']]
    print(top_gender.to_string(index=False))
else:
    print("No gender effects found (check sex coding: should be M/F)")
print()

gender_df.to_parquet(OUTPUT_DIR / "gender_effects.parquet", index=False)
print(f"✅ Saved: gender_effects.parquet")
print()

# ======================== ANALYSIS 6: AGE CORRELATIONS ========================
print("="*70)
print("ANALYSIS 6: AGE CORRELATIONS")
print("="*70)

age_results = []

for node in range(1, N_NODES + 1):
    for metric in METRICS:
        node_age_data = node_df[
            (node_df['node'] == node) &
            (node_df[metric].notna()) &
            (node_df['age'].notna())
        ]
        
        if len(node_age_data) < 10:
            continue
        
        ages = node_age_data['age'].values
        values = node_age_data[metric].values
        
        try:
            # Pearson correlation
            r_pearson, p_pearson = pearsonr(ages, values)
            
            # Spearman correlation (non-parametric)
            r_spearman, p_spearman = spearmanr(ages, values)
            
            age_results.append({
                'node': node,
                'metric': metric,
                'r_pearson': r_pearson,
                'p_pearson': p_pearson,
                'r_spearman': r_spearman,
                'p_spearman': p_spearman,
                'abs_r_pearson': np.abs(r_pearson),
                'direction': 'positive' if r_pearson > 0 else 'negative',
                'significant_pearson': p_pearson < 0.05,
                'significant_spearman': p_spearman < 0.05,
                'n_samples': len(ages)
            })
        except:
            continue

age_df = pd.DataFrame(age_results)
print(f"Analyzed {len(age_df)} node-metric combinations")
print(f"Significant age correlations (p<0.05, Pearson): {age_df['significant_pearson'].sum()}")
print(f"Strong correlations (|r|>0.3): {(age_df['abs_r_pearson'] > 0.3).sum()}")
print()

print("Top 10 nodes with strongest age correlations:")
top_age = age_df.nlargest(10, 'abs_r_pearson')[['node', 'metric', 'r_pearson', 'direction', 'p_pearson']]
print(top_age.to_string(index=False))
print()

age_df.to_parquet(OUTPUT_DIR / "age_correlations.parquet", index=False)
print(f"✅ Saved: age_correlations.parquet")
print()

# ======================== VISUALIZATIONS ========================
print("="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# 1. Heatmap: Significant effects across analyses
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

analyses = [
    ('5-Group', five_group_df, 'significant', '5-Group ANOVA\n(p<0.05)'),
    ('Social', social_df, 'significant', 'Social Effects\n(Alone vs Group)'),
    ('Duration', duration_df, 'significant', 'Duration Effects\n(2w vs 4w)'),
    ('Intervention', binary_df, 'significant', 'Intervention vs Control'),
    ('Gender', gender_df, 'significant', 'Gender Effects'),
    ('Age', age_df, 'significant_pearson', 'Age Correlations')
]

for idx, (name, df, sig_col, title) in enumerate(analyses):
    ax = axes[idx // 3, idx % 3]
    
    # Create node × metric matrix of significance
    if len(df) > 0 and sig_col in df.columns:
        sig_matrix = df.pivot_table(
            index='node',
            columns='metric',
            values=sig_col,
            aggfunc='sum',
            fill_value=0
        )
        
        sns.heatmap(sig_matrix, cmap='YlOrRd', cbar_kws={'label': 'Significant'}, ax=ax)
    else:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
    
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_xlabel('Metric')
    ax.set_ylabel('Node')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'significance_heatmaps.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: significance_heatmaps.png")

# 2. Effect size distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

effect_analyses = [
    ('Social', social_df, 'abs_cohens_d', "Cohen's d"),
    ('Duration', duration_df, 'abs_cohens_d', "Cohen's d"),
    ('Intervention', binary_df, 'abs_cohens_d', "Cohen's d"),
    ('Gender', gender_df, 'abs_cohens_d', "Cohen's d")
]

for idx, (name, df, col, label) in enumerate(effect_analyses):
    ax = axes[idx // 2, idx % 2]
    
    if len(df) > 0 and col in df.columns:
        metric_effects = df.groupby('metric')[col].apply(list)
        
        # Only plot metrics that exist in the data
        available_metrics = [m for m in METRICS if m in metric_effects.index]
        
        if available_metrics:
            bp = ax.boxplot([metric_effects[m] for m in available_metrics], 
                             labels=available_metrics, patch_artist=True)
            
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            
            ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large effect')
            ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium effect')
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
    else:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
    
    ax.set_ylabel(f'Effect Size ({label})', fontweight='bold')
    ax.set_title(f'{name} Effects Distribution', fontweight='bold', fontsize=12)
    ax.grid(alpha=0.3, axis='y')
    if len(df) > 0:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'effect_size_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: effect_size_distributions.png")

# 3. Top nodes summary plot
fig, axes = plt.subplots(3, 2, figsize=(14, 16))

top_node_analyses = [
    ('5-Group', five_group_df, 'eta_squared', 'η²'),
    ('Social', social_df, 'abs_cohens_d', '|d|'),
    ('Duration', duration_df, 'abs_cohens_d', '|d|'),
    ('Intervention', binary_df, 'abs_cohens_d', '|d|'),
    ('Gender', gender_df, 'abs_cohens_d', '|d|'),
    ('Age', age_df, 'abs_r_pearson', '|r|')
]

for idx, (name, df, col, label) in enumerate(top_node_analyses):
    ax = axes[idx // 2, idx % 2]
    
    if len(df) > 0 and col in df.columns:
        top_20 = df.nlargest(min(20, len(df)), col)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_20)))
        
        ax.barh(range(len(top_20)), top_20[col].values, color=colors)
        ax.set_yticks(range(len(top_20)))
        ax.set_yticklabels([f"Node {int(row['node'])} ({row['metric']})" 
                            for _, row in top_20.iterrows()], fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
    
    ax.set_xlabel(f'Effect Size ({label})', fontweight='bold')
    ax.set_title(f'Top 20 Nodes: {name}', fontweight='bold', fontsize=11)
    ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'top_nodes_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: top_nodes_summary.png")

print()

# ======================== SUMMARY STATISTICS ========================
print("="*70)
print("COMPREHENSIVE NODE-LEVEL ANALYSIS SUMMARY")
print("="*70)
print()

summary_stats = {
    '5-Group ANOVA': {
        'total': len(five_group_df),
        'significant': five_group_df.get('significant', pd.Series()).sum() if len(five_group_df) > 0 else 0,
        'large_effect': (five_group_df['eta_squared'] > 0.14).sum() if len(five_group_df) > 0 else 0
    },
    'Social Effects': {
        'total': len(social_df),
        'significant': social_df.get('significant', pd.Series()).sum() if len(social_df) > 0 else 0,
        'large_effect': (social_df['abs_cohens_d'] > 0.8).sum() if len(social_df) > 0 else 0
    },
    'Duration Effects': {
        'total': len(duration_df),
        'significant': duration_df.get('significant', pd.Series()).sum() if len(duration_df) > 0 else 0,
        'large_effect': (duration_df['abs_cohens_d'] > 0.8).sum() if len(duration_df) > 0 else 0
    },
    'Intervention vs Control': {
        'total': len(binary_df),
        'significant': binary_df.get('significant', pd.Series()).sum() if len(binary_df) > 0 else 0,
        'large_effect': (binary_df['abs_cohens_d'] > 0.8).sum() if len(binary_df) > 0 else 0
    },
    'Gender Effects': {
        'total': len(gender_df),
        'significant': gender_df.get('significant', pd.Series()).sum() if len(gender_df) > 0 else 0,
        'large_effect': (gender_df.get('abs_cohens_d', pd.Series()) > 0.8).sum() if len(gender_df) > 0 else 0
    },
    'Age Correlations': {
        'total': len(age_df),
        'significant': age_df.get('significant_pearson', pd.Series()).sum() if len(age_df) > 0 else 0,
        'strong_correlation': (age_df.get('abs_r_pearson', pd.Series()) > 0.3).sum() if len(age_df) > 0 else 0
    }
}

for analysis, stats in summary_stats.items():
    print(f"{analysis}:")
    print(f"  Total analyzed: {stats['total']}")
    if stats['total'] > 0:
        print(f"  Significant: {stats['significant']} ({100*stats['significant']/stats['total']:.1f}%)")
        effect_key = 'strong_correlation' if 'Age' in analysis else 'large_effect'
        print(f"  Large effects: {stats[effect_key]} ({100*stats[effect_key]/stats['total']:.1f}%)")
    else:
        print("  No data available")
    print()

print("="*70)
print(f"✅ ALL OUTPUTS SAVED TO: {OUTPUT_DIR}")
print("="*70)
print()
print("Generated files:")
print("  - five_group_anova.parquet")
print("  - social_effects.parquet")
print("  - duration_effects.parquet")
print("  - intervention_vs_control.parquet")
print("  - gender_effects.parquet")
print("  - age_correlations.parquet")
print("  - significance_heatmaps.png")
print("  - effect_size_distributions.png")
print("  - top_nodes_summary.png")
print()
print("="*70)
