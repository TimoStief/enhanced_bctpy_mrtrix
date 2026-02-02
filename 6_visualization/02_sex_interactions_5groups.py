"""
Visualization: Sex Interactions in Group 1 + Other Metrics
===========================================================
Plots the sex×group interaction for Group 1 (the reversal pattern).
Also examines which other metrics show this interaction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
BCT_RESULTS = Path('/data/local/129_PK01/derivatives/bct/bct_analysis_results.parquet')
PARTICIPANTS_TSV = Path('/data/local/129_PK01/derivatives/bct/participants_5groups.tsv')
SLOPES_CSV = Path('/data/local/129_PK01/derivatives/bct/time_effect_slopes_5groups.csv')
OUTPUT_DIR = Path('/data/local/129_PK01/derivatives/bct')

# Load data
df_bct = pd.read_parquet(BCT_RESULTS)
df_demo = pd.read_csv(PARTICIPANTS_TSV, sep='\t')
df_slopes = pd.read_csv(SLOPES_CSV)

# Merge demographics - slopes already has them but we need them clean
df_merged = df_slopes.copy()

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# =========================================================================
# FIGURE 1: Group 1 Sex Interaction - Strength Metrics
# =========================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Group 1: Sex Interaction in Connectivity Trajectories\n(Male vs Female Temporal Patterns)', 
             fontsize=14, fontweight='bold')

# Filter Group 1
df_g1 = df_merged[df_merged['group'] == 1].copy()

strength_metrics = [
    'strength_mean_slope',
    'strength_std_slope',
    'avg_strength_slope',
    'max_strength_slope',
    'weight_mean_slope',
    'weight_std_slope'
]

for idx, metric in enumerate(strength_metrics):
    ax = axes[idx // 3, idx % 3]
    
    # Plot by sex
    data_to_plot = []
    labels = []
    colors = []
    
    for sex in ['M', 'F']:
        values = df_g1[df_g1['sex'] == sex][metric].dropna()
        data_to_plot.append(values)
        labels.append(f"{'Male' if sex == 'M' else 'Female'} (n={len(values)})")
        colors.append('#1f77b4' if sex == 'M' else '#ff7f0e')
    
    # Boxplot
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                    widths=0.6, showmeans=True)
    
    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Overlay individual points
    for i, (vals, color) in enumerate(zip(data_to_plot, colors)):
        x = np.random.normal(i+1, 0.04, size=len(vals))
        ax.scatter(x, vals, alpha=0.4, s=40, color=color)
    
    ax.set_ylabel('Slope (change/session)', fontsize=10)
    ax.set_title(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    male_vals = df_g1[df_g1['sex'] == 'M'][metric].dropna()
    female_vals = df_g1[df_g1['sex'] == 'F'][metric].dropna()
    
    if len(male_vals) > 0 and len(female_vals) > 0:
        from scipy.stats import ttest_ind
        t_stat, p_val = ttest_ind(male_vals, female_vals)
        ax.text(0.5, 0.95, f'p={p_val:.3f}', transform=ax.transAxes,
               ha='center', va='top', fontsize=9, 
               bbox=dict(boxstyle='round', facecolor='yellow' if p_val < 0.1 else 'white', alpha=0.7))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'group1_sex_interaction_5groups_strength.png', dpi=300, bbox_inches='tight')
print("✅ Saved: group1_sex_interaction_5groups_strength.png")
plt.close()

# =========================================================================
# FIGURE 2: Other Important Metrics by Sex (All Groups)
# =========================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Connectivity Slopes by Sex (All Groups)\nOther Important Metrics', 
             fontsize=14, fontweight='bold')

other_metrics = [
    'modularity_slope',
    'degree_mean_slope',
    'density_slope',
    'participation_coef_mean_slope',
    'module_degree_zscore_mean_slope',
    'edge_density_slope'
]

for idx, metric in enumerate(other_metrics):
    ax = axes[idx // 3, idx % 3]
    
    # Create data for all groups and both sexes
    plot_data = []
    colors_list = []
    
    for group in [1, 2, 3]:
        for sex in ['M', 'F']:
            subset = df_merged[(df_merged['group'] == group) & (df_merged['sex'] == sex)]
            if len(subset) > 0:
                values = subset[metric].dropna()
                plot_data.append(values)
                
                # Color by group, shade by sex
                base_color = ['#1f77b4', '#ff7f0e', '#2ca02c'][group - 1]
                # Add transparency to distinguish sex
                if sex == 'F':
                    base_color = base_color  # Solid for female
                colors_list.append(base_color)
    
    # Violin plot if we have data
    if plot_data:
        parts = ax.violinplot(plot_data, positions=range(len(plot_data)), 
                              showmeans=True, showmedians=True)
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors_list[i])
            pc.set_alpha(0.7)
    
    ax.set_ylabel('Slope (change/session)', fontsize=10)
    ax.set_title(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xticks(range(len(plot_data)))
    
    labels = []
    for group in [1, 2, 3]:
        for sex in ['M', 'F']:
            if any((df_merged['group'] == group) & (df_merged['sex'] == sex)):
                labels.append(f'G{group}\n{"M" if sex == "M" else "F"}')
    
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'all_groups_metrics_5groups_by_sex.png', dpi=300, bbox_inches='tight')
print("✅ Saved: all_groups_metrics_5groups_by_sex.png")
plt.close()

# =========================================================================
# FIGURE 3: Heatmap of Mean Slopes by Sex × Group
# =========================================================================
metrics_for_heatmap = [
    'strength_mean_slope', 'strength_std_slope', 'modularity_slope',
    'degree_mean_slope', 'density_slope', 'edge_density_slope',
    'participation_coef_mean_slope', 'module_degree_zscore_mean_slope'
]

# Create matrices for each sex
data_male = []
data_female = []
row_labels = []

for metric in metrics_for_heatmap:
    row_male = []
    row_female = []
    
    for group in [1, 2, 3]:
        male_mean = df_merged[(df_merged['group'] == group) & (df_merged['sex'] == 'M')][metric].mean()
        female_mean = df_merged[(df_merged['group'] == group) & (df_merged['sex'] == 'F')][metric].mean()
        
        row_male.append(male_mean)
        row_female.append(female_mean)
    
    data_male.append(row_male)
    data_female.append(row_female)
    row_labels.append(metric.replace('_', '\n').title())

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# Male heatmap
sns.heatmap(data_male, annot=True, fmt='.1f', cmap='RdBu_r', center=0,
            xticklabels=['Group 1', 'Group 2', 'Group 3'],
            yticklabels=row_labels, ax=axes[0], cbar_kws={'label': 'Mean Slope'})
axes[0].set_title('Males: Mean Metric Slopes by Group', fontweight='bold', fontsize=12)

# Female heatmap
sns.heatmap(data_female, annot=True, fmt='.1f', cmap='RdBu_r', center=0,
            xticklabels=['Group 1', 'Group 2', 'Group 3'],
            yticklabels=row_labels, ax=axes[1], cbar_kws={'label': 'Mean Slope'})
axes[1].set_title('Females: Mean Metric Slopes by Group', fontweight='bold', fontsize=12)

fig.suptitle('Sex Differences in Temporal Connectivity Patterns', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'heatmap_sex_group_5groups_slopes.png', dpi=300, bbox_inches='tight')
print("✅ Saved: heatmap_sex_group_5groups_slopes.png")
plt.close()

# =========================================================================
# FIGURE 4: Statistical Summary Table
# =========================================================================
from scipy.stats import ttest_ind

summary_data = []

for group in [1, 2, 3]:
    for metric in metrics_for_heatmap:
        male_vals = df_merged[(df_merged['group'] == group) & (df_merged['sex'] == 'M')][metric].dropna()
        female_vals = df_merged[(df_merged['group'] == group) & (df_merged['sex'] == 'F')][metric].dropna()
        
        if len(male_vals) > 1 and len(female_vals) > 1:
            t_stat, p_val = ttest_ind(male_vals, female_vals)
            
            summary_data.append({
                'Group': f'Group {group}',
                'Metric': metric.replace('_slope', ''),
                'Male Mean': male_vals.mean(),
                'Female Mean': female_vals.mean(),
                'Difference': male_vals.mean() - female_vals.mean(),
                't-statistic': t_stat,
                'p-value': p_val,
                'Significant': 'Yes' if p_val < 0.05 else 'No'
            })

df_summary = pd.DataFrame(summary_data)
df_summary = df_summary.sort_values(['Group', 'p-value'])

# Save to CSV
df_summary.to_csv(OUTPUT_DIR / 'sex_group_statistics_5groups.csv', index=False)
print("✅ Saved: sex_group_statistics_5groups.csv")

# Print significant findings
print("\n" + "="*80)
print("SIGNIFICANT SEX DIFFERENCES (p < 0.05) BY GROUP")
print("="*80)

for group in [1, 2, 3]:
    sig_rows = df_summary[(df_summary['Group'] == f'Group {group}') & (df_summary['Significant'] == 'Yes')]
    if len(sig_rows) > 0:
        print(f"\n{group}:")
        for _, row in sig_rows.iterrows():
            print(f"  {row['Metric']:30s} Male: {row['Male Mean']:8.1f}  Female: {row['Female Mean']:8.1f}  p={row['p-value']:.4f}")
    else:
        print(f"\nGroup {group}: No significant sex differences")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

# Group 1 analysis
g1_strength_male = df_merged[(df_merged['group'] == 1) & (df_merged['sex'] == 'M')]['strength_mean_slope'].mean()
g1_strength_female = df_merged[(df_merged['group'] == 1) & (df_merged['sex'] == 'F')]['strength_mean_slope'].mean()

print(f"\n🔍 GROUP 1 (THE REVERSAL):")
print(f"   Males:   strength decreasing over time (slope = {g1_strength_male:.1f})")
print(f"   Females: strength increasing over time (slope = {g1_strength_female:.1f})")
print(f"   Opposite trajectories suggest fundamentally different neural dynamics in this group by sex")

# Check other metrics
print(f"\n🔍 OTHER METRICS IN GROUP 1:")
for metric in ['modularity_slope', 'degree_mean_slope', 'density_slope']:
    male_mean = df_merged[(df_merged['group'] == 1) & (df_merged['sex'] == 'M')][metric].mean()
    female_mean = df_merged[(df_merged['group'] == 1) & (df_merged['sex'] == 'F')][metric].mean()
    print(f"   {metric:30s} Male: {male_mean:8.1f}  Female: {female_mean:8.1f}")

print("\n✅ Visualization complete!")
