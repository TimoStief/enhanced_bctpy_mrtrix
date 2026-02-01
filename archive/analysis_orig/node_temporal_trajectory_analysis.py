"""
Node-Level Temporal Trajectory Analysis
========================================

Analyzes how individual Brainnectome regions (nodes) respond to intervention:
1. Temporal trajectories per node per metric
2. Regional intervention effects maps
3. Hub-specific response analysis (provincial vs connector hubs)
4. Rich-club reorganization
5. Publication-ready brain visualizations

Output: Regional effect maps, hub-specific statistics, node trajectories

Author: Analysis Pipeline
Date: January 2026
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# ======================== CONFIGURATION ========================
BCT_DIR = Path("/data/local/129_PK01/derivatives/bct")
NODE_ANALYSIS_DIR = BCT_DIR / "node_level_analysis"
OUTPUT_DIR = BCT_DIR / "node_trajectory_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

ATLAS = "Brainnectome"
N_NODES = 246

print("="*70)
print("NODE-LEVEL TEMPORAL TRAJECTORY ANALYSIS")
print("="*70)
print(f"Input: {NODE_ANALYSIS_DIR}")
print(f"Output: {OUTPUT_DIR}")
print()

# ======================== LOAD NODE-LEVEL DATA ========================
print("Loading node-level metrics...")

node_metrics_file = NODE_ANALYSIS_DIR / "node_level_metrics.parquet"
if not node_metrics_file.exists():
    raise FileNotFoundError(f"Node metrics file not found: {node_metrics_file}")

node_df = pd.read_parquet(node_metrics_file)
print(f"Loaded {len(node_df)} node-level records")
print(f"Nodes: {node_df['node'].max()} regions")
print(f"Sessions: {node_df['session'].unique()}")
print(f"Groups: {node_df['group'].unique()}")
print()

# Load subject summaries for hub classification
summary_file = NODE_ANALYSIS_DIR / "subject_hub_summaries.parquet"
summary_df = pd.read_parquet(summary_file)
print(f"Loaded {len(summary_df)} subject-session hub summaries")
print()

# ======================== CALCULATE NODAL TRAJECTORIES ========================
print("Calculating node-level temporal trajectories...")

nodal_trajectories = []

for node in range(1, N_NODES + 1):
    node_data = node_df[node_df['node'] == node].copy()
    
    if len(node_data) == 0:
        continue
    
    # Get unique metrics per node
    metrics_list = ['degree', 'strength', 'betweenness', 'clustering',
                   'participation_coef', 'within_module_zscore']
    
    for metric in metrics_list:
        if metric not in node_data.columns:
            continue
        
        # Calculate slopes per group
        for group in node_data['group'].unique():
            group_data = node_data[node_data['group'] == group].copy()
            
            if len(group_data) < 2:
                continue
            
            # Fit linear trajectory
            sessions = group_data['session'].values
            values = group_data[metric].values
            
            # Remove NaN
            valid_idx = ~np.isnan(values)
            if np.sum(valid_idx) < 2:
                continue
            
            sessions_valid = sessions[valid_idx]
            values_valid = values[valid_idx]
            
            # Linear fit
            try:
                coeffs = np.polyfit(sessions_valid, values_valid, 1)
                slope = coeffs[0]
                intercept = coeffs[1]
                
                # R-squared
                y_pred = np.polyval(coeffs, sessions_valid)
                ss_res = np.sum((values_valid - y_pred)**2)
                ss_tot = np.sum((values_valid - np.mean(values_valid))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
                
                # Change magnitude
                min_val = np.min(values_valid)
                max_val = np.max(values_valid)
                change_magnitude = max_val - min_val
                
                # Direction (positive = increase, negative = decrease)
                change_direction = 'increasing' if slope > 0 else 'decreasing'
                
                nodal_trajectories.append({
                    'node': node,
                    'metric': metric,
                    'group': group,
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_squared,
                    'change_magnitude': change_magnitude,
                    'change_direction': change_direction,
                    'n_sessions': np.sum(valid_idx),
                    'mean_value': np.mean(values_valid)
                })
            except:
                continue

trajectory_df = pd.DataFrame(nodal_trajectories)
print(f"Calculated {len(trajectory_df)} node-metric-group trajectories")
print()

# Save trajectories
trajectory_output = OUTPUT_DIR / "node_trajectories.parquet"
trajectory_df.to_parquet(trajectory_output, index=False)
print(f"✅ Saved: {trajectory_output}")
print()

# ======================== IDENTIFY KEY RESPONDING NODES ========================
print("Identifying nodes with strongest intervention effects...")

# Compare intervention vs control for each metric
intervention_groups = [1, 2, 3, 4]  # alone_2w, alone_4w, group_2w, group_4w
control_group = 5

effect_sizes = []

for metric in trajectory_df['metric'].unique():
    for node in range(1, N_NODES + 1):
        # Get intervention slopes
        interv_data = trajectory_df[
            (trajectory_df['node'] == node) &
            (trajectory_df['metric'] == metric) &
            (trajectory_df['group'].isin(intervention_groups))
        ]
        
        control_data = trajectory_df[
            (trajectory_df['node'] == node) &
            (trajectory_df['metric'] == metric) &
            (trajectory_df['group'] == control_group)
        ]
        
        if len(interv_data) == 0 or len(control_data) == 0:
            continue
        
        interv_slopes = interv_data['slope'].values
        control_slopes = control_data['slope'].values
        
        # Remove NaN
        interv_slopes = interv_slopes[~np.isnan(interv_slopes)]
        control_slopes = control_slopes[~np.isnan(control_slopes)]
        
        if len(interv_slopes) == 0 or len(control_slopes) == 0:
            continue
        
        # Effect size (Cohen's d)
        mean_diff = np.mean(interv_slopes) - np.mean(control_slopes)
        pooled_std = np.sqrt((np.std(interv_slopes)**2 + np.std(control_slopes)**2) / 2)
        
        if pooled_std > 0:
            cohens_d = mean_diff / pooled_std
        else:
            cohens_d = 0
        
        # T-test
        if len(interv_slopes) > 1 and len(control_slopes) > 1:
            t_stat, p_val = stats.ttest_ind(interv_slopes, control_slopes)
        else:
            t_stat, p_val = np.nan, np.nan
        
        effect_sizes.append({
            'node': node,
            'metric': metric,
            'intervention_mean_slope': np.mean(interv_slopes),
            'control_mean_slope': np.mean(control_slopes),
            'effect_size_cohens_d': cohens_d,
            't_statistic': t_stat,
            'p_value': p_val,
            'abs_effect_size': np.abs(cohens_d)
        })

effects_df = pd.DataFrame(effect_sizes)
print(f"Calculated {len(effects_df)} intervention vs control effects")
print()

# Top responding nodes
print("Top 10 Nodes with Strongest Intervention Effects:")
top_effects = effects_df.nlargest(10, 'abs_effect_size')[
    ['node', 'metric', 'effect_size_cohens_d', 'p_value']
]
print(top_effects.to_string(index=False))
print()

# Save effects
effects_output = OUTPUT_DIR / "intervention_effect_sizes.parquet"
effects_df.to_parquet(effects_output, index=False)
print(f"✅ Saved: {effects_output}")
print()

# ======================== HUB-SPECIFIC ANALYSIS ========================
print("Analyzing hub-specific intervention responses...")

# Get hub classifications from subject summaries
hub_classifications = []

for node in range(1, N_NODES + 1):
    node_trajectories = trajectory_df[trajectory_df['node'] == node]
    
    if len(node_trajectories) == 0:
        continue
    
    # Get hub types from node-level data (modal hub type across subjects)
    node_hub_data = node_df[node_df['node'] == node]['hub_type'].value_counts()
    
    if len(node_hub_data) == 0:
        continue
    
    modal_hub_type = node_hub_data.idxmax()
    
    # Calculate mean slope across intervention groups
    interv_trajectories = node_trajectories[
        node_trajectories['group'].isin(intervention_groups)
    ]
    
    control_trajectories = node_trajectories[
        node_trajectories['group'] == control_group
    ]
    
    if len(interv_trajectories) > 0 and len(control_trajectories) > 0:
        for metric in node_trajectories['metric'].unique():
            interv_metric = interv_trajectories[interv_trajectories['metric'] == metric]
            control_metric = control_trajectories[control_trajectories['metric'] == metric]
            
            if len(interv_metric) > 0 and len(control_metric) > 0:
                hub_classifications.append({
                    'node': node,
                    'metric': metric,
                    'hub_type': modal_hub_type,
                    'intervention_mean_slope': interv_metric['slope'].mean(),
                    'control_mean_slope': control_metric['slope'].mean(),
                    'slope_difference': (interv_metric['slope'].mean() - 
                                        control_metric['slope'].mean())
                })

hub_df = pd.DataFrame(hub_classifications)

# Compare hub types
print("\nMean Intervention Response by Hub Type:")
hub_comparison = hub_df.groupby('hub_type')[
    ['intervention_mean_slope', 'control_mean_slope', 'slope_difference']
].mean()
print(hub_comparison)
print()

# Statistical test: do hub types respond differently?
hub_types = hub_df['hub_type'].unique()
for metric in hub_df['metric'].unique():
    metric_data = hub_df[hub_df['metric'] == metric]
    
    if len(hub_types) >= 2:
        groups_by_hub = [metric_data[metric_data['hub_type'] == ht]['slope_difference'].values
                        for ht in hub_types]
        groups_by_hub = [g for g in groups_by_hub if len(g) > 0]
        
        if len(groups_by_hub) >= 2:
            f_stat, p_val = stats.f_oneway(*groups_by_hub)
            if p_val < 0.05:
                print(f"✅ Hub type effect on {metric}: F={f_stat:.3f}, p={p_val:.4f}")

# Save hub analysis
hub_output = OUTPUT_DIR / "hub_specific_responses.parquet"
hub_df.to_parquet(hub_output, index=False)
print(f"\n✅ Saved: {hub_output}")
print()

# ======================== REGIONAL EFFECT MAPS ========================
print("Creating regional effect maps...")

# Create effect size matrix (nodes × metrics)
effect_matrix = effects_df.pivot_table(
    index='node',
    columns='metric',
    values='effect_size_cohens_d',
    fill_value=0
)

# Heatmap: Intervention effect sizes per node per metric
fig, ax = plt.subplots(figsize=(12, 20))

sns.heatmap(effect_matrix, cmap='RdBu_r', center=0, 
            cbar_kws={'label': "Cohen's d (Intervention Effect)"}, ax=ax)

ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
ax.set_ylabel('Brainnectome Node', fontsize=12, fontweight='bold')
ax.set_title('Intervention Effect Sizes - Node × Metric\n(Red=Increased, Blue=Decreased with Intervention)',
             fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'regional_effect_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: regional_effect_heatmap.png")

# Create effect map per metric (node rankings)
for metric in trajectory_df['metric'].unique():
    metric_effects = effects_df[effects_df['metric'] == metric].copy()
    metric_effects = metric_effects.sort_values('effect_size_cohens_d', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 14))
    
    colors = ['red' if x > 0 else 'blue' for x in metric_effects['effect_size_cohens_d'].values]
    
    ax.barh(range(len(metric_effects)), metric_effects['effect_size_cohens_d'].values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(metric_effects)))
    ax.set_yticklabels(metric_effects['node'].values, fontsize=8)
    ax.set_xlabel("Cohen's d (Intervention Effect Size)", fontsize=11, fontweight='bold')
    ax.set_ylabel('Brainnectome Node', fontsize=11, fontweight='bold')
    ax.set_title(f'Intervention Effect Sizes: {metric.upper()}\n(Top Responding Nodes)',
                 fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'node_effects_{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: node_effects_{metric}.png")

print()

# ======================== HUB TYPE VISUALIZATION ========================
print("Creating hub-specific response visualizations...")

# Distribution of slopes by hub type
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics_to_plot = hub_df['metric'].unique()[:4]  # First 4 metrics

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx // 2, idx % 2]
    
    metric_hub = hub_df[hub_df['metric'] == metric]
    
    hub_types_list = metric_hub['hub_type'].unique()
    data_by_hub = [metric_hub[metric_hub['hub_type'] == ht]['slope_difference'].values
                  for ht in hub_types_list]
    
    bp = ax.boxplot(data_by_hub, labels=hub_types_list, patch_artist=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Intervention Effect (Slope Difference)', fontsize=10)
    ax.set_title(f'Hub Type Response: {metric}', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'hub_type_response_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: hub_type_response_distributions.png")

# ======================== TRAJECTORY EXAMPLES ========================
print("Creating trajectory example plots...")

# Select top responding nodes for visualization
top_nodes = effects_df.nlargest(6, 'abs_effect_size')['node'].unique()

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for plot_idx, node in enumerate(top_nodes):
    ax = axes[plot_idx]
    
    node_trajs = trajectory_df[
        (trajectory_df['node'] == node) &
        (trajectory_df['metric'] == 'degree')  # Show degree as example
    ]
    
    if len(node_trajs) == 0:
        continue
    
    # Plot trajectories per group
    colors_map = {
        'alone_2w': '#FF6B6B', 'alone_4w': '#FFA07A',
        'group_2w': '#4169E1', 'group_4w': '#87CEEB',
        'control': '#808080',
        1: '#FF6B6B', 2: '#FFA07A',
        3: '#4169E1', 4: '#87CEEB',
        5: '#808080'
    }
    
    for _, traj in node_trajs.iterrows():
        x = np.array([1, 2, 3])
        y = traj['intercept'] + traj['slope'] * x
        
        ax.plot(x, y, marker='o', label=str(traj['group']),
               color=colors_map.get(traj['group'], 'gray'), linewidth=2, markersize=8)
    
    ax.set_xlabel('Session', fontsize=10)
    ax.set_ylabel('Degree', fontsize=10)
    ax.set_title(f'Node {node} - Degree Trajectory', fontsize=11, fontweight='bold')
    ax.set_xticks([1, 2, 3])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'trajectory_examples_top_nodes.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: trajectory_examples_top_nodes.png")

print()

# ======================== SUMMARY STATISTICS ========================
print("="*70)
print("NODE-LEVEL ANALYSIS SUMMARY")
print("="*70)

print(f"\nTotal nodes analyzed: {N_NODES}")
print(f"Nodes with significant intervention effects (|d|>0.5): {np.sum(effects_df['abs_effect_size'] > 0.5)}")
print(f"Nodes with moderate effects (|d|>0.2): {np.sum(effects_df['abs_effect_size'] > 0.2)}")

print("\nTop 5 Most Responsive Nodes (by effect size):")
top_5 = effects_df.nlargest(5, 'abs_effect_size')[['node', 'metric', 'effect_size_cohens_d']]
for idx, row in top_5.iterrows():
    print(f"  Node {int(row['node']):3d} - {row['metric']:20s}: d = {row['effect_size_cohens_d']:6.3f}")

print("\nMetrics with Strongest Overall Effects:")
metric_strength = effects_df.groupby('metric')['abs_effect_size'].mean().sort_values(ascending=False)
for metric, strength in metric_strength.items():
    print(f"  {metric:25s}: avg |d| = {strength:.3f}")

print("\nHub Type Distribution:")
print(node_df['hub_type'].value_counts())

print()
print("="*70)
print(f"✅ ALL OUTPUTS SAVED TO: {OUTPUT_DIR}")
print("="*70)
print("\nGenerated files:")
print("  - node_trajectories.parquet")
print("  - intervention_effect_sizes.parquet")
print("  - hub_specific_responses.parquet")
print("  - regional_effect_heatmap.png")
print("  - node_effects_[metric].png (per metric)")
print("  - hub_type_response_distributions.png")
print("  - trajectory_examples_top_nodes.png")
print()
