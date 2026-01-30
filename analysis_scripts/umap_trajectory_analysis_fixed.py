"""
UMAP + Trajectory Analysis (Fixed)
===================================

Takes the session-level metrics and performs:
1. UMAP dimensionality reduction
2. Trajectory analysis in UMAP space
3. Response phenotyping

Author: Analysis Pipeline
Date: January 2026
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# ======================== LOAD DATA ========================
BCT_DIR = Path("/data/local/129_PK01/derivatives/bct")
comprehensive_dir = BCT_DIR / "comprehensive_analysis"
comprehensive_dir.mkdir(exist_ok=True)

print("="*70)
print("UMAP + TRAJECTORY ANALYSIS (Session-Level)")
print("="*70)

# Load metrics
metrics_file = comprehensive_dir / "complete_metrics_with_graph_theory.parquet"
print(f"Loading metrics from: {metrics_file}")
df = pd.read_parquet(metrics_file)
print(f"Loaded {len(df)} session-level records from {df['subject'].nunique()} subjects")
print()

# ======================== PREPARE FEATURES ========================
print("Preparing features for UMAP...")

# Select features that have no NaN
feature_cols = [col for col in df.columns 
               if col not in ['subject', 'session', 'atlas', 'group', 'sex', 'age', 'density']]

# Check NaN values
nan_cols = df[feature_cols].columns[df[feature_cols].isna().any()]
print(f"Columns with NaN: {nan_cols.tolist()}")

# Remove NaN columns
feature_cols = [col for col in feature_cols if col not in nan_cols]
print(f"Using {len(feature_cols)} features:")
for col in feature_cols:
    print(f"  - {col}")
print()

# Remove rows with any remaining NaN in features or metadata
df_clean = df[feature_cols + ['subject', 'session', 'group', 'sex', 'age']].dropna()

print(f"After removing NaN: {len(df_clean)} sessions")
print()

# ======================== UMAP EMBEDDING ========================
print("Running UMAP dimensionality reduction...")

X = df_clean[feature_cols].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# UMAP with 3D embedding
print("Fitting UMAP (n_neighbors=15, min_dist=0.1, 3D)...")
reducer = UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=3,
    metric='euclidean',
    random_state=42,
    verbose=1,
    n_jobs=-1
)

X_umap = reducer.fit_transform(X_scaled)
print(f"✅ UMAP complete: shape {X_umap.shape}")
print()

# Create UMAP dataframe
umap_df = df_clean.copy()
umap_df['umap_1'] = X_umap[:, 0]
umap_df['umap_2'] = X_umap[:, 1]
umap_df['umap_3'] = X_umap[:, 2]

# Save UMAP results
umap_output = comprehensive_dir / "umap_embedding.parquet"
umap_df.to_parquet(umap_output, index=False)
print(f"✅ Saved: {umap_output}")
print()

# ======================== TRAJECTORY ANALYSIS ========================
print("Analyzing trajectories in UMAP space...")

trajectory_data = []

for subject in umap_df['subject'].unique():
    subject_data = umap_df[umap_df['subject'] == subject].sort_values('session')
    
    if len(subject_data) < 2:
        continue
    
    group = subject_data['group'].iloc[0]
    sex = subject_data['sex'].iloc[0]
    age = subject_data['age'].iloc[0]
    
    # Get UMAP positions for each session
    sessions = subject_data['session'].values
    umap_pos = subject_data[['umap_1', 'umap_2', 'umap_3']].values
    
    # Calculate distances between sessions
    if len(sessions) >= 2:
        dist_1_2 = np.linalg.norm(umap_pos[1] - umap_pos[0])
        
        if len(sessions) >= 3:
            dist_2_3 = np.linalg.norm(umap_pos[2] - umap_pos[1])
            total_distance = dist_1_2 + dist_2_3
            accel_ratio = dist_2_3 / dist_1_2 if dist_1_2 > 0 else 1
            
            # Classify response type
            if accel_ratio < 0.6:
                response_type = 'Decelerating'
            elif accel_ratio < 1.4:
                response_type = 'Linear'
            else:
                response_type = 'Accelerating'
        else:
            dist_2_3 = np.nan
            total_distance = dist_1_2
            accel_ratio = np.nan
            response_type = 'Unknown'
        
        trajectory_data.append({
            'subject': subject,
            'group': group,
            'sex': sex,
            'age': age,
            'early_change': dist_1_2,
            'late_change': dist_2_3,
            'total_distance': total_distance,
            'acceleration_ratio': accel_ratio,
            'response_type': response_type,
            'n_sessions': len(sessions)
        })

trajectory_df = pd.DataFrame(trajectory_data)
print(f"Analyzed {len(trajectory_df)} subjects")
print()

print("Response type distribution:")
print(trajectory_df['response_type'].value_counts())
print()

# Save trajectories
traj_output = comprehensive_dir / "trajectory_analysis.parquet"
trajectory_df.to_parquet(traj_output, index=False)
print(f"✅ Saved: {traj_output}")
print()

# ======================== VISUALIZATIONS ========================
print("Creating visualizations...")

# 1. 3D UMAP by group
fig = plt.figure(figsize=(14, 6))

colors_map = {1: '#FF6B6B', 2: '#FFA07A', 3: '#4169E1', 4: '#87CEEB', 5: '#808080'}
labels_map = {1: 'Alone 2w', 2: 'Alone 4w', 3: 'Group 2w', 4: 'Group 4w', 5: 'Control'}

# Subplot 1: By group
ax1 = fig.add_subplot(121, projection='3d')

for group in sorted(umap_df['group'].unique()):
    group_data = umap_df[umap_df['group'] == group]
    ax1.scatter(group_data['umap_1'], group_data['umap_2'], group_data['umap_3'],
               c=colors_map[group], label=labels_map[group], alpha=0.6, s=50)

ax1.set_xlabel('UMAP-1')
ax1.set_ylabel('UMAP-2')
ax1.set_zlabel('UMAP-3')
ax1.set_title('UMAP Embedding by Group', fontweight='bold', fontsize=12)
ax1.legend()

# Subplot 2: By response type
ax2 = fig.add_subplot(122, projection='3d')

response_colors = {'Accelerating': '#FF4444', 'Linear': '#FFA500', 
                  'Decelerating': '#4169E1', 'Unknown': '#808080'}

# Map response types to UMAP data
response_map = dict(zip(trajectory_df['subject'], trajectory_df['response_type']))
umap_df['response_type'] = umap_df['subject'].map(response_map)

for response_type in ['Decelerating', 'Linear', 'Accelerating', 'Unknown']:
    type_data = umap_df[umap_df['response_type'] == response_type]
    if len(type_data) > 0:
        ax2.scatter(type_data['umap_1'], type_data['umap_2'], type_data['umap_3'],
                   c=response_colors[response_type], label=response_type, alpha=0.6, s=50)

ax2.set_xlabel('UMAP-1')
ax2.set_ylabel('UMAP-2')
ax2.set_zlabel('UMAP-3')
ax2.set_title('UMAP Embedding by Response Type', fontweight='bold', fontsize=12)
ax2.legend()

plt.tight_layout()
plt.savefig(comprehensive_dir / '3d_umap_embedding.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 3d_umap_embedding.png")

# 2. 2D UMAP projections
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, (dim1, dim2) in enumerate([(0, 1), (0, 2), (1, 2)]):
    ax = axes[idx]
    
    for group in sorted(umap_df['group'].unique()):
        group_data = umap_df[umap_df['group'] == group]
        ax.scatter(group_data[f'umap_{dim1+1}'], group_data[f'umap_{dim2+1}'],
                  c=colors_map[group], label=labels_map[group], alpha=0.6, s=50)
    
    ax.set_xlabel(f'UMAP-{dim1+1}')
    ax.set_ylabel(f'UMAP-{dim2+1}')
    ax.set_title(f'UMAP Projection: Dims {dim1+1}-{dim2+1}')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(comprehensive_dir / '2d_umap_projections.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 2d_umap_projections.png")

# 3. Trajectory metrics by group
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, metric in enumerate(['early_change', 'late_change', 'total_distance', 'acceleration_ratio']):
    ax = axes[idx // 2, idx % 2]
    
    valid_data = trajectory_df[trajectory_df[metric].notna()]
    
    groups_list = sorted(valid_data['group'].unique())
    data_by_group = [valid_data[valid_data['group'] == g][metric].values for g in groups_list]
    
    bp = ax.boxplot(data_by_group, labels=[labels_map[g] for g in groups_list], 
                    patch_artist=True)
    
    for patch, group in zip(bp['boxes'], groups_list):
        patch.set_facecolor(colors_map[group])
        patch.set_alpha(0.7)
    
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_", " ").title()} by Group')
    ax.grid(alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(comprehensive_dir / 'trajectory_metrics_by_group.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: trajectory_metrics_by_group.png")

# 4. Response type distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Overall response type
ax1 = axes[0]
response_counts = trajectory_df['response_type'].value_counts()
colors = [response_colors[rt] for rt in response_counts.index]
ax1.bar(response_counts.index, response_counts.values, color=colors, alpha=0.7)
ax1.set_ylabel('Count')
ax1.set_title('Overall Response Type Distribution')
ax1.grid(alpha=0.3, axis='y')

# Response type by group
ax2 = axes[1]
response_by_group = pd.crosstab(trajectory_df['group'], trajectory_df['response_type'])
response_by_group.index = response_by_group.index.map(labels_map)
response_by_group.plot(kind='bar', ax=ax2, 
                       color=[response_colors.get(col, '#808080') for col in response_by_group.columns],
                       alpha=0.7)
ax2.set_ylabel('Count')
ax2.set_xlabel('Group')
ax2.set_title('Response Type Distribution by Group')
ax2.legend(title='Response Type', loc='best')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax2.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(comprehensive_dir / 'response_type_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: response_type_distribution.png")

print()
print("="*70)
print("UMAP + TRAJECTORY ANALYSIS COMPLETE")
print("="*70)
print()
print("Output files:")
print(f"  - umap_embedding.parquet ({len(umap_df)} sessions)")
print(f"  - trajectory_analysis.parquet ({len(trajectory_df)} subjects)")
print("  - 3d_umap_embedding.png")
print("  - 2d_umap_projections.png")
print("  - trajectory_metrics_by_group.png")
print("  - response_type_distribution.png")
print()
print(f"Location: {comprehensive_dir}")
print("="*70)
