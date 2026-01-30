"""
Complete Analysis Pipeline: Graph Metrics + UMAP Trajectories
==============================================================

Implements Path C:
1. Additional graph metrics (path length, global efficiency, small-worldness)
2. UMAP dimensionality reduction 
3. Trajectory analysis in UMAP space
4. Responder classification

Can be run with: nohup python this_script.py > analysis.log 2>&1 &

Author: Analysis Pipeline
Date: January 2026
"""

import os
import numpy as np
import pandas as pd
import bct
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Disable GPU if using for reproducibility
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for remote server
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from umap import UMAP
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ======================== CONFIGURATION ========================
DATA_DIR = Path("/data/local/129_PK01/derivatives/dsistudio_connectomics/connectivity")
BCT_DIR = Path("/data/local/129_PK01/derivatives/bct")
OUTPUT_DIR = BCT_DIR / "comprehensive_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

ATLAS = "Brainnectome"
N_NODES = 246

print("="*70)
print("COMPREHENSIVE ANALYSIS: METRICS + UMAP + TRAJECTORIES")
print("="*70)
print(f"Output directory: {OUTPUT_DIR}")
print()

# ======================== LOAD DATA ========================
print("Loading participant metadata...")
participants_file = BCT_DIR / "participants_5groups.tsv"
participants_df = pd.read_csv(participants_file, sep='\t')
print(f"Loaded {len(participants_df)} participant records")
print()

# ======================== HELPER FUNCTIONS ========================

def load_connectivity_matrix(subject, session, atlas=ATLAS):
    """Load connectivity matrix from DSI Studio .mat file"""
    if not subject.startswith('sub-'):
        subject = f'sub-{subject}'
    
    pattern = f"{subject}_ses-{session}*/tracks_1000k_streamline/by_atlas/{atlas}/*.connectivity.mat"
    matches = list(DATA_DIR.glob(pattern))
    
    if not matches:
        return None
    
    try:
        mat_data = loadmat(matches[0])
        if 'connectivity' in mat_data:
            A = mat_data['connectivity']
        else:
            keys = [k for k in mat_data.keys() if not k.startswith('__')]
            A = mat_data[keys[0]] if keys else None
        
        if A is None:
            return None
            
        A = np.array(A, dtype=float)
        
        if A.shape[0] != N_NODES or A.shape[1] != N_NODES:
            return None
        return A
    except Exception as e:
        return None


def compute_small_worldness(A):
    """
    Compute small-worldness coefficient
    σ = (C/C_random) / (L/L_random)
    where C = clustering coef, L = characteristic path length
    """
    try:
        A_bin = (A > 0).astype(int)
        
        # Real network metrics
        clustering = bct.clustering_coef_bu(A_bin)
        C_real = np.mean(clustering)
        
        # Path length
        D = bct.distance_bin(A_bin)
        if np.all(np.isfinite(D)):
            C_real_path, charpath_real = bct.charpath(D)
        else:
            return np.nan
        
        # Null model (random network with same degree distribution)
        # Use 10 iterations
        C_random_list = []
        L_random_list = []
        
        for _ in range(10):
            A_rand = bct.null_model_und_degree(A_bin)
            c_rand = np.mean(bct.clustering_coef_bu(A_rand))
            D_rand = bct.distance_bin(A_rand)
            
            if np.all(np.isfinite(D_rand)):
                _, l_rand = bct.charpath(D_rand)
                C_random_list.append(c_rand)
                L_random_list.append(l_rand)
        
        if len(C_random_list) == 0:
            return np.nan
            
        C_random = np.mean(C_random_list)
        L_random = np.mean(L_random_list)
        
        # Small-worldness
        if C_random > 0 and L_random > 0:
            sigma = (C_real / C_random) / (charpath_real / L_random)
            return sigma
        else:
            return np.nan
            
    except Exception as e:
        return np.nan


def compute_global_efficiency(A):
    """Global efficiency (BCT function wrapper)"""
    try:
        eff = bct.efficiency_wei(A)
        return eff
    except:
        return np.nan


def compute_characteristic_path_length(A):
    """Characteristic path length from weighted network"""
    try:
        A_bin = (A > 0).astype(int)
        D = bct.distance_bin(A_bin)
        
        if not np.all(np.isfinite(D)):
            return np.nan
            
        C, L = bct.charpath(D)
        return L
    except:
        return np.nan


# ======================== MAIN ANALYSIS LOOP ========================
print("Computing connectivity metrics for all subjects/sessions...")
print()

all_data = []
unique_subjects = participants_df['participant_id'].unique()

for idx, subject in enumerate(unique_subjects):
    if idx % 20 == 0:
        print(f"Progress: {idx}/{len(unique_subjects)} subjects...")
    
    subj_meta = participants_df[participants_df['participant_id'] == subject].iloc[0]
    
    for session in [1, 2, 3]:
        A = load_connectivity_matrix(subject, session)
        
        if A is None:
            continue
        
        # ===== EXISTING METRICS (aggregate) =====
        A_bin = (A > 0).astype(int)
        
        metrics = {
            'subject': subject,
            'session': session,
            'atlas': ATLAS,
            'group': subj_meta.get('group', 'unknown'),
            'sex': subj_meta.get('sex', 'unknown'),
            'age': subj_meta.get('age', np.nan),
        }
        
        # Degree and strength
        try:
            degree_vec = bct.degrees_und(A_bin)
            metrics['degree_mean'] = np.mean(degree_vec)
            metrics['degree_std'] = np.std(degree_vec)
            metrics['degree_max'] = np.max(degree_vec)
            
            strength_vec = bct.strengths_und(A)
            metrics['strength_mean'] = np.mean(strength_vec)
            metrics['strength_std'] = np.std(strength_vec)
            metrics['strength_max'] = np.max(strength_vec)
        except:
            pass
        
        # Clustering
        try:
            clustering = bct.clustering_coef_wu(A)
            metrics['clustering_mean'] = np.mean(clustering)
            metrics['clustering_std'] = np.std(clustering)
        except:
            pass
        
        # Modularity
        try:
            Ci, Q = bct.community_louvain(A_bin)
            metrics['modularity'] = Q
            metrics['n_communities'] = len(np.unique(Ci[~np.isnan(Ci)]))
        except:
            metrics['modularity'] = np.nan
            metrics['n_communities'] = np.nan
        
        # Participation coefficient
        try:
            pc = bct.participation_coef(A, Ci)
            metrics['participation_coef_mean'] = np.mean(pc)
            metrics['participation_coef_std'] = np.std(pc)
        except:
            pass
        
        # Density
        try:
            metrics['density'] = bct.density_und(A_bin)
        except:
            pass
        
        # ===== NEW METRICS (3 graph theory measures) =====
        
        # 1. Characteristic Path Length
        charpath = compute_characteristic_path_length(A)
        metrics['characteristic_path_length'] = charpath
        
        # 2. Global Efficiency
        global_eff = compute_global_efficiency(A)
        metrics['global_efficiency'] = global_eff
        
        # 3. Small-Worldness (requires null model - computationally expensive)
        sigma = compute_small_worldness(A)
        metrics['small_worldness'] = sigma
        
        # Local efficiency
        try:
            local_eff = bct.efficiency_wei(A, local=True)
            metrics['local_efficiency_mean'] = np.mean(local_eff)
        except:
            pass
        
        # Betweenness centrality
        try:
            betweenness = bct.betweenness_wei(A)
            metrics['betweenness_mean'] = np.mean(betweenness)
        except:
            pass
        
        all_data.append(metrics)

print(f"Completed: {len(unique_subjects)} subjects")
print(f"Total records: {len(all_data)}")
print()

# ======================== CREATE ANALYSIS DATAFRAME ========================
print("Creating analysis dataframe...")
analysis_df = pd.DataFrame(all_data)

# Save full metrics
metrics_output = OUTPUT_DIR / "complete_metrics_with_graph_theory.parquet"
analysis_df.to_parquet(metrics_output, index=False)
print(f"✅ Metrics saved: {metrics_output}")
print()

# ======================== UMAP ANALYSIS ========================
print("Running UMAP dimensionality reduction...")

# Select features for UMAP (standardize column names)
feature_cols = [
    'degree_mean', 'degree_std', 'strength_mean', 'strength_std',
    'clustering_mean', 'modularity', 'participation_coef_mean',
    'density', 'characteristic_path_length', 'global_efficiency', 
    'small_worldness', 'local_efficiency_mean', 'betweenness_mean'
]

# Filter to available columns
available_cols = [c for c in feature_cols if c in analysis_df.columns]
print(f"Using {len(available_cols)} features for UMAP:")
for col in available_cols:
    print(f"  - {col}")

# Remove NaN rows
umap_data = analysis_df[available_cols].copy()
valid_idx = ~umap_data.isnull().any(axis=1)
umap_data = umap_data[valid_idx].values
umap_df_full = analysis_df[valid_idx].copy()

print(f"Valid samples for UMAP: {len(umap_data)}")
print()

# Standardize
scaler = StandardScaler()
features_scaled = scaler.fit_transform(umap_data)

# UMAP with good parameters for your data
print("Fitting UMAP (n_neighbors=15, min_dist=0.1)...")
reducer = UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=3,
    metric='euclidean',
    random_state=42,
    verbose=1
)

embedding = reducer.fit_transform(features_scaled)
print("✅ UMAP complete")
print()

# Create UMAP dataframe
umap_df = umap_df_full.copy()
umap_df['umap_1'] = embedding[:, 0]
umap_df['umap_2'] = embedding[:, 1]
umap_df['umap_3'] = embedding[:, 2]

# Save UMAP results
umap_output = OUTPUT_DIR / "umap_embedding.parquet"
umap_df.to_parquet(umap_output, index=False)
print(f"✅ UMAP embedding saved: {umap_output}")
print()

# ======================== TRAJECTORY ANALYSIS ========================
print("Analyzing trajectories in UMAP space...")

trajectory_metrics = []

for subject in umap_df['subject'].unique():
    subj_data = umap_df[umap_df['subject'] == subject].sort_values('session')
    
    if len(subj_data) < 3:
        continue
    
    # Get UMAP positions for each session
    sessions = []
    positions = []
    for _, row in subj_data.iterrows():
        sessions.append(row['session'])
        positions.append([row['umap_1'], row['umap_2'], row['umap_3']])
    
    if len(sessions) != 3:
        continue
    
    s1 = np.array(positions[0])
    s2 = np.array(positions[1])
    s3 = np.array(positions[2])
    
    # Calculate distances
    dist_1_2 = np.linalg.norm(s2 - s1)
    dist_2_3 = np.linalg.norm(s3 - s2)
    total_distance = dist_1_2 + dist_2_3
    
    # Acceleration metric
    if dist_1_2 > 0.01:
        acceleration = dist_2_3 / dist_1_2
    else:
        acceleration = np.nan
    
    # Trajectory direction (vector from session 1 to 3)
    trajectory_vec = s3 - s1
    trajectory_angle = np.linalg.norm(trajectory_vec)
    
    trajectory_metrics.append({
        'subject': subject,
        'group': subj_data.iloc[0]['group'],
        'sex': subj_data.iloc[0]['sex'],
        'age': subj_data.iloc[0]['age'],
        'early_change': dist_1_2,      # Session 1→2
        'late_change': dist_2_3,       # Session 2→3
        'total_distance': total_distance,
        'acceleration': acceleration,   # late/early ratio
        'final_position_distance': trajectory_angle
    })

trajectory_df = pd.DataFrame(trajectory_metrics)

# Classify response types
trajectory_df['response_type'] = pd.cut(
    trajectory_df['acceleration'], 
    bins=[0, 0.6, 1.4, np.inf], 
    labels=['Decelerating', 'Linear', 'Accelerating'],
    include_lowest=True
)

trajectory_df['response_magnitude'] = pd.cut(
    trajectory_df['total_distance'],
    bins=[0, trajectory_df['total_distance'].quantile(0.33), 
          trajectory_df['total_distance'].quantile(0.67), np.inf],
    labels=['Low', 'Medium', 'High'],
    include_lowest=True
)

# Save trajectories
trajectory_output = OUTPUT_DIR / "trajectory_analysis.parquet"
trajectory_df.to_parquet(trajectory_output, index=False)
print(f"✅ Trajectory analysis saved: {trajectory_output}")
print()

# ======================== SUMMARY STATISTICS ========================
print("="*70)
print("TRAJECTORY SUMMARY")
print("="*70)
print("\nResponse Type Distribution:")
print(trajectory_df['response_type'].value_counts())

print("\nResponse Type by Intervention Group:")
print(pd.crosstab(trajectory_df['group'], trajectory_df['response_type'], margins=True))

print("\nResponse Magnitude by Intervention Group:")
print(pd.crosstab(trajectory_df['group'], trajectory_df['response_magnitude'], margins=True))

print("\nMean Trajectory Metrics by Group:")
group_stats = trajectory_df.groupby('group')[
    ['early_change', 'late_change', 'total_distance', 'acceleration']
].mean()
print(group_stats)

print("\nMean Trajectory Metrics by Sex:")
sex_stats = trajectory_df.groupby('sex')[
    ['early_change', 'late_change', 'total_distance', 'acceleration']
].mean()
print(sex_stats)

print()

# ======================== VISUALIZATIONS ========================
print("Creating visualizations...")

# 1. 3D UMAP colored by group
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

colors_map = {
    'alone_2w': '#FF6B6B', 'alone_4w': '#FFA07A',
    'group_2w': '#4169E1', 'group_4w': '#87CEEB',
    'control': '#808080',
    1: '#FF6B6B', 2: '#FFA07A',
    3: '#4169E1', 4: '#87CEEB',
    5: '#808080'
}

for group in umap_df['group'].unique():
    mask = umap_df['group'] == group
    ax.scatter(
        umap_df[mask]['umap_1'],
        umap_df[mask]['umap_2'],
        umap_df[mask]['umap_3'],
        label=str(group),
        s=80,
        alpha=0.6,
        color=colors_map.get(group, 'gray')
    )

ax.set_xlabel('UMAP 1', fontsize=12)
ax.set_ylabel('UMAP 2', fontsize=12)
ax.set_zlabel('UMAP 3', fontsize=12)
ax.legend(loc='upper right', fontsize=10)
ax.set_title('Brain Network Topology - Intervention Response Space\n(colored by group)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'umap_3d_by_group.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ UMAP 3D by group saved")

# 2. 3D UMAP colored by response type
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

response_colors = {
    'Decelerating': '#FF6B6B',
    'Linear': '#FFD700',
    'Accelerating': '#4169E1'
}

for rtype in trajectory_df['response_type'].unique():
    # Get subjects with this response type
    subjs_with_type = trajectory_df[trajectory_df['response_type'] == rtype]['subject'].values
    mask = umap_df['subject'].isin(subjs_with_type)
    
    ax.scatter(
        umap_df[mask]['umap_1'],
        umap_df[mask]['umap_2'],
        umap_df[mask]['umap_3'],
        label=rtype,
        s=80,
        alpha=0.6,
        color=response_colors.get(rtype, 'gray')
    )

ax.set_xlabel('UMAP 1', fontsize=12)
ax.set_ylabel('UMAP 2', fontsize=12)
ax.set_zlabel('UMAP 3', fontsize=12)
ax.legend(loc='upper right', fontsize=10)
ax.set_title('Brain Network Topology - Colored by Response Type\n(Decelerating, Linear, Accelerating)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'umap_3d_by_response_type.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ UMAP 3D by response type saved")

# 3. 2D projections
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel A: By group
for group in umap_df['group'].unique():
    mask = umap_df['group'] == group
    axes[0].scatter(umap_df[mask]['umap_1'], umap_df[mask]['umap_2'],
                   label=str(group), s=60, alpha=0.6,
                   color=colors_map.get(group, 'gray'))

axes[0].set_xlabel('UMAP 1', fontsize=11)
axes[0].set_ylabel('UMAP 2', fontsize=11)
axes[0].set_title('UMAP projection - By Group', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# Panel B: By response type
for rtype in trajectory_df['response_type'].unique():
    subjs_with_type = trajectory_df[trajectory_df['response_type'] == rtype]['subject'].values
    mask = umap_df['subject'].isin(subjs_with_type)
    axes[1].scatter(umap_df[mask]['umap_1'], umap_df[mask]['umap_2'],
                   label=rtype, s=60, alpha=0.6,
                   color=response_colors.get(rtype, 'gray'))

axes[1].set_xlabel('UMAP 1', fontsize=11)
axes[1].set_ylabel('UMAP 2', fontsize=11)
axes[1].set_title('UMAP projection - By Response Type', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'umap_2d_projections.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ UMAP 2D projections saved")

# 4. Trajectory scatter plot
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Early change vs late change
for group in trajectory_df['group'].unique():
    mask = trajectory_df['group'] == group
    axes[0, 0].scatter(trajectory_df[mask]['early_change'],
                       trajectory_df[mask]['late_change'],
                       label=str(group), s=80, alpha=0.6,
                       color=colors_map.get(group, 'gray'))

axes[0, 0].set_xlabel('Early Change (Session 1→2)', fontsize=11)
axes[0, 0].set_ylabel('Late Change (Session 2→3)', fontsize=11)
axes[0, 0].set_title('Trajectory Pattern: Early vs Late Response', fontsize=12, fontweight='bold')
axes[0, 0].axline((0, 0), slope=1, color='red', linestyle='--', alpha=0.5, label='Equal')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(alpha=0.3)

# Acceleration by group
trajectory_df.boxplot(column='acceleration', by='group', ax=axes[0, 1])
axes[0, 1].set_xlabel('Group', fontsize=11)
axes[0, 1].set_ylabel('Acceleration (Late/Early)', fontsize=11)
axes[0, 1].set_title('Acceleration by Group', fontsize=12, fontweight='bold')
plt.sca(axes[0, 1])
plt.xticks(rotation=45)

# Total distance by group
trajectory_df.boxplot(column='total_distance', by='group', ax=axes[1, 0])
axes[1, 0].set_xlabel('Group', fontsize=11)
axes[1, 0].set_ylabel('Total Distance (UMAP units)', fontsize=11)
axes[1, 0].set_title('Total Response Magnitude by Group', fontsize=12, fontweight='bold')
plt.sca(axes[1, 0])
plt.xticks(rotation=45)

# Response type distribution
response_counts = trajectory_df.groupby(['group', 'response_type']).size().unstack(fill_value=0)
response_counts.plot(kind='bar', ax=axes[1, 1], color=['#FF6B6B', '#4169E1', '#FFD700'])
axes[1, 1].set_xlabel('Group', fontsize=11)
axes[1, 1].set_ylabel('Count', fontsize=11)
axes[1, 1].set_title('Response Type Distribution by Group', fontsize=12, fontweight='bold')
axes[1, 1].legend(title='Response Type', fontsize=10)
plt.sca(axes[1, 1])
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'trajectory_analysis_plots.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Trajectory analysis plots saved")

print()
print("="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print(f"\nOutput files saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
print(f"  - complete_metrics_with_graph_theory.parquet")
print(f"  - umap_embedding.parquet")
print(f"  - trajectory_analysis.parquet")
print(f"  - umap_3d_by_group.png")
print(f"  - umap_3d_by_response_type.png")
print(f"  - umap_2d_projections.png")
print(f"  - trajectory_analysis_plots.png")
print()
print("="*70)
