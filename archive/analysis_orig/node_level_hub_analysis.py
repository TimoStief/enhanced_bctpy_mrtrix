"""
Node-Level Brain Network Analysis with Hub Identification
=========================================================

Computes comprehensive node-level metrics for all 246 Brainnectome regions:
- Node strength, degree, betweenness centrality
- Participation coefficient (connector hub measure)
- Within-module z-score (provincial hub measure)
- Hub classification (provincial, connector, kinless, peripheral)
- Rich-club coefficient
- Node-level temporal trajectories
- Regional intervention effects

Author: Analysis Pipeline
Date: January 2026
"""

import os
import numpy as np
import pandas as pd
import bct
from scipy import stats
from scipy.io import loadmat
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# Configuration
# -------------------------------
DATA_DIR = Path("/data/local/129_PK01/derivatives/dsistudio_connectomics/connectivity")
BCT_DIR = Path("/data/local/129_PK01/derivatives/bct")
OUTPUT_DIR = BCT_DIR / "node_level_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

# Hub classification thresholds (standard from literature)
PROVINCIAL_HUB_THRESHOLD = 2.5  # within-module z-score
CONNECTOR_HUB_THRESHOLD = 0.30   # participation coefficient
KINLESS_HUB_THRESHOLD = 0.05     # low participation

ATLAS = "Brainnectome"
N_NODES = 246

print("="*70)
print("NODE-LEVEL BRAIN NETWORK ANALYSIS")
print("="*70)
print(f"Atlas: {ATLAS} ({N_NODES} nodes)")
print(f"Output directory: {OUTPUT_DIR}")
print()

# -------------------------------
# Load Participants Metadata
# -------------------------------
print("Loading participant metadata...")
participants_file = BCT_DIR / "participants_5groups.tsv"
if not participants_file.exists():
    raise FileNotFoundError(f"Participants file not found: {participants_file}")

participants_df = pd.read_csv(participants_file, sep='\t')
print(f"Loaded {len(participants_df)} participant records")
print(f"Groups: {participants_df['group'].value_counts().to_dict()}")
print()

# -------------------------------
# Helper Functions
# -------------------------------

def load_connectivity_matrix(subject, session, atlas=ATLAS):
    """Load connectivity matrix for a subject/session/atlas
    
    Args:
        subject: Subject ID (e.g., 'sub-1291003' or '1291003')
        session: Session number (1, 2, 3)
        atlas: Atlas name
    """
    # Ensure subject has 'sub-' prefix
    if not subject.startswith('sub-'):
        subject = f'sub-{subject}'
    
    # DSI Studio path structure: sub-{subject}_ses-{session}.../*.mat
    pattern = f"{subject}_ses-{session}*/tracks_1000k_streamline/by_atlas/{atlas}/*.connectivity.mat"
    matches = list(DATA_DIR.glob(pattern))
    
    if not matches:
        return None
    
    # Use the first match
    matrix_path = matches[0]
    try:
        mat_data = loadmat(matrix_path)
        # DSI Studio saves matrix as 'connectivity'
        if 'connectivity' in mat_data:
            A = mat_data['connectivity']
        else:
            # Try other possible keys
            keys = [k for k in mat_data.keys() if not k.startswith('__')]
            if keys:
                A = mat_data[keys[0]]
            else:
                return None
        
        A = np.array(A, dtype=float)
        
        if A.shape[0] != N_NODES or A.shape[1] != N_NODES:
            print(f"Warning: Matrix shape mismatch for {subject} ses-{session}: {A.shape}")
            return None
        return A
    except Exception as e:
        print(f"Error loading matrix for {subject} ses-{session}: {e}")
        return None


def compute_node_metrics(A):
    """
    Compute comprehensive node-level metrics
    
    Returns dict with:
    - degree: node degree
    - strength: node strength (weighted degree)
    - betweenness: betweenness centrality
    - clustering: clustering coefficient
    - local_efficiency: local efficiency
    - participation_coef: participation coefficient
    - within_module_zscore: within-module degree z-score
    - hub_type: hub classification
    """
    metrics = {}
    
    # Ensure float type
    A = np.array(A, dtype=float)
    A_bin = (A > 0).astype(int)
    
    # Degree and strength
    metrics['degree'] = bct.degrees_und(A_bin)
    metrics['strength'] = bct.strengths_und(A)
    
    # Clustering coefficient
    try:
        metrics['clustering'] = bct.clustering_coef_wu(A)
    except:
        metrics['clustering'] = np.full(N_NODES, np.nan)
    
    # Local efficiency
    try:
        metrics['local_efficiency'] = bct.efficiency_wei(A, local=True)
    except:
        metrics['local_efficiency'] = np.full(N_NODES, np.nan)
    
    # Betweenness centrality (computationally expensive)
    try:
        metrics['betweenness'] = bct.betweenness_wei(A)
    except:
        metrics['betweenness'] = np.full(N_NODES, np.nan)
    
    # Community detection for hub metrics
    try:
        Ci, Q = bct.community_louvain(A)
        metrics['community'] = Ci
        metrics['modularity'] = Q
        
        # Participation coefficient (connector hub measure)
        metrics['participation_coef'] = bct.participation_coef(A, Ci)
        
        # Within-module degree z-score (provincial hub measure)
        metrics['within_module_zscore'] = bct.module_degree_zscore(A, Ci)
        
        # Hub classification
        metrics['hub_type'] = classify_hubs(
            metrics['participation_coef'],
            metrics['within_module_zscore']
        )
        
    except Exception as e:
        print(f"  Warning: Community detection failed: {e}")
        metrics['community'] = np.full(N_NODES, np.nan)
        metrics['modularity'] = np.nan
        metrics['participation_coef'] = np.full(N_NODES, np.nan)
        metrics['within_module_zscore'] = np.full(N_NODES, np.nan)
        metrics['hub_type'] = np.full(N_NODES, 'unknown')
    
    return metrics


def classify_hubs(participation_coef, within_module_zscore):
    """
    Classify nodes into hub types based on Guimerà & Amaral (2005)
    
    Hub types:
    - provincial_hub: High within-module z-score, low participation
    - connector_hub: High participation, moderate within-module z-score
    - kinless_node: Low on both (non-hub connector)
    - peripheral: Low on both (standard node)
    """
    hub_types = []
    
    for pc, wz in zip(participation_coef, within_module_zscore):
        if np.isnan(pc) or np.isnan(wz):
            hub_types.append('unknown')
        elif wz > PROVINCIAL_HUB_THRESHOLD and pc < CONNECTOR_HUB_THRESHOLD:
            hub_types.append('provincial_hub')
        elif pc > CONNECTOR_HUB_THRESHOLD and wz > 1.0:
            hub_types.append('connector_hub')
        elif pc < KINLESS_HUB_THRESHOLD and wz < 1.0:
            hub_types.append('peripheral')
        else:
            hub_types.append('kinless_node')
    
    return np.array(hub_types)


def compute_rich_club(A):
    """Compute rich-club coefficient across degree thresholds"""
    try:
        A_bin = (A > 0).astype(int)
        rc = bct.rich_club_bu(A_bin)
        return rc
    except:
        return None


# -------------------------------
# Main Analysis Loop
# -------------------------------
print("Computing node-level metrics for all subjects/sessions...")
print()

all_node_data = []
subject_summaries = []

# Get unique subjects
unique_subjects = participants_df['participant_id'].unique()

for idx, subject in enumerate(unique_subjects):
    if idx % 10 == 0:
        print(f"Progress: {idx}/{len(unique_subjects)} subjects processed...")
    
    # Get subject metadata
    subj_meta = participants_df[participants_df['participant_id'] == subject].iloc[0]
    
    # Process each session
    for session in [1, 2, 3]:
        A = load_connectivity_matrix(subject, session)
        
        if A is None:
            continue
        
        # Compute node metrics
        metrics = compute_node_metrics(A)
        
        # Compute rich club (once per matrix, not per node)
        rich_club = compute_rich_club(A)
        
        # Store node-level data
        for node_idx in range(N_NODES):
            node_data = {
                'subject': subject,
                'session': session,
                'atlas': ATLAS,
                'node': node_idx + 1,  # 1-indexed
                'group': subj_meta.get('group', 'unknown'),
                'sex': subj_meta.get('sex', 'unknown'),
                'age': subj_meta.get('age', np.nan),
                'degree': metrics['degree'][node_idx],
                'strength': metrics['strength'][node_idx],
                'betweenness': metrics['betweenness'][node_idx],
                'clustering': metrics['clustering'][node_idx],
                'local_efficiency': metrics['local_efficiency'][node_idx],
                'participation_coef': metrics['participation_coef'][node_idx],
                'within_module_zscore': metrics['within_module_zscore'][node_idx],
                'community': metrics['community'][node_idx],
                'hub_type': metrics['hub_type'][node_idx]
            }
            all_node_data.append(node_data)
        
        # Store subject-level summary
        summary = {
            'subject': subject,
            'session': session,
            'atlas': ATLAS,
            'group': subj_meta.get('group', 'unknown'),
            'sex': subj_meta.get('sex', 'unknown'),
            'age': subj_meta.get('age', np.nan),
            'modularity': metrics['modularity'],
            'n_communities': len(np.unique(metrics['community'][~np.isnan(metrics['community'])])),
            'n_provincial_hubs': np.sum(metrics['hub_type'] == 'provincial_hub'),
            'n_connector_hubs': np.sum(metrics['hub_type'] == 'connector_hub'),
            'n_kinless_nodes': np.sum(metrics['hub_type'] == 'kinless_node'),
            'n_peripheral': np.sum(metrics['hub_type'] == 'peripheral'),
            'mean_participation': np.nanmean(metrics['participation_coef']),
            'mean_within_module_z': np.nanmean(metrics['within_module_zscore']),
            'mean_betweenness': np.nanmean(metrics['betweenness']),
            'mean_strength': np.nanmean(metrics['strength'])
        }
        
        if rich_club is not None:
            summary['rich_club_coef'] = np.nanmean(rich_club) if len(rich_club) > 0 else np.nan
        else:
            summary['rich_club_coef'] = np.nan
            
        subject_summaries.append(summary)

print(f"\nCompleted: {len(unique_subjects)} subjects processed")
print(f"Total node records: {len(all_node_data)}")
print(f"Total subject-session records: {len(subject_summaries)}")
print()

# -------------------------------
# Save Results
# -------------------------------
print("Saving node-level data...")

# Convert to DataFrames
node_df = pd.DataFrame(all_node_data)
summary_df = pd.DataFrame(subject_summaries)

# Save as parquet (efficient compression)
node_output = OUTPUT_DIR / "node_level_metrics.parquet"
summary_output = OUTPUT_DIR / "subject_hub_summaries.parquet"

node_df.to_parquet(node_output, index=False)
summary_df.to_parquet(summary_output, index=False)

print(f"✅ Node-level metrics saved: {node_output}")
print(f"✅ Subject summaries saved: {summary_output}")
print()

# -------------------------------
# Hub Analysis Summary
# -------------------------------
if len(node_df) == 0:
    print("ERROR: No data loaded! Check connectivity matrix paths.")
    print("\nDebugging: Testing file loading for first subject...")
    test_subject = unique_subjects[0] if len(unique_subjects) > 0 else "sub-1291003"
    test_path = list(DATA_DIR.glob(f"{test_subject}_ses-1*/tracks_1000k_streamline/by_atlas/{ATLAS}/*.connectivity.mat"))
    print(f"  Test pattern: {test_subject}_ses-1*/tracks_1000k_streamline/by_atlas/{ATLAS}/*.connectivity.mat")
    print(f"  Matches found: {len(test_path)}")
    if test_path:
        print(f"  Example file: {test_path[0]}")
    exit(1)

print("="*70)
print("HUB ANALYSIS SUMMARY")
print("="*70)

# Count hub types across all sessions
hub_counts = node_df['hub_type'].value_counts()
print("\nHub Type Distribution (all nodes, all sessions):")
for hub_type, count in hub_counts.items():
    pct = 100 * count / len(node_df)
    print(f"  {hub_type}: {count} ({pct:.1f}%)")

# Average hubs per subject
print("\nAverage Hub Counts per Subject-Session:")
print(f"  Provincial hubs: {summary_df['n_provincial_hubs'].mean():.1f} ± {summary_df['n_provincial_hubs'].std():.1f}")
print(f"  Connector hubs: {summary_df['n_connector_hubs'].mean():.1f} ± {summary_df['n_connector_hubs'].std():.1f}")
print(f"  Kinless nodes: {summary_df['n_kinless_nodes'].mean():.1f} ± {summary_df['n_kinless_nodes'].std():.1f}")
print(f"  Peripheral nodes: {summary_df['n_peripheral'].mean():.1f} ± {summary_df['n_peripheral'].std():.1f}")

# Group differences in hub counts
print("\nHub Counts by Intervention Group:")
for group in summary_df['group'].unique():
    if group == 'unknown':
        continue
    group_data = summary_df[summary_df['group'] == group]
    print(f"\n  {group} (n={len(group_data)}):")
    print(f"    Provincial: {group_data['n_provincial_hubs'].mean():.1f} ± {group_data['n_provincial_hubs'].std():.1f}")
    print(f"    Connector: {group_data['n_connector_hubs'].mean():.1f} ± {group_data['n_connector_hubs'].std():.1f}")

print()
print("="*70)
print("NEXT STEPS:")
print("="*70)
print("1. Run temporal trajectory analysis on node-level metrics")
print("2. Identify which specific nodes/hubs show intervention effects")
print("3. Create spatial maps of connectivity change")
print("4. Test hub-specific intervention effects (provincial vs connector)")
print("5. Examine rich-club reorganization across timepoints")
print()
print("Output files ready for visualization and statistical testing!")
print("="*70)
