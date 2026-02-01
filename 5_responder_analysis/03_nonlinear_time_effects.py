#!/usr/bin/env python3
"""
Non-Linear Time Effects Analysis
==================================
Tests polynomial fits (linear, quadratic, cubic) for temporal trajectories.
Identifies trajectory shapes (accelerating, decelerating, plateauing).
Uses polynomial features for improved classification.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Paths
BCT_DIR = Path('/data/local/129_PK01/derivatives/bct')
BCT_RESULTS = BCT_DIR / 'bct_analysis_results.parquet'
PARTICIPANTS = BCT_DIR / 'participants_5groups.tsv'
OUTPUT_DIR = BCT_DIR


def load_data():
    """Load BCT results and participant metadata."""
    print("Loading data...")
    bct_df = pd.read_parquet(BCT_RESULTS)
    participants_df = pd.read_csv(PARTICIPANTS, sep='\t')
    
    bct_df = bct_df.rename(columns={'subject': 'participant_id', 'session': 'session_id'})
    df = bct_df.merge(participants_df, on=['participant_id', 'session_id'], how='inner')
    
    print(f"Loaded {len(df)} rows, {len(df['participant_id'].unique())} subjects")
    return df


def fit_polynomial_trajectories(df):
    """
    Fit linear, quadratic, and cubic polynomials to each metric trajectory.
    Returns R² for each fit + polynomial coefficients.
    """
    print("\nFitting polynomial trajectories...")
    
    # Get numeric columns
    exclude_cols = ['participant_id', 'session_id', 'atlas', 'age', 'sex', 'group', 'nr_sessions',
                    'dsi_metric', 'matrix_type', 'qc_passed', 'qc_warnings']
    metric_cols = [col for col in df.columns if col not in exclude_cols]
    metric_cols = [col for col in metric_cols if pd.api.types.is_numeric_dtype(df[col])]
    
    data_list = []
    
    for subject in df['participant_id'].unique():
        subj_data = df[df['participant_id'] == subject].copy()
        
        if len(subj_data) < 2:
            continue
        
        # Metadata
        age = subj_data['age'].iloc[0]
        sex = subj_data['sex'].iloc[0]
        group = subj_data['group'].iloc[0]
        
        # Brainnectome only
        subj_bn = subj_data[subj_data['atlas'] == 'Brainnectome'].sort_values('session_id')
        
        if len(subj_bn) < 2:
            continue
        
        sessions = subj_bn['session_id'].str.extract(r'ses-(\d+)')[0].astype(int).values
        
        row = {
            'participant_id': subject,
            'age': age,
            'sex': sex,
            'group': group,
            'n_sessions': len(subj_bn)
        }
        
        # Fit polynomials for each metric
        for metric in metric_cols:
            values = subj_bn[metric].values
            if hasattr(values, 'to_numpy'):
                values = values.to_numpy()
            values = np.asarray(values, dtype=float)
            
            if len(values) < 2 or np.any(np.isnan(values)):
                continue
            
            # Linear fit (degree 1)
            try:
                p_linear = np.polyfit(sessions, values, 1)
                r2_linear = np.corrcoef(sessions, values)[0, 1] ** 2
            except:
                continue
            
            # Quadratic fit (degree 2)
            try:
                p_quad = np.polyfit(sessions, values, 2)
                y_pred = np.polyval(p_quad, sessions)
                ss_res = np.sum((values - y_pred) ** 2)
                ss_tot = np.sum((values - np.mean(values)) ** 2)
                r2_quad = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            except:
                r2_quad = r2_linear
            
            # Cubic fit (degree 3)
            try:
                p_cubic = np.polyfit(sessions, values, 3)
                y_pred = np.polyval(p_cubic, sessions)
                ss_res = np.sum((values - y_pred) ** 2)
                ss_tot = np.sum((values - np.mean(values)) ** 2)
                r2_cubic = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            except:
                r2_cubic = r2_quad
            
            # Store results
            row[f'{metric}_linear'] = p_linear[0]  # slope
            row[f'{metric}_quadratic'] = p_quad[0]  # quadratic coeff
            row[f'{metric}_cubic'] = p_cubic[0]      # cubic coeff
            row[f'{metric}_r2_linear'] = r2_linear
            row[f'{metric}_r2_quad'] = r2_quad
            row[f'{metric}_r2_cubic'] = r2_cubic
            row[f'{metric}_nonlinear_benefit'] = max(r2_quad, r2_cubic) - r2_linear
        
        data_list.append(row)
    
    df_metrics = pd.DataFrame(data_list)
    print(f"Calculated polynomial fits for {len(df_metrics)} subjects")
    
    return df_metrics


def classify_trajectory_shapes(df_metrics):
    """Classify trajectory shapes as accelerating, decelerating, linear, or plateauing."""
    print("\nClassifying trajectory shapes...")
    
    # Get quadratic and cubic coefficients
    quad_cols = [col for col in df_metrics.columns if col.endswith('_quadratic')]
    cubic_cols = [col for col in df_metrics.columns if col.endswith('_cubic')]
    
    trajectory_types = {}
    
    for metric_base in set([col.replace('_quadratic', '').replace('_cubic', '') 
                            for col in quad_cols + cubic_cols]):
        quad_col = f'{metric_base}_quadratic'
        cubic_col = f'{metric_base}_cubic'
        
        if quad_col not in df_metrics.columns:
            continue
        
        traj_type = []
        
        for idx, row in df_metrics.iterrows():
            quad = row[quad_col]
            cubic = row[cubic_col]
            
            if np.isnan(quad) or np.isnan(cubic):
                traj_type.append('unknown')
                continue
            
            # Classify based on coefficients
            if abs(cubic) > abs(quad):  # Cubic dominates
                if cubic > 0:
                    traj_type.append('accelerating_strong')
                else:
                    traj_type.append('decelerating_strong')
            elif quad > 0.01:  # Positive quadratic = accelerating
                traj_type.append('accelerating')
            elif quad < -0.01:  # Negative quadratic = decelerating
                traj_type.append('decelerating')
            else:  # Near-zero = linear
                traj_type.append('linear')
        
        trajectory_types[metric_base] = traj_type
    
    # Overall trajectory type per subject (majority vote)
    overall_traj = []
    for idx, row in df_metrics.iterrows():
        trajectories = [traj_type[idx] for traj_type in trajectory_types.values()]
        trajectories = [t for t in trajectories if t != 'unknown']
        
        if not trajectories:
            overall_traj.append('unknown')
        else:
            from collections import Counter
            overall_traj.append(Counter(trajectories).most_common(1)[0][0])
    
    df_metrics['overall_trajectory_type'] = overall_traj
    
    return df_metrics, trajectory_types


def analyze_trajectory_patterns(df_metrics, trajectory_types):
    """Analyze non-linearity across groups."""
    print(f"\n{'='*70}")
    print("NON-LINEAR TIME EFFECTS ANALYSIS")
    print('='*70)
    
    # Overall non-linearity
    nonlinear_benefit = df_metrics[[col for col in df_metrics.columns 
                                     if col.endswith('_nonlinear_benefit')]].values
    nonlinear_benefit = nonlinear_benefit[~np.isnan(nonlinear_benefit)]
    
    print(f"\nOverall Non-Linear Benefit:")
    print(f"  Mean R² improvement (quad/cubic vs linear): {nonlinear_benefit.mean():.4f}")
    print(f"  {(nonlinear_benefit > 0.05).sum() / len(nonlinear_benefit) * 100:.1f}% of metrics benefit from non-linear fit")
    
    # Trajectory distribution
    print(f"\n{'-'*70}")
    print("Distribution of Trajectory Shapes (across all metrics):")
    print('-'*70)
    
    from collections import Counter
    traj_counter = Counter(df_metrics['overall_trajectory_type'])
    for traj_type, count in traj_counter.most_common():
        pct = count / len(df_metrics) * 100
        print(f"  {traj_type:25s}  {count:3d} ({pct:5.1f}%)")
    
    # By intervention group
    print(f"\n{'-'*70}")
    print("Trajectory Types by Intervention Group:")
    print('-'*70)
    
    for group in [1, 2, 3, 4, 5]:
        group_name = {1: 'alone_2w', 2: 'alone_4w', 3: 'group_2w', 4: 'group_4w', 5: 'control'}[group]
        group_data = df_metrics[df_metrics['group'] == group]
        
        print(f"\n{group_name} (n={len(group_data)}):")
        traj_counter = Counter(group_data['overall_trajectory_type'])
        for traj_type, count in traj_counter.most_common(3):
            pct = count / len(group_data) * 100
            print(f"  {traj_type:25s}  {count:2d} ({pct:5.1f}%)")


def compare_linear_vs_nonlinear_features(df_metrics):
    """
    Compare classification accuracy using:
    - Linear features only (slopes)
    - Non-linear features (polynomial coefficients)
    - Combined features
    """
    print(f"\n{'='*70}")
    print("CLASSIFICATION WITH LINEAR vs NON-LINEAR FEATURES")
    print('='*70)
    
    # Prepare feature sets
    linear_cols = [col for col in df_metrics.columns if col.endswith('_linear')]
    quad_cols = [col for col in df_metrics.columns if col.endswith('_quadratic')]
    cubic_cols = [col for col in df_metrics.columns if col.endswith('_cubic')]
    nonlinear_benefit_cols = [col for col in df_metrics.columns if col.endswith('_nonlinear_benefit')]
    
    # Add covariates
    covariate_cols = ['age', 'sex']
    
    # Map sex to numeric
    df_metrics['sex_encoded'] = (df_metrics['sex'] == 'M').astype(int)
    
    # Intervention vs Control classification
    df_test = df_metrics[df_metrics['group'].isin([1, 2, 3, 4, 5])].copy()
    df_test['is_intervention'] = (df_test['group'] != 5).astype(int)
    
    # Remove rows with NaN
    feature_sets = {
        'Linear only': linear_cols + ['age', 'sex_encoded'],
        'Non-linear only (quad+cubic)': quad_cols + cubic_cols + ['age', 'sex_encoded'],
        'Non-linear benefit only': nonlinear_benefit_cols + ['age', 'sex_encoded'],
        'Combined (all)': linear_cols + quad_cols + cubic_cols + ['age', 'sex_encoded']
    }
    
    results = {}
    
    for feature_set_name, feature_cols_set in feature_sets.items():
        # Filter available columns
        available_cols = [col for col in feature_cols_set if col in df_test.columns]
        
        X = df_test[available_cols].values
        y = df_test['is_intervention'].values
        
        # Remove NaN rows
        valid_mask = ~np.any(np.isnan(X), axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(np.unique(y)) < 2 or len(X) < 10:
            print(f"\n{feature_set_name}:")
            print(f"  Insufficient data")
            continue
        
        print(f"\n{feature_set_name}:")
        print(f"  Features: {len(available_cols)}, Samples: {len(X)}")
        
        # RF
        rf = RandomForestClassifier(n_estimators=300, max_depth=10, 
                                     class_weight='balanced', random_state=42, n_jobs=-1)
        cv_rf = cross_val_score(rf, X, y, cv=5)
        
        # SVM
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42)
        cv_svm = cross_val_score(svm, X_scaled, y, cv=5)
        
        print(f"  RF CV Accuracy:  {cv_rf.mean():.3f} ± {cv_rf.std():.3f}")
        print(f"  SVM CV Accuracy: {cv_svm.mean():.3f} ± {cv_svm.std():.3f}")
        
        best = "RF" if cv_rf.mean() > cv_svm.mean() else "SVM"
        improvement = abs(cv_rf.mean() - cv_svm.mean()) / max(cv_rf.mean(), cv_svm.mean()) * 100
        print(f"  Winner: {best} ({improvement:.1f}% difference)")
        
        results[feature_set_name] = {
            'rf_cv_mean': cv_rf.mean(),
            'rf_cv_std': cv_rf.std(),
            'svm_cv_mean': cv_svm.mean(),
            'svm_cv_std': cv_svm.std(),
            'n_features': len(available_cols),
            'n_samples': len(X)
        }
    
    return results


def visualize_trajectory_examples(df, df_metrics):
    """Create visualization of example trajectories (linear vs non-linear)."""
    print("\nCreating trajectory visualizations...")
    
    # Find subjects with high non-linear benefit
    df_metrics['max_nonlinear_benefit'] = df_metrics[
        [col for col in df_metrics.columns if col.endswith('_nonlinear_benefit')]
    ].max(axis=1)
    
    # Select examples
    high_nonlinear = df_metrics.nlargest(3, 'max_nonlinear_benefit')
    low_nonlinear = df_metrics.nsmallest(3, 'max_nonlinear_benefit')
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Plot high non-linearity examples
    for idx, (_, row) in enumerate(high_nonlinear.iterrows()):
        ax = axes[0, idx]
        subject = row['participant_id']
        
        subj_data = df[df['participant_id'] == subject]
        subj_bn = subj_data[subj_data['atlas'] == 'Brainnectome'].sort_values('session_id')
        
        # Pick first valid metric
        metric_cols = [col for col in df.columns if col.endswith('_slope')]
        metric = metric_cols[0] if metric_cols else None
        
        if metric and metric in subj_bn.columns:
            sessions = subj_bn['session_id'].str.extract(r'ses-(\d+)')[0].astype(int).values
            values = subj_bn[metric].values
            
            # Fit polynomials
            p_lin = np.polyfit(sessions, values, 1)
            p_quad = np.polyfit(sessions, values, 2)
            
            x_cont = np.linspace(sessions.min(), sessions.max(), 100)
            y_lin = np.polyval(p_lin, x_cont)
            y_quad = np.polyval(p_quad, x_cont)
            
            ax.scatter(sessions, values, s=100, alpha=0.7, label='Actual')
            ax.plot(x_cont, y_lin, '--', label='Linear', linewidth=2)
            ax.plot(x_cont, y_quad, '-', label='Quadratic', linewidth=2)
            ax.set_title(f'{row["overall_trajectory_type"]}\n(Subj {subject})', fontsize=10)
            ax.set_xlabel('Session')
            ax.set_ylabel('Metric Value')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    
    # Plot low non-linearity examples
    for idx, (_, row) in enumerate(low_nonlinear.iterrows()):
        ax = axes[1, idx]
        subject = row['participant_id']
        
        subj_data = df[df['participant_id'] == subject]
        subj_bn = subj_data[subj_data['atlas'] == 'Brainnectome'].sort_values('session_id')
        
        metric_cols = [col for col in df.columns if col.endswith('_slope')]
        metric = metric_cols[0] if metric_cols else None
        
        if metric and metric in subj_bn.columns:
            sessions = subj_bn['session_id'].str.extract(r'ses-(\d+)')[0].astype(int).values
            values = subj_bn[metric].values
            
            p_lin = np.polyfit(sessions, values, 1)
            p_quad = np.polyfit(sessions, values, 2)
            
            x_cont = np.linspace(sessions.min(), sessions.max(), 100)
            y_lin = np.polyval(p_lin, x_cont)
            y_quad = np.polyval(p_quad, x_cont)
            
            ax.scatter(sessions, values, s=100, alpha=0.7, label='Actual')
            ax.plot(x_cont, y_lin, '--', label='Linear', linewidth=2)
            ax.plot(x_cont, y_quad, '-', label='Quadratic', linewidth=2)
            ax.set_title(f'{row["overall_trajectory_type"]}\n(Subj {subject})', fontsize=10)
            ax.set_xlabel('Session')
            ax.set_ylabel('Metric Value')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    
    plt.suptitle('Example Trajectories: High vs Low Non-Linearity', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'nonlinear_trajectories_examples.png', dpi=300, bbox_inches='tight')
    print("Saved: nonlinear_trajectories_examples.png")
    plt.close()


def main():
    """Run non-linear time effects analysis."""
    print("="*70)
    print("NON-LINEAR TIME EFFECTS ANALYSIS")
    print("="*70)
    
    # Load data
    df = load_data()
    df_metrics = fit_polynomial_trajectories(df)
    df_metrics, trajectory_types = classify_trajectory_shapes(df_metrics)
    
    # Analyze
    analyze_trajectory_patterns(df_metrics, trajectory_types)
    results = compare_linear_vs_nonlinear_features(df_metrics)
    
    # Visualize
    visualize_trajectory_examples(df, df_metrics)
    
    # Save results
    df_metrics.to_csv(OUTPUT_DIR / 'nonlinear_trajectory_metrics.csv', index=False)
    
    with open(OUTPUT_DIR / 'nonlinear_analysis_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("✅ Non-Linear Analysis Complete!")
    print("="*70)
    print(f"\nResults saved:")
    print(f"  - nonlinear_trajectory_metrics.csv")
    print(f"  - nonlinear_analysis_summary.json")
    print(f"  - nonlinear_trajectories_examples.png")


if __name__ == '__main__':
    main()
