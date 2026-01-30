#!/usr/bin/env python3
"""
Responder Phenotyping Analysis
===============================
Identifies responders vs non-responders to intervention based on connectivity changes.

Approach:
1. Calculate "response magnitude" for each subject (combined change across key metrics)
2. Stratify intervention groups into responders (top 50%) vs non-responders (bottom 50%)
3. Compare baseline characteristics, demographics, and trajectories
4. Test if baseline metrics can predict response
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, f_oneway, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
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


def calculate_baseline_and_slopes(df):
    """Calculate baseline values (first session) and slopes for each metric."""
    print("\nCalculating baseline and change metrics...")
    
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
        
        # Baseline (first session) and slopes
        for metric in metric_cols:
            values = subj_bn[metric].values
            if hasattr(values, 'to_numpy'):
                values = values.to_numpy()
            values = np.asarray(values, dtype=float)
            
            if len(values) >= 2 and not np.any(np.isnan(values)):
                # Baseline
                row[f'{metric}_baseline'] = values[0]
                # Slope (rate of change)
                slope = np.polyfit(sessions, values, 1)[0]
                row[f'{metric}_slope'] = slope
                # Absolute change
                row[f'{metric}_change'] = values[-1] - values[0]
        
        data_list.append(row)
    
    df_metrics = pd.DataFrame(data_list)
    print(f"Calculated metrics for {len(df_metrics)} subjects")
    
    return df_metrics


def calculate_response_magnitude(df_metrics, control_group=5):
    """
    Calculate response magnitude for each subject.
    
    Response = how much they differ from control group trajectory.
    Uses top discriminative metrics from our RF analysis.
    """
    print("\nCalculating response magnitude...")
    
    # Top metrics from RF analysis (participation coef, modularity, strength, degree)
    top_metrics = [
        'participation_coef_std_slope',
        'modularity_slope',
        'max_degree_slope',
        'avg_participation_slope',
        'strength_std_slope',
    ]
    
    # Get control group stats
    control_data = df_metrics[df_metrics['group'] == control_group]
    control_slopes = {m: control_data[m].mean() for m in top_metrics if m in df_metrics.columns}
    
    # Calculate response for each subject
    response_magnitudes = []
    
    for idx, row in df_metrics.iterrows():
        # Only for intervention groups
        if row['group'] not in [1, 2, 3, 4]:
            response_magnitudes.append(np.nan)
            continue
        
        # Response = absolute deviation from control group mean
        diffs = []
        for metric in top_metrics:
            if metric in df_metrics.columns and not np.isnan(row[metric]):
                diff = abs(row[metric] - control_slopes[metric])
                diffs.append(diff)
        
        if diffs:
            response_magnitudes.append(np.mean(diffs))
        else:
            response_magnitudes.append(np.nan)
    
    df_metrics['response_magnitude'] = response_magnitudes
    
    return df_metrics, control_slopes


def classify_responders(df_metrics):
    """Classify subjects as responders vs non-responders within each intervention group."""
    print("\nClassifying responders vs non-responders...")
    
    df_metrics['responder_status'] = 'control'  # default
    
    for group in [1, 2, 3, 4]:
        group_mask = df_metrics['group'] == group
        group_data = df_metrics[group_mask].copy()
        
        if len(group_data) < 4:
            continue
        
        # Median split on response magnitude
        median_response = group_data['response_magnitude'].median()
        
        # Classify
        responder_status = []
        for idx, row in group_data.iterrows():
            if row['response_magnitude'] > median_response:
                responder_status.append('responder')
            else:
                responder_status.append('non_responder')
        
        df_metrics.loc[group_mask, 'responder_status'] = responder_status
    
    return df_metrics


def analyze_responder_characteristics(df_metrics):
    """Compare baseline characteristics of responders vs non-responders."""
    print("\n" + "="*70)
    print("RESPONDER PHENOTYPING")
    print("="*70)
    
    results = {}
    
    for group in [1, 2, 3, 4]:
        group_name = {1: 'alone_2w', 2: 'alone_4w', 3: 'group_2w', 4: 'group_4w'}[group]
        
        group_data = df_metrics[df_metrics['group'] == group]
        responders = group_data[group_data['responder_status'] == 'responder']
        non_responders = group_data[group_data['responder_status'] == 'non_responder']
        
        if len(responders) < 2 or len(non_responders) < 2:
            continue
        
        print(f"\n{'='*70}")
        print(f"GROUP {group}: {group_name.upper()}")
        print('='*70)
        
        print(f"\nResponders: n={len(responders)}, Response magnitude: {responders['response_magnitude'].mean():.4f} ± {responders['response_magnitude'].std():.4f}")
        print(f"Non-responders: n={len(non_responders)}, Response magnitude: {non_responders['response_magnitude'].mean():.4f} ± {non_responders['response_magnitude'].std():.4f}")
        
        # ============ Demographic Comparison ============
        print(f"\n{'-'*70}")
        print("Demographic Characteristics")
        print('-'*70)
        
        # Age
        age_resp = responders['age'].values
        age_non = non_responders['age'].values
        if not np.any(np.isnan(age_resp)) and not np.any(np.isnan(age_non)):
            t_stat, p_val = ttest_ind(age_resp, age_non)
            print(f"Age:")
            print(f"  Responders:     {age_resp.mean():.1f} ± {age_resp.std():.1f} years")
            print(f"  Non-responders: {age_non.mean():.1f} ± {age_non.std():.1f} years")
            print(f"  t-test: p={p_val:.4f} {'*' if p_val < 0.05 else ''}")
        
        # Sex
        sex_resp = responders['sex'].value_counts()
        sex_non = non_responders['sex'].value_counts()
        print(f"Sex distribution:")
        print(f"  Responders:     {dict(sex_resp)}")
        print(f"  Non-responders: {dict(sex_non)}")
        
        # ============ Response Patterns ============
        print(f"\n{'-'*70}")
        print("Response Patterns (Top Metrics)")
        print('-'*70)
        
        top_metrics = [
            'participation_coef_std_slope',
            'modularity_slope',
            'max_degree_slope',
            'avg_participation_slope',
            'strength_std_slope',
        ]
        
        for metric in top_metrics:
            if metric not in df_metrics.columns:
                continue
            
            slope_resp = responders[metric].values
            slope_non = non_responders[metric].values
            
            if not (np.any(np.isnan(slope_resp)) and np.any(np.isnan(slope_non))):
                t_stat, p_val = ttest_ind(slope_resp[~np.isnan(slope_resp)], 
                                         slope_non[~np.isnan(slope_non)])
                
                metric_short = metric.replace('_slope', '')
                print(f"{metric_short}:")
                print(f"  Responders:     {slope_resp[~np.isnan(slope_resp)].mean():9.4f} ± {slope_resp[~np.isnan(slope_resp)].std():.4f}")
                print(f"  Non-responders: {slope_non[~np.isnan(slope_non)].mean():9.4f} ± {slope_non[~np.isnan(slope_non)].std():.4f}")
                print(f"  t-test: p={p_val:.4f} {'*' if p_val < 0.05 else ''}")
        
        # Store results
        results[group_name] = {
            'n_responders': len(responders),
            'n_non_responders': len(non_responders),
            'response_mag_responder': responders['response_magnitude'].mean(),
            'response_mag_non_responder': non_responders['response_magnitude'].mean(),
            'age_responder': age_resp.mean() if not np.any(np.isnan(age_resp)) else None,
            'age_non_responder': age_non.mean() if not np.any(np.isnan(age_non)) else None,
        }
    
    return results


def predict_baseline_responders(df_metrics):
    """Can baseline connectivity predict who will respond?"""
    print(f"\n{'='*70}")
    print("BASELINE PREDICTORS OF RESPONSE")
    print("="*70)
    
    for group in [1, 2, 3, 4]:
        group_name = {1: 'alone_2w', 2: 'alone_4w', 3: 'group_2w', 4: 'group_4w'}[group]
        
        group_data = df_metrics[df_metrics['group'] == group].copy()
        
        if len(group_data) < 6:
            print(f"\n{group_name}: Too few samples (n={len(group_data)})")
            continue
        
        # Get baseline metrics and response status
        baseline_cols = [col for col in group_data.columns if col.endswith('_baseline')]
        baseline_cols = [col for col in baseline_cols if col in group_data.columns]
        
        X = group_data[baseline_cols].values
        y = (group_data['responder_status'] == 'responder').values.astype(int)
        
        # Remove NaN rows
        valid_mask = ~np.any(np.isnan(X), axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(np.unique(y)) < 2 or len(X) < 6:
            print(f"\n{group_name}: Cannot classify (n={len(X)}, classes={np.unique(y)})")
            continue
        
        print(f"\n{group_name} (n={len(X)})")
        print('-'*70)
        
        # RF classifier
        rf = RandomForestClassifier(n_estimators=200, max_depth=5, 
                                     class_weight='balanced', random_state=42)
        
        cv_scores = cross_val_score(rf, X, y, cv=min(5, len(X)//2))
        
        print(f"RF CV Accuracy (predicting responders from baseline): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        if cv_scores.mean() > 0.65:
            # Train on full data to get feature importance
            rf.fit(X, y)
            
            print(f"\nTop 5 Predictive Baseline Metrics:")
            feature_imp = pd.DataFrame({
                'metric': baseline_cols,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            for i, row in feature_imp.head(5).iterrows():
                metric = row['metric'].replace('_baseline', '')
                print(f"  {metric:40s}  {row['importance']:.4f}")


def create_visualizations(df_metrics):
    """Create responder phenotyping visualizations."""
    print(f"\n{'='*70}")
    print("Creating visualizations...")
    print('='*70)
    
    group_names = {1: 'Alone 2w', 2: 'Alone 4w', 3: 'Group 2w', 4: 'Group 4w', 5: 'Control'}
    
    # ============ Plot 1: Response Magnitude Distribution ============
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Response magnitude by group
    for group in [1, 2, 3, 4]:
        group_data = df_metrics[df_metrics['group'] == group]
        ax = axes[0]
        ax.scatter([group]*len(group_data), group_data['response_magnitude'], 
                  alpha=0.6, s=100)
    
    axes[0].set_xlabel('Intervention Group')
    axes[0].set_ylabel('Response Magnitude')
    axes[0].set_xticks([1, 2, 3, 4])
    axes[0].set_xticklabels(['Alone 2w', 'Alone 4w', 'Group 2w', 'Group 4w'])
    axes[0].set_title('Response Magnitude by Intervention Group')
    axes[0].grid(alpha=0.3)
    
    # Responders vs non-responders
    intervention_data = df_metrics[df_metrics['group'].isin([1, 2, 3, 4])].copy()
    responder_counts = intervention_data.groupby('group')['responder_status'].value_counts().unstack()
    
    responder_counts.plot(kind='bar', ax=axes[1], color=['#FF6B6B', '#4ECDC4'])
    axes[1].set_xlabel('Intervention Group')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Responder Classification by Group')
    axes[1].set_xticklabels(['Alone 2w', 'Alone 4w', 'Group 2w', 'Group 4w'], rotation=45)
    axes[1].legend(title='Status')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'responder_phenotyping_distribution.png', dpi=300, bbox_inches='tight')
    print("Saved: responder_phenotyping_distribution.png")
    plt.close()
    
    # ============ Plot 2: Age vs Response by Group ============
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for group in [1, 2, 3, 4]:
        group_data = df_metrics[df_metrics['group'] == group]
        responders = group_data[group_data['responder_status'] == 'responder']
        non_responders = group_data[group_data['responder_status'] == 'non_responder']
        
        ax.scatter(responders['age'], responders['response_magnitude'], 
                  label=f'{group_names[group]} (R)', marker='o', s=120, alpha=0.7)
        ax.scatter(non_responders['age'], non_responders['response_magnitude'], 
                  label=f'{group_names[group]} (NR)', marker='x', s=120, alpha=0.7)
    
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Response Magnitude')
    ax.set_title('Age vs Response Magnitude by Intervention Group')
    ax.legend(ncol=2, fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'responder_phenotyping_age.png', dpi=300, bbox_inches='tight')
    print("Saved: responder_phenotyping_age.png")
    plt.close()


def main():
    """Run responder phenotyping analysis."""
    print("="*70)
    print("RESPONDER PHENOTYPING ANALYSIS")
    print("="*70)
    
    # Load and calculate
    df = load_data()
    df_metrics = calculate_baseline_and_slopes(df)
    df_metrics, control_slopes = calculate_response_magnitude(df_metrics)
    df_metrics = classify_responders(df_metrics)
    
    # Analyze
    results = analyze_responder_characteristics(df_metrics)
    predict_baseline_responders(df_metrics)
    
    # Visualize
    create_visualizations(df_metrics)
    
    # Save results
    df_metrics.to_csv(OUTPUT_DIR / 'responder_phenotyping_data.csv', index=False)
    
    with open(OUTPUT_DIR / 'responder_phenotyping_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("✅ Responder Phenotyping Complete!")
    print("="*70)
    print(f"\nResults saved:")
    print(f"  - responder_phenotyping_data.csv")
    print(f"  - responder_phenotyping_summary.json")
    print(f"  - responder_phenotyping_distribution.png")
    print(f"  - responder_phenotyping_age.png")


if __name__ == '__main__':
    main()
