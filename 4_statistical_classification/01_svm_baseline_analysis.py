#!/usr/bin/env python3
"""
SVM Analysis with Different Study Design Variants
==================================================
Tests various grouping strategies:
1. Social effect: alone vs group vs control
2. Duration effect: 2w vs 4w vs control  
3. Intervention vs Control: binary classification

Each variant includes gender-stratified analysis.
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import f_oneway
import json
from pathlib import Path

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
    
    # Rename columns to match
    bct_df = bct_df.rename(columns={'subject': 'participant_id', 'session': 'session_id'})
    
    # Merge
    df = bct_df.merge(participants_df, on=['participant_id', 'session_id'], how='inner')
    
    print(f"Loaded {len(df)} rows, {len(df['participant_id'].unique())} subjects")
    return df


def calculate_slopes(df):
    """Calculate temporal slopes for each metric."""
    print("\nCalculating time effects (slopes across sessions)...")
    
    # Get metric columns - exclude non-numeric and metadata
    exclude_cols = ['participant_id', 'session_id', 'atlas', 'age', 'sex', 'group', 'nr_sessions',
                    'dsi_metric', 'matrix_type', 'qc_passed', 'qc_warnings']
    
    # Get numeric columns only
    metric_cols = [col for col in df.columns if col not in exclude_cols]
    # Filter to numeric types
    metric_cols = [col for col in metric_cols if pd.api.types.is_numeric_dtype(df[col])]
    
    print(f"Found {len(metric_cols)} numeric metrics")
    
    slopes_data = []
    
    for subject in df['participant_id'].unique():
        subj_data = df[df['participant_id'] == subject].copy()
        
        if len(subj_data) < 2:
            continue
            
        # Get metadata
        age = subj_data['age'].iloc[0]
        sex = subj_data['sex'].iloc[0]
        group = subj_data['group'].iloc[0]
        n_sessions = len(subj_data)
        
        # Extract Brainnectome atlas only
        subj_bn = subj_data[subj_data['atlas'] == 'Brainnectome'].sort_values('session_id')
        
        if len(subj_bn) < 2:
            continue
        
        sessions = subj_bn['session_id'].str.extract(r'ses-(\d+)')[0].astype(int).values
        
        slopes = {'participant_id': subject, 'age': age, 'sex': sex, 'group': group, 'n_sessions': n_sessions}
        
        for metric in metric_cols:
            values = subj_bn[metric].values
            # Convert to numpy array if needed
            if hasattr(values, 'to_numpy'):
                values = values.to_numpy()
            values = np.asarray(values, dtype=float)
            
            if len(values) >= 2 and not np.any(np.isnan(values)):
                slope = np.polyfit(sessions, values, 1)[0]
                slopes[f'{metric}_slope'] = slope
        
        slopes_data.append(slopes)
    
    slopes_df = pd.DataFrame(slopes_data)
    print(f"Calculated slopes for {len(slopes_df)} subjects")
    
    return slopes_df


def remap_groups_social(group):
    """Remap to social effect: alone vs group vs control."""
    if group in [1, 2]:  # alone_2w, alone_4w
        return 'alone'
    elif group in [3, 4]:  # group_2w, group_4w
        return 'group'
    elif group == 5:
        return 'control'
    return np.nan


def remap_groups_duration(group):
    """Remap to duration effect: 2w vs 4w vs control."""
    if group in [1, 3]:  # alone_2w, group_2w
        return '2w'
    elif group in [2, 4]:  # alone_4w, group_4w
        return '4w'
    elif group == 5:
        return 'control'
    return np.nan


def remap_groups_intervention(group):
    """Remap to intervention vs control: binary."""
    if group in [1, 2, 3, 4]:
        return 'intervention'
    elif group == 5:
        return 'control'
    return np.nan


def encode_labels(labels, label_type):
    """Encode string labels to integers."""
    if label_type == 'social':
        mapping = {'alone': 0, 'group': 1, 'control': 2}
    elif label_type == 'duration':
        mapping = {'2w': 0, '4w': 1, 'control': 2}
    elif label_type == 'intervention':
        mapping = {'intervention': 0, 'control': 1}
    
    return np.array([mapping[label] if label in mapping else np.nan for label in labels])


def prepare_features(slopes_df, label_type='social'):
    """Prepare feature matrix and labels."""
    print(f"\nPreparing features for {label_type} analysis...")
    
    # Apply group remapping
    if label_type == 'social':
        slopes_df = slopes_df.copy()
        slopes_df['group_new'] = slopes_df['group'].apply(remap_groups_social)
    elif label_type == 'duration':
        slopes_df = slopes_df.copy()
        slopes_df['group_new'] = slopes_df['group'].apply(remap_groups_duration)
    elif label_type == 'intervention':
        slopes_df = slopes_df.copy()
        slopes_df['group_new'] = slopes_df['group'].apply(remap_groups_intervention)
    
    # Encode sex
    sex_mapping = {'F': 0, 'M': 1}
    slopes_df['sex_encoded'] = slopes_df['sex'].map(sex_mapping)
    
    # Get feature columns (slopes + covariates)
    feature_cols = [col for col in slopes_df.columns if col.endswith('_slope')]
    feature_cols += ['age', 'sex_encoded', 'n_sessions']
    
    # Create feature matrix
    X = slopes_df[feature_cols].values
    y_str = slopes_df['group_new'].values
    y = encode_labels(y_str, label_type)
    
    # Remove NaN rows
    valid_mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
    X = X[valid_mask]
    y = y[valid_mask].astype(int)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    if label_type == 'social':
        print(f"  0=alone ({np.sum(y==0)}), 1=group ({np.sum(y==1)}), 2=control ({np.sum(y==2)})")
    elif label_type == 'duration':
        print(f"  0=2w ({np.sum(y==0)}), 1=4w ({np.sum(y==1)}), 2=control ({np.sum(y==2)})")
    elif label_type == 'intervention':
        print(f"  0=intervention ({np.sum(y==0)}), 1=control ({np.sum(y==1)})")
    
    return X, y, feature_cols, slopes_df[valid_mask]


def train_svm(X, y, label_type):
    """Train SVM classifier with cross-validation."""
    print(f"\n{'='*60}")
    print(f"SVM Classifier - {label_type.upper()} Effect")
    print('='*60)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train SVM
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42)
    
    # Cross-validation
    cv_scores = cross_val_score(svm, X_scaled, y, cv=5)
    print(f"\nCross-Validation Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"Per-fold scores: {[f'{s:.3f}' for s in cv_scores]}")
    
    # Train on full data
    svm.fit(X_scaled, y)
    y_pred = svm.predict(X_scaled)
    
    print(f"\nTraining Accuracy: {(y_pred == y).mean():.3f}")
    
    # Classification report
    if label_type == 'social':
        target_names = ['Alone', 'Group', 'Control']
    elif label_type == 'duration':
        target_names = ['2 weeks', '4 weeks', 'Control']
    elif label_type == 'intervention':
        target_names = ['Intervention', 'Control']
    
    print("\n" + "-"*60)
    print("Classification Report:")
    print("-"*60)
    print(classification_report(y, y_pred, target_names=target_names, digits=2))
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    return {
        'svm': svm,
        'scaler': scaler,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'train_acc': (y_pred == y).mean(),
        'y_pred': y_pred,
        'confusion_matrix': cm
    }


def stratified_analysis(X, y, sex_values, label_type):
    """Run stratified analysis by gender."""
    print(f"\n{'='*60}")
    print(f"Gender-Stratified Analysis - {label_type.upper()} Effect")
    print('='*60)
    
    results = {}
    
    for sex_label, sex_code in [('Male', 1), ('Female', 0)]:
        mask = sex_values == sex_code
        X_sex = X[mask]
        y_sex = y[mask]
        
        if len(np.unique(y_sex)) < 2:
            print(f"\n⚠️  {sex_label}: Not enough classes for classification")
            continue
        
        print(f"\n{sex_label} (n={len(X_sex)})")
        print("-"*60)
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sex)
        
        # SVM
        svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42)
        
        # CV
        cv_scores = cross_val_score(svm, X_scaled, y_sex, cv=5)
        print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Train
        svm.fit(X_scaled, y_sex)
        y_pred = svm.predict(X_scaled)
        
        print(f"Training Accuracy: {(y_pred == y_sex).mean():.3f}")
        
        # Store
        results[sex_label] = {
            'n': len(X_sex),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_acc': (y_pred == y_sex).mean(),
            'distribution': np.bincount(y_sex).tolist()
        }
    
    return results


def feature_importance_anova(X, y, feature_cols):
    """Calculate feature importance using ANOVA F-test."""
    print(f"\n{'='*60}")
    print("Feature Importance (ANOVA F-scores)")
    print('='*60)
    
    scores = []
    for i, feature in enumerate(feature_cols):
        groups = [X[y == label, i] for label in np.unique(y)]
        f_stat, p_val = f_oneway(*groups)
        scores.append({'feature': feature, 'f_score': f_stat, 'p_value': p_val})
    
    scores_df = pd.DataFrame(scores).sort_values('f_score', ascending=False)
    
    print("\nTop 15 Features:")
    print(scores_df.head(15).to_string(index=False))
    
    return scores_df


def main():
    """Run all variant analyses."""
    print("="*60)
    print("SVM Design Variant Analysis")
    print("="*60)
    
    # Load data
    df = load_data()
    slopes_df = calculate_slopes(df)
    
    all_results = {}
    
    # ==========================================
    # Variant 1: Social Effect (alone vs group vs control)
    # ==========================================
    print("\n" + "="*60)
    print("VARIANT 1: Social Effect")
    print("="*60)
    
    X, y, feature_cols, slopes_filtered = prepare_features(slopes_df, 'social')
    
    # Overall analysis
    results_social = train_svm(X, y, 'social')
    
    # Feature importance
    features_social = feature_importance_anova(X, y, feature_cols)
    
    # Gender stratification
    sex_col_idx = feature_cols.index('sex_encoded')
    stratified_social = stratified_analysis(X, y, X[:, sex_col_idx], 'social')
    
    all_results['social'] = {
        'overall': {
            'cv_mean': results_social['cv_mean'],
            'cv_std': results_social['cv_std'],
            'train_acc': results_social['train_acc'],
            'confusion_matrix': results_social['confusion_matrix'].tolist()
        },
        'stratified': stratified_social,
        'top_features': features_social.head(10).to_dict('records')
    }
    
    # Save slopes
    slopes_social = slopes_filtered.copy()
    slopes_social['group_variant'] = slopes_social['group_new']
    slopes_social.to_csv(OUTPUT_DIR / 'time_effect_slopes_social.csv', index=False)
    features_social.to_csv(OUTPUT_DIR / 'feature_importance_social.csv', index=False)
    
    # ==========================================
    # Variant 2: Duration Effect (2w vs 4w vs control)
    # ==========================================
    print("\n" + "="*60)
    print("VARIANT 2: Duration Effect")
    print("="*60)
    
    X, y, feature_cols, slopes_filtered = prepare_features(slopes_df, 'duration')
    
    # Overall analysis
    results_duration = train_svm(X, y, 'duration')
    
    # Feature importance
    features_duration = feature_importance_anova(X, y, feature_cols)
    
    # Gender stratification
    sex_col_idx = feature_cols.index('sex_encoded')
    stratified_duration = stratified_analysis(X, y, X[:, sex_col_idx], 'duration')
    
    all_results['duration'] = {
        'overall': {
            'cv_mean': results_duration['cv_mean'],
            'cv_std': results_duration['cv_std'],
            'train_acc': results_duration['train_acc'],
            'confusion_matrix': results_duration['confusion_matrix'].tolist()
        },
        'stratified': stratified_duration,
        'top_features': features_duration.head(10).to_dict('records')
    }
    
    # Save
    slopes_duration = slopes_filtered.copy()
    slopes_duration['group_variant'] = slopes_duration['group_new']
    slopes_duration.to_csv(OUTPUT_DIR / 'time_effect_slopes_duration.csv', index=False)
    features_duration.to_csv(OUTPUT_DIR / 'feature_importance_duration.csv', index=False)
    
    # ==========================================
    # Variant 3: Intervention vs Control (binary)
    # ==========================================
    print("\n" + "="*60)
    print("VARIANT 3: Intervention vs Control")
    print("="*60)
    
    X, y, feature_cols, slopes_filtered = prepare_features(slopes_df, 'intervention')
    
    # Overall analysis
    results_intervention = train_svm(X, y, 'intervention')
    
    # Feature importance
    features_intervention = feature_importance_anova(X, y, feature_cols)
    
    # Gender stratification
    sex_col_idx = feature_cols.index('sex_encoded')
    stratified_intervention = stratified_analysis(X, y, X[:, sex_col_idx], 'intervention')
    
    all_results['intervention'] = {
        'overall': {
            'cv_mean': results_intervention['cv_mean'],
            'cv_std': results_intervention['cv_std'],
            'train_acc': results_intervention['train_acc'],
            'confusion_matrix': results_intervention['confusion_matrix'].tolist()
        },
        'stratified': stratified_intervention,
        'top_features': features_intervention.head(10).to_dict('records')
    }
    
    # Save
    slopes_intervention = slopes_filtered.copy()
    slopes_intervention['group_variant'] = slopes_intervention['group_new']
    slopes_intervention.to_csv(OUTPUT_DIR / 'time_effect_slopes_intervention.csv', index=False)
    features_intervention.to_csv(OUTPUT_DIR / 'feature_importance_intervention.csv', index=False)
    
    # ==========================================
    # Summary Comparison
    # ==========================================
    print("\n" + "="*60)
    print("SUMMARY: Variant Comparison")
    print("="*60)
    
    print("\nOverall CV Accuracy:")
    print(f"  Social (alone/group/control):      {all_results['social']['overall']['cv_mean']:.3f} ± {all_results['social']['overall']['cv_std']:.3f}")
    print(f"  Duration (2w/4w/control):          {all_results['duration']['overall']['cv_mean']:.3f} ± {all_results['duration']['overall']['cv_std']:.3f}")
    print(f"  Intervention vs Control:           {all_results['intervention']['overall']['cv_mean']:.3f} ± {all_results['intervention']['overall']['cv_std']:.3f}")
    
    print("\nGender-Stratified CV Accuracy:")
    for variant in ['social', 'duration', 'intervention']:
        print(f"\n  {variant.upper()}:")
        if 'Male' in all_results[variant]['stratified']:
            print(f"    Male:   {all_results[variant]['stratified']['Male']['cv_mean']:.3f} ± {all_results[variant]['stratified']['Male']['cv_std']:.3f} (n={all_results[variant]['stratified']['Male']['n']})")
        if 'Female' in all_results[variant]['stratified']:
            print(f"    Female: {all_results[variant]['stratified']['Female']['cv_mean']:.3f} ± {all_results[variant]['stratified']['Female']['cv_std']:.3f} (n={all_results[variant]['stratified']['Female']['n']})")
    
    # Save summary
    with open(OUTPUT_DIR / 'svm_design_variants_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ Analysis Complete!")
    print("="*60)
    print(f"\nResults saved to {OUTPUT_DIR}:")
    print("  - svm_design_variants_summary.json")
    print("  - time_effect_slopes_social.csv")
    print("  - feature_importance_social.csv")
    print("  - time_effect_slopes_duration.csv")
    print("  - feature_importance_duration.csv")
    print("  - time_effect_slopes_intervention.csv")
    print("  - feature_importance_intervention.csv")


if __name__ == '__main__':
    main()
