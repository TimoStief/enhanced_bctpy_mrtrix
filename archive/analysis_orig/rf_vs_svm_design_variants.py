#!/usr/bin/env python3
"""
Random Forest vs SVM Analysis with Design Variants
===================================================
Compares RF and SVM across three grouping strategies:
1. Social effect: alone vs group vs control
2. Duration effect: 2w vs 4w vs control  
3. Intervention vs Control: binary classification

Each variant includes gender-stratified analysis.
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
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
    if group in [1, 2]:
        return 'alone'
    elif group in [3, 4]:
        return 'group'
    elif group == 5:
        return 'control'
    return np.nan


def remap_groups_duration(group):
    """Remap to duration effect: 2w vs 4w vs control."""
    if group in [1, 3]:
        return '2w'
    elif group in [2, 4]:
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
    slopes_df = slopes_df.copy()
    if label_type == 'social':
        slopes_df['group_new'] = slopes_df['group'].apply(remap_groups_social)
    elif label_type == 'duration':
        slopes_df['group_new'] = slopes_df['group'].apply(remap_groups_duration)
    elif label_type == 'intervention':
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
    
    return X, y, feature_cols, slopes_df[valid_mask]


def train_rf_vs_svm(X, y, feature_cols, label_type):
    """Train both RF and SVM with cross-validation and comparison."""
    print(f"\n{'='*70}")
    print(f"RF vs SVM Comparison - {label_type.upper()}")
    print('='*70)
    
    # ============ RANDOM FOREST ============
    print(f"\n🌲 RANDOM FOREST")
    print('-'*70)
    
    rf = RandomForestClassifier(n_estimators=500, max_depth=10, 
                                 class_weight='balanced', random_state=42, n_jobs=-1)
    
    cv_scores_rf = cross_val_score(rf, X, y, cv=5)
    print(f"Cross-Validation Accuracy: {cv_scores_rf.mean():.3f} ± {cv_scores_rf.std():.3f}")
    print(f"Per-fold scores: {[f'{s:.3f}' for s in cv_scores_rf]}")
    
    rf.fit(X, y)
    y_pred_rf = rf.predict(X)
    print(f"Training Accuracy: {(y_pred_rf == y).mean():.3f}")
    
    rf_importance = rf.feature_importances_
    
    # ============ SVM ============
    print(f"\n🔵 SVM (RBF)")
    print('-'*70)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42)
    
    cv_scores_svm = cross_val_score(svm, X_scaled, y, cv=5)
    print(f"Cross-Validation Accuracy: {cv_scores_svm.mean():.3f} ± {cv_scores_svm.std():.3f}")
    print(f"Per-fold scores: {[f'{s:.3f}' for s in cv_scores_svm]}")
    
    svm.fit(X_scaled, y)
    y_pred_svm = svm.predict(X_scaled)
    print(f"Training Accuracy: {(y_pred_svm == y).mean():.3f}")
    
    # ============ COMPARISON ============
    print(f"\n{'='*70}")
    print("📊 MODEL COMPARISON")
    print('='*70)
    
    rf_mean = cv_scores_rf.mean()
    svm_mean = cv_scores_svm.mean()
    
    print(f"\nRandom Forest CV Accuracy: {rf_mean:.3f} ± {cv_scores_rf.std():.3f}")
    print(f"SVM CV Accuracy:           {svm_mean:.3f} ± {cv_scores_svm.std():.3f}")
    
    if rf_mean > svm_mean:
        improvement = (rf_mean - svm_mean) / svm_mean * 100
        print(f"\n✅ RF outperforms SVM by {improvement:.1f}%")
        best_model = 'RF'
    else:
        improvement = (svm_mean - rf_mean) / rf_mean * 100
        print(f"\n✅ SVM outperforms RF by {improvement:.1f}%")
        best_model = 'SVM'
    
    # ============ CLASSIFICATION REPORTS ============
    print(f"\n{'='*70}")
    print("Random Forest - Classification Report")
    print('-'*70)
    
    if label_type == 'social':
        target_names = ['Alone', 'Group', 'Control']
    elif label_type == 'duration':
        target_names = ['2 weeks', '4 weeks', 'Control']
    elif label_type == 'intervention':
        target_names = ['Intervention', 'Control']
    
    print(classification_report(y, y_pred_rf, target_names=target_names, digits=2))
    
    print(f"{'='*70}")
    print("SVM - Classification Report")
    print('-'*70)
    print(classification_report(y, y_pred_svm, target_names=target_names, digits=2))
    
    # ============ TOP FEATURES ============
    print(f"\n{'='*70}")
    print("🎯 Top 10 Features - Random Forest Importance (Gini)")
    print('='*70)
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_importance
    }).sort_values('importance', ascending=False)
    
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:40s}  {row['importance']:.4f}")
    
    return {
        'label_type': label_type,
        'best_model': best_model,
        'rf_cv_mean': rf_mean,
        'rf_cv_std': cv_scores_rf.std(),
        'svm_cv_mean': svm_mean,
        'svm_cv_std': cv_scores_svm.std(),
        'rf_train_acc': (y_pred_rf == y).mean(),
        'svm_train_acc': (y_pred_svm == y).mean(),
        'feature_importance': feature_importance.head(10).to_dict('records')
    }


def stratified_comparison(X, y, sex_values, feature_cols, label_type):
    """Compare RF vs SVM by gender."""
    print(f"\n{'='*70}")
    print(f"Gender-Stratified Comparison - {label_type.upper()}")
    print('='*70)
    
    results = {}
    
    for sex_label, sex_code in [('Male', 1), ('Female', 0)]:
        mask = sex_values == sex_code
        X_sex = X[mask]
        y_sex = y[mask]
        
        if len(np.unique(y_sex)) < 2:
            print(f"\n⚠️  {sex_label}: Not enough classes")
            continue
        
        print(f"\n{sex_label.upper()} (n={len(X_sex)})")
        print('-'*70)
        
        # RF
        rf = RandomForestClassifier(n_estimators=500, max_depth=10, 
                                     class_weight='balanced', random_state=42, n_jobs=-1)
        cv_rf = cross_val_score(rf, X_sex, y_sex, cv=5)
        
        # SVM
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sex)
        svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42)
        cv_svm = cross_val_score(svm, X_scaled, y_sex, cv=5)
        
        print(f"Random Forest CV Accuracy: {cv_rf.mean():.3f} ± {cv_rf.std():.3f}")
        print(f"SVM CV Accuracy:           {cv_svm.mean():.3f} ± {cv_svm.std():.3f}")
        
        better = "RF" if cv_rf.mean() > cv_svm.mean() else "SVM"
        print(f"Winner: {better}")
        
        results[sex_label] = {
            'n': len(X_sex),
            'rf_cv_mean': cv_rf.mean(),
            'rf_cv_std': cv_rf.std(),
            'svm_cv_mean': cv_svm.mean(),
            'svm_cv_std': cv_svm.std(),
            'better': better
        }
    
    return results


def main():
    """Run all variant analyses with RF vs SVM comparison."""
    print("="*70)
    print("Random Forest vs SVM Design Variant Analysis")
    print("="*70)
    
    # Load data
    df = load_data()
    slopes_df = calculate_slopes(df)
    
    all_results = {}
    
    # ==========================================
    # Variant 1: Social Effect
    # ==========================================
    print("\n" + "="*70)
    print("VARIANT 1: SOCIAL EFFECT (alone vs group vs control)")
    print("="*70)
    
    X, y, feature_cols, slopes_filtered = prepare_features(slopes_df, 'social')
    
    results_social = train_rf_vs_svm(X, y, feature_cols, 'social')
    sex_col_idx = feature_cols.index('sex_encoded')
    stratified_social = stratified_comparison(X, y, X[:, sex_col_idx], feature_cols, 'social')
    
    all_results['social'] = {**results_social, 'stratified': stratified_social}
    
    # ==========================================
    # Variant 2: Duration Effect
    # ==========================================
    print("\n" + "="*70)
    print("VARIANT 2: DURATION EFFECT (2w vs 4w vs control)")
    print("="*70)
    
    X, y, feature_cols, slopes_filtered = prepare_features(slopes_df, 'duration')
    
    results_duration = train_rf_vs_svm(X, y, feature_cols, 'duration')
    sex_col_idx = feature_cols.index('sex_encoded')
    stratified_duration = stratified_comparison(X, y, X[:, sex_col_idx], feature_cols, 'duration')
    
    all_results['duration'] = {**results_duration, 'stratified': stratified_duration}
    
    # ==========================================
    # Variant 3: Intervention vs Control
    # ==========================================
    print("\n" + "="*70)
    print("VARIANT 3: INTERVENTION vs CONTROL (binary)")
    print("="*70)
    
    X, y, feature_cols, slopes_filtered = prepare_features(slopes_df, 'intervention')
    
    results_intervention = train_rf_vs_svm(X, y, feature_cols, 'intervention')
    sex_col_idx = feature_cols.index('sex_encoded')
    stratified_intervention = stratified_comparison(X, y, X[:, sex_col_idx], feature_cols, 'intervention')
    
    all_results['intervention'] = {**results_intervention, 'stratified': stratified_intervention}
    
    # ==========================================
    # Summary Table
    # ==========================================
    print("\n" + "="*70)
    print("SUMMARY: RF vs SVM Across All Variants")
    print("="*70)
    
    print("\nOverall CV Accuracy:")
    print(f"\n{'Variant':<25} {'Random Forest':<20} {'SVM':<20} {'Winner':<10}")
    print("-"*75)
    
    for variant in ['social', 'duration', 'intervention']:
        rf_acc = f"{all_results[variant]['rf_cv_mean']:.3f} ± {all_results[variant]['rf_cv_std']:.3f}"
        svm_acc = f"{all_results[variant]['svm_cv_mean']:.3f} ± {all_results[variant]['svm_cv_std']:.3f}"
        winner = all_results[variant]['best_model']
        print(f"{variant:<25} {rf_acc:<20} {svm_acc:<20} {winner:<10}")
    
    print("\n\nGender-Stratified CV Accuracy:")
    for variant in ['social', 'duration', 'intervention']:
        print(f"\n{variant.upper()}:")
        if 'Male' in all_results[variant]['stratified']:
            male = all_results[variant]['stratified']['Male']
            print(f"  Male (n={male['n']}):")
            print(f"    RF:  {male['rf_cv_mean']:.3f} ± {male['rf_cv_std']:.3f}")
            print(f"    SVM: {male['svm_cv_mean']:.3f} ± {male['svm_cv_std']:.3f}")
            print(f"    Winner: {male['better']}")
        
        if 'Female' in all_results[variant]['stratified']:
            female = all_results[variant]['stratified']['Female']
            print(f"  Female (n={female['n']}):")
            print(f"    RF:  {female['rf_cv_mean']:.3f} ± {female['rf_cv_std']:.3f}")
            print(f"    SVM: {female['svm_cv_mean']:.3f} ± {female['svm_cv_std']:.3f}")
            print(f"    Winner: {female['better']}")
    
    # Save summary
    with open(OUTPUT_DIR / 'rf_vs_svm_design_variants.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ Analysis Complete!")
    print("="*70)
    print(f"\nResults saved to {OUTPUT_DIR}/rf_vs_svm_design_variants.json")


if __name__ == '__main__':
    main()
