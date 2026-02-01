#!/usr/bin/env python3
"""
SVM Analysis for Group Classification Based on Time Effects and Covariates

This script:
1. Calculates time-dependent changes in brain connectivity metrics across sessions
2. Includes age and gender as covariates
3. Uses SVM to classify groups based on temporal patterns
4. Analyzes which metrics show significant group-time interactions
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import json
import os

def load_data(bct_file, participants_file):
    """Load BCT results and participant metadata."""
    print("Loading data...")
    bct_df = pd.read_parquet(bct_file)
    participants_df = pd.read_csv(participants_file, sep='\t')
    
    # Clean subject IDs
    bct_df['subject'] = bct_df['subject'].str.replace('sub-', '')
    bct_df['session_num'] = bct_df['session'].str.replace('ses-', '').astype(int)
    bct_df['participant_id'] = 'sub-' + bct_df['subject']
    
    return bct_df, participants_df

def prepare_time_effect_data(bct_df, participants_df):
    """
    Calculate time-dependent changes in metrics.
    For each subject, compute the slope of change across sessions.
    """
    print("\nCalculating time effects (slopes across sessions)...")
    
    # Merge BCT results with participants data
    merged = bct_df.merge(
        participants_df[['participant_id', 'group', 'age', 'sex']],
        on='participant_id',
        how='left'
    )
    
    # Get metric columns (exclude metadata)
    metric_cols = [col for col in merged.columns if col not in [
        'subject', 'session', 'session_num', 'atlas', 'dsi_metric', 'matrix_type',
        'qc_passed', 'qc_warnings', 'participant_id', 'group', 'age', 'sex'
    ]]
    
    print(f"Found {len(metric_cols)} metrics")
    
    # Calculate slopes for each subject and metric
    subjects = merged['participant_id'].unique()
    slopes_list = []
    
    for subject in subjects:
        subject_data = merged[merged['participant_id'] == subject].sort_values('session_num')
        
        if len(subject_data) < 2:
            continue  # Need at least 2 sessions
        
        # Get participant info (same for all sessions)
        group = subject_data['group'].iloc[0]
        age = subject_data['age'].iloc[0]
        sex = subject_data['sex'].iloc[0]
        atlas = subject_data['atlas'].iloc[0]
        
        slope_dict = {
            'subject': subject,
            'group': group,
            'age': age,
            'sex': sex,
            'atlas': atlas,
            'n_sessions': len(subject_data)
        }
        
        # Calculate slope for each metric
        sessions = subject_data['session_num'].values
        for metric in metric_cols:
            values = subject_data[metric].values
            
            # Linear regression slope
            if len(sessions) >= 2 and not np.isnan(values).any():
                slope = np.polyfit(sessions, values, 1)[0]
                slope_dict[f'{metric}_slope'] = slope
        
        slopes_list.append(slope_dict)
    
    slopes_df = pd.DataFrame(slopes_list)
    print(f"Calculated slopes for {len(slopes_df)} subjects")
    
    return slopes_df, metric_cols

def encode_categorical_features(slopes_df):
    """Encode gender and group as numerical features."""
    slopes_df = slopes_df.copy()
    
    # Encode sex (M/F)
    le_sex = LabelEncoder()
    slopes_df['sex_encoded'] = le_sex.fit_transform(slopes_df['sex'])
    
    print(f"\nSex encoding: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")
    
    return slopes_df

def prepare_features_and_target(slopes_df, metric_cols):
    """Prepare feature matrix and target variable."""
    print("\nPreparing features...")
    
    # Get slope columns (for each metric)
    slope_cols = [col for col in slopes_df.columns if col.endswith('_slope')]
    
    # Feature matrix: slopes + covariates (age, sex, n_sessions)
    feature_cols = slope_cols + ['age', 'sex_encoded', 'n_sessions']
    
    X = slopes_df[feature_cols].fillna(0)
    y = slopes_df['group'].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Feature columns: {feature_cols[:5]}... (and {len(feature_cols)-5} more)")
    print(f"Target (group) distribution: {np.bincount(y)}")
    
    return X, y, feature_cols

def train_svm_classifier(X, y):
    """Train SVM with cross-validation."""
    print("\n" + "="*60)
    print("Training SVM Classifier (Time Effects + Covariates)")
    print("="*60)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # SVM with RBF kernel
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    
    # 5-fold cross-validation
    cv_scores = cross_val_score(svm, X_scaled, y, cv=5, scoring='accuracy')
    
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"Per-fold scores: {[f'{s:.3f}' for s in cv_scores]}")
    
    # Train on full data
    svm.fit(X_scaled, y)
    y_pred = svm.predict(X_scaled)
    
    print(f"\nTraining Accuracy: {(y_pred == y).mean():.3f}")
    print(f"\n{'-'*60}")
    print("Classification Report:")
    print(f"{'-'*60}")
    print(classification_report(y, y_pred, target_names=[f'Group {int(g)}' for g in np.unique(y)]))
    
    return svm, scaler, X_scaled, y, y_pred, cv_scores

def analyze_feature_importance(X, y, feature_cols, metric_cols):
    """Analyze which features (metrics) contribute to group discrimination."""
    print("\n" + "="*60)
    print("Feature Importance Analysis (using ANOVA)")
    print("="*60)
    
    # ANOVA F-scores for each feature
    f_scores = []
    p_values = []
    
    for i, feature in enumerate(feature_cols):
        x_col = X.iloc[:, i]
        
        # Group by target and compute F-statistic
        groups = [x_col[y == g] for g in np.unique(y)]
        f_stat, p_val = stats.f_oneway(*groups)
        
        f_scores.append(f_stat)
        p_values.append(p_val)
    
    # Sort by F-score
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'f_score': f_scores,
        'p_value': p_values
    }).sort_values('f_score', ascending=False)
    
    print("\nTop 15 Discriminative Features:")
    print(importance_df.head(15).to_string(index=False))
    
    # Identify which metrics (not covariates) are most important
    metric_importance = importance_df[importance_df['feature'].str.endswith('_slope')].head(10)
    print(f"\nTop 10 Time-Dependent Metrics:")
    print(metric_importance.to_string(index=False))
    
    return importance_df

def save_results(output_dir, slopes_df, svm, scaler, X_scaled, y, y_pred, cv_scores, 
                 importance_df, feature_cols):
    """Save analysis results."""
    print(f"\n{'='*60}")
    print("Saving Results")
    print(f"{'='*60}")
    
    # Summary statistics
    results_summary = {
        'analysis_type': 'Group Classification Based on Time Effects + Covariates',
        'model': 'SVM (RBF kernel)',
        'cv_accuracy_mean': float(cv_scores.mean()),
        'cv_accuracy_std': float(cv_scores.std()),
        'training_accuracy': float((y_pred == y).mean()),
        'n_support_vectors': int(len(svm.support_vectors_)),
        'n_samples': int(len(y)),
        'n_features': int(X_scaled.shape[1]),
        'groups': [int(g) for g in np.unique(y)],
        'features': feature_cols,
        'top_features': importance_df.head(10)['feature'].tolist(),
        'top_features_f_scores': importance_df.head(10)['f_score'].tolist()
    }
    
    with open(f'{output_dir}/svm_time_effect_classifier.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save slopes dataframe
    slopes_df.to_csv(f'{output_dir}/time_effect_slopes.csv', index=False)
    
    # Save importance scores
    importance_df.to_csv(f'{output_dir}/feature_importance_scores.csv', index=False)
    
    print(f"\n✅ Results saved to {output_dir}:")
    print(f"   - svm_time_effect_classifier.json")
    print(f"   - time_effect_slopes.csv")
    print(f"   - feature_importance_scores.csv")

def main():
    """Main analysis pipeline."""
    # File paths
    bct_file = '/data/local/129_PK01/derivatives/bct/bct_analysis_results.parquet'
    participants_file = '/data/local/129_PK01/derivatives/bct/orig_participants.tsv'
    output_dir = '/data/local/129_PK01/derivatives/bct'
    
    # Load data
    bct_df, participants_df = load_data(bct_file, participants_file)
    
    # Calculate time effects
    slopes_df, metric_cols = prepare_time_effect_data(bct_df, participants_df)
    
    # Encode categorical features
    slopes_df = encode_categorical_features(slopes_df)
    
    # Prepare features and target
    X, y, feature_cols = prepare_features_and_target(slopes_df, metric_cols)
    
    # Train SVM
    svm, scaler, X_scaled, y, y_pred, cv_scores = train_svm_classifier(X, y)
    
    # Feature importance analysis
    importance_df = analyze_feature_importance(X, y, feature_cols, metric_cols)
    
    # Save results
    save_results(output_dir, slopes_df, svm, scaler, X_scaled, y, y_pred, 
                 cv_scores, importance_df, feature_cols)
    
    print(f"\n{'='*60}")
    print("✅ Time Effect Analysis Complete!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
