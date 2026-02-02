"""
Stratified Time-Effect SVM Analysis
====================================
Separate SVM models for each sex and age group to examine
if group classification (based on temporal connectivity changes)
differs by demographic factors.

Outputs comparison of model performance across strata.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Paths
BCT_RESULTS = Path('/data/local/129_PK01/derivatives/bct/bct_analysis_results.parquet')
PARTICIPANTS_TSV = Path('/data/local/129_PK01/derivatives/bct/orig_participants.tsv')
OUTPUT_DIR = Path('/data/local/129_PK01/derivatives/bct')


def load_data():
    """Load BCT results and participant demographics."""
    df_bct = pd.read_parquet(BCT_RESULTS)
    df_demo = pd.read_csv(PARTICIPANTS_TSV, sep='\t')
    
    print(f"BCT results shape: {df_bct.shape}")
    print(f"Demographics shape: {df_demo.shape}")
    
    return df_bct, df_demo


def prepare_time_effect_data(df_bct, df_demo):
    """Calculate metric slopes per subject across sessions."""
    # Get unique subjects and session info
    subjects = df_bct['subject'].unique()
    
    slopes_dict = {}
    for subject in subjects:
        subj_data = df_bct[df_bct['subject'] == subject].sort_values('session')
        
        # Skip subjects with < 2 sessions
        if len(subj_data) < 2:
            continue
        
        # Convert session labels to numeric (ses-1 -> 1, etc)
        sessions = np.array([int(s.split('-')[1]) for s in subj_data['session']])
        
        slopes_dict[subject] = {'n_sessions': len(subj_data)}
        
        # Calculate slopes for all metrics
        metric_cols = [col for col in subj_data.columns 
                      if col not in ['subject', 'session', 'atlas']]
        
        for metric in metric_cols:
            values = pd.to_numeric(subj_data[metric], errors='coerce').values
            if np.all(np.isfinite(values)):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    slope = np.polyfit(sessions, values, 1)[0]
                slopes_dict[subject][f'{metric}_slope'] = slope
    
    # Convert to DataFrame
    slopes_df = pd.DataFrame.from_dict(slopes_dict, orient='index')
    slopes_df['subject'] = slopes_df.index
    
    # Merge with demographics
    df_merged = slopes_df.merge(df_demo[['participant_id', 'group', 'age', 'sex']], 
                                left_on='subject', right_on='participant_id', how='left')
    
    print(f"\nSlopes calculated for {len(df_merged)} subjects")
    print(f"Merged with demographics: {df_merged.shape}")
    
    return df_merged


def encode_categorical_features(df):
    """Encode categorical variables."""
    df_encoded = df.copy()
    
    le_sex = LabelEncoder()
    df_encoded['sex_encoded'] = le_sex.fit_transform(df_encoded['sex'].fillna('M'))
    sex_mapping = dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))
    
    print(f"\nSex encoding: {sex_mapping}")
    
    return df_encoded, sex_mapping


def prepare_features_and_target(df_encoded, exclude_cols=None):
    """Prepare feature matrix and target variable."""
    if exclude_cols is None:
        exclude_cols = ['subject', 'participant_id', 'group', 'age', 'sex', 'n_sessions', 'age_group']
    
    feature_cols = [col for col in df_encoded.columns if col not in exclude_cols and col != 'sex_encoded']
    feature_cols.append('sex_encoded')
    feature_cols.append('age')
    feature_cols.append('n_sessions')
    
    # Remove any NaN columns
    feature_cols = [col for col in feature_cols if col in df_encoded.columns]
    
    X = df_encoded[feature_cols].fillna(0)
    y = df_encoded['group'].fillna(-1)  # -1 for missing groups
    
    # Keep only subjects with valid groups
    mask = y >= 0
    X = X[mask]
    y = y[mask]
    
    return X, y, feature_cols


def train_svm_classifier(X, y, stratify_by=None):
    """Train SVM classifier with cross-validation."""
    # Remove rows with NaN in target
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    if len(y) < 5:
        return None
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train SVM with cross-validation
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    
    n_folds = min(5, len(np.unique(y)))  # At least 2 samples per fold
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Cross-validation scores
    cv_scores = cross_val_score(svm, X_scaled, y, cv=cv, scoring='accuracy')
    
    # Train on full data
    svm.fit(X_scaled, y)
    train_acc = svm.score(X_scaled, y)
    
    # Predictions for detailed metrics
    y_pred = svm.predict(X_scaled)
    
    return {
        'model': svm,
        'scaler': scaler,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'train_acc': train_acc,
        'y_pred': y_pred,
        'y_true': y,
        'feature_cols': X.columns.tolist(),
        'n_samples': len(y),
        'n_features': X.shape[1]
    }


def compute_feature_importance(X, y, feature_cols):
    """Compute ANOVA F-scores for feature importance."""
    importance_scores = []
    
    for feature_col in feature_cols:
        groups = [X[y == group][feature_col].values for group in np.unique(y) if group >= 0]
        
        if len(groups) > 1 and all(len(g) > 0 for g in groups):
            f_score, p_value = stats.f_oneway(*groups)
            importance_scores.append({
                'feature': feature_col,
                'f_score': f_score,
                'p_value': p_value
            })
    
    importance_df = pd.DataFrame(importance_scores).sort_values('f_score', ascending=False)
    return importance_df


def run_stratified_analysis(df_merged, df_encoded, feature_cols):
    """Run SVM analysis stratified by sex and age."""
    results_summary = {}
    
    # =========================================================================
    # 1. STRATIFIED BY SEX
    # =========================================================================
    print("\n" + "="*70)
    print("STRATIFIED ANALYSIS BY SEX")
    print("="*70)
    
    sex_results = {}
    for sex_value in ['M', 'F']:
        sex_label = 'Male' if sex_value == 'M' else 'Female'
        df_sex = df_encoded[df_encoded['sex'] == sex_value]
        
        if len(df_sex) < 5:
            print(f"\n⚠️ {sex_label}: Only {len(df_sex)} subjects, skipping...")
            continue
        
        print(f"\n{sex_label} (n={len(df_sex)}):")
        print("-" * 70)
        
        X_sex, y_sex, _ = prepare_features_and_target(df_sex)
        
        if len(np.unique(y_sex)) < 2:
            print(f"  ⚠️ Only 1 group represented, skipping...")
            continue
        
        result = train_svm_classifier(X_sex, y_sex)
        
        if result:
            sex_results[sex_label] = result
            print(f"  CV Accuracy: {result['cv_mean']:.3f} ± {result['cv_std']:.3f}")
            print(f"  Training Accuracy: {result['train_acc']:.3f}")
            print(f"  Per-fold scores: {[f'{s:.3f}' for s in result['cv_scores']]}")
            
            # Classification report
            print(f"\n  Classification Report:")
            print(f"  {classification_report(result['y_true'], result['y_pred'], zero_division=0)}")
            
            # Feature importance for this stratum
            importance = compute_feature_importance(X_sex, y_sex, result['feature_cols'])
            print(f"\n  Top 5 Features for {sex_label}:")
            for idx, row in importance.head(5).iterrows():
                print(f"    {row['feature']:40s} f={row['f_score']:6.3f} p={row['p_value']:.4f}")
    
    results_summary['by_sex'] = sex_results
    
    # =========================================================================
    # 2. STRATIFIED BY AGE GROUP
    # =========================================================================
    print("\n" + "="*70)
    print("STRATIFIED ANALYSIS BY AGE GROUP")
    print("="*70)
    
    age_results = {}
    df_encoded['age_group'] = pd.cut(df_encoded['age'], 
                                     bins=[0, 40, 100], 
                                     labels=['Young (<40)', 'Older (≥40)'])
    
    for age_group in ['Young (<40)', 'Older (≥40)']:
        df_age = df_encoded[df_encoded['age_group'] == age_group]
        
        if len(df_age) < 5:
            print(f"\n⚠️ {age_group}: Only {len(df_age)} subjects, skipping...")
            continue
        
        print(f"\n{age_group} (n={len(df_age)}):")
        print("-" * 70)
        
        X_age, y_age, _ = prepare_features_and_target(df_age)
        
        if len(np.unique(y_age)) < 2:
            print(f"  ⚠️ Only 1 group represented, skipping...")
            continue
        
        result = train_svm_classifier(X_age, y_age)
        
        if result:
            age_results[age_group] = result
            print(f"  CV Accuracy: {result['cv_mean']:.3f} ± {result['cv_std']:.3f}")
            print(f"  Training Accuracy: {result['train_acc']:.3f}")
            print(f"  Per-fold scores: {[f'{s:.3f}' for s in result['cv_scores']]}")
            
            # Classification report
            print(f"\n  Classification Report:")
            print(f"  {classification_report(result['y_true'], result['y_pred'], zero_division=0)}")
            
            # Feature importance for this stratum
            importance = compute_feature_importance(X_age, y_age, result['feature_cols'])
            print(f"\n  Top 5 Features for {age_group}:")
            for idx, row in importance.head(5).iterrows():
                print(f"    {row['feature']:40s} f={row['f_score']:6.3f} p={row['p_value']:.4f}")
    
    results_summary['by_age'] = age_results
    
    # =========================================================================
    # 3. CROSS-STRATIFICATION: SEX × GROUPS
    # =========================================================================
    print("\n" + "="*70)
    print("COMPARISON: TIME EFFECTS BY SEX × GROUP")
    print("="*70)
    
    comparison_data = []
    for sex_value in ['M', 'F']:
        for group in np.unique(df_encoded['group']):
            if group < 0:
                continue
            
            subset = df_encoded[(df_encoded['sex'] == sex_value) & (df_encoded['group'] == group)]
            
            if len(subset) >= 3:
                sex_label = 'Male' if sex_value == 'M' else 'Female'
                group_label = f'Group {int(group)}'
                
                # Get strength metric slopes
                strength_cols = [col for col in subset.columns if 'strength' in col and 'slope' in col]
                
                for col in strength_cols[:3]:  # Top 3 strength metrics
                    mean_slope = subset[col].mean()
                    std_slope = subset[col].std()
                    comparison_data.append({
                        'sex': sex_label,
                        'group': group_label,
                        'metric': col,
                        'n': len(subset),
                        'mean_slope': mean_slope,
                        'std_slope': std_slope
                    })
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        print("\n" + df_comparison.to_string(index=False))
        df_comparison.to_csv(OUTPUT_DIR / 'stratified_sex_group_comparison.csv', index=False)
        print(f"\n✅ Saved to stratified_sex_group_comparison.csv")
    
    return results_summary


def save_results(results_summary):
    """Save stratified analysis results."""
    # Convert non-serializable objects
    summary_for_json = {
        'stratified_by_sex': {},
        'stratified_by_age': {}
    }
    
    for sex_label, result in results_summary.get('by_sex', {}).items():
        summary_for_json['stratified_by_sex'][sex_label] = {
            'n_samples': result['n_samples'],
            'n_features': result['n_features'],
            'cv_mean': float(result['cv_mean']),
            'cv_std': float(result['cv_std']),
            'train_accuracy': float(result['train_acc']),
            'cv_scores': [float(s) for s in result['cv_scores']]
        }
    
    for age_group, result in results_summary.get('by_age', {}).items():
        summary_for_json['stratified_by_age'][age_group] = {
            'n_samples': result['n_samples'],
            'n_features': result['n_features'],
            'cv_mean': float(result['cv_mean']),
            'cv_std': float(result['cv_std']),
            'train_accuracy': float(result['train_acc']),
            'cv_scores': [float(s) for s in result['cv_scores']]
        }
    
    with open(OUTPUT_DIR / 'stratified_analysis_summary.json', 'w') as f:
        json.dump(summary_for_json, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ Stratified Analysis Complete!")
    print("="*70)
    print(f"\n✅ Results saved to {OUTPUT_DIR}:")
    print(f"   - stratified_analysis_summary.json")
    print(f"   - stratified_sex_group_comparison.csv")


def main():
    """Run full stratified analysis pipeline."""
    print("="*70)
    print("STRATIFIED TIME-EFFECT ANALYSIS")
    print("="*70)
    
    # Load data
    df_bct, df_demo = load_data()
    
    # Calculate slopes
    df_merged = prepare_time_effect_data(df_bct, df_demo)
    
    # Encode categorical variables
    df_encoded, sex_mapping = encode_categorical_features(df_merged)
    
    # Get feature columns
    _, _, feature_cols = prepare_features_and_target(df_encoded)
    
    # Run stratified analyses
    results_summary = run_stratified_analysis(df_merged, df_encoded, feature_cols)
    
    # Save results
    save_results(results_summary)


if __name__ == '__main__':
    main()
