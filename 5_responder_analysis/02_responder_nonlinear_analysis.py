#!/usr/bin/env python3
"""
Non-Linear Responders Analysis
================================
Identifies subjects with strong non-linear trajectory deviations.

Responder types:
1. Accelerators: Strong cubic/quadratic positive effects (delayed response)
2. Decelerators: Strong negative curvature (early plateau)
3. Stable: Weak non-linear effects (linear responders)

For each type, compares baseline characteristics and intervention responses.
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Paths
BCT_DIR = Path('/data/local/129_PK01/derivatives/bct')
OUTPUT_DIR = BCT_DIR


def load_nonlinear_metrics():
    """Load pre-calculated non-linear trajectory metrics."""
    print("Loading non-linear trajectory metrics...")
    df_metrics = pd.read_csv(BCT_DIR / 'nonlinear_trajectory_metrics.csv')
    print(f"Loaded {len(df_metrics)} subjects")
    return df_metrics


def calculate_nonlinear_responder_types(df_metrics):
    """
    Classify subjects into responder types based on polynomial coefficients.
    
    Response magnitude = max non-linear deviation from linear fit
    """
    print("\nCalculating non-linear responder types...")
    
    # Get all quadratic and cubic coefficients
    quad_cols = [col for col in df_metrics.columns if col.endswith('_quadratic')]
    cubic_cols = [col for col in df_metrics.columns if col.endswith('_cubic')]
    
    responder_types = []
    nonlinear_magnitudes = []
    acceleration_scores = []
    
    for idx, row in df_metrics.iterrows():
        # Get all quad and cubic coefficients
        quads = [row[col] for col in quad_cols if not np.isnan(row[col])]
        cubics = [row[col] for col in cubic_cols if not np.isnan(row[col])]
        
        if not quads or not cubics:
            responder_types.append('unknown')
            nonlinear_magnitudes.append(np.nan)
            acceleration_scores.append(np.nan)
            continue
        
        # Magnitude of non-linear effect (average absolute coefficients)
        quad_mag = np.mean(np.abs(quads))
        cubic_mag = np.mean(np.abs(cubics))
        nonlinear_mag = quad_mag + cubic_mag
        
        # Acceleration score (positive = accelerating, negative = decelerating)
        quad_sign = np.mean(quads)
        cubic_sign = np.mean(cubics)
        accel_score = cubic_sign + quad_sign
        
        nonlinear_magnitudes.append(nonlinear_mag)
        acceleration_scores.append(accel_score)
        
        # Classify type
        if nonlinear_mag < np.nanpercentile(nonlinear_magnitudes, 33):
            responder_types.append('stable')
        elif accel_score > 0:
            responder_types.append('accelerator')
        else:
            responder_types.append('decelerator')
    
    df_metrics['nonlinear_magnitude'] = nonlinear_magnitudes
    df_metrics['acceleration_score'] = acceleration_scores
    df_metrics['responder_type'] = responder_types
    
    return df_metrics


def analyze_nonlinear_responder_phenotypes(df_metrics):
    """Analyze characteristics of different non-linear responder types."""
    print(f"\n{'='*70}")
    print("NON-LINEAR RESPONDER PHENOTYPES")
    print('='*70)
    
    results = {}
    
    # Overall distribution
    type_counts = df_metrics['responder_type'].value_counts()
    print(f"\nOverall Distribution:")
    for rtype, count in type_counts.items():
        pct = count / len(df_metrics) * 100
        print(f"  {rtype:15s}  {count:3d} ({pct:5.1f}%)")
    
    # By intervention group
    print(f"\n{'-'*70}")
    print("Distribution by Intervention Group:")
    print('-'*70)
    
    for group in [1, 2, 3, 4, 5]:
        group_name = {1: 'alone_2w', 2: 'alone_4w', 3: 'group_2w', 4: 'group_4w', 5: 'control'}[group]
        group_data = df_metrics[df_metrics['group'] == group]
        
        print(f"\n{group_name} (n={len(group_data)}):")
        
        type_dist = group_data['responder_type'].value_counts()
        for rtype, count in type_dist.items():
            pct = count / len(group_data) * 100
            print(f"  {rtype:15s}  {count:3d} ({pct:5.1f}%)")
    
    # Detailed comparison of responder types
    print(f"\n{'='*70}")
    print("Detailed Comparison of Responder Types")
    print('='*70)
    
    for rtype in ['accelerator', 'decelerator', 'stable']:
        type_data = df_metrics[df_metrics['responder_type'] == rtype]
        
        if len(type_data) < 2:
            continue
        
        print(f"\n{rtype.upper()} (n={len(type_data)})")
        print('-'*70)
        
        # Non-linear metrics
        print(f"Non-linear Characteristics:")
        print(f"  Nonlinear Magnitude: {type_data['nonlinear_magnitude'].mean():.6f} ± {type_data['nonlinear_magnitude'].std():.6f}")
        print(f"  Acceleration Score:  {type_data['acceleration_score'].mean():.6f} ± {type_data['acceleration_score'].std():.6f}")
        
        # Demographics
        print(f"\nDemographics:")
        age = type_data['age'].values
        if not np.any(np.isnan(age)):
            print(f"  Age: {age.mean():.1f} ± {age.std():.1f} years")
        
        sex_dist = type_data['sex'].value_counts()
        print(f"  Sex: {dict(sex_dist)}")
        
        # Group distribution
        print(f"\nIntervention Distribution:")
        group_dist = type_data['group'].value_counts()
        for g, count in group_dist.items():
            gname = {1: 'alone_2w', 2: 'alone_4w', 3: 'group_2w', 4: 'group_4w', 5: 'control'}[g]
            pct = count / len(type_data) * 100
            print(f"  {gname:15s}  {count:2d} ({pct:5.1f}%)")
        
        # Store results
        results[rtype] = {
            'n': len(type_data),
            'nonlinear_mag': type_data['nonlinear_magnitude'].mean(),
            'accel_score': type_data['acceleration_score'].mean(),
            'age': age.mean() if not np.any(np.isnan(age)) else None,
        }
    
    return results


def compare_responder_types_across_groups(df_metrics):
    """Compare responder types within intervention vs control."""
    print(f"\n{'='*70}")
    print("RESPONDER TYPES: INTERVENTION vs CONTROL")
    print('='*70)
    
    intervention = df_metrics[df_metrics['group'].isin([1, 2, 3, 4])]
    control = df_metrics[df_metrics['group'] == 5]
    
    print(f"\nIntervention Groups (n={len(intervention)}):")
    print(f"  Accelerators: {(intervention['responder_type'] == 'accelerator').sum()} ({(intervention['responder_type'] == 'accelerator').sum() / len(intervention) * 100:.1f}%)")
    print(f"  Decelerators: {(intervention['responder_type'] == 'decelerator').sum()} ({(intervention['responder_type'] == 'decelerator').sum() / len(intervention) * 100:.1f}%)")
    print(f"  Stable:       {(intervention['responder_type'] == 'stable').sum()} ({(intervention['responder_type'] == 'stable').sum() / len(intervention) * 100:.1f}%)")
    
    print(f"\nControl Group (n={len(control)}):")
    print(f"  Accelerators: {(control['responder_type'] == 'accelerator').sum()} ({(control['responder_type'] == 'accelerator').sum() / len(control) * 100:.1f}%)")
    print(f"  Decelerators: {(control['responder_type'] == 'decelerator').sum()} ({(control['responder_type'] == 'decelerator').sum() / len(control) * 100:.1f}%)")
    print(f"  Stable:       {(control['responder_type'] == 'stable').sum()} ({(control['responder_type'] == 'stable').sum() / len(control) * 100:.1f}%)")
    
    # Statistical test: are intervention groups enriched for accelerators/decelerators?
    print(f"\n{'-'*70}")
    print("Statistical Test (Chi-square for distribution difference):")
    
    from scipy.stats import chi2_contingency
    contingency = pd.crosstab(
        df_metrics['group'].isin([1, 2, 3, 4]),
        df_metrics['responder_type']
    )
    chi2, p_val, dof, expected = chi2_contingency(contingency)
    print(f"  Chi-square statistic: {chi2:.4f}")
    print(f"  p-value: {p_val:.4f} {'*' if p_val < 0.05 else ''}")


def classify_nonlinear_types(df_metrics):
    """Can we predict intervention membership from responder type?"""
    print(f"\n{'='*70}")
    print("CLASSIFICATION: Can Responder Type Predict Intervention?")
    print('='*70)
    
    # Create features from responder types (one-hot encoding)
    X = pd.get_dummies(df_metrics['responder_type'], prefix='responder')
    
    # Add demographics
    X['age'] = df_metrics['age']
    X['sex'] = (df_metrics['sex'] == 'M').astype(int)
    
    # Target
    y = (df_metrics['group'] != 5).astype(int)  # 1 = intervention, 0 = control
    
    # Remove NaN
    valid_mask = ~X.isna().any(axis=1) & ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"\nSamples: {len(X)}")
    print(f"Intervention: {y.sum()}, Control: {(1-y).sum()}")
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # RF
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, 
                                 class_weight='balanced', random_state=42, n_jobs=-1)
    cv_rf = cross_val_score(rf, X_scaled, y, cv=5)
    
    print(f"\nRandom Forest CV Accuracy: {cv_rf.mean():.3f} ± {cv_rf.std():.3f}")
    print(f"Per-fold scores: {[f'{s:.3f}' for s in cv_rf]}")
    
    # Train to get feature importance
    rf.fit(X_scaled, y)
    feature_imp = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop Features for Predicting Intervention:")
    for i, row in feature_imp.head(5).iterrows():
        print(f"  {row['feature']:30s}  {row['importance']:.4f}")


def visualize_nonlinear_responders(df_metrics):
    """Visualize non-linear responder types."""
    print("\nCreating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Nonlinear magnitude by type
    ax = axes[0, 0]
    df_metrics.boxplot(column='nonlinear_magnitude', by='responder_type', ax=ax)
    ax.set_title('Non-Linear Magnitude by Responder Type')
    ax.set_xlabel('Responder Type')
    ax.set_ylabel('Non-Linear Magnitude')
    plt.sca(ax)
    plt.xticks(rotation=45)
    
    # Plot 2: Acceleration score by type
    ax = axes[0, 1]
    df_metrics.boxplot(column='acceleration_score', by='responder_type', ax=ax)
    ax.set_title('Acceleration Score by Responder Type')
    ax.set_xlabel('Responder Type')
    ax.set_ylabel('Acceleration Score')
    plt.sca(ax)
    plt.xticks(rotation=45)
    
    # Plot 3: Responder type distribution by group
    ax = axes[1, 0]
    group_names = {1: 'Alone 2w', 2: 'Alone 4w', 3: 'Group 2w', 4: 'Group 4w', 5: 'Control'}
    df_plot = df_metrics.copy()
    df_plot['group_name'] = df_plot['group'].map(group_names)
    
    type_by_group = pd.crosstab(df_plot['group_name'], df_plot['responder_type'])
    type_by_group.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#95E1D3'])
    ax.set_title('Responder Type Distribution by Intervention Group')
    ax.set_xlabel('Intervention Group')
    ax.set_ylabel('Count')
    ax.legend(title='Responder Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.sca(ax)
    plt.xticks(rotation=45)
    
    # Plot 4: Scatter - acceleration vs magnitude
    ax = axes[1, 1]
    colors = {'accelerator': '#FF6B6B', 'decelerator': '#4ECDC4', 'stable': '#95E1D3'}
    for rtype in ['accelerator', 'decelerator', 'stable']:
        data = df_metrics[df_metrics['responder_type'] == rtype]
        ax.scatter(data['nonlinear_magnitude'], data['acceleration_score'], 
                  label=rtype, s=100, alpha=0.6, color=colors.get(rtype, 'gray'))
    
    ax.set_xlabel('Non-Linear Magnitude')
    ax.set_ylabel('Acceleration Score')
    ax.set_title('Responder Type: Magnitude vs Acceleration')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'nonlinear_responders_phenotypes.png', dpi=300, bbox_inches='tight')
    print("Saved: nonlinear_responders_phenotypes.png")
    plt.close()


def main():
    """Run non-linear responders analysis."""
    print("="*70)
    print("NON-LINEAR RESPONDERS ANALYSIS")
    print("="*70)
    
    # Load and classify
    df_metrics = load_nonlinear_metrics()
    df_metrics = calculate_nonlinear_responder_types(df_metrics)
    
    # Analyze
    results = analyze_nonlinear_responder_phenotypes(df_metrics)
    compare_responder_types_across_groups(df_metrics)
    classify_nonlinear_types(df_metrics)
    
    # Visualize
    visualize_nonlinear_responders(df_metrics)
    
    # Save
    df_metrics.to_csv(OUTPUT_DIR / 'nonlinear_responders_classified.csv', index=False)
    
    with open(OUTPUT_DIR / 'nonlinear_responders_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("✅ Non-Linear Responders Analysis Complete!")
    print("="*70)
    print(f"\nResults saved:")
    print(f"  - nonlinear_responders_classified.csv")
    print(f"  - nonlinear_responders_summary.json")
    print(f"  - nonlinear_responders_phenotypes.png")


if __name__ == '__main__':
    main()
