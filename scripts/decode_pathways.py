#!/usr/bin/env python3
"""
Main decoding analysis: Compare object information in dorsal vs ventral pathways
across hemispheres using the THINGs dataset.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, permutation_test_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_config():
    """Load analysis configuration."""
    config_path = Path(__file__).parent.parent / "configs" / "analysis_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_roi_data(subject_id, session_id):
    """Load preprocessed ROI data."""
    data_file = Path("results") / f"sub-{subject_id}_ses-{session_id}_roi_data.pkl"
    
    if not data_file.exists():
        raise FileNotFoundError(f"No preprocessed data found: {data_file}")
    
    with open(data_file, 'rb') as f:
        roi_data = pickle.load(f)
    
    return roi_data

def prepare_data_for_decoding(roi_data, roi_name):
    """Prepare data for classification."""
    data = roi_data[roi_name]['data']  # trials x voxels
    labels = roi_data[roi_name]['labels']  # stimulus names
    
    # Encode labels as integers
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    
    return X, y, le

def run_classification(X, y, config):
    """Run cross-validated classification."""
    
    # Set up classifier
    clf = SVC(kernel='linear', C=1.0, random_state=42)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=config['analysis']['cv_folds'], 
                        shuffle=True, random_state=42)
    
    # Main classification
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    
    # Permutation test for significance
    score, perm_scores, pvalue = permutation_test_score(
        clf, X, y, cv=cv, scoring='accuracy',
        n_permutations=config['analysis']['n_permutations'],
        random_state=42, n_jobs=-1
    )
    
    return {
        'cv_scores': cv_scores,
        'mean_accuracy': cv_scores.mean(),
        'std_accuracy': cv_scores.std(),
        'permutation_score': score,
        'permutation_pvalue': pvalue,
        'null_distribution': perm_scores
    }

def compare_pathways(roi_data, config):
    """Compare decoding accuracy across pathways and hemispheres."""
    
    results = {}
    
    # ROI pairs to compare
    roi_names = ['dorsal_left', 'dorsal_right', 'ventral_left', 'ventral_right']
    
    print("Running classification for each ROI...")
    
    for roi_name in roi_names:
        print(f"\nAnalyzing {roi_name}...")
        
        # Prepare data
        X, y, le = prepare_data_for_decoding(roi_data, roi_name)
        
        print(f"  Data shape: {X.shape}")
        print(f"  Number of categories: {len(le.classes_)}")
        
        # Run classification
        result = run_classification(X, y, config)
        results[roi_name] = result
        
        print(f"  Accuracy: {result['mean_accuracy']:.3f} ± {result['std_accuracy']:.3f}")
        print(f"  P-value: {result['permutation_pvalue']:.3f}")
    
    return results

def analyze_pathway_differences(results):
    """Analyze differences between pathways and hemispheres."""
    
    # Extract mean accuracies
    accuracies = {roi: res['mean_accuracy'] for roi, res in results.items()}
    
    # Compare within hemisphere: dorsal vs ventral
    left_diff = accuracies['dorsal_left'] - accuracies['ventral_left']
    right_diff = accuracies['dorsal_right'] - accuracies['ventral_right']
    
    # Compare across hemisphere: left vs right
    dorsal_hemi_diff = accuracies['dorsal_left'] - accuracies['dorsal_right']
    ventral_hemi_diff = accuracies['ventral_left'] - accuracies['ventral_right']
    
    comparisons = {
        'left_dorsal_vs_ventral': left_diff,
        'right_dorsal_vs_ventral': right_diff,
        'dorsal_left_vs_right': dorsal_hemi_diff,
        'ventral_left_vs_right': ventral_hemi_diff
    }
    
    return comparisons

def plot_results(results, comparisons, subject_id, session_id):
    """Create visualization of results."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Decoding accuracies by ROI
    roi_names = list(results.keys())
    accuracies = [results[roi]['mean_accuracy'] for roi in roi_names]
    errors = [results[roi]['std_accuracy'] for roi in roi_names]
    
    bars = ax1.bar(roi_names, accuracies, yerr=errors, capsize=5)
    ax1.set_ylabel('Classification Accuracy')
    ax1.set_title(f'Object Decoding Accuracy by ROI\nSub-{subject_id}/Ses-{session_id}')
    ax1.axhline(y=1/720, color='red', linestyle='--', label='Chance (1/720)')  # THINGs has 720 categories
    ax1.legend()
    plt.setp(ax1.get_xticklabels(), rotation=45)
    
    # Color code by pathway
    colors = ['blue', 'blue', 'red', 'red']  # dorsal=blue, ventral=red
    for bar, color in zip(bars, colors):
        bar.set_color(color)
        bar.set_alpha(0.7)
    
    # Plot 2: Pathway comparisons
    comp_names = list(comparisons.keys())
    comp_values = list(comparisons.values())
    
    ax2.bar(comp_names, comp_values)
    ax2.set_ylabel('Accuracy Difference')
    ax2.set_title('Pathway Comparisons')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.setp(ax2.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = Path("results") / f"sub-{subject_id}_ses-{session_id}_decoding_results.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_file}")
    
    return fig

def save_results(results, comparisons, subject_id, session_id):
    """Save results to files."""
    
    # Create summary DataFrame
    summary_data = []
    for roi_name, result in results.items():
        summary_data.append({
            'subject': subject_id,
            'session': session_id,
            'roi': roi_name,
            'pathway': roi_name.split('_')[0],
            'hemisphere': roi_name.split('_')[1],
            'mean_accuracy': result['mean_accuracy'],
            'std_accuracy': result['std_accuracy'],
            'p_value': result['permutation_pvalue']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary
    summary_file = Path("results") / f"sub-{subject_id}_ses-{session_id}_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    
    # Save detailed results
    detailed_file = Path("results") / f"sub-{subject_id}_ses-{session_id}_detailed_results.pkl"
    with open(detailed_file, 'wb') as f:
        pickle.dump({'results': results, 'comparisons': comparisons}, f)
    
    print(f"Saved summary: {summary_file}")
    print(f"Saved detailed results: {detailed_file}")
    
    return summary_df

def decode_subject(subject_id, session_id):
    """Main analysis pipeline for one subject/session."""
    
    print(f"\n=== Decoding Analysis: sub-{subject_id}/ses-{session_id} ===")
    
    # Load configuration
    config = load_config()
    
    # Load preprocessed data
    roi_data = load_roi_data(subject_id, session_id)
    
    # Run pathway comparison analysis
    results = compare_pathways(roi_data, config)
    
    # Analyze differences
    comparisons = analyze_pathway_differences(results)
    
    # Create visualizations
    plot_results(results, comparisons, subject_id, session_id)
    
    # Save results
    summary_df = save_results(results, comparisons, subject_id, session_id)
    
    # Print summary
    print("\n=== RESULTS SUMMARY ===")
    print(summary_df[['roi', 'mean_accuracy', 'p_value']].to_string(index=False))
    
    print("\n=== PATHWAY COMPARISONS ===")
    for comp_name, diff in comparisons.items():
        print(f"{comp_name}: {diff:+.3f}")
    
    print(f"\nAnalysis complete for sub-{subject_id}/ses-{session_id}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python decode_pathways.py <subject_id> <session_id>")
        print("Example: python decode_pathways.py 01 01")
        sys.exit(1)
    
    subject_id = sys.argv[1]
    session_id = sys.argv[2]
    
    decode_subject(subject_id, session_id)