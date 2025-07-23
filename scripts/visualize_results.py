#!/usr/bin/env python3
"""
Visualize and aggregate results across subjects/sessions.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_all_results():
    """Load results from all processed subjects/sessions."""
    
    results_dir = Path("results")
    summary_files = list(results_dir.glob("*_summary.csv"))
    
    if not summary_files:
        raise FileNotFoundError("No summary files found. Run decoding analysis first.")
    
    # Load and combine all summary files
    all_summaries = []
    for file in summary_files:
        df = pd.read_csv(file)
        all_summaries.append(df)
    
    combined_df = pd.concat(all_summaries, ignore_index=True)
    print(f"Loaded results from {len(summary_files)} sessions")
    
    return combined_df

def plot_pathway_comparison(df):
    """Create main pathway comparison plot."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Overall accuracy by pathway and hemisphere
    ax1 = axes[0, 0]
    sns.barplot(data=df, x='pathway', y='mean_accuracy', hue='hemisphere', ax=ax1)
    ax1.set_title('Decoding Accuracy by Pathway and Hemisphere')
    ax1.set_ylabel('Classification Accuracy')
    
    # Plot 2: Individual subject points
    ax2 = axes[0, 1]
    sns.stripplot(data=df, x='pathway', y='mean_accuracy', hue='hemisphere', 
                  dodge=True, size=8, ax=ax2)
    ax2.set_title('Individual Session Results')
    ax2.set_ylabel('Classification Accuracy')
    
    # Plot 3: Statistical significance
    ax3 = axes[1, 0]
    significant = df[df['p_value'] < 0.05]
    sns.countplot(data=significant, x='roi', ax=ax3)
    ax3.set_title('Significant Results (p < 0.05)')
    ax3.set_ylabel('Number of Sessions')
    plt.setp(ax3.get_xticklabels(), rotation=45)
    
    # Plot 4: Pathway differences within subjects
    ax4 = axes[1, 1]
    
    # Calculate dorsal-ventral differences for each session
    differences = []
    for subject in df['subject'].unique():
        for session in df[df['subject'] == subject]['session'].unique():
            sess_data = df[(df['subject'] == subject) & (df['session'] == session)]
            
            for hemi in ['left', 'right']:
                dorsal_acc = sess_data[(sess_data['pathway'] == 'dorsal') & 
                                     (sess_data['hemisphere'] == hemi)]['mean_accuracy'].values
                ventral_acc = sess_data[(sess_data['pathway'] == 'ventral') & 
                                      (sess_data['hemisphere'] == hemi)]['mean_accuracy'].values
                
                if len(dorsal_acc) > 0 and len(ventral_acc) > 0:
                    diff = dorsal_acc[0] - ventral_acc[0]
                    differences.append({
                        'subject': subject,
                        'session': session,
                        'hemisphere': hemi,
                        'dorsal_minus_ventral': diff
                    })
    
    diff_df = pd.DataFrame(differences)
    if not diff_df.empty:
        sns.boxplot(data=diff_df, x='hemisphere', y='dorsal_minus_ventral', ax=ax4)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax4.set_title('Dorsal - Ventral Accuracy Difference')
        ax4.set_ylabel('Accuracy Difference')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = Path("results") / "pathway_comparison_summary.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved summary plot: {plot_file}")
    
    return fig

def run_statistical_tests(df):
    """Run statistical tests comparing pathways and hemispheres."""
    
    print("\n=== STATISTICAL TESTS ===")
    
    # Test 1: Dorsal vs Ventral (collapsed across hemispheres)
    dorsal_acc = df[df['pathway'] == 'dorsal']['mean_accuracy']
    ventral_acc = df[df['pathway'] == 'ventral']['mean_accuracy']
    
    if len(dorsal_acc) > 0 and len(ventral_acc) > 0:
        stat, p = stats.ttest_ind(dorsal_acc, ventral_acc)
        print(f"Dorsal vs Ventral: t={stat:.3f}, p={p:.3f}")
        print(f"  Dorsal mean: {dorsal_acc.mean():.3f} ± {dorsal_acc.std():.3f}")
        print(f"  Ventral mean: {ventral_acc.mean():.3f} ± {ventral_acc.std():.3f}")
    
    # Test 2: Left vs Right (collapsed across pathways)
    left_acc = df[df['hemisphere'] == 'left']['mean_accuracy']
    right_acc = df[df['hemisphere'] == 'right']['mean_accuracy']
    
    if len(left_acc) > 0 and len(right_acc) > 0:
        stat, p = stats.ttest_ind(left_acc, right_acc)
        print(f"\nLeft vs Right: t={stat:.3f}, p={p:.3f}")
        print(f"  Left mean: {left_acc.mean():.3f} ± {left_acc.std():.3f}")
        print(f"  Right mean: {right_acc.mean():.3f} ± {right_acc.std():.3f}")
    
    # Test 3: Interaction effect (2x2 ANOVA would be ideal, but simple comparisons for now)
    print(f"\n=== ROI-SPECIFIC RESULTS ===")
    for roi in df['roi'].unique():
        roi_data = df[df['roi'] == roi]
        print(f"{roi}: {roi_data['mean_accuracy'].mean():.3f} ± {roi_data['mean_accuracy'].std():.3f}")
        print(f"  Significant sessions: {len(roi_data[roi_data['p_value'] < 0.05])}/{len(roi_data)}")

def create_summary_table(df):
    """Create a summary table of results."""
    
    # Group by ROI and calculate statistics
    summary = df.groupby('roi').agg({
        'mean_accuracy': ['mean', 'std', 'count'],
        'p_value': lambda x: (x < 0.05).sum()
    }).round(3)
    
    # Flatten column names
    summary.columns = ['mean_acc', 'std_acc', 'n_sessions', 'n_significant']
    
    # Add chance level comparison
    chance_level = 1/720  # THINGs has 720 categories
    summary['above_chance'] = summary['mean_acc'] > chance_level
    
    print("\n=== SUMMARY TABLE ===")
    print(summary.to_string())
    
    # Save table
    table_file = Path("results") / "summary_table.csv"
    summary.to_csv(table_file)
    print(f"\nSaved summary table: {table_file}")
    
    return summary

def main():
    """Main visualization pipeline."""
    
    print("Loading and visualizing results...")
    
    # Load all results
    df = load_all_results()
    
    # Create main visualization
    plot_pathway_comparison(df)
    
    # Run statistical tests
    run_statistical_tests(df)
    
    # Create summary table
    create_summary_table(df)
    
    print("\nVisualization complete!")
    print("Check the 'results' directory for plots and summary files.")

if __name__ == "__main__":
    main()