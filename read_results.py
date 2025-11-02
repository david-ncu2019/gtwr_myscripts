#!/usr/bin/env python3
"""
GTNNWR Results Analysis
Usage: python read_results.py <path_to_coefficients.csv>
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_results(filepath):
    """Analyze GTNNWR results and display visualizations."""
    
    # Load data
    try:
        results = pd.read_csv(filepath)
        print(f"✓ Loaded: {filepath}")
        print(f"Shape: {results.shape}\n")
    except FileNotFoundError:
        print(f"✗ Error: File not found - {filepath}")
        sys.exit(1)
    
    # Identify columns (flexible naming)
    y_col = [c for c in results.columns if c.startswith('Pred_')][0]
    actual_col = y_col.replace('Pred_', '')
    coef_cols = [c for c in results.columns if c.startswith('coef_')]
    
    # Performance metrics by dataset
    print("=" * 50)
    print("MODEL PERFORMANCE BY DATASET")
    print("=" * 50)

    for dataset_name in ['train', 'valid', 'test']:
        mask = results['dataset_belong'] == dataset_name
        actual = results.loc[mask, actual_col]
        predicted = results.loc[mask, y_col]
        
        r2 = 1 - np.sum((actual - predicted)**2) / np.sum((actual - actual.mean())**2)
        rmse = np.sqrt(np.mean((actual - predicted)**2))
        mae = np.mean(np.abs(actual - predicted))
        
        print(f"\n{dataset_name.upper()}:")
        print(f"  R²:   {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  N:    {mask.sum()}")

    # Overall (for reference)
    print(f"\nOVERALL (all data):")
    actual_all = results[actual_col]
    predicted_all = results[y_col]
    r2_all = 1 - np.sum((actual_all - predicted_all)**2) / np.sum((actual_all - actual_all.mean())**2)
    print(f"  R²:   {r2_all:.4f}")
    
    # Data splits
    if 'dataset_belong' in results.columns:
        print(f"\nData Distribution:")
        print(results['dataset_belong'].value_counts())
    
    # Coefficient statistics
    print("\n" + "=" * 50)
    print("COEFFICIENT STATISTICS")
    print("=" * 50)
    for coef in coef_cols:
        print(f"{coef}:")
        print(f"  Range: [{results[coef].min():.4f}, {results[coef].max():.4f}]")
        print(f"  Mean:  {results[coef].mean():.4f}")
        print(f"  Std:   {results[coef].std():.4f}")
    
    # Station-level summary (if applicable)
    if 'STATION' in results.columns:
        print("\n" + "=" * 50)
        print("STATION-LEVEL SUMMARY (Top 5)")
        print("=" * 50)
        station_summary = results.groupby('STATION').agg({
            coef_cols[0]: ['mean', 'std'],
            actual_col: 'mean',
            y_col: 'mean'
        }).round(4)
        print(station_summary.head())
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Predicted vs Actual
    axes[0,0].scatter(actual, predicted, alpha=0.5, s=20)
    axes[0,0].plot([actual.min(), actual.max()], 
                   [actual.min(), actual.max()], 'r--', lw=2)
    axes[0,0].set_xlabel(f'Actual {actual_col}', fontsize=11)
    axes[0,0].set_ylabel(f'Predicted {actual_col}', fontsize=11)
    axes[0,0].set_title(f'Predictions vs Actual (R² = {r2:.3f})', fontsize=12, fontweight='bold')
    axes[0,0].grid(alpha=0.3)
    
    # 2. Spatial distribution (if spatial columns exist)
    if 'X_TWD97' in results.columns and 'Y_TWD97' in results.columns:
        scatter = axes[0,1].scatter(results['X_TWD97'], results['Y_TWD97'], 
                                   c=results[coef_cols[0]], cmap='RdBu_r', 
                                   alpha=0.7, s=30)
        axes[0,1].set_xlabel('X_TWD97', fontsize=11)
        axes[0,1].set_ylabel('Y_TWD97', fontsize=11)
        axes[0,1].set_title(f'Spatial Distribution: {coef_cols[0]}', 
                           fontsize=12, fontweight='bold')
        plt.colorbar(scatter, ax=axes[0,1], label='Coefficient Value')
    else:
        axes[0,1].text(0.5, 0.5, 'No spatial columns found', 
                      ha='center', va='center', transform=axes[0,1].transAxes)
        axes[0,1].set_title('Spatial Distribution (N/A)')
    
    # 3. Temporal evolution (if monthly column exists)
    if 'monthly' in results.columns:
        monthly_coef = results.groupby('monthly')[coef_cols[0]].mean()
        axes[1,0].plot(monthly_coef.index, monthly_coef.values, 'o-', linewidth=2)
        axes[1,0].set_xlabel('Month', fontsize=11)
        axes[1,0].set_ylabel(f'Avg {coef_cols[0]}', fontsize=11)
        axes[1,0].set_title('Temporal Evolution of Coefficients', 
                           fontsize=12, fontweight='bold')
        axes[1,0].grid(alpha=0.3)
    else:
        axes[1,0].text(0.5, 0.5, 'No temporal column found', 
                      ha='center', va='center', transform=axes[1,0].transAxes)
        axes[1,0].set_title('Temporal Evolution (N/A)')
    
    # 4. Residuals
    residuals = actual - predicted
    axes[1,1].scatter(predicted, residuals, alpha=0.5, s=20)
    axes[1,1].axhline(y=0, color='red', linestyle='--', lw=2)
    axes[1,1].set_xlabel(f'Predicted {actual_col}', fontsize=11)
    axes[1,1].set_ylabel('Residuals', fontsize=11)
    axes[1,1].set_title('Residual Plot', fontsize=12, fontweight='bold')
    axes[1,1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n✓ Analysis complete!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_results.py <path_to_coefficients.csv>")
        print("Example: python read_results.py ./results/coefficients_20250707.csv")
        sys.exit(1)
    
    analyze_results(sys.argv[1])