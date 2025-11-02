import pandas as pd
import numpy as np
import os

def analyze_gtnnwr_results(mainfolder, timestamp="20250708_102940"):
    """Analyze GTNNWR training results and provide recommendations."""
    
    # Load results
    train = pd.read_csv(os.path.join(mainfolder, f'gtnnwr_output_{timestamp}_train.csv'))
    valid = pd.read_csv(os.path.join(mainfolder, f'gtnnwr_output_{timestamp}_valid.csv'))
    test = pd.read_csv(os.path.join(mainfolder, f'gtnnwr_output_{timestamp}_test.csv'))
    
    # Performance summary
    print("GTNNWR Performance Analysis")
    print("=" * 40)
    
    # Average metrics
    train_r2 = train['r_square'].mean()
    valid_r2 = valid['r_square'].mean()
    test_r2 = test['r_square'].mean()
    
    print(f"Average R²:")
    print(f"  Train: {train_r2:.4f}")
    print(f"  Valid: {valid_r2:.4f}")
    print(f"  Test:  {test_r2:.4f}")
    
    # RMSE
    print(f"\nAverage RMSE:")
    print(f"  Train: {train['rmse'].mean():.4f}")
    print(f"  Valid: {valid['rmse'].mean():.4f}")
    print(f"  Test:  {test['rmse'].mean():.4f}")
    
    # Overfitting check
    overfitting = train_r2 - valid_r2
    print(f"\nOverfitting Check:")
    print(f"  Train-Valid R² gap: {overfitting:.4f}")
    
    if overfitting > 0.15:
        print("  ⚠ OVERFITTING detected")
    elif overfitting > 0.1:
        print("  ⚠ Mild overfitting")
    else:
        print("  ✓ No significant overfitting")
    
    # Worst stations
    print(f"\nWorst performing stations (validation):")
    worst = valid.nsmallest(3, 'r_square')
    for _, row in worst.iterrows():
        print(f"  {row.iloc[0]}: R²={row['r_square']:.3f}, RMSE={row['rmse']:.4f}")
    
    # Best stations
    print(f"\nBest performing stations (validation):")
    best = valid.nlargest(3, 'r_square')
    for _, row in best.iterrows():
        print(f"  {row.iloc[0]}: R²={row['r_square']:.3f}, RMSE={row['rmse']:.4f}")
    
    # Performance distribution
    print(f"\nPerformance Distribution:")
    r2_ranges = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1)]
    for low, high in r2_ranges:
        count = ((valid['r_square'] >= low) & (valid['r_square'] < high)).sum()
        print(f"  R² {low:.1f}-{high:.1f}: {count} stations")
    
    # Recommendations
    print(f"\nRecommendations:")
    print("=" * 20)
    
    if valid_r2 < 0.5:
        print("• Low accuracy - increase model capacity:")
        print("  dense_layers: [[6], [256, 128, 64]]")
        print("  start_lr: 0.01")
    
    if overfitting > 0.1:
        print("• Reduce overfitting:")
        print(f"  drop_out: {0.4 if overfitting < 0.15 else 0.5}")
        print("  weight_decay: 0.001")
    
    if valid['r_square'].std() > 0.2:
        print("• High variation across stations:")
        print("  Add spatial features or increase regularization")
    
    if test_r2 < valid_r2 - 0.05:
        print("• Poor generalization:")
        print("  Increase early_stop or reduce model complexity")
    
    return {
        'train_r2': train_r2,
        'valid_r2': valid_r2, 
        'test_r2': test_r2,
        'overfitting': overfitting
    }

# Run analysis
if __name__ == "__main__":
    # Update timestamp to match your files
    mainfolder = r"D:\1000_SCRIPTS\003_Project002\20250222_GTWR001\8_GNNWTR\Layer_1~CUMDISP\result_summary\gtnnwr_output_20250708_133857"
    timestamp = "20250708_133857"
    results = analyze_gtnnwr_results(mainfolder, timestamp)