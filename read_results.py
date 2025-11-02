import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the main results file (use your actual timestamp)
results = pd.read_csv(r"D:\1000_SCRIPTS\003_Project002\20250222_GTWR001\8_GNNWTR\gtnnwr_output_20250707_225421\coefficients_20250707_225421.csv")

print("Dataset Overview:")
print(f"Shape: {results.shape}")
print(f"Columns: {list(results.columns)}")

# Basic performance metrics
actual = results['Layer_1']
predicted = results['Pred_Layer_1']
r2 = 1 - np.sum((actual - predicted)**2) / np.sum((actual - actual.mean())**2)
rmse = np.sqrt(np.mean((actual - predicted)**2))

print(f"\nModel Performance:")
print(f"R²: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# Check data splits
print(f"\nData Distribution:")
print(results['dataset_belong'].value_counts())

# Key result columns to examine:
print(f"\nKey Results Preview:")
key_cols = ['STATION', 'monthly', 'Layer_1', 'Pred_Layer_1', 'coef_CUMDISP', 'bias']
print(results[key_cols].head())

# Spatial variation of coefficients
print(f"\nCoefficient Statistics:")
print(f"CUMDISP coefficient range: {results['coef_CUMDISP'].min():.4f} to {results['coef_CUMDISP'].max():.4f}")
print(f"Coefficient std: {results['coef_CUMDISP'].std():.4f}")

# Station-level analysis
station_summary = results.groupby('STATION').agg({
    'coef_CUMDISP': ['mean', 'std'],
    'Layer_1': 'mean',
    'Pred_Layer_1': 'mean'
}).round(4)

print(f"\nStation-level Summary (first 5 stations):")
print(station_summary.head())

# Quick visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Predicted vs Actual
axes[0,0].scatter(actual, predicted, alpha=0.6)
axes[0,0].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
axes[0,0].set_xlabel('Actual Layer_1')
axes[0,0].set_ylabel('Predicted Layer_1')
axes[0,0].set_title(f'Predictions vs Actual (R² = {r2:.3f})')

# 2. Spatial distribution of coefficients
scatter = axes[0,1].scatter(results['X_TWD97'], results['Y_TWD97'], 
                           c=results['coef_CUMDISP'], cmap='RdBu_r', alpha=0.7)
axes[0,1].set_xlabel('X_TWD97')
axes[0,1].set_ylabel('Y_TWD97')
axes[0,1].set_title('Spatial Distribution of CUMDISP Coefficients')
plt.colorbar(scatter, ax=axes[0,1])

# 3. Temporal evolution
monthly_coef = results.groupby('monthly')['coef_CUMDISP'].mean()
axes[1,0].plot(monthly_coef.index, monthly_coef.values)
axes[1,0].set_xlabel('Month')
axes[1,0].set_ylabel('Average CUMDISP Coefficient')
axes[1,0].set_title('Temporal Evolution of Coefficients')

# 4. Residuals
residuals = actual - predicted
axes[1,1].scatter(predicted, residuals, alpha=0.6)
axes[1,1].axhline(y=0, color='red', linestyle='--')
axes[1,1].set_xlabel('Predicted Layer_1')
axes[1,1].set_ylabel('Residuals')
axes[1,1].set_title('Residual Plot')

plt.tight_layout()
plt.show()

# Export station summaries
station_summary.to_csv('station_analysis.csv')
print(f"\n✓ Station summary saved to station_analysis.csv")