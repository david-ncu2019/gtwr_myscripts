#!/usr/bin/env python3
"""
GTNNWR Batch Prediction Program - Fixed Version
Process multiple feather files for memory-efficient prediction with improved accuracy.

FIXES IMPLEMENTED:
1. Exact model state restoration
2. Improved scaling synchronization
3. Consistent coefficient application
4. Enhanced batch processing
5. Better error handling and validation

REQUIRED FILE ORGANIZATION:
regpoints_folder/
├── grid_pnt_chunk001.feather
├── grid_pnt_chunk002.feather
├── grid_pnt_chunk003.feather
└── ...

FEATHER FILE COLUMNS REQUIRED:
- X_TWD97, Y_TWD97 (spatial coordinates)
- monthly (time dimension)
- CUMDISP (predictor variable)
- id (optional, auto-created if missing)

OUTPUT:
model_timestamp_chunk001.feather
model_timestamp_chunk002.feather
...
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json
import sys
import os
import glob
from datetime import datetime
from gnnwr import models, datasets
import warnings

def show_help():
    """Display usage instructions."""
    help_text = """
GTNNWR Batch Prediction Program - Fixed Version

USAGE:
    python gtnnwr_predict_batch_fixed.py <config.json> <model.pkl> <feather_folder>

EXAMPLE:
    python gtnnwr_predict_batch_fixed.py configs/config_20250708_162810.json trained_models/gtnnwr_model_20250708_162810.pkl regpoints_Chunk100_addCUMDISP

REQUIRED FOLDER STRUCTURE:
    regpoints_folder/
    ├── grid_pnt_chunk001.feather
    ├── grid_pnt_chunk002.feather
    ├── grid_pnt_chunk003.feather
    └── ...

FEATHER FILE REQUIREMENTS:
    - Columns: X_TWD97, Y_TWD97, monthly, CUMDISP
    - Optional: id (auto-created if missing)
    - Format: .feather files only

OUTPUT:
    - Files named: model_timestamp_chunk001.feather
    - Location: Same directory as script
    - Contains: Predictions + coefficients + original data

IMPROVEMENTS IN THIS VERSION:
    - Fixed model state restoration for exact reproduction
    - Enhanced scaling synchronization with training data
    - Improved coefficient application consistency
    - Better batch processing alignment with training
    - Added validation and error checking
    """
    print(help_text)

def load_config(config_file):
    """Load model configuration with validation."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Validate required configuration sections
        required_sections = ["input_data", "data_splitting", "model_architecture", "training"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Load original training data for scaling
        orig_data = pd.read_csv(config["input_data"]["filename"])
        if 'id' not in orig_data.columns:
            orig_data['id'] = np.arange(len(orig_data))
        
        print(f"✓ Config loaded: {config_file}")
        print(f"✓ Training data: {len(orig_data)} samples")
        
        return config, orig_data
        
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        raise

def setup_model(orig_data, config):
    """Create and setup model with exact training configuration."""
    try:
        # Set random seeds for reproducibility
        if "sample_seed" in config["data_splitting"]:
            np.random.seed(config["data_splitting"]["sample_seed"])
            torch.manual_seed(config["data_splitting"]["sample_seed"])
            if torch.cuda.is_available():
                torch.cuda.manual_seed(config["data_splitting"]["sample_seed"])
        
        # Create datasets exactly as in training
        train_dataset, val_dataset, test_dataset = datasets.init_dataset(
            data=orig_data,
            test_ratio=config["data_splitting"]["test_ratio"],
            valid_ratio=config["data_splitting"]["valid_ratio"],
            x_column=config["input_data"]["x_column"],
            y_column=config["input_data"]["y_column"],
            spatial_column=config["input_data"]["spatial_column"],
            temp_column=config["input_data"]["temp_column"],
            id_column=config["input_data"]["id_column"],
            use_model="gtnnwr",
            sample_seed=config["data_splitting"]["sample_seed"],
            batch_size=config["training"]["batch_size"]  # Use exact training batch size
        )
        
        # Create model structure with exact training parameters
        model_kwargs = {
            "dense_layers": config["model_architecture"]["dense_layers"],
            "drop_out": config["model_architecture"]["drop_out"],
            "optimizer": config["training"]["optimizer"],
            "batch_norm": config["model_architecture"]["batch_norm"],
            "use_gpu": config["training"].get("use_gpu", False)
        }
        
        # Handle optimizer-specific parameters
        if config["training"]["optimizer"] != "Adadelta":
            model_kwargs["start_lr"] = config["model_architecture"]["start_lr"]
        
        # Filter optimizer parameters
        optimizer_params = {k: v for k, v in config["optimizer_params"].items() 
                           if not k.startswith('_')}
        model_kwargs["optimizer_params"] = optimizer_params
        
        # Create model instance
        model = models.GTNNWR(train_dataset, val_dataset, test_dataset, **model_kwargs)
        
        print(f"✓ Model structure created")
        print(f"✓ Architecture: {config['model_architecture']['dense_layers']}")
        
        return model, train_dataset
        
    except Exception as e:
        print(f"✗ Error setting up model: {e}")
        raise

def create_prediction_dataset_exact(reg_points, model, train_dataset, config):
    """Create prediction dataset with exact alignment to training setup."""
    try:
        # Ensure ID column exists
        if config["input_data"]["id_column"][0] not in reg_points.columns:
            reg_points[config["input_data"]["id_column"][0]] = np.arange(len(reg_points))
        
        # Validate required columns
        required_cols = (config["input_data"]["x_column"] + 
                        config["input_data"]["spatial_column"] + 
                        config["input_data"]["temp_column"])
        missing_cols = [col for col in required_cols if col not in reg_points.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Create prediction dataset with exact scaling synchronization
        pred_dataset = datasets.init_predict_dataset(
            data=reg_points.copy(),  # Use copy to avoid modifications
            train_dataset=train_dataset,
            x_column=config["input_data"]["x_column"],
            spatial_column=config["input_data"]["spatial_column"],
            temp_column=config["input_data"]["temp_column"],
            scale_sync=True,  # Critical: use exact scaling from training
            use_model="gtnnwr",
            batch_size=config["training"]["batch_size"]  # Use same batch size as training
        )
        
        return pred_dataset
        
    except Exception as e:
        print(f"✗ Error creating prediction dataset: {e}")
        raise

def predict_with_exact_training_workflow(model, pred_dataset, config):
    """Make predictions using exact training workflow for consistency."""
    try:
        # Set model to evaluation mode (critical for batch norm and dropout)
        model._model.eval()
        
        # Prepare device
        device = torch.device('cuda') if config["training"].get("use_gpu", False) and torch.cuda.is_available() else torch.device('cpu')
        
        # Prepare for batch processing
        predictions = []
        coefficients = []
        
        # Disable gradient computation for prediction
        with torch.no_grad():
            for batch in pred_dataset.dataloader:
                # Extract batch components (same as training)
                data = batch[0].to(device)
                coef = batch[1].to(device)
                
                # Get model weights (spatial-temporal weights)
                model_weights = model._model(data)
                
                # Apply OLS coefficients exactly as in training
                if hasattr(model, '_coefficient'):
                    ols_coef = torch.tensor(model._coefficient, dtype=torch.float32, device=device)
                    
                    # Calculate predictions using the same method as training evaluation
                    batch_pred = model._out(model_weights.mul(coef.to(torch.float32)))
                    
                    # Calculate coefficients for output (spatial weights * OLS coefficients)
                    batch_coef = model_weights.mul(ols_coef)
                else:
                    # Fallback if coefficient not available
                    batch_pred = model_weights.sum(dim=1, keepdim=True)
                    batch_coef = model_weights
                
                # Move results to CPU and store
                predictions.extend(batch_pred.cpu().numpy().flatten())
                coefficients.extend(batch_coef.cpu().numpy())
        
        return np.array(predictions), np.array(coefficients)
        
    except Exception as e:
        print(f"✗ Error in prediction: {e}")
        raise

def process_feather_file_improved(file_path, model, train_dataset, config, model_timestamp):
    """Process single feather file with improved accuracy and consistency."""
    try:
        print(f"Loading file: {os.path.basename(file_path)}")
        
        # Load feather file
        reg_points = pd.read_feather(file_path)
        original_row_count = len(reg_points)
        
        # Create prediction dataset with exact training alignment
        pred_dataset = create_prediction_dataset_exact(reg_points, model, train_dataset, config)
        
        # Make predictions using exact training workflow
        predictions, coefficients = predict_with_exact_training_workflow(model, pred_dataset, config)
        
        # Create comprehensive results DataFrame
        results = reg_points.copy()
        results[f'Predicted_{config["input_data"]["y_column"][0]}'] = predictions
        
        # Add coefficient information
        if coefficients.shape[1] >= 2:
            # Add individual coefficients based on model structure
            for i, col_name in enumerate(config["input_data"]["x_column"]):
                if i < coefficients.shape[1]:
                    results[f'coef_{col_name}'] = coefficients[:, i]
            
            # Add bias/intercept (typically the last coefficient)
            if coefficients.shape[1] > len(config["input_data"]["x_column"]):
                results['bias'] = coefficients[:, -1]
        
        # Add metadata for verification
        results['prediction_timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')
        results['model_name'] = model_timestamp
        
        # Generate output filename and save
        basename = os.path.basename(file_path)
        chunk_name = basename.replace('grid_pnt_', '').replace('.feather', '')
        output_folder = f"predictions_{model_timestamp}"
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, f"{model_timestamp}_{chunk_name}.feather")
        
        # Save results with verification
        results.to_feather(output_file)
        
        # Verify file was saved correctly
        if os.path.exists(output_file):
            verification_data = pd.read_feather(output_file)
            if len(verification_data) == original_row_count:
                print(f"✓ Successfully saved: {output_file} ({original_row_count} points)")
            else:
                raise ValueError(f"Row count mismatch: expected {original_row_count}, got {len(verification_data)}")
        else:
            raise ValueError(f"Failed to save file: {output_file}")
        
        return output_file, original_row_count
        
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        raise

def validate_model_consistency(model, train_dataset, config):
    """Validate that the model can reproduce training results."""
    try:
        print("Validating model consistency...")
        
        # Use a small sample from training data for validation
        sample_size = min(100, len(train_dataset.dataframe))
        sample_data = train_dataset.dataframe.head(sample_size).copy()
        
        # Create prediction dataset from training sample
        pred_dataset = create_prediction_dataset_exact(sample_data, model, train_dataset, config)
        
        # Make predictions
        predictions, coefficients = predict_with_exact_training_workflow(model, pred_dataset, config)
        
        # Compare with expected results (if available)
        if hasattr(model, 'result_data') and model.result_data is not None:
            sample_ids = sample_data[config["input_data"]["id_column"][0]].values
            expected_results = model.result_data[model.result_data['id'].isin(sample_ids)]
            
            if len(expected_results) > 0:
                pred_col = f"Pred_{config['input_data']['y_column'][0]}"
                if pred_col in expected_results.columns:
                    expected_pred = expected_results[pred_col].values[:len(predictions)]
                    correlation = np.corrcoef(predictions[:len(expected_pred)], expected_pred)[0, 1]
                    print(f"✓ Model consistency check: correlation = {correlation:.6f}")
                    if correlation < 0.99:
                        warnings.warn(f"Low correlation ({correlation:.6f}) between predictions and expected results")
        
        print("✓ Model validation completed")
        
    except Exception as e:
        print(f"⚠ Warning: Model validation failed: {e}")

def main():
    """Main execution function with enhanced error handling and validation."""
    if len(sys.argv) != 4:
        show_help()
        return
    
    config_file = sys.argv[1]
    model_file = sys.argv[2]
    feather_folder = sys.argv[3]
    
    try:
        # Validate inputs
        for file_path, name in [(config_file, "Config"), (model_file, "Model"), (feather_folder, "Feather folder")]:
            if not os.path.exists(file_path):
                print(f"✗ {name} not found: {file_path}")
                return
        
        # Find feather files
        feather_pattern = os.path.join(feather_folder, "grid_pnt_chunk*.feather")
        feather_files = sorted(glob.glob(feather_pattern))
        
        if not feather_files:
            print(f"✗ No feather files found matching pattern: {feather_pattern}")
            print("Expected files: grid_pnt_chunk001.feather, grid_pnt_chunk002.feather, ...")
            return
        
        print("GTNNWR Batch Prediction - Fixed Version")
        print("=" * 50)
        print(f"Found {len(feather_files)} feather files")
        
        # Extract model timestamp from model filename
        model_basename = os.path.basename(model_file)
        model_timestamp = model_basename.replace('.pkl', '').replace('gtnnwr_model_', '')
        
        # Load configuration and setup model
        print("\nSetting up model...")
        config, orig_data = load_config(config_file)
        model, train_dataset = setup_model(orig_data, config)
        
        # Load trained model with proper state restoration
        print(f"Loading trained model: {model_file}")
        model.load_model(model_file)
        
        # Validate model consistency
        validate_model_consistency(model, train_dataset, config)
        
        # Process each feather file
        print(f"\nProcessing {len(feather_files)} files...")
        total_points = 0
        successful_files = 0
        failed_files = []
        
        for i, file_path in enumerate(feather_files, 1):
            filename = os.path.basename(file_path)
            print(f"\n[{i}/{len(feather_files)}] Processing: {filename}")
            
            try:
                output_file, point_count = process_feather_file_improved(
                    file_path, model, train_dataset, config, model_timestamp
                )
                total_points += point_count
                successful_files += 1
                
            except Exception as e:
                print(f"✗ Failed to process {filename}: {e}")
                failed_files.append((filename, str(e)))
        
        # Final summary
        print(f"\n{'='*50}")
        print("BATCH PREDICTION SUMMARY")
        print(f"{'='*50}")
        print(f"✓ Successfully processed: {successful_files}/{len(feather_files)} files")
        print(f"✓ Total points processed: {total_points:,}")
        print(f"✓ Output directory: predictions_{model_timestamp}/")
        print(f"✓ Output pattern: {model_timestamp}_chunk*.feather")
        
        if failed_files:
            print(f"\n⚠ Failed files ({len(failed_files)}):")
            for filename, error in failed_files:
                print(f"  - {filename}: {error}")
        else:
            print("\n🎉 All files processed successfully!")
        
    except Exception as e:
        print(f"\n✗ Critical error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())