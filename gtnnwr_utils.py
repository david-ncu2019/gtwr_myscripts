#!/usr/bin/env python3
"""
Utility Functions for GTNNWR Training Pipeline
Provides modular components for configuration, data loading, and results management.
"""

import json
import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import logging
from gnnwr import datasets


class ConfigManager:
    """
    Manages configuration loading, validation, and merging for GTNNWR training.
    
    Handles JSON config files, parameter validation, and intelligent merging
    of original and resume configurations.
    """
    
    @staticmethod
    def load_config(config_file: str = "gtnnwr_config.json") -> Optional[Dict[str, Any]]:
        """
        Load configuration from JSON file.
        
        Args:
            config_file: Path to configuration JSON file
            
        Returns:
            Configuration dictionary or None if loading fails
        """
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
            print(f"✓ Configuration loaded from {config_file}")
            return config
        except FileNotFoundError:
            print(f"✗ Config file {config_file} not found")
            print("Creating default config file...")
            ConfigManager.create_default_config(config_file)
            return ConfigManager.load_config(config_file)
        except json.JSONDecodeError as e:
            print(f"✗ Error parsing config file: {e}")
            return None
    
    @staticmethod
    def create_default_config(filename: str = "gtnnwr_config.json") -> None:
        """
        Create a default configuration file with reasonable settings.
        
        Args:
            filename: Path where to save the default configuration
        """
        default_config = {
            "input_data": {
                "filename": "data.csv",
                "x_column": ["DIFFDISP"],
                "y_column": ["Layer_1"],
                "spatial_column": ["X_TWD97", "Y_TWD97"],
                "temp_column": ["monthly"],
                "id_column": ["id"],
            },
            "data_splitting": {
                "test_ratio": 0.15,
                "valid_ratio": 0.15,
                "sample_seed": 42,
            },
            "model_architecture": {
                "dense_layers": [[3], [32, 16]],
                "drop_out": 0.3,
                "batch_norm": True,
                "start_lr": 0.001,
            },
            "training": {
                "optimizer": "AdamW",
                "max_epoch": 10000,
                "early_stop": 1000,
                "batch_size": 64,
                "use_gpu": "auto",
            },
            "optimizer_params": {
                "scheduler": "MultiStepLR",
                "scheduler_milestones": [2000, 4000, 6000, 8000],
                "scheduler_gamma": 0.8,
                "weight_decay": 0.0001,
            },
            "checkpoint": {
                "save_frequency": 100,  # Save checkpoint every N epochs
                "keep_count": 5,        # Number of checkpoints to keep
                "auto_resume": True,    # Automatically resume from latest checkpoint
            },
            "output": {
                "save_model": True,
                "save_coefficients": True,
                "save_predictions": True,
                "output_prefix": "gtnnwr",
            },
            "experiment": {
                "name": "default_experiment",
                "description": "Default GTNNWR training experiment",
                "notes": "Created automatically",
            },
        }
        
        with open(filename, "w") as f:
            json.dump(default_config, f, indent=2)
        print(f"✓ Default config created: {filename}")
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """
        Validate configuration parameters for completeness and correctness.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        required_sections = [
            "input_data",
            "data_splitting", 
            "model_architecture",
            "training",
        ]
        
        for section in required_sections:
            if section not in config:
                print(f"✗ Missing required config section: {section}")
                return False
        
        # Validate input data section
        input_data = config["input_data"]
        required_input_keys = ["filename", "x_column", "y_column", "spatial_column", "temp_column"]
        for key in required_input_keys:
            if key not in input_data:
                print(f"✗ Missing required input_data key: {key}")
                return False
        
        # Validate data splitting ratios
        data_split = config["data_splitting"]
        test_ratio = data_split.get("test_ratio", 0.15)
        valid_ratio = data_split.get("valid_ratio", 0.15)
        
        if test_ratio + valid_ratio >= 1.0:
            print(f"✗ Invalid data split: test_ratio ({test_ratio}) + valid_ratio ({valid_ratio}) >= 1.0")
            return False
        
        # Check GPU setting and auto-configure if needed
        if config["training"]["use_gpu"] == "auto":
            config["training"]["use_gpu"] = torch.cuda.is_available()
        
        # Validate model architecture
        arch = config["model_architecture"]
        dense_layers = arch.get("dense_layers", [[3], [32, 16]])
        if not isinstance(dense_layers, list) or len(dense_layers) != 2:
            print(f"✗ Invalid dense_layers format: must be list of two lists")
            return False
        
        print(f"✓ Configuration validated")
        return True
    
    @staticmethod
    def merge_configs(
        original_config: Dict[str, Any], 
        resume_config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Intelligently merge original and resume configurations.
        
        Args:
            original_config: Configuration from checkpoint
            resume_config: New configuration for resuming
            
        Returns:
            Tuple of (merged_config, list_of_changes)
        """
        merged_config = original_config.copy()
        changes = []
        
        # Allow certain training parameters to be updated
        updatable_paths = [
            ("training", "max_epoch"),
            ("training", "early_stop"), 
            ("training", "batch_size"),
            ("optimizer_params", "weight_decay"),
            ("checkpoint", "save_frequency"),
            ("experiment", "notes"),
        ]
        
        for section, key in updatable_paths:
            if (section in resume_config and 
                key in resume_config[section] and
                section in original_config and
                original_config[section].get(key) != resume_config[section][key]):
                
                old_val = original_config[section][key]
                new_val = resume_config[section][key]
                merged_config[section][key] = new_val
                changes.append(f"Updated {section}.{key}: {old_val} → {new_val}")
        
        return merged_config, changes


class DataManager:
    """
    Manages data loading and dataset preparation for GTNNWR training.
    
    Handles CSV loading, preprocessing, and dataset initialization with
    proper train/validation/test splits.
    """
    
    @staticmethod
    def load_data(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Load and preprocess input data from CSV file.
        
        Args:
            config: Configuration dictionary containing input data settings
            
        Returns:
            Loaded DataFrame or None if loading fails
        """
        filename = config["input_data"]["filename"]
        try:
            data = pd.read_csv(filename)
            
            # Add ID column if not present
            if "id" not in data.columns:
                data["id"] = np.arange(len(data))
            
            print(f"✓ Loaded {len(data)} observations from {filename}")
            
            # Print data summary if standard columns exist
            if "STATION" in data.columns and "monthly" in data.columns:
                print(f"✓ {data['STATION'].nunique()} stations, {data['monthly'].nunique()} time points")
            
            return data
            
        except FileNotFoundError:
            print(f"✗ Error: {filename} not found")
            return None
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return None
    
    @staticmethod
    def setup_datasets(
        data: pd.DataFrame, 
        config: Dict[str, Any]
    ) -> Tuple[Any, Any, Any]:
        """
        Initialize train/validation/test datasets from loaded data.
        
        Args:
            data: Input DataFrame
            config: Configuration dictionary
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        print("Setting up datasets...")
        
        train_dataset, val_dataset, test_dataset = datasets.init_dataset(
            data=data,
            test_ratio=config["data_splitting"]["test_ratio"],
            valid_ratio=config["data_splitting"]["valid_ratio"],
            x_column=config["input_data"]["x_column"],
            y_column=config["input_data"]["y_column"],
            spatial_column=config["input_data"]["spatial_column"],
            temp_column=config["input_data"]["temp_column"],
            id_column=config["input_data"]["id_column"],
            use_model="gtnnwr",
            sample_seed=config["data_splitting"]["sample_seed"],
            batch_size=config["training"]["batch_size"],
        )
        
        print(f"✓ Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset


class ResultsManager:
    """
    Manages saving of training results, models, and experiment metadata.
    
    Handles directory creation, file saving, and experiment tracking with
    comprehensive output organization.
    """
    
    def __init__(self, timestamp: str):
        """
        Initialize results manager with timestamp for consistent naming.
        
        Args:
            timestamp: Timestamp string for file naming
        """
        self.timestamp = timestamp
        self.logger = logging.getLogger(__name__)
    
    def setup_output_directories(self, config: Dict[str, Any]) -> Dict[str, Path]:
        """
        Create and organize output directories for experiment results.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary mapping directory types to Path objects
        """
        output_prefix = config["output"]["output_prefix"]
        
        directories = {
            "output": Path("gtnnwr_model_output") / f"{output_prefix}_output_{self.timestamp}",
            "configs": Path("gtnnwr_configs"),
            "models": Path("gtnnwr_trained_models"),
            "checkpoints": Path("checkpoints"),
        }
        
        for dir_type, dir_path in directories.items():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        return directories
    
    def save_model_files(
        self, 
        model, 
        config: Dict[str, Any], 
        directories: Dict[str, Path]
    ) -> None:
        """
        Save trained model files in multiple formats.
        
        Args:
            model: Trained GTNNWR model
            config: Configuration dictionary
            directories: Output directory paths
        """
        if not config["output"]["save_model"]:
            return
        
        output_prefix = config["output"]["output_prefix"]
        models_dir = directories["models"]
        
        try:
            # Save complete model
            # model_file = models_dir / f"{output_prefix}_model_{self.timestamp}.pkl"
            # torch.save(model._model, model_file)
            # self.logger.info(f"Model saved: {model_file}")
            
            # Save model state dict as backup
            state_file = models_dir / f"{output_prefix}_model_state_{self.timestamp}.pkl"
            torch.save(model._model.state_dict(), state_file)
            self.logger.info(f"Model state saved: {state_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
    
    def save_predictions_and_coefficients(
        self, 
        model, 
        config: Dict[str, Any], 
        directories: Dict[str, Path]
    ) -> None:
        """
        Save model predictions and coefficient estimates.
        
        Args:
            model: Trained GTNNWR model
            config: Configuration dictionary
            directories: Output directory paths
        """
        output_dir = directories["output"]
        
        try:
            # Save predictions
            if config["output"]["save_predictions"]:
                results_data = model.reg_result(only_return=True)
                if results_data is not None:
                    results_file = output_dir / f"predictions_{self.timestamp}.csv"
                    results_data.to_csv(results_file, index=False)
                    print(f"✓ Predictions saved: {results_file}")
            
            # Save coefficients
            if config["output"]["save_coefficients"]:
                try:
                    coef_data = model.getCoefs()
                    coef_file = output_dir / f"coefficients_{self.timestamp}.csv"
                    coef_data.to_csv(coef_file, index=False)
                    print(f"✓ Coefficients saved: {coef_file}")
                except Exception as e:
                    print(f"⚠ Could not extract coefficients: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def save_experiment_config(
        self, 
        config: Dict[str, Any], 
        directories: Dict[str, Path],
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save experiment configuration with metadata.
        
        Args:
            config: Configuration dictionary
            directories: Output directory paths
            additional_info: Additional metadata to include
        """
        configs_dir = directories["configs"]
        config_file = configs_dir / f"config_{self.timestamp}.json"
        
        # Create enhanced config with run information
        enhanced_config = config.copy()
        enhanced_config["run_info"] = {
            "timestamp": self.timestamp,
            "experiment_name": config["experiment"]["name"],
            "description": config["experiment"]["description"],
            "pytorch_version": torch.__version__,
            "gpu_available": torch.cuda.is_available(),
        }
        
        if additional_info:
            enhanced_config["run_info"].update(additional_info)
        
        with open(config_file, "w") as f:
            json.dump(enhanced_config, f, indent=2)
        
        print(f"✓ Configuration saved: {config_file}")
    
    def save_all_results(
        self, 
        model, 
        config: Dict[str, Any], 
        additional_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save complete experiment results including model, predictions, and metadata.
        
        Args:
            model: Trained GTNNWR model
            config: Configuration dictionary
            additional_info: Additional run information
            
        Returns:
            Path to main output directory
        """
        print("Saving experiment results...")
        
        # Setup directories
        directories = self.setup_output_directories(config)
        
        # Save all components
        self.save_model_files(model, config, directories)
        self.save_predictions_and_coefficients(model, config, directories)
        self.save_experiment_config(config, directories, additional_info)
        
        output_dir = directories["output"]
        print(f"✓ All outputs saved to: {output_dir}")
        
        return str(output_dir)


class ExperimentTracker:
    """
    Tracks and summarizes experiment information for easy reference.
    
    Provides experiment logging, progress tracking, and summary generation
    for GTNNWR training runs.
    """
    
    @staticmethod
    def print_experiment_summary(
        config: Dict[str, Any], 
        timestamp: str, 
        output_dir: str,
        training_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Print comprehensive experiment summary.
        
        Args:
            config: Configuration dictionary
            timestamp: Experiment timestamp
            output_dir: Output directory path
            training_info: Additional training information
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"Name: {config['experiment']['name']}")
        print(f"Description: {config['experiment']['description']}")
        print(f"Timestamp: {timestamp}")
        print(f"Target: {config['input_data']['y_column'][0]} ~ {config['input_data']['x_column'][0]}")
        print(f"Architecture: {config['model_architecture']['dense_layers']}")
        print(f"Optimizer: {config['training']['optimizer']}")
        print(f"Scheduler: {config['optimizer_params'].get('scheduler', 'None')}")
        
        if training_info:
            print(f"Final Epoch: {training_info.get('final_epoch', 'N/A')}")
            print(f"Best Validation R²: {training_info.get('best_r2', 'N/A'):.4f}")
            print(f"Training Time: {training_info.get('training_time', 'N/A')}")
        
        print(f"Results: {output_dir}")
        print("=" * 60)
    
    @staticmethod
    def setup_logging(log_level: int = logging.INFO) -> None:
        """
        Setup logging configuration for experiment tracking.
        
        Args:
            log_level: Logging level (default: INFO)
        """
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f'gtnnwr_experiment_{datetime.now().strftime("%Y%m%d")}.log')
            ]
        )


def get_device_info() -> Dict[str, Any]:
    """
    Get comprehensive device and environment information.
    
    Returns:
        Dictionary containing device and environment details
    """
    device_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    
    if torch.cuda.is_available():
        device_info["cuda_version"] = torch.version.cuda
        device_info["gpu_names"] = [torch.cuda.get_device_name(i) 
                                   for i in range(torch.cuda.device_count())]
    
    return device_info


def validate_file_paths(config: Dict[str, Any]) -> bool:
    """
    Validate that all required file paths exist and are accessible.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if all paths are valid, False otherwise
    """
    # Check input data file
    input_file = config["input_data"]["filename"]
    if not Path(input_file).exists():
        print(f"✗ Input data file not found: {input_file}")
        return False
    
    # Check write permissions for output directories
    output_dirs = ["gtnnwr_model_output", "gtnnwr_configs", "gtnnwr_trained_models", "checkpoints"]
    for output_dir in output_dirs:
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        except PermissionError:
            print(f"✗ No write permission for directory: {output_dir}")
            return False
    
    return True