#!/usr/bin/env python3
"""
GTNNWR Analysis with External Configuration
Load parameters from JSON config file for easy experimentation.

Usage:
    python gtnnwr_analysis.py [config_file.json]
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from gnnwr import datasets, models

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


def load_config(config_file="gtnnwr_config.json"):
    """Load configuration from JSON file."""
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
        print(f"✓ Configuration loaded from {config_file}")
        return config
    except FileNotFoundError:
        print(f"✗ Config file {config_file} not found")
        print("Creating default config file...")
        create_default_config(config_file)
        return load_config(config_file)
    except json.JSONDecodeError as e:
        print(f"✗ Error parsing config file: {e}")
        return None


def create_default_config(filename="gtnnwr_config.json"):
    """Create default configuration file."""
    default_config = {
        "input_data": {
            "filename": "20250504_GTWR_InputData_MLCW_InSAR_Layer_1.csv",
            "x_column": ["CUMDISP"],
            "y_column": ["Layer_1"],
            "spatial_column": ["X_TWD97", "Y_TWD97"],
            "temp_column": ["monthly"],
            "id_column": ["id"],
        },
        "data_splitting": {
            "test_ratio": 0.15,
            "valid_ratio": 0.1,
            "sample_seed": 42,
        },
        "model_architecture": {
            "dense_layers": [[2], [64, 32]],
            "drop_out": 0.3,
            "batch_norm": True,
            "start_lr": 0.01,
        },
        "training": {
            "optimizer": "Adam",
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
        "output": {
            "save_model": True,
            "save_coefficients": True,
            "save_predictions": True,
            "output_prefix": "gtnnwr",
        },
        "experiment": {
            "name": "Layer1_CUMDISP_baseline",
            "description": "Baseline GTNNWR analysis for Layer_1 ~ CUMDISP relationship",
            "notes": "Initial run with conservative parameters",
        },
    }

    with open(filename, "w") as f:
        json.dump(default_config, f, indent=2)
    print(f"✓ Default config created: {filename}")


def load_data(config):
    """Load and preprocess input data."""
    filename = config["input_data"]["filename"]
    try:
        data = pd.read_csv(filename)
        data["id"] = np.arange(len(data))
        print(f"✓ Loaded {len(data)} observations from {filename}")
        print(
            f"✓ {data['STATION'].nunique()} stations, {data['monthly'].nunique()} time points"
        )
        return data
    except FileNotFoundError:
        print(f"✗ Error: {filename} not found")
        return None


def validate_config(config):
    """Validate configuration parameters."""
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

    # Check GPU setting
    if config["training"]["use_gpu"] == "auto":
        config["training"]["use_gpu"] = torch.cuda.is_available()

    print(f"✓ Configuration validated")
    return True


def run_gtnnwr(data, config):
    """Run GTNNWR model with configuration."""

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

    print(
        f"✓ Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    # Create model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config["experiment"]["name"]
    model_name = f"{exp_name}_{timestamp}"

    # Filter optimizer parameters (remove commented parameters starting with _)
    optimizer_params = {
        k: v
        for k, v in config["optimizer_params"].items()
        if not k.startswith("_")
    }

    # Handle start_lr based on optimizer type
    optimizer_type = config["training"]["optimizer"]
    model_kwargs = {
        "dense_layers": config["model_architecture"]["dense_layers"],
        "drop_out": config["model_architecture"]["drop_out"],
        "optimizer": optimizer_type,
        "optimizer_params": optimizer_params,
        "batch_norm": config["model_architecture"]["batch_norm"],
        "write_path": f"./results_{timestamp}",
        "model_name": model_name,
        "use_gpu": config["training"]["use_gpu"],
    }

    # Only include start_lr for optimizers that require it
    if optimizer_type != "Adadelta":
        model_kwargs["start_lr"] = config["model_architecture"]["start_lr"]

    print("Initializing GTNNWR model...")
    gtnnwr_model = models.GTNNWR(
        train_dataset, val_dataset, test_dataset, **model_kwargs
    )

    arch = config["model_architecture"]["dense_layers"]
    print(f"✓ Model: {arch}, Optimizer: {config['training']['optimizer']}")
    print(f"✓ Using {'GPU' if config['training']['use_gpu'] else 'CPU'}")

    # Train model
    print("Starting training...")
    gtnnwr_model.run(
        max_epoch=config["training"]["max_epoch"],
        early_stop=config["training"]["early_stop"],
    )

    print("✓ Training completed!")
    return gtnnwr_model, timestamp


def save_results(model, timestamp, config):
    """Save model results and configuration."""

    print("Saving results...")

    # Create directories
    output_prefix = config["output"]["output_prefix"]
    output_dir = f"{output_prefix}_output_{timestamp}"
    configs_dir = "configs"
    models_dir = "trained_models"

    for directory in [output_dir, configs_dir, models_dir]:
        os.makedirs(directory, exist_ok=True)

    # Save trained model
    if config["output"]["save_model"]:
        try:
            model_file = os.path.join(
                models_dir, f"{output_prefix}_model_{timestamp}.pkl"
            )
            # Save the entire model using PyTorch's save method
            torch.save(model._model, model_file)
            print(f"✓ Trained model saved: {model_file}")
        except Exception as e:
            print(f"⚠ Could not save trained model: {e}")
            # Try alternative saving method
            try:
                alt_file = os.path.join(
                    models_dir, f"{output_prefix}_model_state_{timestamp}.pkl"
                )
                torch.save(model._model.state_dict(), alt_file)
                print(f"✓ Model state dict saved: {alt_file}")
            except Exception as e2:
                print(f"⚠ Alternative save also failed: {e2}")

    # Save results
    try:
        if config["output"]["save_predictions"]:
            results_data = model.reg_result(only_return=True)
            if results_data is not None:
                results_file = os.path.join(
                    output_dir, f"predictions_{timestamp}.csv"
                )
                results_data.to_csv(results_file, index=False)
                print(f"✓ Predictions saved: {results_file}")

        if config["output"]["save_coefficients"]:
            try:
                coef_data = model.getCoefs()
                coef_file = os.path.join(
                    output_dir, f"coefficients_{timestamp}.csv"
                )
                coef_data.to_csv(coef_file, index=False)
                print(f"✓ Coefficients saved: {coef_file}")
            except:
                print("⚠ Could not extract coefficients")

    except Exception as e:
        print(f"⚠ Error saving results: {e}")

    # Save configuration with experiment info
    config_file = os.path.join(configs_dir, f"config_{timestamp}.json")
    config_copy = config.copy()
    config_copy["run_info"] = {
        "timestamp": timestamp,
        "experiment_name": config["experiment"]["name"],
        "description": config["experiment"]["description"],
    }

    with open(config_file, "w") as f:
        json.dump(config_copy, f, indent=2)

    print(f"✓ Configuration saved: {config_file}")
    print(f"✓ All outputs in: {output_dir}")

    return output_dir


def print_experiment_summary(config, timestamp, output_dir):
    """Print experiment summary."""
    print("\n" + "=" * 50)
    print("EXPERIMENT SUMMARY")
    print("=" * 50)
    print(f"Name: {config['experiment']['name']}")
    print(f"Description: {config['experiment']['description']}")
    print(f"Timestamp: {timestamp}")
    print(
        f"Target: {config['input_data']['y_column'][0]} ~ {config['input_data']['x_column'][0]}"
    )
    print(f"Architecture: {config['model_architecture']['dense_layers']}")
    print(f"Results: {output_dir}")
    print("=" * 50)


def main():
    """Main execution function."""

    # Check for config file argument
    config_file = "gtnnwr_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    print("GTNNWR Analysis with External Config")
    print("=" * 40)

    # Load configuration
    config = load_config(config_file)
    if config is None:
        return

    if not validate_config(config):
        return

    # Load data
    data = load_data(config)
    if data is None:
        return

    # Run model
    model, timestamp = run_gtnnwr(data, config)

    # Save results
    output_dir = save_results(model, timestamp, config)

    # Print summary
    print_experiment_summary(config, timestamp, output_dir)


if __name__ == "__main__":
    main()