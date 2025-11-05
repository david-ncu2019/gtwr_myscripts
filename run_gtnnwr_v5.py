#!/usr/bin/env python3
"""
GTNNWR Training with Checkpoint Support
Enhanced training script with checkpoint saving/loading and modular architecture.

Usage:
    python run_gtnnwr.py [config_file.json] [--resume] [--checkpoint_path PATH]
    
Examples:
    # Start fresh training
    python run_gtnnwr.py model_train_config.json
    
    # Resume from latest checkpoint
    python run_gtnnwr.py model_train_config.json --resume
    
    # Resume from specific checkpoint
    python run_gtnnwr.py model_train_config.json --resume --checkpoint_path checkpoints/checkpoint_epoch_500.pt
"""

import sys
import argparse
import time
from datetime import datetime
from typing import Optional, Dict, Any, List

import torch
import numpy as np
from gnnwr import models

from checkpoint_manager import CheckpointManager
from gtnnwr_utils import (
    ConfigManager, 
    DataManager, 
    ResultsManager, 
    ExperimentTracker,
    get_device_info,
    validate_file_paths
)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


def setup_model_from_config(
    train_dataset, 
    val_dataset, 
    test_dataset, 
    config: Dict[str, Any], 
    timestamp: str
) -> models.GTNNWR:
    """
    Initialize GTNNWR model from configuration.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset  
        test_dataset: Test dataset
        config: Configuration dictionary
        timestamp: Timestamp for model naming
        
    Returns:
        Initialized GTNNWR model
    """
    # Generate model name
    exp_name = config["experiment"]["name"]
    model_name = f"{exp_name}_{timestamp}"
    
    # Filter optimizer parameters (remove commented parameters)
    optimizer_params = {
        k: v for k, v in config["optimizer_params"].items()
        if not k.startswith("_")
    }
    
    # Prepare model arguments
    optimizer_type = config["training"]["optimizer"]
    model_kwargs = {
        "dense_layers": config["model_architecture"]["dense_layers"],
        "drop_out": config["model_architecture"]["drop_out"],
        "optimizer": optimizer_type,
        "optimizer_params": optimizer_params,
        "batch_norm": config["model_architecture"]["batch_norm"],
        "model_save_path": "./gtnnwr_trained_models",
        "write_path": f"./gtnnwr_model_results/results_{timestamp}",
        "model_name": model_name,
        "use_gpu": config["training"]["use_gpu"],
    }
    
    # Add start_lr for optimizers that require it
    if optimizer_type != "Adadelta":
        model_kwargs["start_lr"] = config["model_architecture"]["start_lr"]
    
    print("Initializing GTNNWR model...")
    gtnnwr_model = models.GTNNWR(
        train_dataset, val_dataset, test_dataset, **model_kwargs
    )
    
    arch = config["model_architecture"]["dense_layers"]
    print(f"✓ Model: {arch}, Optimizer: {optimizer_type}")
    print(f"✓ Using {'GPU' if config['training']['use_gpu'] else 'CPU'}")
    
    return gtnnwr_model


def train_with_checkpointing(
    model: models.GTNNWR, 
    config: Dict[str, Any], 
    checkpoint_manager: CheckpointManager,
    resume_from_epoch: int = 0
) -> Dict[str, Any]:
    """
    Train model with automatic checkpoint saving.
    
    Args:
        model: GTNNWR model to train
        config: Configuration dictionary
        checkpoint_manager: Checkpoint manager instance
        resume_from_epoch: Starting epoch for resume
        
    Returns:
        Dictionary with training statistics
    """
    print("Starting training with checkpoint support...")
    
    max_epoch = config["training"]["max_epoch"]
    early_stop = config["training"]["early_stop"]
    checkpoint_freq = config.get("checkpoint", {}).get("save_frequency", 100)
    
    start_time = time.time()
    
    # Custom training loop with checkpoint saving
    try:
        # Hook into the original run method but with checkpoint callbacks
        original_valid = model._GTNNWR__valid if hasattr(model, '_GTNNWR__valid') else model.__class__.__dict__.get('_GTNNWR__valid')
        
        # Store reference for checkpoint saving
        def checkpoint_callback():
            current_epoch = model._epoch
            if current_epoch > 0 and current_epoch % checkpoint_freq == 0:
                checkpoint_manager.save_checkpoint(
                    model, 
                    current_epoch, 
                    config,
                    f"checkpoint_epoch_{current_epoch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                )
                
                # Cleanup old checkpoints
                checkpoint_manager.cleanup_old_checkpoints(
                    keep_count=config.get("checkpoint", {}).get("keep_count", 5)
                )
        
        # Monkey patch validation method to include checkpointing
        def enhanced_valid(self):
            # Call original validation
            if hasattr(self, '_GTNNWR__valid'):
                self._GTNNWR__valid()
            else:
                # Fallback to basic validation logic
                pass
                
            # Save checkpoint periodically
            checkpoint_callback()
        
        # Replace validation method temporarily
        original_method = getattr(model, f"_{model.__class__.__name__}__valid", None)
        if original_method:
            setattr(model, f"_{model.__class__.__name__}__valid", lambda: enhanced_valid(model))
        
        # Run training
        model.run(
            max_epoch=max_epoch,
            early_stop=early_stop
        )
        
        # Restore original method
        if original_method:
            setattr(model, f"_{model.__class__.__name__}__valid", original_method)
    
    except Exception as e:
        print(f"Training interrupted: {e}")
        # Save emergency checkpoint
        checkpoint_manager.save_checkpoint(
            model, 
            model._epoch, 
            config,
            f"emergency_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        )
        raise
    
    training_time = time.time() - start_time
    
    # Load best model for final evaluation
    print("Loading best model for final evaluation...")
    best_model_path = f"{model._modelSavePath}/{model._modelName}.pkl"
    model.load_model(best_model_path)
    print(f"✓ Loaded best model from: {best_model_path}")
    
    # Save final checkpoint
    final_checkpoint_name = f"final_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    checkpoint_manager.save_checkpoint(model, model._epoch, config, final_checkpoint_name)
    
    training_stats = {
        "final_epoch": model._epoch,
        "best_r2": float(model._bestr2) if hasattr(model._bestr2, 'item') else model._bestr2,
        "training_time": f"{training_time:.2f} seconds",
        "training_time_minutes": f"{training_time/60:.2f} minutes",
    }
    
    return training_stats


def run_experiment(
    config_file: str, 
    resume: bool = False, 
    checkpoint_path: Optional[str] = None
) -> None:
    """
    Run complete GTNNWR experiment with optional checkpoint resume.
    
    Args:
        config_file: Path to configuration JSON file
        resume: Whether to resume from checkpoint
        checkpoint_path: Specific checkpoint path (optional)
    """
    # Setup logging
    ExperimentTracker.setup_logging()
    
    # Load and validate configuration
    config = ConfigManager.load_config(config_file)
    if config is None:
        return
    
    if not ConfigManager.validate_config(config):
        return
    
    if not validate_file_paths(config):
        return
    
    # Print environment information
    device_info = get_device_info()
    print(f"✓ Environment: PyTorch {device_info['pytorch_version']}, "
          f"CUDA {'available' if device_info['cuda_available'] else 'not available'}")
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager()
    
    # Handle resume logic
    resume_from_epoch = 0
    original_config = None
    
    if resume:
        if checkpoint_path is None:
            checkpoint_path = checkpoint_manager.find_latest_checkpoint()
        
        if checkpoint_path is None:
            print("✗ No checkpoint found for resume. Starting fresh training.")
            resume = False
        else:
            print(f"✓ Found checkpoint: {checkpoint_path}")
            
            try:
                checkpoint_data = checkpoint_manager.load_checkpoint(checkpoint_path)
                original_config = checkpoint_data.get('config', {})
                resume_from_epoch = checkpoint_data.get('epoch', 0)
                
                # Validate checkpoint compatibility
                is_compatible, warnings = checkpoint_manager.validate_checkpoint_compatibility(
                    checkpoint_data, config
                )
                
                if warnings:
                    print("⚠ Checkpoint compatibility warnings:")
                    for warning in warnings:
                        print(f"  - {warning}")
                
                if not is_compatible:
                    print("✗ Checkpoint incompatible with current config. Starting fresh training.")
                    resume = False
                else:
                    # Merge configurations
                    config, changes = ConfigManager.merge_configs(original_config, config)
                    if changes:
                        print("✓ Configuration updates for resume:")
                        for change in changes:
                            print(f"  - {change}")
                    
                    print(f"✓ Resuming from epoch {resume_from_epoch}")
                    
            except Exception as e:
                print(f"✗ Failed to load checkpoint: {e}")
                print("Starting fresh training.")
                resume = False
    
    # Load data and setup datasets
    data = DataManager.load_data(config)
    if data is None:
        return
    
    train_dataset, val_dataset, test_dataset = DataManager.setup_datasets(data, config)
    
    # Create model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = setup_model_from_config(train_dataset, val_dataset, test_dataset, config, timestamp)
    
    # Restore model state if resuming
    if resume and checkpoint_path:
        try:
            checkpoint_data = checkpoint_manager.load_checkpoint(checkpoint_path)
            checkpoint_manager.restore_model_state(model, checkpoint_data)
            print("✓ Model state restored from checkpoint")
        except Exception as e:
            print(f"✗ Failed to restore model state: {e}")
            print("Starting fresh training.")
    
    # Train model
    training_stats = train_with_checkpointing(
        model, config, checkpoint_manager, resume_from_epoch
    )
    
    # Save results
    results_manager = ResultsManager(timestamp)
    additional_info = {
        "resumed_from_checkpoint": resume,
        "checkpoint_path": checkpoint_path if resume else None,
        "device_info": device_info,
        **training_stats
    }
    
    output_dir = results_manager.save_all_results(model, config, additional_info)
    
    # Print experiment summary
    ExperimentTracker.print_experiment_summary(config, timestamp, output_dir, training_stats)
    
    print("✓ Experiment completed successfully!")


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="GTNNWR Training with Checkpoint Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_gtnnwr.py config.json                    # Fresh training
  python run_gtnnwr.py config.json --resume           # Resume from latest
  python run_gtnnwr.py config.json --resume --checkpoint_path checkpoints/checkpoint_epoch_500.pt
        """
    )
    
    parser.add_argument(
        "config_file", 
        nargs="?", 
        default="gtnnwr_config.json",
        help="Path to configuration JSON file (default: gtnnwr_config.json)"
    )
    
    parser.add_argument(
        "--resume", 
        action="store_true",
        help="Resume training from latest checkpoint"
    )
    
    parser.add_argument(
        "--checkpoint_path", 
        type=str,
        help="Specific checkpoint file to resume from"
    )
    
    args = parser.parse_args()
    
    print("GTNNWR Training with Checkpoint Support")
    print("=" * 45)
    
    try:
        run_experiment(args.config_file, args.resume, args.checkpoint_path)
    except KeyboardInterrupt:
        print("\n✗ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()