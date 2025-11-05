#!/usr/bin/env python3
"""
Checkpoint Manager for GTNNWR Training
Provides comprehensive checkpoint saving and loading functionality for resuming training.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import logging


class CheckpointManager:
    """
    Manages checkpoint saving and loading for GTNNWR model training.
    
    Handles complete training state preservation including:
    - Model weights and architecture
    - Optimizer state (momentum, learning rates)
    - Scheduler state (step counts, learning rate schedules)
    - Training metadata (epoch, metrics, loss history)
    - Random states for reproducibility
    """
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def save_checkpoint(
        self, 
        model, 
        epoch: int, 
        config: Dict[str, Any],
        checkpoint_name: Optional[str] = None
    ) -> str:
        """
        Save complete training state to checkpoint file.
        
        Args:
            model: GTNNWR model instance
            epoch: Current training epoch
            config: Training configuration dictionary
            checkpoint_name: Optional custom checkpoint name
            
        Returns:
            Path to saved checkpoint file
        """
        # Generate checkpoint filename
        if checkpoint_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_epoch_{epoch}_{timestamp}.pt"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Gather complete training state
        checkpoint_data = {
            # Training progress
            'epoch': epoch,
            'current_epoch': model._epoch,
            'no_update_epoch': model._noUpdateEpoch,
            
            # Model state
            'model_state_dict': model._model.state_dict() if hasattr(model._model, 'state_dict') else None,
            'model_architecture': {
                'dense_layers': model._dense_layers,
                'insize': model._insize,
                'outsize': model._outsize,
                'drop_out': model._drop_out,
                'batch_norm': model._batch_norm,
            },
            
            # Optimizer and scheduler state
            'optimizer_state_dict': model._optimizer.state_dict() if model._optimizer else None,
            'optimizer_name': model._optimizer_name,
            'scheduler_state_dict': model._scheduler.state_dict() if model._scheduler else None,
            
            # Training metrics and history
            'best_r2': float(model._bestr2) if hasattr(model._bestr2, 'item') else model._bestr2,
            'best_train_r2': float(model._besttrainr2) if hasattr(model._besttrainr2, 'item') else model._besttrainr2,
            'train_loss_history': model._trainLossList,
            'valid_loss_history': model._validLossList,
            
            # Model configuration
            'start_lr': model._start_lr,
            'coefficient': model._coefficient,
            'model_name': model._modelName,
            'model_save_path': model._modelSavePath,
            
            # Training configuration
            'config': config,
            
            # Random states for reproducibility
            'random_states': {
                'torch_rng_state': torch.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            },
            
            # Metadata
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
            'device': str(next(model._model.parameters()).device) if hasattr(model._model, 'parameters') else 'cpu',
        }
        
        try:
            torch.save(checkpoint_data, checkpoint_path)
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Also save a "latest" checkpoint for easy resuming
            latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
            torch.save(checkpoint_data, latest_path)
            
            return str(checkpoint_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load checkpoint data from file.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary containing complete training state
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def restore_model_state(self, model, checkpoint_data: Dict[str, Any]) -> None:
        """
        Restore model training state from checkpoint data.
        
        Args:
            model: GTNNWR model instance to restore
            checkpoint_data: Checkpoint data dictionary
        """
        try:
            # Restore model weights
            if checkpoint_data.get('model_state_dict'):
                model._model.load_state_dict(checkpoint_data['model_state_dict'])
                self.logger.info("Model weights restored")
            
            # Restore optimizer state
            if checkpoint_data.get('optimizer_state_dict') and model._optimizer:
                model._optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                self.logger.info("Optimizer state restored")
            
            # Restore scheduler state
            if checkpoint_data.get('scheduler_state_dict') and model._scheduler:
                model._scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                self.logger.info("Scheduler state restored")
            
            # Restore training progress
            model._epoch = checkpoint_data.get('current_epoch', 0)
            model._noUpdateEpoch = checkpoint_data.get('no_update_epoch', 0)
            
            # Restore metrics
            model._bestr2 = checkpoint_data.get('best_r2', float('-inf'))
            model._besttrainr2 = checkpoint_data.get('best_train_r2', float('-inf'))
            
            # Restore training history
            model._trainLossList = checkpoint_data.get('train_loss_history', [])
            model._validLossList = checkpoint_data.get('valid_loss_history', [])
            
            # Restore random states for reproducibility
            if 'random_states' in checkpoint_data:
                random_states = checkpoint_data['random_states']
                
                if random_states.get('torch_rng_state') is not None:
                    torch.set_rng_state(random_states['torch_rng_state'])
                
                if random_states.get('numpy_rng_state') is not None:
                    np.random.set_state(random_states['numpy_rng_state'])
                
                if random_states.get('cuda_rng_state') is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state(random_states['cuda_rng_state'])
            
            self.logger.info("Model state fully restored from checkpoint")
            
        except Exception as e:
            self.logger.error(f"Failed to restore model state: {e}")
            raise
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """
        Find the most recent checkpoint in the checkpoint directory.
        
        Returns:
            Path to latest checkpoint file, or None if no checkpoints found
        """
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        if latest_path.exists():
            return str(latest_path)
        
        # Fallback: find most recent checkpoint by filename
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoint_files:
            return None
        
        # Sort by modification time
        latest_file = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        return str(latest_file)
    
    def validate_checkpoint_compatibility(
        self, 
        checkpoint_data: Dict[str, Any], 
        current_config: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that checkpoint is compatible with current configuration.
        
        Args:
            checkpoint_data: Loaded checkpoint data
            current_config: Current training configuration
            
        Returns:
            Tuple of (is_compatible, list_of_warnings)
        """
        warnings = []
        is_compatible = True
        
        # Check model architecture compatibility
        if 'model_architecture' in checkpoint_data:
            arch = checkpoint_data['model_architecture']
            current_arch = current_config.get('model_architecture', {})
            
            if arch.get('dense_layers') != current_arch.get('dense_layers'):
                warnings.append("Model architecture differs between checkpoint and config")
                is_compatible = False
            
            if arch.get('drop_out') != current_arch.get('drop_out'):
                warnings.append("Dropout rate differs between checkpoint and config")
        
        # Check optimizer compatibility
        checkpoint_optimizer = checkpoint_data.get('optimizer_name')
        current_optimizer = current_config.get('training', {}).get('optimizer')
        
        if checkpoint_optimizer != current_optimizer:
            warnings.append(f"Optimizer changed: {checkpoint_optimizer} → {current_optimizer}")
        
        # Check data compatibility
        checkpoint_config = checkpoint_data.get('config', {})
        input_data_keys = ['x_column', 'y_column', 'spatial_column', 'temp_column']
        
        for key in input_data_keys:
            checkpoint_val = checkpoint_config.get('input_data', {}).get(key)
            current_val = current_config.get('input_data', {}).get(key)
            
            if checkpoint_val != current_val:
                warnings.append(f"Input data configuration differs: {key}")
                is_compatible = False
        
        return is_compatible, warnings
    
    def cleanup_old_checkpoints(self, keep_count: int = 5) -> None:
        """
        Remove old checkpoint files, keeping only the most recent ones.
        
        Args:
            keep_count: Number of recent checkpoints to keep
        """
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        
        if len(checkpoint_files) <= keep_count:
            return
        
        # Sort by modification time, newest first
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # Remove old checkpoints
        for old_checkpoint in checkpoint_files[keep_count:]:
            try:
                old_checkpoint.unlink()
                self.logger.info(f"Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                self.logger.warning(f"Failed to remove old checkpoint {old_checkpoint}: {e}")