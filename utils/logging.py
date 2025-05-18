import os
import json
import time
import torch
from datetime import datetime
import numpy as np 

class Logger:
    """
    Logger for training process.
    """
    def __init__(self, log_dir):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
        
        # Create metrics file
        self.metrics_file = os.path.join(log_dir, f"metrics_{timestamp}.json")
        self.metrics = []
        
        # Initialize files
        with open(self.log_file, 'w') as f:
            f.write(f"Training started at {timestamp}\n")
            f.write(f"Log directory: {log_dir}\n\n")
    
    def log_metrics(self, epoch, train_metrics, val_metrics, lr, epoch_time):
        """
        Log metrics for one epoch.
        
        Args:
            epoch: Current epoch
            train_metrics: Dictionary with training metrics
            val_metrics: Dictionary with validation metrics
            lr: Current learning rate
            epoch_time: Time taken for the epoch
        """
        # Create epoch summary
        summary = {
            'epoch': epoch,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'train': self._convert_to_serializable(train_metrics),
            'val': self._convert_to_serializable(val_metrics),
            'learning_rate': float(lr),
            'epoch_time': float(epoch_time)
        }
        
        # Add to metrics list
        self.metrics.append(summary)
        
        # Save metrics to file
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Log to text file
        with open(self.log_file, 'a') as f:
            f.write(f"\nEpoch {epoch+1} - Time: {epoch_time:.1f}s - LR: {lr:.6f}\n")
            
            # Log training metrics
            f.write("  Training:\n")
            f.write(f"    Loss: {train_metrics['loss']['total']:.4f} ")
            f.write(f"(Reg: {train_metrics['loss']['regression']:.4f}, ")
            f.write(f"Cls: {train_metrics['loss']['classification']:.4f})\n")
            f.write(f"    MAE: {train_metrics['regression']['mae']:.0f} ")
            f.write(f"RMSE: {train_metrics['regression']['rmse']:.0f} ")
            f.write(f"Accuracy: {train_metrics['classification']['accuracy']:.4f}\n")
            
            # Log validation metrics
            f.write("  Validation:\n")
            f.write(f"    Loss: {val_metrics['loss']['total']:.4f} ")
            f.write(f"(Reg: {val_metrics['loss']['regression']:.4f}, ")
            f.write(f"Cls: {val_metrics['loss']['classification']:.4f})\n")
            f.write(f"    MAE: {val_metrics['regression']['mae']:.0f} ")
            f.write(f"RMSE: {val_metrics['regression']['rmse']:.0f} ")
            f.write(f"Accuracy: {val_metrics['classification']['accuracy']:.4f}\n")

    def _convert_to_serializable(self, obj):
        """
        Convert all values in a nested dictionary to JSON serializable types.
        
        Args:
            obj: Dictionary or value to convert
            
        Returns:
            Converted object
        """
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(x) for x in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_to_serializable(obj.tolist())
        else:
            return obj


def save_checkpoint(model, optimizer, epoch, metrics, save_dir, filename='checkpoint.pt'):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        metrics: Validation metrics
        save_dir: Directory to save checkpoint
        filename: Checkpoint filename
    """
    # Make sure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Create checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics
    }
    
    # Save checkpoint
    torch.save(checkpoint, os.path.join(save_dir, filename))


class ContrastiveLogger:
    """
    Logger for contrastive training process.
    """
    def __init__(self, log_dir):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"contrastive_log_{timestamp}.txt")
        
        # Create metrics file
        self.metrics_file = os.path.join(log_dir, f"contrastive_metrics_{timestamp}.json")
        self.metrics = []
        
        # Initialize files
        with open(self.log_file, 'w') as f:
            f.write(f"Contrastive training started at {timestamp}\n")
            f.write(f"Log directory: {log_dir}\n\n")
    
    def log_metrics(self, epoch, train_metrics, val_metrics, lr, epoch_time):
        """
        Log metrics for one epoch.
        
        Args:
            epoch: Current epoch
            train_metrics: Dictionary with training metrics
            val_metrics: Dictionary with validation metrics
            lr: Current learning rate
            epoch_time: Time taken for the epoch
        """
        # Create epoch summary
        summary = {
            'epoch': epoch,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'train': self._convert_to_serializable(train_metrics),
            'val': self._convert_to_serializable(val_metrics),
            'learning_rate': float(lr),
            'epoch_time': float(epoch_time)
        }
        
        # Add to metrics list
        self.metrics.append(summary)
        
        # Save metrics to file
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Log to text file
        with open(self.log_file, 'a') as f:
            f.write(f"\nEpoch {epoch+1} - Time: {epoch_time:.1f}s - LR: {lr:.6f}\n")
            
            # Log training metrics
            f.write("  Training:\n")
            f.write(f"    Loss: {train_metrics['loss']:.4f} ")
            f.write(f"Accuracy: {train_metrics['accuracy']:.4f}\n")
            
            # Log validation metrics
            f.write("  Validation:\n")
            f.write(f"    Loss: {val_metrics['loss']:.4f} ")
            f.write(f"Accuracy: {val_metrics['accuracy']:.4f}\n")

    def _convert_to_serializable(self, obj):
        """
        Convert all values in a nested dictionary to JSON serializable types.
        
        Args:
            obj: Dictionary or value to convert
            
        Returns:
            Converted object
        """
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(x) for x in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_to_serializable(obj.tolist())
        else:
            return obj