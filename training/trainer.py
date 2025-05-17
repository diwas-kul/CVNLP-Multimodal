import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict
import json
import gc
from datetime import datetime

from utils.metrics import compute_metrics, compute_detailed_metrics
from utils.logging import Logger, save_checkpoint
from training.losses import CombinedLoss

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('Agg')

class Trainer:
    """
    Trainer class for property price prediction model.
    """
    def __init__(self, model, train_loader, val_loader, config, preprocessing_params, experiment_dir, multimodal=False, multi_lr=False):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            config: Configuration dictionary
            preprocessing_params: Dictionary with preprocessing parameters
            experiment_dir: Directory to save logs and checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.preprocessing_params = preprocessing_params
        self.experiment_dir = experiment_dir
        self.device = next(model.parameters()).device
        
        # Create logger
        self.logger = Logger(experiment_dir)
        
        # Loss function
        self.criterion = CombinedLoss(
            regression_weight=config['train']['regression_weight'],
            classification_weight=config['train']['classification_weight']
        )
        
        # Optimizer
        if multimodal and multi_lr:
            image_encoder_params = list(model.image_encoder.parameters())
            text_encoder_params = list(model.text_encoder.parameters())
            fusion_params = [p for n, p in model.named_parameters() 
                        if not any(n.startswith(x) for x in ['image_encoder', 'text_encoder'])]

            self.optimizer = optim.AdamW([
                {'params': image_encoder_params, 'lr': config['model']['image_lr']},
                {'params': text_encoder_params, 'lr': config['model']['text_lr']},
                {'params': fusion_params, 'lr': config['model']['fusion_lr']}
            ], weight_decay=config['train']['weight_decay'])
            pass
        else:
            self.optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=config['train']['learning_rate'],
                weight_decay=config['train']['weight_decay']
            )
        
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=config['train']['scheduler_factor'],
            patience=config['train']['scheduler_patience'],
        )
        
        # Save configuration
        self._save_config()
        
        # Initialize best metrics
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.early_stopping_patience = config['train']['early_stopping_patience']
        
        # Initialize memory tracking
        self.peak_memory = 0
        
        # Log model complexity
        self._log_model_complexity()
    
    def _prefetch_data(self, dataloader, num_batches=3):
        """
        Prefetch a few batches of data to warm up the dataloader.
        
        Args:
            dataloader: DataLoader to prefetch from
            num_batches: Number of batches to prefetch
        """
        prefetch_iter = iter(dataloader)
        prefetched_batches = []
        
        for _ in range(min(num_batches, len(dataloader))):
            try:
                batch = next(prefetch_iter)
                prefetched_batches.append(batch)
            except StopIteration:
                break
                
        return prefetched_batches
    
    def _log_model_complexity(self):
        """Log model complexity statistics with modality awareness"""
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Model complexity:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
        
        # Estimate memory usage
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        
        # Determine which modalities are being used
        use_text = self.config['data'].get('use_text', False)
        use_images = self.config['data'].get('use_images', True)
        
        # Estimate activation memory (rough estimate based on batch size)
        batch_size = self.config['train']['batch_size']
        activation_size = 0
        
        # For images, use the image encoder feature dimension
        if use_images:
            activation_size += batch_size * self.model.image_encoder.feature_dim * 4  # 4 bytes per float32
        
        # For text, use the text encoder feature dimension
        if use_text:
            # If the model has a text_encoder attribute
            if hasattr(self.model, 'text_encoder'):
                activation_size += batch_size * self.model.text_encoder.feature_dim * 4  # 4 bytes per float32
        
        total_memory = (param_size + buffer_size + activation_size) / 1024 / 1024  # MB
        
        print(f"  Estimated memory usage: {total_memory:.2f} MB")
        
        # Save to log file
        with open(os.path.join(self.experiment_dir, 'model_complexity.txt'), 'w') as f:
            # Write modality information
            if use_text and use_images:
                f.write("Modality: Multimodal (Images + Text)\n")
            elif use_images:
                f.write("Modality: Images only\n")
            elif use_text:
                f.write("Modality: Text only\n")
            
            # Write model-specific information based on modality
            if use_images:
                f.write(f"Image Model: {self.config['model']['encoder_type']}, Pooling: {self.config['model']['pooling_type']}\n")
            
            if use_text:
                f.write(f"Text Model: {self.config['model'].get('text_encoder_model', 'all-mpnet-base-v2')}\n")
            
            if use_text and use_images:
                f.write(f"Fusion: {self.config['model'].get('fusion_type', 'concat')}\n")
            
            # Write parameter counts
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Trainable parameters: {trainable_params:,}\n")
            f.write(f"Non-trainable parameters: {total_params - trainable_params:,}\n")
            f.write(f"Estimated memory usage: {total_memory:.2f} MB\n")
    
    def _save_config(self):
        """Save experiment configuration."""
        # Combine model configuration and training configuration
        experiment_config = {
            'model': self.config['model'],
            'train': self.config['train'],
            'data': self.config['data'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'device': str(self.device)
        }
        
        # Save to file
        with open(os.path.join(self.experiment_dir, 'config.json'), 'w') as f:
            json.dump(experiment_config, f, indent=2)
    
    def _track_memory(self, step=None):
        """Track and log peak memory usage"""
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            
            if peak_memory > self.peak_memory:
                self.peak_memory = peak_memory
                print(f"New peak memory: {self.peak_memory:.2f} MB")
            
            if step is not None and step % 50 == 0:
                print(f"Memory at step {step}: Current: {current_memory:.2f} MB, Peak: {peak_memory:.2f} MB")
            
            return current_memory, peak_memory
        return 0, 0
    
    def train(self, num_epochs):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train for
            
        Returns:
            history: Dictionary with training history
        """
        history = defaultdict(list)
        
        print(f"Starting training for {num_epochs} epochs...")
        
        # Prefetch data before starting training to warm up data loaders
        print("Warming up data loaders...")
        _ = self._prefetch_data(self.train_loader)
        _ = self._prefetch_data(self.val_loader)
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            train_metrics = self._train_epoch(epoch)
            gc.collect()
            
            val_metrics = self._validate_epoch(epoch)
            gc.collect()
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss']['total'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Track metrics
            for k, v in train_metrics.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        history[f'train_{k}_{kk}'].append(vv)
                else:
                    history[f'train_{k}'].append(v)
                    
            for k, v in val_metrics.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        history[f'val_{k}_{kk}'].append(vv)
                else:
                    history[f'val_{k}'].append(v)
                    
            history['learning_rate'].append(current_lr)
            
            # Check for improvement
            if val_metrics['loss']['total'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']['total']
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                
                # Save best model
                save_checkpoint(
                    self.model, 
                    self.optimizer, 
                    epoch, 
                    val_metrics,
                    self.experiment_dir,
                    filename='best_model.pt'
                )
                print(f"✓ New best model saved! (val_loss: {self.best_val_loss:.4f})")
            else:
                self.epochs_without_improvement += 1
                
            # Save checkpoint periodically
            if (epoch + 1) % self.config['logging']['save_checkpoint_frequency'] == 0:
                save_checkpoint(
                    self.model, 
                    self.optimizer, 
                    epoch, 
                    val_metrics,
                    self.experiment_dir,
                    filename=f'checkpoint_epoch_{epoch+1}.pt'
                )
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch summary
            self._print_epoch_summary(epoch, num_epochs, train_metrics, val_metrics, current_lr, epoch_time)
            
            # Log metrics
            self.logger.log_metrics(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lr=current_lr,
                epoch_time=epoch_time
            )
            
            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs without improvement.")
                break
        
        # Load best model
        self._load_best_model()
        
        return dict(history)

    def _train_epoch(self, epoch):
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch
            
        Returns:
            metrics: Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0
        loss_components = defaultdict(float)
        regression_predictions = []
        regression_targets = []
        classification_predictions = []
        classification_targets = []
        
        print(f"Starting epoch {epoch+1} training...")
        epoch_start_time = time.time()
        
        # Determine which modalities are being used
        use_text = self.config['data'].get('use_text', False)
        use_images = self.config['data'].get('use_images', True)
        
        # Initialize dataloader iterator
        loader_iter = iter(self.train_loader)
        
        # Process first batch separately to initialize tqdm accurately
        try:
            print("Loading first batch...")
            first_batch_start = time.time()
            batch = next(loader_iter)
            print(f"First batch loaded in {time.time() - first_batch_start:.2f}s")
            
            # Process first batch - prepare data based on modalities
            model_inputs, reg_targets, cls_targets = self._prepare_batch(batch, use_text, use_images)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(*model_inputs)
            
            # Compute loss
            loss, losses = self.criterion(outputs, (reg_targets, cls_targets))
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Record losses
            total_loss += loss.item()
            for k, v in losses.items():
                loss_components[k] += v
            
            # Record predictions
            regression_predictions.extend(outputs['regression'].detach().cpu().numpy())
            regression_targets.extend(reg_targets.cpu().numpy())
            
            classification_predictions.extend(
                outputs['classification'].argmax(dim=1).detach().cpu().numpy()
            )
            classification_targets.extend(cls_targets.cpu().numpy())
            
            # Clear memory
            del batch, model_inputs, reg_targets, cls_targets, outputs, loss
            
            # Initialize tqdm with the remaining batches
            remaining_batches = len(self.train_loader) - 1
            
            # Now create tqdm for the remaining batches
            train_pbar = tqdm(
                range(remaining_batches),
                desc=f"Epoch {epoch+1} [Train]", 
                leave=False, 
                dynamic_ncols=True
            )
            
            # Process remaining batches
            for _ in train_pbar:
                try:
                    batch_start = time.time()
                    batch = next(loader_iter)
                    
                    # Prepare batch based on modalities
                    model_inputs, reg_targets, cls_targets = self._prepare_batch(batch, use_text, use_images)
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(*model_inputs)
                    
                    # Compute loss
                    loss, losses = self.criterion(outputs, (reg_targets, cls_targets))
                    
                    # Backward pass and optimize
                    loss.backward()
                    self.optimizer.step()
                    
                    # Record losses
                    total_loss += loss.item()
                    for k, v in losses.items():
                        loss_components[k] += v
                    
                    # Record predictions
                    regression_predictions.extend(outputs['regression'].detach().cpu().numpy())
                    regression_targets.extend(reg_targets.cpu().numpy())
                    
                    classification_predictions.extend(
                        outputs['classification'].argmax(dim=1).detach().cpu().numpy()
                    )
                    classification_targets.extend(cls_targets.cpu().numpy())
                    
                    # Track memory
                    # if (step + 1) % 20 == 0:
                    #     self._track_memory(step)
                    
                    # Update progress bar
                    batch_time = time.time() - batch_start
                    train_pbar.set_postfix({
                        "loss": f"{losses['total']:.4f}",
                        "reg_loss": f"{losses['regression']:.4f}",
                        "cls_loss": f"{losses['classification']:.4f}",
                        "batch_time": f"{batch_time:.2f}s"
                    })
                    
                    # Clear memory
                    del batch, model_inputs, reg_targets, cls_targets, outputs, loss
                    
                except StopIteration:
                    break
                    
        except StopIteration:
            print("Warning: DataLoader is empty!")
            return {}
        
        # Compute metrics
        metrics = compute_metrics(
            regression_predictions=np.array(regression_predictions),
            regression_targets=np.array(regression_targets),
            classification_predictions=np.array(classification_predictions),
            classification_targets=np.array(classification_targets),
            preprocessing_params=self.preprocessing_params
        )
        
        # Calculate average losses
        num_batches = len(self.train_loader)
        metrics['loss'] = {k: v / num_batches for k, v in loss_components.items()}
        
        # Calculate total epoch time
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} training completed in {epoch_time:.2f}s")
        
        return metrics

    def _validate_epoch(self, epoch):
        """
        Validate for one epoch.
        
        Args:
            epoch: Current epoch
            
        Returns:
            metrics: Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0
        loss_components = defaultdict(float)
        regression_predictions = []
        regression_targets = []
        classification_predictions = []
        classification_targets = []
        
        print(f"Starting epoch {epoch+1} validation...")
        val_start_time = time.time()
        
        # Determine which modalities are being used
        use_text = self.config['data'].get('use_text', False)
        use_images = self.config['data'].get('use_images', True)
        
        # Initialize dataloader iterator
        loader_iter = iter(self.val_loader)
        
        # Process first batch separately
        try:
            print("Loading first validation batch...")
            first_batch_start = time.time()
            batch = next(loader_iter)
            print(f"First validation batch loaded in {time.time() - first_batch_start:.2f}s")
            
            # Process first batch - prepare data based on modalities
            with torch.no_grad():
                model_inputs, reg_targets, cls_targets = self._prepare_batch(batch, use_text, use_images)
                
                # Forward pass
                outputs = self.model(*model_inputs)
                
                # Compute loss
                loss, losses = self.criterion(outputs, (reg_targets, cls_targets))
                
                # Record losses
                total_loss += loss.item()
                for k, v in losses.items():
                    loss_components[k] += v
                
                # Record predictions
                regression_predictions.extend(outputs['regression'].cpu().numpy())
                regression_targets.extend(reg_targets.cpu().numpy())
                
                classification_predictions.extend(
                    outputs['classification'].argmax(dim=1).cpu().numpy()
                )
                classification_targets.extend(cls_targets.cpu().numpy())
                
                # Clear memory
                del batch, model_inputs, reg_targets, cls_targets, outputs, loss
            
            # Initialize tqdm with the remaining batches
            remaining_batches = len(self.val_loader) - 1
            
            # Create tqdm for the remaining batches
            val_pbar = tqdm(
                range(remaining_batches),
                desc=f"Epoch {epoch+1} [Valid]", 
                leave=False, 
                dynamic_ncols=True
            )
            
            # Process remaining batches
            with torch.no_grad():
                for _ in val_pbar:
                    try:
                        batch_start = time.time()
                        batch = next(loader_iter)
                        
                        # Prepare batch based on modalities
                        model_inputs, reg_targets, cls_targets = self._prepare_batch(batch, use_text, use_images)
                        
                        # Forward pass
                        outputs = self.model(*model_inputs)
                        
                        # Compute loss
                        loss, losses = self.criterion(outputs, (reg_targets, cls_targets))
                        
                        # Record losses
                        total_loss += loss.item()
                        for k, v in losses.items():
                            loss_components[k] += v
                        
                        # Record predictions
                        regression_predictions.extend(outputs['regression'].cpu().numpy())
                        regression_targets.extend(reg_targets.cpu().numpy())
                        
                        classification_predictions.extend(
                            outputs['classification'].argmax(dim=1).cpu().numpy()
                        )
                        classification_targets.extend(cls_targets.cpu().numpy())
                        
                        # Update progress bar
                        batch_time = time.time() - batch_start
                        val_pbar.set_postfix({
                            "loss": f"{losses['total']:.4f}",
                            "reg_loss": f"{losses['regression']:.4f}",
                            "cls_loss": f"{losses['classification']:.4f}",
                            "batch_time": f"{batch_time:.2f}s"
                        })
                        
                        # Clear memory
                        del batch, model_inputs, reg_targets, cls_targets, outputs, loss
                        
                    except StopIteration:
                        break
            
        except StopIteration:
            print("Warning: Validation DataLoader is empty!")
            return {}
        
        # Compute basic metrics
        metrics = compute_metrics(
            regression_predictions=np.array(regression_predictions),
            regression_targets=np.array(regression_targets),
            classification_predictions=np.array(classification_predictions),
            classification_targets=np.array(classification_targets),
            preprocessing_params=self.preprocessing_params
        )
        
        # Calculate average losses
        num_batches = len(self.val_loader)
        metrics['loss'] = {k: v / num_batches for k, v in loss_components.items()}
        
        # Compute detailed metrics
        detailed_metrics = compute_detailed_metrics(
            regression_predictions=np.array(regression_predictions),
            regression_targets=np.array(regression_targets),
            classification_predictions=np.array(classification_predictions),
            classification_targets=np.array(classification_targets),
            preprocessing_params=self.preprocessing_params
        )
        
        # Save detailed metrics for this epoch
        self._save_epoch_detailed_metrics(
            epoch, 
            detailed_metrics,
            regression_predictions,
            regression_targets,
            classification_predictions,
            classification_targets
        )
        
        # Calculate total validation time
        val_time = time.time() - val_start_time
        print(f"Epoch {epoch+1} validation completed in {val_time:.2f}s")
        
        return metrics


    def _prepare_batch(self, batch, use_text, use_images):
        """Prepare batch inputs for the model based on modalities."""
        
        if use_text and use_images:
            # Both modalities
            images, masks, text_inputs, reg_targets, cls_targets = batch
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Move text tensors to device
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            
            reg_targets = reg_targets.to(self.device)
            cls_targets = cls_targets.to(self.device)
            model_inputs = (images, masks, text_inputs)
            
        elif use_images:
            # Images only - stays the same
            images, masks, reg_targets, cls_targets = batch
            images = images.to(self.device)
            masks = masks.to(self.device)
            reg_targets = reg_targets.to(self.device)
            cls_targets = cls_targets.to(self.device)
            model_inputs = (images, masks)
            
        elif use_text:
            # Text only
            text_inputs, reg_targets, cls_targets = batch
            
            # Move text tensors to device
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            
            reg_targets = reg_targets.to(self.device)
            cls_targets = cls_targets.to(self.device)
            model_inputs = (text_inputs,)
        
        return model_inputs, reg_targets, cls_targets

    def _load_best_model(self):
        """Load the best model from checkpoint"""
        best_model_path = os.path.join(self.experiment_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            try:
                # First attempt with weights_only=True (default in PyTorch 2.6+)
                checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}")
            except Exception as e:
                print(f"Error loading model with default settings: {e}")
                try:
                    torch.serialization.add_safe_globals(['numpy.core.multiarray.scalar'])
                    checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')} with weights_only=False")
                except Exception as e2:
                    print(f"Failed to load model with weights_only=False: {e2}")
                    print("Warning: Could not load best model checkpoint")
        else:
            print("Warning: Could not find best model checkpoint")
    
    def _print_epoch_summary(self, epoch, num_epochs, train_metrics, val_metrics, lr, epoch_time):
        """Print summary of epoch results."""
        # Determine if this is the best model so far
        is_best = val_metrics['loss']['total'] <= self.best_val_loss
        best_marker = "✓" if is_best else " "
        
        # Print epoch header
        print(f"\nEpoch {epoch+1}/{num_epochs} - {epoch_time:.1f}s - LR: {lr:.6f} {best_marker}")
        
        # Print regression metrics
        print(f"  Regression:")
        print(f"    Train: Loss={train_metrics['loss']['regression']:.4f}, MAE={train_metrics['regression']['mae']:.0f}, RMSE={train_metrics['regression']['rmse']:.0f}")
        print(f"    Val:   Loss={val_metrics['loss']['regression']:.4f}, MAE={val_metrics['regression']['mae']:.0f}, RMSE={val_metrics['regression']['rmse']:.0f}")
        
        # Print classification metrics
        print(f"  Classification:")
        print(f"    Train: Loss={train_metrics['loss']['classification']:.4f}, Acc={train_metrics['classification']['accuracy']:.4f}")
        print(f"    Val:   Loss={val_metrics['loss']['classification']:.4f}, Acc={val_metrics['classification']['accuracy']:.4f}")
        
        # Print combined loss
        print(f"  Combined Loss:")
        print(f"    Train: {train_metrics['loss']['total']:.4f}")
        print(f"    Val:   {val_metrics['loss']['total']:.4f}")
        
        # Print epochs without improvement (if applicable)
        if self.epochs_without_improvement > 0:
            print(f"  No improvement for {self.epochs_without_improvement} epochs (best: {self.best_val_loss:.4f} at epoch {self.best_epoch+1})")
        
        # Print memory usage
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            print(f"  Memory usage: Current: {current_memory:.1f} MB, Peak: {peak_memory:.1f} MB")


    def _save_epoch_detailed_metrics(self, epoch, detailed_metrics, reg_preds, reg_targets, cls_preds, cls_targets):
        """
        Save detailed metrics for the current epoch.
        
        Args:
            epoch: Current epoch number
            detailed_metrics: Dictionary with detailed metrics
            reg_preds: Regression predictions
            reg_targets: Regression targets
            cls_preds: Classification predictions
            cls_targets: Classification targets
        """

        
        # Create epoch directory
        epoch_dir = os.path.join(self.experiment_dir, f'epoch_{epoch+1}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        # 1. Save detailed metrics as JSON
        with open(os.path.join(epoch_dir, 'detailed_metrics.json'), 'w') as f:
            # Convert numpy types to native types for JSON serialization
            json.dump(self._convert_to_serializable(detailed_metrics), f, indent=2)
        
        # 2. Save classification report as CSV
        if detailed_metrics['classification']['report'] is not None:
            try:
                # Convert the classification report to a DataFrame
                report_df = pd.DataFrame(detailed_metrics['classification']['report']).T
                report_df.to_csv(os.path.join(epoch_dir, 'classification_report.csv'))
                
                # Also save as text file for easy reading
                with open(os.path.join(epoch_dir, 'classification_report.txt'), 'w') as f:
                    class_names = self.preprocessing_params['class_names']
                    f.write(detailed_metrics['classification']['report'])
            except Exception as e:
                print(f"Error saving classification report: {e}")
        
        # 3. Save confusion matrix as CSV
        if detailed_metrics['classification']['confusion_matrix'] is not None:
            try:
                # Save as CSV
                conf_matrix = np.array(detailed_metrics['classification']['confusion_matrix'])
                conf_df = pd.DataFrame(conf_matrix, 
                                    index=self.preprocessing_params['class_names'],
                                    columns=self.preprocessing_params['class_names'])
                conf_df.to_csv(os.path.join(epoch_dir, 'confusion_matrix.csv'))
                
                # Also save visualization
                plt.figure(figsize=(12, 10))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                            xticklabels=self.preprocessing_params['class_names'],
                            yticklabels=self.preprocessing_params['class_names'])
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'Confusion Matrix - Epoch {epoch+1}')
                plt.tight_layout()
                plt.savefig(os.path.join(epoch_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Error saving confusion matrix: {e}")
        
        # 4. Save price predictions plot
        try:
            # Denormalize predictions and targets
            denorm_preds = self.preprocessing_params['price_mean'] + self.preprocessing_params['price_std'] * np.array(reg_preds)
            denorm_targets = self.preprocessing_params['price_mean'] + self.preprocessing_params['price_std'] * np.array(reg_targets)
            
            # Save predictions as CSV
            pred_df = pd.DataFrame({
                'actual': denorm_targets,
                'predicted': denorm_preds,
                'error': denorm_preds - denorm_targets,
                'percent_error': (denorm_preds - denorm_targets) / denorm_targets * 100
            })
            pred_df.to_csv(os.path.join(epoch_dir, 'predictions.csv'), index=False)
            
            # Create scatter plot
            plt.figure(figsize=(10, 8))
            plt.scatter(denorm_targets, denorm_preds, alpha=0.5)
            
            # Plot perfect prediction line
            max_val = max(np.max(denorm_preds), np.max(denorm_targets))
            min_val = min(np.min(denorm_preds), np.min(denorm_targets))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            # Add labels and title
            plt.xlabel('Actual Price')
            plt.ylabel('Predicted Price')
            plt.title(f'Price Predictions - Epoch {epoch+1}\nMAE: {detailed_metrics["regression"]["mae"]:.0f}, RMSE: {detailed_metrics["regression"]["rmse"]:.0f}')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(epoch_dir, 'price_predictions.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error creating price predictions plot: {e}")
        
        # 5. Save a summary text file
        try:
            with open(os.path.join(epoch_dir, 'summary.txt'), 'w') as f:
                f.write(f"Epoch {epoch+1} - Detailed Metrics Summary\n")
                f.write("=" * 50 + "\n\n")
                
                # Regression metrics
                f.write("Regression Metrics:\n")
                f.write(f"  MAE: {detailed_metrics['regression']['mae']:.2f}\n")
                f.write(f"  RMSE: {detailed_metrics['regression']['rmse']:.2f}\n")
                f.write(f"  R²: {detailed_metrics['regression']['r2']:.4f}\n")
                f.write(f"  Mean Percent Error: {detailed_metrics['regression']['mean_percent_error']:.2f}%\n")
                f.write(f"  Median Percent Error: {detailed_metrics['regression']['median_percent_error']:.2f}%\n")
                f.write(f"  Max Absolute Error: {detailed_metrics['regression']['max_ae']:.2f}\n")
                f.write(f"  Price Range: €{detailed_metrics['regression']['min_price']:.2f} - €{detailed_metrics['regression']['max_price']:.2f}\n\n")
                
                # Classification metrics
                f.write("Classification Metrics:\n")
                f.write(f"  Accuracy: {detailed_metrics['classification']['accuracy']:.4f}\n\n")
                
                # Detailed metrics are saved in the CSV files
                f.write("Detailed metrics saved in:\n")
                f.write(f"  - classification_report.csv\n")
                f.write(f"  - classification_report.txt\n")
                f.write(f"  - confusion_matrix.csv\n")
                f.write(f"  - confusion_matrix.png\n")
                f.write(f"  - predictions.csv\n")
                f.write(f"  - price_predictions.png\n")
        except Exception as e:
            print(f"Error creating summary file: {e}")
        
        print(f"Saved detailed metrics for epoch {epoch+1} to {epoch_dir}")


    def _convert_to_serializable(self, obj):
        """
        Convert non-serializable types to JSON-serializable types.
        
        Args:
            obj: Object to convert
            
        Returns:
            Serializable object
        """
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(x) for x in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_to_serializable(obj.tolist())
        else:
            return obj