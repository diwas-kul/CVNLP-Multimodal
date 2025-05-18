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

from utils.logging import ContrastiveLogger,  save_checkpoint
from training.losses import InfoNCELoss

class ContrastiveTrainer:
    """
    Trainer for contrastive learning.
    """
    def __init__(self, model, train_loader, val_loader, config, experiment_dir):
        """
        Initialize trainer.
        
        Args:
            model: ContrastiveModel
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            config: Configuration dictionary
            experiment_dir: Directory to save logs and checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.experiment_dir = experiment_dir
        self.device = next(model.parameters()).device
        
        # Create logger
        self.logger = ContrastiveLogger(experiment_dir)
        
        # Loss function
        self.criterion = InfoNCELoss(temperature=config['contrastive']['temperature'])
        
        # Optimizer
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['contrastive']['learning_rate'],
            weight_decay=config['contrastive']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=config['contrastive']['scheduler_factor'],
            patience=config['contrastive']['scheduler_patience'],
        )
        
        # Save configuration
        self._save_config()
        
        # Initialize best metrics
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.early_stopping_patience = config['contrastive']['early_stopping_patience']
    
    def _save_config(self):
        """Save experiment configuration."""
        # Combine model configuration and training configuration
        experiment_config = {
            'model': self.config['model'],
            'contrastive': self.config['contrastive'],
            'data': self.config['data'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'device': str(self.device)
        }
        
        # Save to file
        with open(os.path.join(self.experiment_dir, 'contrastive_config.json'), 'w') as f:
            json.dump(experiment_config, f, indent=2)
    
    def train(self, num_epochs):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train for
            
        Returns:
            history: Dictionary with training history
        """
        history = defaultdict(list)
        
        print(f"Starting contrastive pretraining for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            train_metrics = self._train_epoch(epoch)
            gc.collect()
            
            val_metrics = self._validate_epoch(epoch)
            gc.collect()
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Track metrics
            for k, v in train_metrics.items():
                history[f'train_{k}'].append(v)
                    
            for k, v in val_metrics.items():
                history[f'val_{k}'].append(v)
                    
            history['learning_rate'].append(current_lr)
            
            # Check for improvement
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                
                # Save best model
                save_checkpoint(
                    self.model, 
                    self.optimizer, 
                    epoch, 
                    val_metrics,
                    self.experiment_dir,
                    filename='best_contrastive_model.pt'
                )
                print(f"✓ New best model saved! (val_loss: {self.best_val_loss:.4f})")
            else:
                self.epochs_without_improvement += 1
                
            # Save checkpoint periodically
            if (epoch + 1) % self.config['contrastive']['save_checkpoint_frequency'] == 0:
                save_checkpoint(
                    self.model, 
                    self.optimizer, 
                    epoch, 
                    val_metrics,
                    self.experiment_dir,
                    filename=f'contrastive_checkpoint_epoch_{epoch+1}.pt'
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
        total_accuracy = 0
        
        print(f"Starting epoch {epoch+1} training...")
        epoch_start_time = time.time()
        
        # Initialize tqdm progress bar
        train_pbar = tqdm(
            enumerate(self.train_loader),
            desc=f"Epoch {epoch+1} [Train]", 
            leave=False, 
            dynamic_ncols=True,
            total=len(self.train_loader)
        )
        
        # Process batches
        for step, batch in train_pbar:
            # Prepare batch
            images, masks, text_inputs, _, _ = batch
            images = images.to(self.device)
            masks = masks.to(self.device)
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, masks, text_inputs)
            
            # Compute loss
            loss = self.criterion(outputs['image_proj'], outputs['text_proj'])
            
            # Compute accuracy
            similarity = self.model.compute_similarity(outputs['image_proj'], outputs['text_proj'])
            batch_size = images.size(0)
            labels = torch.arange(batch_size, device=self.device)
            i2t_accuracy = (torch.argmax(similarity, dim=1) == labels).float().mean().item()
            t2i_accuracy = (torch.argmax(similarity, dim=0) == labels).float().mean().item()
            accuracy = (i2t_accuracy + t2i_accuracy) / 2
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Record metrics
            total_loss += loss.item()
            total_accuracy += accuracy
            
            # Update progress bar
            train_pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{accuracy:.4f}"
            })
            
            # Clear memory
            del batch, images, masks, text_inputs, outputs, loss
        
        # Calculate average metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_accuracy = total_accuracy / len(self.train_loader)
        
        # Calculate total epoch time
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} training completed in {epoch_time:.2f}s")
        
        return {'loss': avg_loss, 'accuracy': avg_accuracy}
    
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
        total_accuracy = 0
        
        print(f"Starting epoch {epoch+1} validation...")
        
        # Initialize tqdm progress bar
        val_pbar = tqdm(
            enumerate(self.val_loader),
            desc=f"Epoch {epoch+1} [Valid]", 
            leave=False, 
            dynamic_ncols=True,
            total=len(self.val_loader)
        )
        
        # Process batches
        with torch.no_grad():
            for step, batch in val_pbar:
                # Prepare batch
                images, masks, text_inputs, _, _ = batch
                images = images.to(self.device)
                masks = masks.to(self.device)
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                
                # Forward pass
                outputs = self.model(images, masks, text_inputs)
                
                # Compute loss
                loss = self.criterion(outputs['image_proj'], outputs['text_proj'])
                
                # Compute accuracy
                similarity = self.model.compute_similarity(outputs['image_proj'], outputs['text_proj'])
                batch_size = images.size(0)
                labels = torch.arange(batch_size, device=self.device)
                i2t_accuracy = (torch.argmax(similarity, dim=1) == labels).float().mean().item()
                t2i_accuracy = (torch.argmax(similarity, dim=0) == labels).float().mean().item()
                accuracy = (i2t_accuracy + t2i_accuracy) / 2
                
                # Record metrics
                total_loss += loss.item()
                total_accuracy += accuracy
                
                # Update progress bar
                val_pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{accuracy:.4f}"
                })
                
                # Clear memory
                del batch, images, masks, text_inputs, outputs, loss
        
        # Calculate average metrics
        avg_loss = total_loss / len(self.val_loader)
        avg_accuracy = total_accuracy / len(self.val_loader)
        
        return {'loss': avg_loss, 'accuracy': avg_accuracy}
    
    def _print_epoch_summary(self, epoch, num_epochs, train_metrics, val_metrics, lr, epoch_time):
        """Print summary of epoch results."""
        # Determine if this is the best model so far
        is_best = val_metrics['loss'] <= self.best_val_loss
        best_marker = "✓" if is_best else " "
        
        # Print epoch header
        print(f"\nEpoch {epoch+1}/{num_epochs} - {epoch_time:.1f}s - LR: {lr:.6f} {best_marker}")
        
        # Print metrics
        print(f"  Contrastive Loss:")
        print(f"    Train: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"    Val:   {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
        
        # Print epochs without improvement (if applicable)
        if self.epochs_without_improvement > 0:
            print(f"  No improvement for {self.epochs_without_improvement} epochs (best: {self.best_val_loss:.4f} at epoch {self.best_epoch+1})")

    def _save_encoders(self):
        """Save image and text encoders separately"""
        # Create directory for encoders
        encoders_dir = os.path.join(self.experiment_dir, 'encoders')
        os.makedirs(encoders_dir, exist_ok=True)
        
        # Save image encoder
        image_encoder_state = {
            'model_state_dict': self.model.image_encoder.state_dict(),
            'config': self.config['model']
        }
        torch.save(image_encoder_state, os.path.join(encoders_dir, 'image_encoder.pt'))
        
        # Save text encoder
        text_encoder_state = {
            'model_state_dict': self.model.text_encoder.state_dict(),
            'config': self.config['model']
        }
        torch.save(text_encoder_state, os.path.join(encoders_dir, 'text_encoder.pt'))
        
        # Save projection heads in case we want to use them later
        projection_heads_state = {
            'image_projection': self.model.image_projection.state_dict(),
            'text_projection': self.model.text_projection.state_dict(),
            'config': self.config['contrastive']
        }
        torch.save(projection_heads_state, os.path.join(encoders_dir, 'projection_heads.pt'))
        
        print(f"Saved encoders to {encoders_dir}")