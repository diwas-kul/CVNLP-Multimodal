import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    """
    Combined loss function for both regression and classification tasks.
    """
    def __init__(self, regression_weight=1.0, classification_weight=1.0):
        super().__init__()
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        
        self.regression_loss = nn.MSELoss()
        self.classification_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets):
        """
        Compute combined loss.
        
        Args:
            outputs: Dictionary with 'regression' and 'classification' outputs
            targets: Tuple of (regression_targets, classification_targets)
            
        Returns:
            total_loss: Combined loss
            losses: Dictionary with individual losses
        """
        regression_targets, classification_targets = targets
        
        reg_loss = self.regression_loss(outputs['regression'], regression_targets)
        cls_loss = self.classification_loss(outputs['classification'], classification_targets)
        
        # Combine losses
        total_loss = self.regression_weight * reg_loss + self.classification_weight * cls_loss
        
        # Return individual losses for logging
        losses = {
            'total': total_loss.item(),
            'regression': reg_loss.item(),
            'classification': cls_loss.item()
        }
        
        return total_loss, losses
    
# Contrastive loss
class InfoNCELoss(nn.Module):
    """
    InfoNCE (NT-Xent) loss for contrastive learning.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, image_proj, text_proj):
        """
        Compute InfoNCE loss.
        
        Args:
            image_proj: Image projections [B, D]
            text_proj: Text projections [B, D]
            
        Returns:
            loss: InfoNCE loss
        """
        # Compute similarity matrices
        image_text_sim = torch.matmul(image_proj, text_proj.t()) / self.temperature
        
        # Create labels (diagonal is positive pairs)
        batch_size = image_proj.shape[0]
        labels = torch.arange(batch_size, device=image_proj.device)
        
        # Compute cross entropy loss (both directions)
        loss_i2t = F.cross_entropy(image_text_sim, labels)
        loss_t2i = F.cross_entropy(image_text_sim.t(), labels)
        
        # Average the losses
        loss = (loss_i2t + loss_t2i) / 2.0
        
        return loss