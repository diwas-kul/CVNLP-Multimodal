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