import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report, confusion_matrix
import math

from data.data_utils import denormalize_predictions

def compute_metrics(regression_predictions, regression_targets, 
                   classification_predictions, classification_targets,
                   preprocessing_params):
    """
    Compute evaluation metrics for both regression and classification tasks.
    
    Args:
        regression_predictions: Normalized regression predictions
        regression_targets: Normalized regression targets
        classification_predictions: Classification predictions (class indices)
        classification_targets: Classification targets (class indices)
        preprocessing_params: Dictionary with preprocessing parameters
        
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    # Denormalize regression predictions and targets for real-world metrics
    denorm_predictions = denormalize_predictions(regression_predictions, preprocessing_params)
    denorm_targets = denormalize_predictions(regression_targets, preprocessing_params)
    
    # Regression metrics
    mae = mean_absolute_error(denorm_targets, denorm_predictions)
    rmse = math.sqrt(mean_squared_error(denorm_targets, denorm_predictions))
    
    # Classification metrics
    accuracy = accuracy_score(classification_targets, classification_predictions)
    
    # Combine metrics
    metrics = {
        'regression': {
            'mae': mae,
            'rmse': rmse
        },
        'classification': {
            'accuracy': accuracy
        }
    }
    
    return metrics

def compute_detailed_metrics(regression_predictions, regression_targets, 
                            classification_predictions, classification_targets,
                            preprocessing_params):
    """
    Compute detailed evaluation metrics for analysis.
    
    Args:
        regression_predictions: Normalized regression predictions
        regression_targets: Normalized regression targets
        classification_predictions: Classification predictions (class indices)
        classification_targets: Classification targets (class indices)
        preprocessing_params: Dictionary with preprocessing parameters
        
    Returns:
        metrics: Dictionary with detailed evaluation metrics
    """
    # Basic metrics
    metrics = compute_metrics(
        regression_predictions, regression_targets,
        classification_predictions, classification_targets,
        preprocessing_params
    )
    
    # Denormalize regression predictions and targets
    denorm_predictions = denormalize_predictions(regression_predictions, preprocessing_params)
    denorm_targets = denormalize_predictions(regression_targets, preprocessing_params)
    
    # Additional regression metrics
    abs_errors = np.abs(denorm_predictions - denorm_targets)
    metrics['regression']['median_ae'] = np.median(abs_errors)
    metrics['regression']['max_ae'] = np.max(abs_errors)
    metrics['regression']['std_ae'] = np.std(abs_errors)
    metrics['regression']['r2'] = 1 - np.sum((denorm_targets - denorm_predictions) ** 2) / np.sum((denorm_targets - np.mean(denorm_targets)) ** 2)
    
    # Price ranges
    metrics['regression']['min_price'] = np.min(denorm_targets)
    metrics['regression']['max_price'] = np.max(denorm_targets)
    metrics['regression']['mean_price'] = np.mean(denorm_targets)
    
    # Percent error
    percent_errors = abs_errors / denorm_targets * 100
    metrics['regression']['mean_percent_error'] = np.mean(percent_errors)
    metrics['regression']['median_percent_error'] = np.median(percent_errors)
    
    # Additional classification metrics
    try:
        class_names = preprocessing_params['class_names']
        cls_report = classification_report(
            classification_targets, classification_predictions,
            target_names=class_names, output_dict=True
        )
        metrics['classification']['report'] = cls_report
        
        # Confusion matrix
        conf_matrix = confusion_matrix(classification_targets, classification_predictions)
        metrics['classification']['confusion_matrix'] = conf_matrix.tolist()
    except:
        # In case of issues with classification metrics
        metrics['classification']['report'] = None
        metrics['classification']['confusion_matrix'] = None
    
    return metrics