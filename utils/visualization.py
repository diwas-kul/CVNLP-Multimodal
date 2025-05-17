import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')

def plot_training_history(history, output_dir=None):
    """
    Plot training history.
    
    Args:
        history: Dictionary with training history
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(16, 12))
    
    # Plot loss
    plt.subplot(2, 3, 1)
    plt.plot(history['train_loss_total'], label='Train Loss')
    plt.plot(history['val_loss_total'], label='Val Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot regression loss
    plt.subplot(2, 3, 2)
    plt.plot(history['train_loss_regression'], label='Train')
    plt.plot(history['val_loss_regression'], label='Val')
    plt.title('Regression Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot classification loss
    plt.subplot(2, 3, 3)
    plt.plot(history['train_loss_classification'], label='Train')
    plt.plot(history['val_loss_classification'], label='Val')
    plt.title('Classification Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot MAE
    plt.subplot(2, 3, 4)
    plt.plot(history['train_regression_mae'], label='Train')
    plt.plot(history['val_regression_mae'], label='Val')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    # Plot RMSE
    plt.subplot(2, 3, 5)
    plt.plot(history['train_regression_rmse'], label='Train')
    plt.plot(history['val_regression_rmse'], label='Val')
    plt.title('Root Mean Squared Error')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(2, 3, 6)
    plt.plot(history['train_classification_accuracy'], label='Train')
    plt.plot(history['val_classification_accuracy'], label='Val')
    plt.title('Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'training_history.png'))

def plot_predictions(predictions, targets, output_dir=None):
    """
    Plot regression predictions vs actual values.
    
    Args:
        predictions: Predicted values
        targets: Actual values
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(targets, predictions, alpha=0.5)
    
    # Add diagonal line (perfect predictions)
    min_val = min(min(targets), min(predictions))
    max_val = max(max(targets), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add regression line
    z = np.polyfit(targets, predictions, 1)
    p = np.poly1d(z)
    plt.plot(targets, p(targets), 'b-', alpha=0.7)
    
    plt.title('Predicted vs Actual Prices')
    plt.xlabel('Actual Price (€)')
    plt.ylabel('Predicted Price (€)')
    plt.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(targets, predictions)[0, 1]
    plt.annotate(f'Correlation: {correlation:.3f}', 
                xy=(0.05, 0.95), xycoords='axes fraction', 
                ha='left', va='top', 
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'price_predictions.png'))
    
def plot_confusion_matrix(y_true, y_pred, class_names, output_dir=None):
    """
    Plot confusion matrix for classification results.
    
    Args:
        y_true: True class indices
        y_pred: Predicted class indices
        class_names: List of class names
        output_dir: Directory to save plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
def plot_error_distribution(predictions, targets, output_dir=None):
    """
    Plot distribution of prediction errors.
    
    Args:
        predictions: Predicted values
        targets: Actual values
        output_dir: Directory to save plot
    """
    errors = predictions - targets
    abs_errors = np.abs(errors)
    
    plt.figure(figsize=(15, 6))
    
    # Error distribution
    plt.subplot(1, 2, 1)
    sns.histplot(errors, kde=True)
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error')
    plt.axvline(x=0, color='r', linestyle='--')
    
    # Plot errors vs actual values
    plt.subplot(1, 2, 2)
    plt.scatter(targets, abs_errors, alpha=0.5)
    plt.title('Absolute Error vs Actual Price')
    plt.xlabel('Actual Price (€)')
    plt.ylabel('Absolute Error')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'error_distribution.png'))