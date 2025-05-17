import os
import sys
import time
import optuna
import yaml
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import run_experiment


def objective(trial, config_path):
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        config_path: Path to hyperparameter search configuration
        
    Returns:
        metric: Metric to optimize (validation loss)
    """
    # Load hyperparameter search configuration
    with open(config_path, 'r') as f:
        hp_config = yaml.safe_load(f)
    
    # Load base model configuration
    with open(hp_config['base_config'], 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Extract search space
    search_space = hp_config['search_space']
    
    # Define trial parameters
    params = {}
    for param_name, param_values in search_space.items():
        if isinstance(param_values, list):
            params[param_name] = trial.suggest_categorical(param_name, param_values)
        elif isinstance(param_values, dict) and 'low' in param_values and 'high' in param_values:
            # For numerical ranges
            if param_values.get('log', False):
                params[param_name] = trial.suggest_float(
                    param_name, param_values['low'], param_values['high'], log=True)
            else:
                params[param_name] = trial.suggest_float(
                    param_name, param_values['low'], param_values['high'])
    
    # Update model configuration with trial parameters
    for param, value in params.items():
        if param in ['pooling_type', 'freeze_backbone']:
            model_config['model'][param] = value
        elif param in ['learning_rate', 'weight_decay', 'balanced_sampler']:
            model_config['train'][param] = value
    
    # Determine batch size
    freeze_backbone = params['freeze_backbone']
    freeze_text_encoder = params.get('freeze_text_encoder', None)
    batch_size_config = hp_config.get('batch_size_config', {})

    # First try the multimodal format if both parameters are present
    if freeze_text_encoder is not None:
        freeze_key = f"{str(freeze_backbone).lower()}-{str(freeze_text_encoder).lower()}"
        if freeze_key in batch_size_config:
            batch_size = batch_size_config[freeze_key]
            print(f"Using batch size {batch_size} for freeze config {freeze_key}")
        else:
            # Try alternative formats for the key
            freeze_key = f"{freeze_backbone}-{freeze_text_encoder}"
            if freeze_key in batch_size_config:
                batch_size = batch_size_config[freeze_key]
                print(f"Using batch size {batch_size} for freeze config {freeze_key}")
            else:
                # Fall back to default based on freeze_backbone only
                batch_size = 32 if freeze_backbone else 12
                print(f"Using default batch size {batch_size} based on freeze_backbone={freeze_backbone}")
    else:
        # Single modality case - use original logic
        if freeze_backbone and str(True) in batch_size_config:
            batch_size = batch_size_config[str(True)]
        elif not freeze_backbone and str(False) in batch_size_config:
            batch_size = batch_size_config[str(False)]
        elif True in batch_size_config:
            batch_size = batch_size_config[True]
        elif False in batch_size_config:
            batch_size = batch_size_config[False]
        else:
            # Default batch sizes if not specified
            batch_size = 32 if freeze_backbone else 16
        print(f"Using batch size {batch_size} for single modality model with freeze_backbone={freeze_backbone}")


    if 'fusion_type' in params:
        model_config['model']['fusion_type'] = params['fusion_type']
    if 'modality_dropout' in params:
        model_config['model']['modality_dropout'] = params['modality_dropout']
    if 'freeze_text_encoder' in params:
        model_config['model']['freeze_text_encoder'] = params['freeze_text_encoder']
    if 'text_lr' in params:
        model_config['model']['text_lr'] = params['text_lr']
    if 'image_lr' in params:
        model_config['model']['image_lr'] = params['image_lr']
    if 'fusion_lr' in params:
        model_config['model']['fusion_lr'] = params['fusion_lr']
    if 'dropout_rate' in params:
        model_config['model']['dropout_rate'] = params['dropout_rate']

    model_config['train']['batch_size'] = batch_size
    model_config['train']['test_batch_size'] = batch_size
    
    # Override training settings for hyperparameter search
    model_config['train']['num_epochs'] = hp_config.get('max_epochs', 15)
    model_config['train']['early_stopping_patience'] = hp_config.get('early_stopping_patience', 5)
    
    # Create trial directory name
    model_type = model_config['model']['encoder_type']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    balanced_sampler_str = "balanced_sampler_" if  model_config['train']["balanced_sampler"] else ""
    trial_name = f"{model_type}_{balanced_sampler_str}trial_{trial.number}_{timestamp}"
    
    # Set experiment name
    model_config['experiment'] = trial_name
    
    # Add trial info to config
    model_config['optuna_trial'] = {
        'number': trial.number,
        'params': params,
        'batch_size': batch_size,
        'timestamp': timestamp
    }
    
    # Save trial config for reference
    os.makedirs('config/trials', exist_ok=True)
    with open(f'config/trials/{trial_name}.yaml', 'w') as f:
        yaml.dump(model_config, f)
    
    # Run experiment
    try:
        print(f"\nStarting Trial {trial.number}")
        print(f"Parameters: {params}")
        print(f"Batch Size: {batch_size} (derived from freeze_backbone={freeze_backbone})")
        
        result = run_experiment(model_config)
        
        # Report intermediate values if history exists
        if result and 'history' in result and result['history']:
            history = result['history']
            if 'val_loss_total' in history:
                for epoch, val_loss in enumerate(history['val_loss_total']):
                    # Get metrics for this epoch
                    val_mae = history.get('val_regression_mae', [0] * len(history['val_loss_total']))[epoch]
                    val_acc = history.get('val_classification_accuracy', [0] * len(history['val_loss_total']))[epoch]
                    
                    # Report to Optuna
                    trial.report(val_loss, epoch)
                    
                    # Store additional metrics
                    trial.set_user_attr(f'epoch_{epoch}_val_mae', float(val_mae))
                    trial.set_user_attr(f'epoch_{epoch}_val_accuracy', float(val_acc))
                    
                    # Early stopping by Optuna (if enabled)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                    
        # Store final metrics for easy access
        if result and 'final_metrics' in result:
            final_metrics = result['final_metrics']
            if 'regression' in final_metrics:
                trial.set_user_attr('final_mae', float(final_metrics['regression']['mae']))
                trial.set_user_attr('final_rmse', float(final_metrics['regression']['rmse']))
                trial.set_user_attr('final_r2', float(final_metrics['regression']['r2']))
            
            if 'classification' in final_metrics:
                trial.set_user_attr('final_accuracy', float(final_metrics['classification']['accuracy']))
        
        # Store batch size information
        trial.set_user_attr('batch_size', batch_size)
                        
        # Return validation loss as the objective to minimize
        return result.get('best_val_loss', float('inf'))
    
    except optuna.exceptions.TrialPruned:
        # Handle pruned trials
        print(f"Trial {trial.number} pruned.")
        raise  # Re-raise to let Optuna handle it
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return float('inf')  # Return a large value on error


def save_study_results(study, hp_config):
    """
    Save study results to file.
    
    Args:
        study: Optuna study object
        hp_config: Hyperparameter search configuration
    """
    # Extract model type from the base config
    with open(hp_config['base_config'], 'r') as f:
        base_config = yaml.safe_load(f)
    model_type = base_config['model']['encoder_type']
    
    # Create output directory
    output_dir = hp_config.get('results_dir', 'results/hyperparameter_search')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save study statistics
    study_stats = {
        'model_type': model_type,
        'best_trial': {
            'number': study.best_trial.number,
            'params': study.best_trial.params,
            'batch_size': study.best_trial.user_attrs.get('batch_size', 'unknown'),
            'value': study.best_trial.value,
            'user_attrs': study.best_trial.user_attrs
        },
        'datetime': timestamp,
        'n_trials': len(study.trials),
        'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'search_config': hp_config,
        'all_trials': [{
            'number': t.number,
            'params': t.params,
            'batch_size': t.user_attrs.get('batch_size', 'unknown'),
            'value': t.value,
            'user_attrs': t.user_attrs,
            'state': str(t.state)
        } for t in study.trials]
    }
    
    # Save as JSON
    json_path = f'{output_dir}/{model_type}_study_results_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(study_stats, f, indent=2)
    
    # Create a summary text file
    text_path = f'{output_dir}/{model_type}_study_summary_{timestamp}.txt'
    with open(text_path, 'w') as f:
        f.write(f"Hyperparameter Search Summary - {model_type} - {timestamp}\n")
        f.write("="*80 + "\n\n")
        
        f.write("Best Trial Parameters:\n")
        for param, value in study.best_trial.params.items():
            f.write(f"  {param}: {value}\n")
        
        batch_size = study.best_trial.user_attrs.get('batch_size', 'unknown')
        f.write(f"  batch_size: {batch_size} (derived from freeze_backbone)\n")
        
        f.write(f"\nBest Validation Loss: {study.best_trial.value:.4f}\n")
        
        # Write user attributes
        for key, value in study.best_trial.user_attrs.items():
            if key.startswith('final_'):
                f.write(f"{key.replace('final_', 'Final ')}: {value:.4f}\n")
        
        f.write("\n" + "-"*80 + "\n\n")
        f.write("All Trials Sorted by Performance:\n\n")
        
        # Sort trials by performance
        sorted_trials = sorted(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE], 
            key=lambda t: t.value
        )
        
        for i, trial in enumerate(sorted_trials):
            f.write(f"Rank {i+1}. Trial {trial.number} - Loss: {trial.value:.4f}\n")
            for param, value in trial.params.items():
                f.write(f"  {param}: {value}\n")
            
            batch_size = trial.user_attrs.get('batch_size', 'unknown')
            f.write(f"  batch_size: {batch_size} (derived from freeze_backbone)\n")
            
            # Add final metrics if available
            for key, value in trial.user_attrs.items():
                if key.startswith('final_'):
                    f.write(f"  {key.replace('final_', '')}: {value:.4f}\n")
            
            f.write("\n")
    
    print(f"Study results saved to:")
    print(f"  JSON: {json_path}")
    print(f"  Text summary: {text_path}")
    
    # Create a yaml file with the best configuration
    best_config_path = f'{output_dir}/{model_type}_best_config_{timestamp}.yaml'
    
    # Load the base config again
    with open(hp_config['base_config'], 'r') as f:
        best_config = yaml.safe_load(f)
    
    # Update with best parameters
    for param, value in study.best_trial.params.items():
        if param in ['pooling_type', 'freeze_backbone']:
            best_config['model'][param] = value
        elif param in ['learning_rate', 'weight_decay', 'balanced_sampler']:
            best_config['train'][param] = value
    
    # Set the batch size based on freeze_backbone
    batch_size = study.best_trial.user_attrs.get('batch_size')
    if batch_size:
        best_config['train']['batch_size'] = batch_size
        best_config['train']['test_batch_size'] = batch_size
    
    # Save best configuration
    with open(best_config_path, 'w') as f:
        yaml.dump(best_config, f)
    
    print(f"  Best configuration: {best_config_path}")


def main():
    """Run hyperparameter search using Optuna."""
    parser = argparse.ArgumentParser(description='Hyperparameter optimization')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to hyperparameter search configuration')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
    
    # Load hyperparameter search configuration
    with open(args.config, 'r') as f:
        hp_config = yaml.safe_load(f)
    
    # Get study settings
    study_name = hp_config.get('study_name', f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    n_trials = hp_config.get('n_trials', 16)
    optimization_direction = hp_config.get('optimization_direction', 'minimize')
    timeout = hp_config.get('timeout', None)
    
    # Extract model type from the base config
    with open(hp_config['base_config'], 'r') as f:
        base_config = yaml.safe_load(f)
    model_type = base_config['model']['encoder_type']
    
    print(f"Starting hyperparameter search for {model_type}")
    print(f"Study name: {study_name}")
    print(f"Number of trials: {n_trials}")
    print(f"Optimization direction: {optimization_direction}")
    print(f"Max epochs per trial: {hp_config.get('max_epochs', 15)}")
    print(f"Batch size configuration: {hp_config.get('batch_size_config', 'default')}")
    
    # Create sampler
    search_space = hp_config['search_space']
    if n_trials >= 2 ** len(search_space):
        print("Using GridSampler for exhaustive search")
        sampler = optuna.samplers.GridSampler(search_space)
    else:
        print("Using TPESampler for efficient search")
        sampler = optuna.samplers.TPESampler(seed=args.seed)
    
    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction=optimization_direction,
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Run optimization
    start_time = time.time()
    study.optimize(
        lambda trial: objective(trial, args.config),
        n_trials=n_trials,
        timeout=timeout
    )
    total_time = time.time() - start_time
    
    # Print results
    print(f"\nStudy completed in {total_time/3600:.2f} hours")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_trial.value:.4f}")
    print("Best hyperparameters:")
    for param, value in study.best_trial.params.items():
        print(f"  {param}: {value}")
    
    batch_size = study.best_trial.user_attrs.get('batch_size', 'unknown')
    print(f"  batch_size: {batch_size} (derived from freeze_backbone)")
    
    # Print final metrics if available
    for key, value in study.best_trial.user_attrs.items():
        if key.startswith('final_'):
            print(f"{key.replace('final_', 'Final ')}: {value:.4f}")
    
    # Save results
    save_study_results(study, hp_config)


if __name__ == "__main__":
    main()