import os
import torch
import torch.multiprocessing as mp
import yaml
import json
import argparse
from torch.utils.data import DataLoader
import pandas as pd
from datetime import datetime
import random
import numpy as np
import warnings
from PIL import Image, ImageFile
import matplotlib
from data.data_utils import generate_property_description
matplotlib.use('Agg')

# Silence the DecompressionBombWarning
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# Set PyTorch memory optimization flags
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False  # Set to True only if reproducibility is critical

from data.data_utils import preprocess_dataframes, save_preprocessing_params, create_balanced_sampler
from models.property_model import PropertyPriceModel
from training.dataset import PropertyImageDataset
from training.trainer import Trainer
from utils.visualization import plot_training_history, plot_predictions, plot_confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser(description='Train property price prediction model')
    parser.add_argument('--config', type=str, default='config/resnet50_attention.yaml',
                        help='Path to config file')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Experiment name (default: timestamp)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (reduced dataset size)')
    return parser.parse_args()


def safe_load_checkpoint(checkpoint_path, device):
    """Safely load checkpoint with PyTorch 2.6+ compatibility."""
    try:
        # First try with default settings
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint with default settings: {e}")
        # Add numpy.core.multiarray.scalar to safe globals
        torch.serialization.add_safe_globals(['numpy.core.multiarray.scalar'])
        try:
            # Try with weights_only=False
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            return checkpoint
        except Exception as e2:
            print(f"Failed to load checkpoint with weights_only=False: {e2}")
            raise RuntimeError(f"Failed to load checkpoint: {checkpoint_path}")


def seed_everything(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Make CUDA operations deterministic
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """Set random seed for each worker"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def collate_fn_images_only(batch):
    return PropertyImageDataset.collate_fn(batch, use_text=False, use_images=True)

def collate_fn_text_only(batch):
    return PropertyImageDataset.collate_fn(batch, use_text=True, use_images=False)

def collate_fn_multimodal(batch):
    return PropertyImageDataset.collate_fn(batch, use_text=True, use_images=True)

# For hyperparameter search, we can call the run_experiment function directly
def run_experiment(config=None):
    """
    Run a single experiment with the given configuration.
    
    Args:
        config: Configuration dictionary (if None, load from command-line args)
        
    Returns:
        results: Dictionary with experiment results
    """

    results = {
        'best_val_loss': float('inf'),
        'best_epoch': -1,
        'history': {},
        'experiment_dir': None
    }

    # If no config provided, parse arguments
    if config is None:
        args = parse_args()
        
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            
        # Set experiment name
        if args.experiment:
            experiment_name = args.experiment
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"
            
        # Set debug mode
        debug_mode = args.debug
        
        # Set random seed
        seed = args.seed
    else:
        # Use provided config
        experiment_name = config.get('experiment', f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        debug_mode = config.get('debug', False)
        seed = config.get('seed', 42)
    
    # Set random seed
    seed_everything(seed)
    
    # Create experiment directory
    log_dir = config['logging']['log_dir']
    experiment_dir = os.path.join(log_dir, experiment_name)
    
    # If the base directory already exists, add a numeric suffix
    if os.path.exists(experiment_dir):
        suffix = 1
        while True:
            experiment_name_with_suffix = f"{experiment_name}_{suffix:03d}"
            experiment_dir = os.path.join(log_dir, experiment_name_with_suffix)
            if not os.path.exists(experiment_dir):
                break
            suffix += 1
        experiment_name = experiment_name_with_suffix
    
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"Experiment directory: {experiment_dir}")
    results['experiment_dir'] = experiment_dir

    # Save configuration
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    data_path = config['data']['data_path']
    train_file = os.path.join(data_path, config['data']['train_file'])
    test_file = os.path.join(data_path, config['data']['test_file'])
    
    train_df_raw = pd.read_csv(train_file)
    test_df_raw = pd.read_csv(test_file)
    
    # Use subset of data in debug mode
    if debug_mode:
        print("Debug mode: using reduced dataset size")
        train_df_raw = train_df_raw.sample(min(1000, len(train_df_raw)), random_state=seed)
        test_df_raw = test_df_raw.sample(min(100, len(test_df_raw)), random_state=seed)
    
    print(f"Training set shape: {train_df_raw.shape}")
    print(f"Testing set shape: {test_df_raw.shape}")
    
    # Preprocess data
    train_df, test_df, preprocessing_params = preprocess_dataframes(train_df_raw, test_df_raw, config)
    
    # Save preprocessing parameters
    save_preprocessing_params(preprocessing_params, experiment_dir)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------- DATASETS AND LOADERS -----------------------------------------------------------------------

    # Create datasets
    train_dataset = PropertyImageDataset(
        dataframe=train_df,
        data_root=data_path,
        max_images=config['data']['max_images'] if config['data'].get('use_images', False) else 0,
        random_sample=config['data']['random_sample'] if config['data'].get('use_images', False) else False,
        phase='train',
        use_text=config['data'].get('use_text', False),
        use_images=config['data'].get('use_images', False),
        text_generator_fn=generate_property_description if config['data'].get('use_text', False) else None,
        text_model_name = config['model'].get('text_encoder_model', None),
        max_text_length = config['model'].get('max_text_length', None)
    )
    
    test_dataset = PropertyImageDataset(
        dataframe=test_df,
        data_root=data_path,
        max_images=config['data']['max_images'] if config['data'].get('use_images', False) else 0,
        random_sample=False,
        phase='test',
        use_text=config['data'].get('use_text', False),
        use_images=config['data'].get('use_images', False),
        text_generator_fn=generate_property_description if config['data'].get('use_text', False) else None,
        text_model_name = config['model'].get('text_encoder_model', None),
        max_text_length = config['model'].get('max_text_length', None)
    )
    
    if config['train']['balanced_sampler']:
        print("Using balanced sampling for training...")
        train_sampler = create_balanced_sampler(train_dataset)
        train_shuffle = False  # When using a sampler, shuffle must be False
    else:
        train_sampler = None
        train_shuffle = True

    use_text = config['data'].get('use_text', False)
    use_images = config['data'].get('use_images', True)

    if use_text and use_images:
        collate_fn = collate_fn_multimodal
    elif use_text:
        collate_fn = collate_fn_text_only
    else:
        collate_fn = collate_fn_images_only
    
    # Create data loaders with fixed random seed
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=train_shuffle,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=config['train']['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
        prefetch_factor=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['train']['test_batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['train']['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
        prefetch_factor=2
    )
    
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------- MODEL --------------------------------------------------------------------------
    # Create model
    model_params = {
        'num_classes': preprocessing_params['num_classes'],
        'use_text': config['data'].get('use_text', False),
        'use_images': config['data'].get('use_images', False),
        'dropout_rate': config['model']['dropout_rate'],
        'pretrained': config['model'].get('use_pretrained', True),
    }
    image_checkpoint = None
    text_checkpoint = None

    # Add image-specific parameters only if using images
    if config['data'].get('use_images', False):
        model_params.update({
            'encoder_type': config['model']['encoder_type'],
            'pooling_type': config['model']['pooling_type'],
            'freeze_backbone': config['model']['freeze_backbone'],
        })
        image_checkpoint = torch.load(config['model']['image_checkpoint'], weights_only=True) if config['model'].get('image_checkpoint', False) else False

    # Add text-specific parameters only if using text
    if config['data'].get('use_text', False):
        model_params.update({
            'text_encoder_model': config['model'].get('text_encoder_model', 'all-mpnet-base-v2'),
            'text_encoder_type': config['model']['text_encoder_type'],
            'freeze_text_encoder': config['model'].get('freeze_text_encoder', False)
        })
        text_checkpoint = torch.load(config['model']['text_checkpoint'], weights_only=True) if config['model'].get('text_checkpoint', False) else False

    # Add fusion parameters only if using both modalities
    if config['data'].get('use_text', False) and config['data'].get('use_images', False):
        model_params.update({
            'fusion_type': config['model'].get('fusion_type', 'concat'),
            'modality_dropout': config['model'].get('modality_dropout', False)
        })
            
    model = PropertyPriceModel(**model_params)

    if image_checkpoint:
        model.image_encoder.load_state_dict(image_checkpoint['model_state_dict']['image_encoder'])
    if text_checkpoint:
        # Get the text encoder state dict
        checkpoint_state_dict = text_checkpoint['model_state_dict']
        
        # Create a new state dict with only the compatible parts
        text_encoder_state_dict = {}
        
        # Get current model state dict to check shapes
        current_model_state_dict = model.text_encoder.state_dict()
        
        # Extract keys with 'text_encoder.' prefix
        for k, v in checkpoint_state_dict.items():
            if k.startswith('text_encoder.'):
                # Remove the 'text_encoder.' prefix
                new_key = k[len('text_encoder.'):]
                
                # Check if shapes match
                if new_key in current_model_state_dict:
                    if current_model_state_dict[new_key].shape == v.shape:
                        text_encoder_state_dict[new_key] = v
                    else:
                        print(f"Skipping due to shape mismatch: {new_key}, " 
                            f"checkpoint: {v.shape}, model: {current_model_state_dict[new_key].shape}")
        
        # Load the compatible weights
        missing_keys, unexpected_keys = model.text_encoder.load_state_dict(text_encoder_state_dict, strict=False)
        print(f"Loaded text encoder with {len(text_encoder_state_dict)} compatible layers")
        print(f"Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")

    model = model.to(device)
    
    # Print model summary
    print(f"\nModel Architecture:")
    print(model)
    print(f"Number of classes: {preprocessing_params['num_classes']}")

    # Print modality information
    use_text = config['data'].get('use_text', False)
    use_images = config['data'].get('use_images', True)
    if use_text and use_images:
        print("Modality: Multimodal (Images + Text)")
    elif use_images:
        print("Modality: Images only")
    elif use_text:
        print("Modality: Text only")

    # Print image-specific information if images are used
    if use_images:
        print(f"Image Encoder: {config['model']['encoder_type']}")
        print(f"Pooling: {config['model']['pooling_type']}")
        print(f"Freeze backbone: {config['model']['freeze_backbone']}")

    # Print text-specific information if text is used
    if use_text:
        print(f"Text Encoder: {config['model'].get('text_encoder_model', 'all-mpnet-base-v2')}")

    # Print fusion information if both modalities are used
    if use_text and use_images:
        print(f"Fusion type: {config['model'].get('fusion_type', 'concat')}")

    # Print dropout rate (common parameter)
    print(f"Dropout rate: {config['model']['dropout_rate']}")
    
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        config=config,
        preprocessing_params=preprocessing_params,
        experiment_dir=experiment_dir,
        multimodal=use_text and use_images,
        multi_lr=config['model'].get('multi_lr', False)
    )
    
    # Train model
    history = trainer.train(config['train']['num_epochs'])
    results['history'] = history
    results['best_val_loss'] = trainer.best_val_loss
    results['best_epoch'] = trainer.best_epoch
    
    try:
        # Plot training history
        plot_training_history(history, experiment_dir)
        
        # Final evaluation
        print("\nFinal Evaluation:")
        from utils.metrics import compute_detailed_metrics
        
        model.eval()
        all_reg_preds = []
        all_reg_targets = []
        all_cls_preds = []
        all_cls_targets = []
        
        # Modality tracking (for multimodal models)
        img_contributions = []
        txt_contributions = []
        
        # Load best model for evaluation
        best_model_path = os.path.join(experiment_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            try:
                checkpoint = safe_load_checkpoint(best_model_path, device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')} for final evaluation")
            except Exception as e:
                print(f"Warning: Could not load best model for final evaluation: {e}")
                print("Using last trained model state for evaluation")
        else:
            print("Warning: Best model checkpoint not found, using last trained model state")
        
        # Determine which modalities are being used
        use_text = config['data'].get('use_text', False)
        use_images = config['data'].get('use_images', True)
        
        # Evaluate the model
        with torch.no_grad():
            for batch in test_loader:
                # Prepare batch based on modalities (similar to Trainer._prepare_batch)
                if use_text and use_images:
                    # Both modalities
                    images, masks, text_inputs, reg_targets, cls_targets = batch
                    images = images.to(device)
                    masks = masks.to(device)
                    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                    outputs = model(images, masks, text_inputs)
                    
                    # Track modality contributions if available
                    if hasattr(model, 'img_contribution') and hasattr(model, 'txt_contribution'):
                        img_contributions.append(model.img_contribution.item())
                        txt_contributions.append(model.txt_contribution.item())
                    
                elif use_images:
                    # Images only
                    images, masks, reg_targets, cls_targets = batch
                    images = images.to(device)
                    masks = masks.to(device)
                    outputs = model(images, masks)
                    
                elif use_text:
                    # Text only
                    text_inputs, reg_targets, cls_targets = batch
                    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                    outputs = model(text_inputs)
                
                # Record predictions
                all_reg_preds.extend(outputs['regression'].cpu().numpy())
                all_reg_targets.extend(reg_targets.cpu().numpy())
                all_cls_preds.extend(outputs['classification'].argmax(dim=1).cpu().numpy())
                all_cls_targets.extend(cls_targets.cpu().numpy())
                
                # Clear memory
                del batch, outputs
        
        all_reg_preds = np.array(all_reg_preds)
        all_reg_targets = np.array(all_reg_targets)
        all_cls_preds = np.array(all_cls_preds)
        all_cls_targets = np.array(all_cls_targets)
        
        # Compute detailed metrics
        metrics = compute_detailed_metrics(
            regression_predictions=all_reg_preds,
            regression_targets=all_reg_targets,
            classification_predictions=all_cls_preds,
            classification_targets=all_cls_targets,
            preprocessing_params=preprocessing_params
        )
        
        # Add modality contribution metrics if available
        if img_contributions and txt_contributions:
            avg_img_contrib = np.mean(img_contributions)
            avg_txt_contrib = np.mean(txt_contributions)
            contrib_ratio = avg_txt_contrib / (avg_img_contrib + 1e-6)
            
            metrics['modality_contributions'] = {
                'image': float(avg_img_contrib),
                'text': float(avg_txt_contrib),
                'text_to_image_ratio': float(contrib_ratio)
            }
            
            print(f"\nModality Contributions:")
            print(f"  Image: {avg_img_contrib:.4f}")
            print(f"  Text: {avg_txt_contrib:.4f}")
            print(f"  Text/Image Ratio: {contrib_ratio:.2f}")
        
        # Add metrics to results
        results['final_metrics'] = metrics
        
        # Save detailed metrics
        try:
            with open(os.path.join(experiment_dir, 'final_metrics.json'), 'w') as f:
                # Convert numpy types to native types for JSON serialization
                metrics_json = json.dumps(metrics, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
                f.write(metrics_json)
        except Exception as e:
            print(f"Error saving metrics to JSON: {e}")
        
        # Print summary
        print(f"Regression Metrics:")
        print(f"  MAE: {metrics['regression']['mae']:.0f}")
        print(f"  RMSE: {metrics['regression']['rmse']:.0f}")
        print(f"  R²: {metrics['regression']['r2']:.4f}")
        print(f"  Mean Percent Error: {metrics['regression']['mean_percent_error']:.2f}%")
        
        print(f"\nClassification Metrics:")
        print(f"  Accuracy: {metrics['classification']['accuracy']:.4f}")
        
        # Try to plot predictions
        try:
            # Plot predictions
            denorm_preds = preprocessing_params['price_mean'] + preprocessing_params['price_std'] * all_reg_preds
            denorm_targets = preprocessing_params['price_mean'] + preprocessing_params['price_std'] * all_reg_targets
            
            plot_predictions(denorm_preds, denorm_targets, experiment_dir)
            
            # Plot confusion matrix
            plot_confusion_matrix(
                all_cls_targets, 
                all_cls_preds,
                preprocessing_params['class_names'],
                experiment_dir
            )
        except Exception as e:
            print(f"Error in visualization: {e}")
        
    except Exception as e:
        print(f"Error in final evaluation: {e}")
    
    # Clean up data loaders
    try:
        train_loader._iterator = None
        test_loader._iterator = None
    except:
        pass
    
    # Clear memory
    torch.cuda.empty_cache()
    
    print(f"\nTraining complete! Results saved to {experiment_dir}")
    
    return results

if __name__ == "__main__":
    # Use spawn instead of fork for multiprocessing
    mp.set_start_method('spawn', force=True)
    run_experiment()