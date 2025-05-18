import os
import torch
import yaml
import argparse
from datetime import datetime
import numpy as np
import random
from torch.utils.data import DataLoader
import pandas as pd

from models.contrastive_model import ContrastiveModel
from training.contrastive_trainer import ContrastiveTrainer
from training.dataset import PropertyImageDataset
from data.data_utils import preprocess_dataframes, save_preprocessing_params
from data.data_utils import generate_property_description
from utils.visualization import plot_contrastive_history

import warnings
from PIL import Image
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

def parse_args():
    parser = argparse.ArgumentParser(description='Contrastive pretraining for property price prediction')
    parser.add_argument('--config', type=str, default='config/contrastive_pretraining.yaml',
                        help='Path to config file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (reduced dataset size)')
    return parser.parse_args()

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

def save_compatible_encoders(model, experiment_dir, epoch, val_metrics):
    """
    Save encoders in a format compatible with the main.py loading mechanism.
    """
    # Create directory for encoders
    encoders_dir = os.path.join(experiment_dir, 'encoders')
    os.makedirs(encoders_dir, exist_ok=True)
    
    # Save image encoder in a format that matches the expected format in main.py
    image_encoder_state = {
        'model_state_dict': {
            'image_encoder': model.image_encoder.state_dict()
        },
        'epoch': epoch,
        'metrics': val_metrics,
        'type': 'contrastive_pretrained'
    }
    torch.save(image_encoder_state, os.path.join(encoders_dir, 'image_encoder_checkpoint.pt'))
    
    # Save text encoder in a format compatible with the main.py loading mechanism
    # We save with 'text_encoder.' prefix to match the expected format
    text_encoder_dict = {}
    for k, v in model.text_encoder.state_dict().items():
        text_encoder_dict[f'text_encoder.{k}'] = v
    
    text_encoder_state = {
        'model_state_dict': text_encoder_dict,
        'epoch': epoch,
        'metrics': val_metrics,
        'type': 'contrastive_pretrained'
    }
    torch.save(text_encoder_state, os.path.join(encoders_dir, 'text_encoder_checkpoint.pt'))
    
    print(f"Saved compatible encoder checkpoints to {encoders_dir}")
    
    # Also save a config example for using these encoders
    example_config = {
        'model': {
            'image_checkpoint': os.path.join(encoders_dir, 'image_encoder_checkpoint.pt'),
            'text_checkpoint': os.path.join(encoders_dir, 'text_encoder_checkpoint.pt'),
            'freeze_backbone': True,  # Recommend freezing the pretrained encoders
            'freeze_text_encoder': True
        }
    }
    
    with open(os.path.join(experiment_dir, 'example_usage.yaml'), 'w') as f:
        yaml.dump(example_config, f, default_flow_style=False)

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    seed_everything(args.seed)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"contrastive_{timestamp}"
    
    # Create experiment directory
    log_dir = config['logging']['log_dir']
    experiment_dir = os.path.join(log_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"Experiment directory: {experiment_dir}")
    
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
    if args.debug:
        print("Debug mode: using reduced dataset size")
        train_df_raw = train_df_raw.sample(min(1000, len(train_df_raw)), random_state=args.seed)
        test_df_raw = test_df_raw.sample(min(100, len(test_df_raw)), random_state=args.seed)
    
    print(f"Training set shape: {train_df_raw.shape}")
    print(f"Testing set shape: {test_df_raw.shape}")
    
    # Preprocess data
    train_df, test_df, preprocessing_params = preprocess_dataframes(train_df_raw, test_df_raw, config)
    
    # Save preprocessing parameters
    save_preprocessing_params(preprocessing_params, experiment_dir)
    
    # Create datasets
    train_dataset = PropertyImageDataset(
        dataframe=train_df,
        data_root=data_path,
        max_images=config['data']['max_images'],
        random_sample=config['data']['random_sample'],
        phase='train',
        use_text=True,
        use_images=True,
        text_generator_fn=generate_property_description,
        text_model_name=config['model']['text_encoder_model'],
        max_text_length=config['model'].get('max_text_length', 512)
    )
    
    test_dataset = PropertyImageDataset(
        dataframe=test_df,
        data_root=data_path,
        max_images=config['data']['max_images'],
        random_sample=False,
        phase='test',
        use_text=True,
        use_images=True,
        text_generator_fn=generate_property_description,
        text_model_name=config['model']['text_encoder_model'],
        max_text_length=config['model'].get('max_text_length', 512)
    )
    
    # Create data loaders
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    # Create a custom collate function for multimodal data
    def collate_fn_multimodal(batch):
        return PropertyImageDataset.collate_fn(batch, use_text=True, use_images=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['contrastive']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers'],
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn_multimodal
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['contrastive']['batch_size'],
        shuffle=False,
        num_workers=config['train']['num_workers'],
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn_multimodal
    )
    
    # Create model
    model = ContrastiveModel(
        encoder_type=config['model']['encoder_type'],
        text_encoder_model=config['model']['text_encoder_model'],
        text_encoder_type=config['model']['text_encoder_type'],
        freeze_backbone=config['model']['freeze_backbone'],
        freeze_text_encoder=config['model']['freeze_text_encoder'],
        projection_dim=config['contrastive']['projection_dim'],
        temperature=config['contrastive']['temperature'],
        pretrained=config['model'].get('use_pretrained', True)
    )
    
    model = model.to(device)
    
    # Create trainer
    trainer = ContrastiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        config=config,
        experiment_dir=experiment_dir
    )
    
    # Train model
    history = trainer.train(config['contrastive']['num_epochs'])
    
    # Plot training history
    plot_contrastive_history(history, experiment_dir)
    
    # Save compatible encoders for easy loading with main.py
    save_compatible_encoders(
        model=model,
        experiment_dir=experiment_dir,
        epoch=trainer.best_epoch,
        val_metrics={'contrastive_loss': trainer.best_val_loss}
    )
    
    # Save compatible encoders for easy loading with main.py
    save_compatible_encoders(
        model=model,
        experiment_dir=experiment_dir,
        epoch=trainer.best_epoch,
        val_metrics={'contrastive_loss': trainer.best_val_loss}
    )
    
    print(f"Training complete! Results saved to {experiment_dir}")
    print(f"To use the pretrained encoders with your existing setup, add this to your config:")
    print(f"model:")
    print(f"  image_checkpoint: '{os.path.join(experiment_dir, 'encoders', 'image_encoder_checkpoint.pt')}'")
    print(f"  text_checkpoint: '{os.path.join(experiment_dir, 'encoders', 'text_encoder_checkpoint.pt')}'")
    print(f"  freeze_backbone: true  # Recommended to freeze pretrained encoders")
    print(f"  freeze_text_encoder: true")

if __name__ == "__main__":
    main()