import os
import subprocess
import argparse

def main():
    """Run hyperparameter search for both ResNet50 and ViT models."""
    parser = argparse.ArgumentParser(description='Run hyperparameter search for multiple models')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Define configurations
    configs = [
        # Without Balanced Sampling
        "config/hyperparameters/resnet_50_no_balanced_sampler.yaml",
        "config/hyperparameters/vit_b_no_balanced_sampler.yaml",

        # With Balanced Sampling
        "config/hyperparameters/resnet_50_balanced_sampler.yaml",
        "config/hyperparameters/vit_b_balanced_sampler.yaml",
    ]
    
    # Run search for each model
    for config in configs:
        print("="*80)
        print(f"Running hyperparameter search with config: {config}")
        print("="*80)
        
        # Run the hyperparameter search script
        subprocess.run([
            "python3", "hyperparameter_search.py",
            "--config", config,
            "--seed", str(args.seed)
        ])
        
        print("\n\n")

if __name__ == "__main__":
    main()