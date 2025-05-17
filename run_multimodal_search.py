import os
import subprocess
import argparse

def main():
    """Run hyperparameter search for multimodal models."""
    parser = argparse.ArgumentParser(description='Run hyperparameter search for multimodal models')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Define configurations
    config = "config/hyperparameters/multimodal_search_without_freeze.yaml"
    
    print("="*80)
    print(f"Running multimodal hyperparameter search with config: {config}")
    print("="*80)
    
    # Run the hyperparameter search script
    subprocess.run([
        "python3", "hyperparameter_search.py",
        "--config", config,
        "--seed", str(args.seed),
    ])

if __name__ == "__main__":
    main()