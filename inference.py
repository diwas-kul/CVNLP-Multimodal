import os
import torch
import argparse
import json
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

from models.property_model import PropertyPriceModel
from data.data_utils import load_preprocessing_params

def parse_args():
    parser = argparse.ArgumentParser(description='Inference for property price prediction model')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory with model checkpoint and preprocessing parameters')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory with property images')
    parser.add_argument('--max_images', type=int, default=15,
                        help='Maximum number of images to use')
    return parser.parse_args()

def load_model(model_dir, device):
    """
    Load trained model and preprocessing parameters.
    
    Args:
        model_dir: Directory with model checkpoint and preprocessing parameters
        device: Device to load model on
        
    Returns:
        model: Loaded model
        preprocessing_params: Preprocessing parameters
    """
    # Load preprocessing parameters
    params_path = os.path.join(model_dir, 'preprocessing_params.json')
    preprocessing_params = load_preprocessing_params(params_path)
    
    # Load model configuration
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create model
    model = PropertyPriceModel(
        num_classes=preprocessing_params['num_classes'],
        encoder_type=config['model']['encoder_type'],
        pooling_type=config['model']['pooling_type'],
        freeze_backbone=False,  # Not needed for inference
        dropout_rate=config['model']['dropout_rate'],
        pretrained=False  # Not needed for inference
    )
    
    # Load checkpoint
    checkpoint_path = os.path.join(model_dir, 'best_model.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, preprocessing_params

def predict_property_price(model, image_paths, preprocessing_params, device, max_images=15):
    """
    Predict property price and category from images.
    
    Args:
        model: Trained model
        image_paths: List of image paths
        preprocessing_params: Preprocessing parameters
        device: Device to use
        max_images: Maximum number of images to use
        
    Returns:
        dict: Prediction results
    """
    # Prepare image transform
    transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
    # Load images
    images = []
    for img_path in image_paths[:max_images]:
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            images.append(img_tensor)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    # Check if we have any images
    if not images:
        return {"error": "No valid images found"}
    
    # Create attention mask
    num_images = len(images)
    attention_mask = torch.zeros(max_images, dtype=torch.bool)
    attention_mask[:num_images] = True
    
    # Pad images if needed
    if num_images < max_images:
        padding = [torch.zeros_like(images[0]) for _ in range(max_images - num_images)]
        images.extend(padding)
    
    # Stack images and add batch dimension
    images_tensor = torch.stack(images).unsqueeze(0)  # [1, N, C, H, W]
    attention_mask = attention_mask.unsqueeze(0)      # [1, N]
    
    # Move to device
    images_tensor = images_tensor.to(device)
    attention_mask = attention_mask.to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(images_tensor, attention_mask)
    
    # Process regression output
    normalized_price = outputs['regression'].cpu().numpy()[0]
    predicted_price = normalized_price * preprocessing_params['price_std'] + preprocessing_params['price_mean']
    
    # Process classification output
    cls_probs = torch.softmax(outputs['classification'], dim=1).cpu().numpy()[0]
    predicted_class_idx = np.argmax(cls_probs)
    predicted_class = preprocessing_params['class_names'][predicted_class_idx]
    
    # Get top 3 class probabilities
    top_indices = np.argsort(cls_probs)[::-1][:3]
    top_classes = [preprocessing_params['class_names'][i] for i in top_indices]
    top_probs = [cls_probs[i] for i in top_indices]
    
    return {
        "predicted_price": float(predicted_price),
        "predicted_class": predicted_class,
        "top_classes": top_classes,
        "top_probabilities": [float(p) for p in top_probs],
        "num_images_used": num_images
    }

def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and preprocessing parameters
    print(f"Loading model from {args.model_dir}...")
    model, preprocessing_params = load_model(args.model_dir, device)
    
    # Get image paths
    image_dir = args.image_dir
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_paths = [os.path.join(image_dir, f) for f in image_files]
    
    print(f"Found {len(image_paths)} images in {image_dir}")
    
    # Predict
    result = predict_property_price(
        model, 
        image_paths, 
        preprocessing_params, 
        device, 
        max_images=args.max_images
    )
    
    # Print results
    print("\nPrediction Results:")
    print(f"Predicted Price: €{result['predicted_price']:.2f}")
    print(f"Predicted Price Category: {result['predicted_class']}")
    print("\nTop 3 Price Categories:")
    for i, (cls, prob) in enumerate(zip(result['top_classes'], result['top_probabilities'])):
        print(f"  {i+1}. {cls} ({prob:.2%})")
    
    # Save results
    output_file = os.path.join(args.image_dir, "prediction_result.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()