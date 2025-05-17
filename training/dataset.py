from functools import partial
import os
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image, ImageFile
import time
from transformers import BertTokenizer, DistilBertTokenizer
from data.transforms import BackTranslator, TextTransforms, RandomTextTransform
import pandas as pd

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Helper function to double dataset with back-translations
def augment_dataframe_with_backtranslation(df, text_generator_fn=None, batch_size=32, save_path=None, force_regenerate=False):
    """
    Double the size of a dataframe by adding back-translated versions of text descriptions.
    
    Args:
        df: DataFrame to augment
        text_generator_fn: Function to generate text descriptions if 'generated_text' column doesn't exist
        batch_size: Batch size for translation
        save_path: Path to save/load the augmented dataframe
        force_regenerate: Whether to force regeneration even if saved file exists
        
    Returns:
        Augmented DataFrame with original and back-translated examples
    """
    # Check if saved file exists
    if save_path and not force_regenerate:
        try:
            saved_df = pd.read_csv(save_path)
            print(f"Loaded pre-augmented dataset from {save_path} with {len(saved_df)} examples")
            return saved_df
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f"No existing augmented dataset found at {save_path}, generating new one...")
    
    # First ensure we have the generated_text column
    if 'generated_text' not in df.columns:
        if text_generator_fn is None:
            raise ValueError("text_generator_fn must be provided if 'generated_text' column doesn't exist")
        print("Generating original text descriptions...")
        df['generated_text'] = df.apply(text_generator_fn, axis=1)
    
    # Get original text descriptions
    original_texts = df['generated_text'].tolist()
    
    # Initialize back-translator with lighter model
    back_translator = BackTranslator()
    
    # Perform back-translation
    print(f"Back-translating {len(original_texts)} texts...")
    back_translated_texts = back_translator.back_translate(original_texts, batch_size)
    
    # Create a copy of the original dataframe for the augmented examples
    aug_df = df.copy()
    aug_df['generated_text'] = back_translated_texts
    aug_df['is_backtranslated'] = True
    
    # Add is_backtranslated column to original df (all False)
    df['is_backtranslated'] = False
    
    # Concatenate original and augmented dataframes
    combined_df = pd.concat([df, aug_df], ignore_index=True)
    
    print(f"Dataset doubled from {len(df)} to {len(combined_df)} examples")
    
    # Save to CSV if path provided
    if save_path:
        combined_df.to_csv(save_path, index=False)
        print(f"Saved augmented dataset to {save_path}")
    
    return combined_df


class PropertyImageDataset(Dataset):
    """
    Dataset for property price prediction from multiple images.
    """
    def __init__(self, 
                 dataframe, 
                 data_root,
                 max_images=15, 
                 transform=None, 
                 random_sample=True,
                 phase='train',
                 use_text=False,
                 use_images=True,
                 text_generator_fn=None,
                 text_model_name=None,
                 max_text_length=None,
                 text_transform=None,
                 use_backtranslation=True,
                 backtranslation_cache_path="aug/augmented.csv"):
        """
        Args:
            dataframe: DataFrame containing property data
            data_root: Root directory for image data
            max_images: Maximum number of images to use per property
            transform: Image transforms
            random_sample: Whether to randomly sample images when count > max_images
            phase: 'train' or 'test'
            use_text: Whether to use text descriptions
            use_images: Whether to use images
            text_generator_fn: Function to generate text descriptions from dataframe rows
            model_name: Name of the pretrained model to use for tokenizer
            max_text_length: Maximum length of tokenized text
            text_transform: Text transforms
            use_backtranslation: Whether to double the dataset with back-translations
        """
        self.df = dataframe
        self.data_root = data_root
        self.max_images = max_images
        self.random_sample = random_sample
        self.phase = phase
        self.use_text = use_text
        self.use_images = use_images
        self.text_generator_fn = text_generator_fn
        self.max_text_length = max_text_length
        self.text_transform = text_transform

        assert use_text or use_images, "At least one of use_text or use_images must be True"

        # If using images, cache directory structure to speed up file access (might be more efficient for IO)
        if self.use_images:
            self._cache_directories()
        
        # If using text, pre load all text descriptions
        if self.use_text:
            # Backtranslation to double the data
            if use_backtranslation and phase == 'train':
                self.df = augment_dataframe_with_backtranslation(
                    self.df, text_generator_fn=text_generator_fn, save_path=backtranslation_cache_path
                )
                # Text descriptions will now be in the generated_text column
                self.text_descriptions = self.df['generated_text'].tolist()
            else:
                if 'generated_text' in self.df.columns:
                    self.text_descriptions = self.df['generated_text'].tolist()
                elif self.text_generator_fn is not None:
                    print("Generating text descriptions...")
                    self.df['generated_text'] = self.df.apply(self.text_generator_fn, axis=1)
                    self.text_descriptions = self.df['generated_text'].tolist()
                else:
                    raise ValueError("Text generation function must be provided if 'generated_text' column doesn't exist")


            if 'distilbert' in text_model_name.lower():
                self.tokenizer = DistilBertTokenizer.from_pretrained(text_model_name)
            elif 'bert' in text_model_name.lower():
                self.tokenizer = BertTokenizer.from_pretrained(text_model_name)
            else:
                raise ValueError("Model not supported")
            

            # Set up text transformations (only for training)
            if self.phase == 'train' and text_transform is None:
                # Define the transform functions for random deletion and synonym replacement
                transforms_list = [
                    partial(TextTransforms.random_deletion, p=0.1),
                    partial(TextTransforms.synonym_replacement, p=0.2)
                ]
                self.text_transform = RandomTextTransform(
                    transforms=transforms_list,
                    p=0.5
                )
            else:
                self.text_transform = text_transform


        if self.use_images:
            # Define transforms based on phase
            if transform is None:
                if phase == 'train':
                    # Data augmentation for training
                    self.transform = transforms.Compose([
                        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225]),
                    ])
                else:
                    # Only resize and normalize for validation/testing
                    self.transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])
                    ])
            else:
                self.transform = transform
            
        # Stats for logging
        self.load_time_stats = {"get_files": 0.0, "load_images": 0.0, "transform": 0.0}
        self.num_calls = 0
    
    def _cache_directories(self):
        """Cache directory structure to improve file access performance"""
        self.directory_cache = {}
        
        # Create a set of all unique (data_source, directory_name) pairs
        unique_dirs = set(zip(self.df['data_source'], self.df['directory_name']))
        
        print(f"Caching {len(unique_dirs)} unique property directories...")
        for data_source, directory_name in unique_dirs:
            img_dir = os.path.join(self.data_root, data_source, directory_name)
            if os.path.exists(img_dir):
                try:
                    image_files = [
                        f for f in os.listdir(img_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                    ]
                    
                    self.directory_cache[(data_source, directory_name)] = image_files
                except Exception as e:
                    print(f"Error caching directory {img_dir}: {e}")
                    self.directory_cache[(data_source, directory_name)] = []
            else:
                self.directory_cache[(data_source, directory_name)] = []
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """Get a single property sample with its images"""
        start_time = time.time()
        
        # Get sample from dataframe
        sample = self.df.iloc[idx]
        
        # Get targets
        regression_target = torch.tensor(sample['price_normalized'], dtype=torch.float32)
        classification_target = torch.tensor(sample['price_cat_encoded'], dtype=torch.long)


        if self.use_text:

            text_description = self.text_descriptions[idx]
            if text_description is None or text_description == "":
                raise ValueError(f"Empty text description at index {idx}")

            if self.text_transform is not None and self.phase == 'train':
                text_description = self.text_transform(text_description)

            text_inputs = self.tokenizer(
                text=text_description,
                add_special_tokens=True,
                max_length=self.max_text_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Remove the batch dimension (will be added by collate_fn)
            text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}


        # Process images if using images
        images_tensor = None
        attention_mask = None
        
        if self.use_images:
            # Get image paths
            data_source = sample['data_source']
            directory_name = sample['directory_name']
            img_dir = os.path.join(self.data_root, data_source, directory_name)
            
            get_files_start = time.time()
            
            # Get image files from cache
            cache_key = (data_source, directory_name)
            if cache_key in self.directory_cache:
                image_files = self.directory_cache[cache_key]
            else:
                # Fallback to direct file system access if not in cache
                if os.path.exists(img_dir):
                    image_files = [
                        f for f in os.listdir(img_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                    ]
                else:
                    image_files = []
                
                # Add to cache
                self.directory_cache[cache_key] = image_files
            
            # Sample if needed
            if len(image_files) > self.max_images:
                if self.random_sample and self.phase == 'train':
                    image_files = random.sample(image_files, self.max_images)
                else:
                    # Take first max_images (highest ranked)
                    image_files = image_files[:self.max_images]
            
            get_files_time = time.time() - get_files_start
            self.load_time_stats["get_files"] += get_files_time
            
            # Load and transform images
            load_images_start = time.time()
            images = []
            for img_file in image_files[:self.max_images]:
                img_path = os.path.join(img_dir, img_file)
                try:
                    # Open and resize in one step to reduce memory usage
                    with Image.open(img_path) as img_original:
                        img = img_original.copy().convert('RGB')
                        transform_start = time.time()
                        img_tensor = self.transform(img)
                        self.load_time_stats["transform"] += time.time() - transform_start
                        images.append(img_tensor)
                        del img
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    continue
            
            self.load_time_stats["load_images"] += time.time() - load_images_start
            
            # Create attention mask and pad to max_images
            num_images = len(images)
            attention_mask = torch.zeros(self.max_images, dtype=torch.bool)
            attention_mask[:num_images] = True
            
            # Handle empty case
            if num_images == 0:
                raise FileNotFoundError(f"No valid images found for {img_dir}")
            
            # Pad images if needed
            if num_images < self.max_images:
                padding = [torch.zeros_like(images[0]) for _ in range(self.max_images - num_images)]
                images.extend(padding)
            
            # Stack images
            images_tensor = torch.stack(images)  # [N, C, H, W]
        
        # Return based on which modalities are enabled
        if self.use_images and self.use_text:
            return images_tensor, attention_mask, text_inputs, regression_target, classification_target
        elif self.use_images:
            return images_tensor, attention_mask, regression_target, classification_target
        elif self.use_text:
            return text_inputs, regression_target, classification_target
    
    @staticmethod
    def collate_fn(batch, use_text=False, use_images=True):
        """
        Custom collate function for the dataloader.
        """
        if use_images and use_text:
            # Both modalities
            images, masks, text_inputs, regression_targets, classification_targets = zip(*batch)
            images = torch.stack(images)  # [B, N, C, H, W]
            masks = torch.stack(masks)    # [B, N]
            regression_targets = torch.stack(regression_targets)  # [B]
            classification_targets = torch.stack(classification_targets)  # [B]
            
            # Collate text inputs
            text_batch = {}
            for key in text_inputs[0].keys():
                text_batch[key] = torch.stack([item[key] for item in text_inputs])
            
            return images, masks, text_batch, regression_targets, classification_targets
        
        elif use_images:
            # Images only
            images, masks, regression_targets, classification_targets = zip(*batch)
            images = torch.stack(images)  # [B, N, C, H, W]
            masks = torch.stack(masks)    # [B, N]
            regression_targets = torch.stack(regression_targets)  # [B]
            classification_targets = torch.stack(classification_targets)  # [B]
            return images, masks, regression_targets, classification_targets
        
        elif use_text:
            # Text only
            text_inputs, regression_targets, classification_targets = zip(*batch)
            regression_targets = torch.stack(regression_targets)  # [B]
            classification_targets = torch.stack(classification_targets)  # [B]
            
            # Collate text inputs
            text_batch = {}
            for key in text_inputs[0].keys():
                text_batch[key] = torch.stack([item[key] for item in text_inputs])
            
            return text_batch, regression_targets, classification_targets