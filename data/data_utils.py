import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import WeightedRandomSampler

def preprocess_dataframes(train_df, test_df, config):
    """
    Preprocess the dataframes:
    - Add normalized price column based on training data statistics
    - Encode price categories
    - Return the normalization parameters for later use
    
    Args:
        train_df: Training dataframe
        test_df: Testing dataframe
        config: Configuration dictionary
        
    Returns:
        train_df: Processed training dataframe
        test_df: Processed testing dataframe
        preprocessing_params: Dictionary with preprocessing parameters
    """
    # Get target column names
    regression_target = config['data']['regression_target']
    classification_target = config['data']['classification_target']
    
    # Normalize regression target
    price_mean = train_df[regression_target].mean()
    price_std = train_df[regression_target].std()
    
    train_df['price_normalized'] = (train_df[regression_target] - price_mean) / price_std
    test_df['price_normalized'] = (test_df[regression_target] - price_mean) / price_std
    
    # Encode classification target
    label_encoder = LabelEncoder()
    train_df['price_cat_encoded'] = label_encoder.fit_transform(train_df[classification_target])
    test_df['price_cat_encoded'] = label_encoder.transform(test_df[classification_target])
    
    # Store preprocessing parameters
    preprocessing_params = {
        'price_mean': float(price_mean),
        'price_std': float(price_std),
        'num_classes': len(label_encoder.classes_),
        'class_names': label_encoder.classes_.tolist(),
        'label_encoder': label_encoder
    }
    
    print(f"Price normalization: mean = {price_mean:.2f}, std = {price_std:.2f}")
    print(f"Normalized price range: [{train_df['price_normalized'].min():.2f}, {train_df['price_normalized'].max():.2f}]")
    print(f"Number of price categories: {preprocessing_params['num_classes']}")
    
    return train_df, test_df, preprocessing_params

def save_preprocessing_params(preprocessing_params, output_dir):
    """
    Save preprocessing parameters to a file.
    
    Args:
        preprocessing_params: Dictionary with preprocessing parameters
        output_dir: Directory to save the parameters
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a copy of the params without the LabelEncoder (not JSON serializable)
    params_to_save = {k: v for k, v in preprocessing_params.items() if k != 'label_encoder'}
    
    with open(os.path.join(output_dir, 'preprocessing_params.json'), 'w') as f:
        json.dump(params_to_save, f, indent=2)
    
    print(f"Preprocessing parameters saved to {os.path.join(output_dir, 'preprocessing_params.json')}")

def load_preprocessing_params(params_path):
    """
    Load preprocessing parameters from a file.
    
    Args:
        params_path: Path to the preprocessing parameters file
        
    Returns:
        preprocessing_params: Dictionary with preprocessing parameters
    """
    with open(params_path, 'r') as f:
        preprocessing_params = json.load(f)
    
    # Recreate label encoder if needed
    if 'class_names' in preprocessing_params:
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(preprocessing_params['class_names'])
        preprocessing_params['label_encoder'] = label_encoder
    
    return preprocessing_params

def denormalize_predictions(normalized_predictions, preprocessing_params):
    """
    Convert normalized predictions back to original scale.
    
    Args:
        normalized_predictions: Normalized model outputs
        preprocessing_params: Dictionary with preprocessing parameters
        
    Returns:
        Predictions in original price scale
    """
    return normalized_predictions * preprocessing_params['price_std'] + preprocessing_params['price_mean']

def decode_class_predictions(class_indices, preprocessing_params):
    """
    Convert class indices to class names.
    
    Args:
        class_indices: Predicted class indices
        preprocessing_params: Dictionary with preprocessing parameters
        
    Returns:
        Class names corresponding to the indices
    """
    return preprocessing_params['label_encoder'].inverse_transform(class_indices)


def create_balanced_sampler(dataset):
    """
    Create a balanced sampler for handling class imbalance.
    
    Args:
        dataset: Dataset to sample from
        
    Returns:
        sampler: WeightedRandomSampler for balanced class sampling
    """
    import torch
    import numpy as np
    from torch.utils.data import WeightedRandomSampler
    import time
    
    print("Creating balanced sampler...")
    start_time = time.time()
    
    # Get all target labels
    targets = []
    for i in range(len(dataset)):
        try:
            cls_target = dataset.df.iloc[i]['price_cat_encoded']
            targets.append(cls_target)
        except:
            _, _, _, cls_target = dataset[i]
            targets.append(cls_target.item())
        
        # Print progress for large datasets
        if (i + 1) % 1000 == 0:
            print(f"Processed {i+1}/{len(dataset)} samples for balanced sampling")
    
    # Convert to numpy array
    targets = np.array(targets)
    
    # Calculate class weights (inverse frequency)
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    
    # Print class distribution
    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {class_weights}")
    
    # Assign weights to samples
    sample_weights = class_weights[targets]
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    print(f"Created balanced sampler in {time.time() - start_time:.2f}s")
    
    return sampler


def generate_property_description(row):
    """
    Generates a textual description for a property based on a DataFrame row,
    using the actual year built and omitting zero bedrooms.

    Args:
        row (pd.Series): A row from the property DataFrame.
    Returns:
        str: A descriptive paragraph about the property.
    """
    description_parts = []

    # --- Property Type ---
    property_type_val = row.get('property_type')
    if pd.notna(property_type_val):
        if property_type_val.lower() == 'apartment':
            description_parts.append("This is an apartment")
        else:
            description_parts.append(f"This is a {property_type_val.lower()}")
    else:
        description_parts.append("This is a property") # Fallback

    # --- Location ---
    address_str_parts = []
    # Prioritize combined 'address' column if available and not empty
    if pd.notna(row.get('address')) and str(row.get('address')).strip():
        address_str_parts.append(f"located at {str(row.get('address')).strip()}")
    else: # Construct from individual parts if 'address' is missing or empty
        street = str(row.get('Street', '')).strip()
        number = str(row.get('Number', '')).strip()
        zip_code = str(row.get('ZIP', '')).strip()
        city = str(row.get('City', '')).strip()

        location_details = []
        if street:
            full_street = street
            if number:
                full_street += f" {number}"
            location_details.append(full_street)
        
        city_zip_parts = []
        if zip_code:
            city_zip_parts.append(zip_code)
        if city:
            city_zip_parts.append(city)
        
        if city_zip_parts:
            location_details.append(" ".join(city_zip_parts))
            
        if location_details:
            address_str_parts.append(f"located in {', '.join(location_details)}") # Simplified "in"

    if address_str_parts:
        description_parts.extend(address_str_parts)

    # --- Year Built ---
    year_built_val = row.get('year_built')
    if pd.notna(year_built_val):
        try:
            year = int(year_built_val)
            if year > 1000: # Simple check for a valid-looking year
                description_parts.append(f"built in {year}")
        except ValueError:
            pass # If year_built is not a convertible number, skip it.


    # --- Features (Bedrooms, Bathrooms, Habitable Space, Ground Space) ---
    features_clauses = []

    # Bedrooms
    bedrooms_val = row.get('bedrooms')
    if pd.notna(bedrooms_val):
        bedrooms = int(bedrooms_val)
        if bedrooms == 1:
            features_clauses.append("1 bedroom")
        elif bedrooms > 1:
            features_clauses.append(f"{bedrooms} bedrooms")
        # If bedrooms is 0, it's omitted as per your instruction.

    # Bathrooms
    bathrooms_val = row.get('bathrooms')
    if pd.notna(bathrooms_val):
        bathrooms = int(bathrooms_val)
        if bathrooms == 1:
            features_clauses.append("1 bathroom")
        elif bathrooms > 1:
            features_clauses.append(f"{bathrooms} bathrooms")

    # Habitable Space
    habitable_space_val = row.get('habitable_space_sqm')
    if pd.notna(habitable_space_val):
        habitable_space = int(habitable_space_val)
        if habitable_space > 0 : # Only add if space is greater than 0
            features_clauses.append(f"{habitable_space} sqm of habitable space")

    # Ground Space
    ground_space_val = row.get('ground_space_sqm')
    if pd.notna(ground_space_val):
        ground_space = int(ground_space_val)
        # Only mention ground space if it's significant and relevant
        is_house = str(row.get('property_type', '')).lower() == 'house'
        # Add if ground_space > 0 AND (it's a house OR it's different from habitable space)
        if ground_space > 0 and (is_house or (not is_house and ground_space != habitable_space_val)):
            features_clauses.append(f"a ground space of {ground_space} sqm")

    # --- EPC Label ---
    epc_clause = ""
    epc_label_val = row.get('epc_label')
    if pd.notna(epc_label_val) and str(epc_label_val).strip():
        epc_clause = f"The property has an Energy Performance Certificate (EPC) rating of {str(epc_label_val).strip()}."

    # --- Combine description parts into a paragraph ---
    if not description_parts: # Should not happen if property_type is always there
        return "No description available."

    # Connect initial parts (type, location, year built)
    # "This is an apartment located at X built in Y."
    # "This is a house built in Z."
    # "This is an apartment located in City."
    
    paragraph = description_parts[0] # "This is a/an property_type"
    
    # Add location if it exists
    if any("located at" in s or "located in" in s for s in description_parts[1:]):
        loc_index = -1
        for i, part in enumerate(description_parts[1:], 1):
            if "located at" in part or "located in" in part:
                loc_index = i
                break
        if loc_index != -1:
            paragraph += ", " + description_parts[loc_index]
            # Remove it from list so it's not processed again
            description_parts.pop(loc_index)


    # Add year built if it exists
    if any("built in" in s for s in description_parts[1:]):
        year_index = -1
        for i, part in enumerate(description_parts[1:], 1):
             if "built in" in part:
                year_index = i
                break
        if year_index != -1:
            # Check if paragraph already has a comma from location
            if ", located" in paragraph:
                 paragraph += ", " + description_parts[year_index]
            else:
                 paragraph += " " + description_parts[year_index] # e.g. "This is a house built in 1990"
            description_parts.pop(year_index)


    # Add features
    if features_clauses:
        paragraph += ". It features " + ", ".join(features_clauses[:-1]) + \
                     (" and " if len(features_clauses) > 1 else "") + features_clauses[-1]
    
    paragraph += "." # End the main description sentence(s).

    # Add EPC clause as a separate sentence if it exists
    if epc_clause:
        paragraph += " " + epc_clause

    # Clean up multiple periods or spaces that might have formed.
    paragraph = paragraph.replace("..", ".").replace(". .", ".").replace("  ", " ").strip()
    
    return paragraph