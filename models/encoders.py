import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import vit_b_16, resnet50, vit_l_16
from torchvision.models import ViT_B_16_Weights, ResNet50_Weights, ViT_L_16_Weights
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizerFast, BertModel

class ImageEncoder(nn.Module):
    """
    Image encoder model that extracts features from images.
    
    Supports different backbone architectures:
    - ResNet50
    - Vision Transformer (ViT-B/16)
    """
    def __init__(self, encoder_type='resnet50', pretrained=True, out_dim=512, freeze_backbone=False):
        super().__init__()
        self.encoder_type = encoder_type
        
        if encoder_type == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V1
            model = resnet50(weights=weights)
            self.model_feature_dim = 2048
            self.encoder = nn.Sequential(*list(model.children())[:-1])
        elif encoder_type == 'vit_b_16':
            weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = vit_b_16(weights=weights)
            self.model_feature_dim = 768
        elif encoder_type == 'vit_l_16':
            weights = ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = vit_l_16(weights=weights)
            self.model_feature_dim = 1024
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        
        # Freeze backbone
        if encoder_type == 'resnet50':
            if freeze_backbone:
                for p in self.encoder.parameters():
                    p.requires_grad = False
        else:
            if freeze_backbone:
                for p in self.model.parameters():
                    p.requires_grad = False

        self.adaptation_layer = nn.Sequential(
            nn.Linear(self.model_feature_dim, out_dim),
            nn.ReLU()
        )
        self.feature_dim = out_dim
            
    def forward(self, x):
        B, C, H, W = x.shape
        
        if self.encoder_type == 'resnet50':
            features = self.encoder(x).squeeze(-1).squeeze(-1)  # [B, 2048]
            adapted_features = self.adaptation_layer(features)  # Apply adaptation here
        else:  # ViT
            # patch embeddings
            patch_embeddings = self.model._process_input(x)
            batch_class_token = self.model.class_token.expand(x.shape[0], -1, -1)
            tokens = torch.cat([batch_class_token, patch_embeddings], dim=1)
            encoded_patches = self.model.encoder(tokens)
            patch_tokens = encoded_patches[:, 1:]  # Skip the class token at index 0
            features = torch.mean(patch_tokens, dim=1)  # [B, 768]
            adapted_features = self.adaptation_layer(features)
        return adapted_features
    

class TextEncoder(nn.Module):
    """
    Text encoder model that extracts features from property descriptions.
    
    Supports multiple encoder types:
    - bert (BERT models from transformers)
    """
    def __init__(self, model_name='bert-base-uncased', 
                 encoder_type='bert',
                 freeze_base=True,
                 max_length=128,
                 out_dim=512):
        """
        Args:
            model_name: Name of the pre-trained model ('bert-base-uncased')
            encoder_type: Type of encoder ('bert')
            freeze_base: Whether to freeze the base transformer weights
            max_length: Maximum sequence length for BERT tokenization
        """
        super().__init__()
        self.model_name = model_name
        self.encoder_type = encoder_type
        self.max_length = max_length

        print(f"Using encoder type {encoder_type} and model {model_name}")
        self.model = BertModel.from_pretrained(model_name)
        
        # Freeze BERT weights if requested
        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.adaptation_layer = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, out_dim),
            nn.ReLU()
        )
        self.feature_dim = out_dim
        
    def forward(self, text_inputs):
        """
        Encode the input texts.
        
        Args:
            text_inputs: Dictionary containing 'input_ids' and 'attention_mask' from tokenizer
            
        Returns:
            tensor of shape [batch_size, feature_dim]
        """
        # Extract tensors from the text_inputs dictionary
        input_ids = text_inputs['input_ids']
        attention_mask = text_inputs['attention_mask']
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.pooler_output
        adapted_embeddings = self.adaptation_layer(embeddings)
        
        return adapted_embeddings