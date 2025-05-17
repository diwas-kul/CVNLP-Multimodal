import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoders import ImageEncoder, TextEncoder
from models.pooling import AttentionPooling, MeanPooling
import random


class CrossModalAttention(nn.Module):
    def __init__(self, img_dim, text_dim, num_heads=4):
        super().__init__()
        self.img2text = nn.MultiheadAttention(
            embed_dim=img_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.text2img = nn.MultiheadAttention(
            embed_dim=text_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.norm_img = nn.LayerNorm(img_dim)
        self.norm_text = nn.LayerNorm(text_dim)
        
    def forward(self, img_features, text_features):
        # Add sequence dimension
        img_seq = img_features.unsqueeze(1)  # [B, 1, D_img]
        text_seq = text_features.unsqueeze(1)  # [B, 1, D_text]
        
        # Cross-attention in both directions
        img_attended, _ = self.img2text(query=img_seq, key=text_seq, value=text_seq)
        text_attended, _ = self.text2img(query=text_seq, key=img_seq, value=img_seq)
        
        # Residual connections and normalization
        img_output = self.norm_img(img_features + img_attended.squeeze(1))
        text_output = self.norm_text(text_features + text_attended.squeeze(1))
        
        return img_output, text_output


class PropertyPriceModel(nn.Module):
    """
    Model for property price prediction from multiple images and/or text.
    """
    def __init__(self, 
                num_classes,
                use_text=False,
                use_images=False,
                # Image-specific parameters (only required if use_images=True)
                encoder_type="resnet50",
                pooling_type='mean',
                freeze_backbone=False,
                # Text-specific parameters (only required if use_text=True)
                text_encoder_model="bert-base-uncased",
                text_encoder_type='bert',  # New parameter
                freeze_text_encoder=True,
                # Common parameters
                dropout_rate=0.3,
                pretrained=True,
                fusion_type='concat',
                # Multimodal
                modality_weights=[0.4,0.6],
                modality_dropout=False):
        """
        Args:
            num_classes: Number of price categories for classification
            use_text: Whether to use text as input
            use_images: Whether to use images as input
            
            # Image-specific parameters
            encoder_type: Type of image encoder ('resnet50', 'vit_b_16')
            pooling_type: Type of pooling for images ('attention', 'mean')
            freeze_backbone: Whether to freeze the backbone network
            
            # Text-specific parameters
            text_encoder_model: Name of the pre-trained model
            text_encoder_type: Type of text encoder ('bert')
            freeze_text_encoder: Whether to freeze or fine-tune the text encoder
            
            # Common parameters
            dropout_rate: Dropout rate for the fully connected layers
            pretrained: Whether to use pretrained backbone
            fusion_type: How to combine modalities ('concat')
        """
        super().__init__()

        # Use at least one modality
        assert use_text or use_images, "At least one modality must be enabled"
        
        self.use_text = use_text
        self.use_images = use_images
        self.fusion_type = fusion_type

        self.use_text = use_text
        self.use_images = use_images
        self.fusion_type = fusion_type
        
        # Initialize feature dimensions
        img_feature_dim = 0
        text_feature_dim = 0
        
        # Image encoder (only if images are used)
        if use_images:
            # Validate required parameters
            if encoder_type is None:
                raise ValueError("encoder_type must be provided when use_images=True")
            if pooling_type is None:
                raise ValueError("pooling_type must be provided when use_images=True")
            if freeze_backbone is None:
                raise ValueError("freeze_backbone must be provided when use_images=True")
                
            self.image_encoder = ImageEncoder(encoder_type, pretrained, out_dim=512, freeze_backbone=freeze_backbone)
            img_feature_dim = self.image_encoder.feature_dim
            
            # Pooling mechanism
            if pooling_type == 'attention':
                self.pool = AttentionPooling(img_feature_dim)
            elif pooling_type == 'mean':
                self.pool = MeanPooling()
            else:
                raise ValueError(f"Unknown pooling: {pooling_type}")
        
        # Text encoder (only if text is used)
        if use_text:
            if text_encoder_model is None:
                raise ValueError("text_encoder_model must be provided when use_text=True")
                
            self.text_encoder = TextEncoder(
                model_name=text_encoder_model,
                encoder_type=text_encoder_type,
                freeze_base=freeze_text_encoder,
                out_dim=512
            )
            text_feature_dim = self.text_encoder.feature_dim
        
        # NOW set the feature_dim based on modalities
        if use_images and not use_text:
            self.feature_dim = img_feature_dim
        elif use_text and not use_images:
            self.feature_dim = text_feature_dim
        
        # Feature fusion (only after BOTH encoders are initialized)
        if use_text and use_images:
            if fusion_type == 'concat':
                self.feature_dim = img_feature_dim + text_feature_dim
            elif fusion_type == 'attention':
                # Cross-modal attention mechanism
                self.cross_modal_attention = CrossModalAttention(
                    img_dim=img_feature_dim,
                    text_dim=text_feature_dim,
                    num_heads=4
                )
                self.feature_dim = img_feature_dim + text_feature_dim
            else:
                raise ValueError(f"Unknown fusion: {fusion_type}")
            
            self.modality_weights = nn.Parameter(torch.tensor(modality_weights))
            self.modality_dropout = modality_dropout
        
        # Feature fusion
        if use_text and use_images:
            if fusion_type == 'concat':
                self.feature_dim = img_feature_dim + text_feature_dim
            elif fusion_type == 'attention':
                # Cross-modal attention mechanism
                self.cross_attention_img2txt = nn.MultiheadAttention(
                    embed_dim=img_feature_dim, 
                    num_heads=4, 
                    batch_first=True
                )
                self.cross_attention_txt2img = nn.MultiheadAttention(
                    embed_dim=text_feature_dim, 
                    num_heads=4, 
                    batch_first=True
                )
                self.layer_norm_img = nn.LayerNorm(img_feature_dim)
                self.layer_norm_txt = nn.LayerNorm(text_feature_dim)
                self.feature_dim = img_feature_dim + text_feature_dim
            else:
                raise ValueError(f"Unknown fusion: {fusion_type}")
            
            self.modality_weights = nn.Parameter(torch.tensor(modality_weights))  # [image_weight, text_weight]
            self.modality_dropout = modality_dropout
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 1)
        )
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, *args):
        """
        Forward pass handling different input modalities
        
        Input format depends on enabled modalities:
        - Images only: (image_set, mask)
        - Text only: (text_list)
        - Both: (image_set, mask, text_list)
            
        Returns:
            dict: Dictionary with regression and classification outputs
        """
        if self.use_images and self.use_text:
            # Both modalities
            image_set, mask, text_inputs = args
            
            # Process images
            B, N, C, H, W = image_set.shape
            x = image_set.view(B * N, C, H, W)
            img_features = self.image_encoder(x).view(B, N, -1)  # [B, N, D_img]
            img_features = self.pool(img_features, mask)  # [B, D_img]
            
            # Process text
            text_features = self.text_encoder(text_inputs)  # [B, D_text]
            
            # Normalize
            img_features = F.normalize(img_features, p=2, dim=1)
            text_features = F.normalize(text_features, p=2, dim=1)


            # Implement modality dropout during training if enabled
            if self.training and self.modality_dropout and random.random() < 0.3:
                if random.random() < 0.5:
                    # Drop image modality (with probability 0.15)
                    img_features = torch.zeros_like(img_features)
                else:
                    # Drop text modality (with probability 0.15)
                    text_features = torch.zeros_like(text_features)

            # Combine features based on fusion type
            if self.fusion_type == 'concat':
                # Simple weighted concatenation
                img_weighted = self.modality_weights[0] * img_features
                text_weighted = self.modality_weights[1] * text_features
                features = torch.cat([img_weighted, text_weighted], dim=1)
                
                # Track modality contributions
                if not self.training:
                    with torch.no_grad():
                        self.img_contribution = torch.norm(img_weighted, dim=1).mean().unsqueeze(0)
                        self.txt_contribution = torch.norm(text_weighted, dim=1).mean().unsqueeze(0)
                
            elif self.fusion_type == 'attention':
                # Apply cross-modal attention
                img_attended, text_attended = self.cross_modal_attention(img_features, text_features)
                
                # Apply modality weights and concatenate
                img_weighted = self.modality_weights[0] * img_attended
                text_weighted = self.modality_weights[1] * text_attended
                features = torch.cat([img_weighted, text_weighted], dim=1)
                
                # Track modality contributions
                if not self.training:
                    with torch.no_grad():
                        self.img_contribution = torch.norm(img_weighted, dim=1).mean().unsqueeze(0)
                        self.txt_contribution = torch.norm(text_weighted, dim=1).mean().unsqueeze(0)
            else:
                raise ValueError(f"Unknown fusion type: {self.fusion_type}")

                
        elif self.use_images:
            # Images only
            image_set, mask = args
            
            # Process images
            B, N, C, H, W = image_set.shape
            x = image_set.view(B * N, C, H, W)
            img_features = self.image_encoder(x).view(B, N, -1)  # [B, N, D_img]
            features = self.pool(img_features, mask)  # [B, D_img]
            
        elif self.use_text:
            # Text only
            text_inputs = args[0]
            
            # Process text
            features = self.text_encoder(text_inputs)  # [B, D_text]
        
        # Shared layers and heads
        shared_features = self.shared(features)  # [B, 512]
        regression_output = self.regression_head(shared_features).squeeze(-1)  # [B]
        classification_output = self.classification_head(shared_features)  # [B, num_classes]
        
        return {
            'regression': regression_output,
            'classification': classification_output
        }