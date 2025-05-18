import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoders import ImageEncoder, TextEncoder

class ContrastiveModel(nn.Module):
    """
    Contrastive learning model for image-text pairs.
    """
    def __init__(self, 
                 encoder_type="resnet50",
                 text_encoder_model="bert-base-uncased",
                 text_encoder_type='bert',
                 freeze_backbone=False,
                 freeze_text_encoder=False,
                 projection_dim=256,
                 temperature=0.07,
                 pretrained=True):
        """
        Args:
            encoder_type: Type of image encoder
            text_encoder_model: Name of text encoder model
            text_encoder_type: Type of text encoder
            freeze_backbone: Whether to freeze image encoder backbone
            freeze_text_encoder: Whether to freeze text encoder
            projection_dim: Dimension of projection space
            temperature: Temperature parameter for contrastive loss
            pretrained: Whether to use pretrained encoders
        """
        super().__init__()
        
        print(f"Initializing ContrastiveModel with freeze_backbone={freeze_backbone}, freeze_text_encoder={freeze_text_encoder}")
        
        # Image encoder
        self.image_encoder = ImageEncoder(encoder_type, pretrained, out_dim=512, freeze_backbone=freeze_backbone)
        
        # Text encoder
        self.text_encoder = TextEncoder(
            model_name=text_encoder_model,
            encoder_type=text_encoder_type,
            freeze_base=freeze_text_encoder,
            out_dim=512
        )
        
        # Projection heads
        self.image_projection = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, projection_dim),
        )
        
        self.text_projection = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, projection_dim),
        )
        
        self.temperature = temperature
        
    def forward(self, images, masks, text_inputs):
        """
        Forward pass for contrastive learning.
        
        Args:
            images: Batch of images [B, N, C, H, W]
            masks: Masks for images [B, N]
            text_inputs: Text inputs dictionary
            
        Returns:
            dict: Dictionary with embeddings and similarity scores
        """
        # Process images
        B, N, C, H, W = images.shape
        x = images.view(B * N, C, H, W)
        img_features = self.image_encoder(x).view(B, N, -1)  # [B, N, D_img]
        
        # Apply mean pooling over the mask
        mask_expanded = masks.unsqueeze(-1).expand(-1, -1, img_features.size(-1))
        masked_features = img_features * mask_expanded
        mask_sum = mask_expanded.sum(dim=1, keepdim=True)
        mask_sum = torch.clamp(mask_sum, min=1.0)  # Avoid division by zero
        image_embeddings = (masked_features.sum(dim=1) / mask_sum.squeeze(1))  # [B, D_img]
        
        # Process text
        text_embeddings = self.text_encoder(text_inputs)  # [B, D_text]
        
        # Project embeddings to the same space
        image_proj = self.image_projection(image_embeddings)  # [B, projection_dim]
        text_proj = self.text_projection(text_embeddings)  # [B, projection_dim]
        
        # Normalize projections (crucial for contrastive learning)
        image_proj = F.normalize(image_proj, p=2, dim=1)
        text_proj = F.normalize(text_proj, p=2, dim=1)
        
        return {
            'image_embeddings': image_embeddings,
            'text_embeddings': text_embeddings,
            'image_proj': image_proj,
            'text_proj': text_proj
        }
    
    def compute_similarity(self, image_proj, text_proj):
        """
        Compute similarity matrix between image and text projections.
        
        Args:
            image_proj: Image projections [B, D]
            text_proj: Text projections [B, D]
            
        Returns:
            similarity: Similarity matrix [B, B]
        """
        # Compute cosine similarity
        similarity = torch.matmul(image_proj, text_proj.t()) / self.temperature
        return similarity