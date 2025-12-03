import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
import torch.nn.functional as F


class ViTCoordinateRegressor(nn.Module):
    def __init__(
        self,
        model_name="google/vit-base-patch16-224",
        img_size=256,
        patch_size=16,
        use_pretrained=True,
    ):
        super().__init__()

        # Configuration adjustments for 256x256
        config = ViTConfig.from_pretrained(model_name)
        config.image_size = img_size
        config.num_channels = 3  # Grayscale, but we'll convert to 3 channels

        # Initialize ViT backbone
        if use_pretrained:
            # Load pretrained weights (ignore size mismatches)
            self.vit = ViTModel.from_pretrained(
                model_name, config=config, ignore_mismatched_sizes=True
            )
        else:
            self.vit = ViTModel(config)

        # Patch embedding adjustment for 256x256
        # ViT-base expects 224x224, but we can adapt
        num_patches = (img_size // patch_size) ** 2
        self.vit.embeddings.num_patches = num_patches
        # self.vit.embeddings.position_embeddings = nn.Embedding(
        #     num_patches + 1, config.hidden_size  # +1 for [CLS] token
        # )
        self.vit.embeddings.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.hidden_size)
        )
        nn.init.trunc_normal_(self.vit.embeddings.position_embeddings, std=0.02)

        # self.vit.embeddings.patch_embeddings.projection = nn.Conv2d(
        #     1, config.hidden_size, kernel_size=patch_size, stride=patch_size
        # )

        # Enhanced regression head
        hidden_size = config.hidden_size

        self.regression_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 2),  # Output: (x, y)
        )

        # Initialize regression head
        self._init_weights(self.regression_head)

    def _init_weights(self, module):
        """Initialize weights for the regression head"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 3, 256, 256)
               Note: We'll convert grayscale to 3 channels if needed
        Returns:
            coords: Normalized coordinates [batch_size, 2]
        """
        # ViT forward pass
        outputs = self.vit(x)

        # Use [CLS] token representation
        cls_token = outputs.last_hidden_state[:, 0, :]

        # Regression to coordinates
        coords = self.regression_head(cls_token)

        # Apply sigmoid to bound outputs to [0, 1]
        coords = torch.sigmoid(coords)

        return coords
