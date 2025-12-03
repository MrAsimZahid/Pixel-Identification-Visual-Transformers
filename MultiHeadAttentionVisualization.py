import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


def visualize_attention_maps(model, image_tensor, layer_idx=-1):
    """
    Visualize which image patches the model attends to
    """
    model.eval()
    
    # Forward pass with attention outputs
    with torch.no_grad():
        outputs = model.vit(
            image_tensor, 
            output_attentions=True
        )
    
    # Get attention from specified layer
    attention = outputs.attentions[layer_idx]  # [batch, heads, seq_len, seq_len]
    attention = attention.mean(dim=1)  # Average over heads
    attention = attention[0]  # First batch
    
    # [CLS] token attention to patches
    cls_attention = attention[0, 1:]  # Shape: [num_patches]
    
    # Reshape to patch grid (16x16 for 256x256 with patch size 16)
    patch_grid_size = 256 // 16
    attention_map = cls_attention.reshape(patch_grid_size, patch_grid_size)
    
    # Upsample to original image size
    attention_map_resized = cv2.resize(
        attention_map.cpu().numpy(),
        (256, 256),
        interpolation=cv2.INTER_CUBIC
    )
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Original image
    axes[0].imshow(image_tensor[0].permute(1, 2, 0).cpu().numpy()[:,:,0], cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attention heatmap
    im = axes[1].imshow(attention_map_resized, cmap='hot')
    axes[1].set_title('ViT Attention Heatmap')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    plt.show()
    
    return attention_map_resized