# Pixel Identification with Vision Transformers (ViT)

A deep learning project that identifies the brightest pixel in grayscale checkerboard images using Vision Transformers (ViT) from Hugging Face Transformers library. This project demonstrates coordinate regression using attention-based models on synthetic and real image data.

## Project Overview

The goal of this project is to train a Vision Transformer model to predict the (x, y) coordinates of the brightest pixel in black-and-white spectrum checkerboard images. The model performs pixel-level coordinate regression, converting the image understanding capabilities of ViT into precise localization predictions.

### Key Features

- **Vision Transformer Backbone**: Uses Google's ViT-base-patch16-224 pretrained model
- **Coordinate Regression**: Outputs normalized (0-1) pixel coordinates
- **Synthetic Data Generation**: Creates diverse training data with checkerboard patterns and bright spots
- **Monte Carlo Dropout Uncertainty**: Implements Bayesian estimation for prediction confidence
- **Data Augmentation**: Applies geometric transformations while preserving brightness information
- **Attention Visualization**: Visualizes which image patches the model attends to
- **Comprehensive Evaluation**: Reports pixel-level error metrics

## Project Structure

```
pixel_identification/
├── main.py                          # Main training pipeline
├── ViTBackbone.py                   # ViT coordinate regressor model
├── Trainer.py                       # Training loop and validation logic
├── SyntheticDataGeneration.py       # Synthetic dataset generation
├── DataAugmentation.py              # Dataset class with augmentations
├── inferenceVisualization.py        # Prediction and visualization utilities
├── UncertainityEstimation.py        # Bayesian uncertainty estimation
├── MultiHeadAttentionVisualization.py # Attention map visualization
├── best_model.pth                   # Trained model checkpoint
├── requirements.txt                 # Python dependencies
└── data/
    ├── train/                       # Real training images (txt annotations)
    ├── train_checker/               # Synthetic checkerboard dataset
    ├── train_synthetic_64/          # 64x64 synthetic images
    └── train_synthetic_256/         # 256x256 synthetic images
```

## Module Descriptions

### `ViTBackbone.py`

Implements the core model architecture `ViTCoordinateRegressor`:

- Adapts pretrained ViT-base for 256×256 image input (original: 224×224)
- Uses the [CLS] token representation for global image understanding
- Multi-layer regression head: `LayerNorm → Linear → GELU → Dropout → ... → 2D output`
- Sigmoid activation bounds predictions to [0, 1] normalized coordinate space

### `Trainer.py`

Training orchestration class `CoordinateRegressionTrainer`:

- Supports multiple loss functions: MSE, SmoothL1, combined with margin penalty
- Differential learning rates for ViT backbone and regression head
- Cosine Annealing with Warm Restarts scheduling
- Gradient clipping and pixel-level error tracking
- Validation loop with comprehensive metrics reporting

### `SyntheticDataGeneration.py`

Generates diverse synthetic training data:

- **Checkerboard Pattern**: Random intensity squares with identified brightest pixel
- **Single Pixel**: Image with one bright pixel at random/specified location
- **Bright Spot**: Gaussian intensity distributions for realistic localization
- Saves generated images with ground truth annotations

### `DataAugmentation.py`

Dataset class `BrightestPixelDataset`:

- Supports both synthetic and real image modes
- Albumentations for geometric transformations: flips, rotations, shifts
- Keypoint tracking during augmentations to maintain coordinate accuracy
- Converts images to 3-channel tensors for ViT input
- Optional synthetic bright spot augmentation during training

### `inferenceVisualization.py`

Prediction and evaluation utilities class `BrightestPixelPredictor`:

- Loads and runs inference on trained models
- Batch prediction support
- Visualization with overlaid predictions and ground truth
- Zoomed region inspection
- Comprehensive evaluation metrics: mean/median/std error, accuracy thresholds

### `UncertainityEstimation.py`

Implements `BayesianViTRegressor` for uncertainty quantification:

- Monte Carlo Dropout: Multiple forward passes with active dropout
- Returns mean prediction and standard deviation uncertainty
- Useful for confidence-calibrated predictions

### `MultiHeadAttentionVisualization.py`

Attention analysis function:

- Visualizes multi-head attention patterns from ViT layers
- Heatmap resized to original image dimensions
- Highlights which image patches influence the final prediction

### `main.py`

Complete training pipeline:

1. Configuration setup (batch size, epochs, learning rate, image size)
2. Synthetic dataset generation (5000 training + 500 validation samples)
3. DataLoader creation with batch processing
4. Model initialization with pretrained weights
5. Training loop with validation and checkpointing
6. Evaluation on validation set
7. Visualization of sample predictions

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)

### Setup

```bash
# Clone or navigate to the project directory
cd pixel_identification

# Install dependencies
pip install -r requirements.txt

# Additional required packages (add to requirements.txt):
# torch>=1.9.0
# torchvision>=0.10.0
# transformers>=4.0.0
# numpy>=1.19.0
# Pillow>=8.0.0
# matplotlib>=3.3.0
# opencv-python>=4.5.0
```

### Current Dependencies (in requirements.txt)

- `albumentations` - Advanced image augmentation library

### Additional Dependencies (needed, not in requirements.txt)

- `torch` - PyTorch deep learning framework
- `torchvision` - Computer vision utilities
- `transformers` - Hugging Face transformers (for ViT)
- `numpy` - Numerical computing
- `Pillow` - Image processing
- `matplotlib` - Visualization
- `opencv-python` - Image processing utilities (cv2)

## Usage

### Training

Run the main training pipeline:

```bash
python main.py
```

This will:

1. Generate 5000 synthetic training images and 500 validation images
2. Initialize ViT model with pretrained weights
3. Train for 50 epochs with validation
4. Save the best model to `best_model.pth`
5. Evaluate on validation set
6. Generate visualization samples

### Configuration

Modify the config dictionary in `main.py`:

```python
config = {
    "data_dir": "./data/train",           # Real training data directory
    "val_dir": "./data/val",              # Real validation data directory
    "batch_size": 32,                     # Batch size for training
    "epochs": 50,                         # Number of training epochs
    "learning_rate": 1e-4,                # Base learning rate
    "img_size": 64,                       # Input image size (64x64 or 256x256)
    "use_synthetic": True,                # Generate synthetic data
    "synthetic_samples": 5000,            # Number of synthetic samples
    "save_dir": "./data/train_checker",   # Save synthetic data
}
```

### Inference

Use trained model for predictions:

```python
from inferenceVisualization import BrightestPixelPredictor
import numpy as np

# Load model
predictor = BrightestPixelPredictor("best_model.pth")

# Predict on single image (256x256 grayscale, values 0-255)
image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
pred_x, pred_y = predictor.predict(image)
print(f"Predicted brightest pixel: ({pred_x}, {pred_y})")

# Visualize
predictor.visualize_prediction(image, pred_coords=(pred_x, pred_y), save_path="result.png")
```

### Uncertainty Estimation

Get prediction confidence using Monte Carlo Dropout:

```python
from UncertainityEstimation import BayesianViTRegressor
from inferenceVisualization import BrightestPixelPredictor

# Wrap model with uncertainty estimation
base_model = BrightestPixelPredictor("best_model.pth").model
bayesian_model = BayesianViTRegressor(base_model, n_samples=10)

# Forward pass returns mean and uncertainty
mean_pred, uncertainty = bayesian_model(image_tensor, n_samples=10)
```

### Attention Visualization

Visualize model attention patterns:

```python
from MultiHeadAttentionVisualization import visualize_attention_maps
from inferenceVisualization import BrightestPixelPredictor

predictor = BrightestPixelPredictor("best_model.pth")
attention_map = visualize_attention_maps(predictor.model, image_tensor)
```

## Model Architecture

### ViT Backbone

- **Base Model**: `google/vit-base-patch16-224`
- **Input Size**: 256×256×3 (converted from grayscale)
- **Patch Size**: 16×16 pixels
- **Number of Patches**: 256 (16×16 grid)
- **Hidden Dimension**: 768
- **Number of Attention Heads**: 12
- **Number of Transformer Layers**: 12

### Regression Head

```
Input (768D) → LayerNorm → Linear(768→384) → GELU → Dropout(0.2)
→ Linear(384→192) → GELU → Dropout(0.1) → Linear(192→2) → Sigmoid
```

Output: Normalized coordinates [x, y] in range [0, 1]

## Training Details

### Loss Functions

1. **MSE**: Standard mean squared error for coordinate regression
2. **SmoothL1**: Robust L1-norm loss
3. **Combined**: MSE + margin penalty for out-of-bounds predictions

### Optimizer

- **Algorithm**: AdamW with weight decay
- **Learning Rates**:
  - ViT backbone: lr × 0.1 (frozen/low learning rate)
  - Regression head: lr (default: 1e-4)
- **Weight Decay**: 1e-4

### Learning Rate Schedule

- **Type**: Cosine Annealing with Warm Restarts
- **T_0**: 10 epochs (initial restart period)
- **T_mult**: 2 (restart period multiplier)
- **eta_min**: 1e-6 (minimum learning rate)

## Evaluation Metrics

The model reports the following metrics on validation set:

- **Mean Pixel Error**: Average Euclidean distance from ground truth (pixels)
- **Median Pixel Error**: Robust measure of central tendency
- **Std Pixel Error**: Standard deviation of error distribution
- **Max Pixel Error**: Worst-case prediction error
- **Accuracy @5px**: % of predictions within 5 pixels
- **Accuracy @10px**: % of predictions within 10 pixels

## Dataset Description

### Synthetic Data

- **Checkerboard Pattern**: 8×8 grid with random grayscale intensities
- **Dimensions**: 64×64 or 256×256 pixels
- **Format**: Grayscale (single channel)
- **Annotations**: (x, y) coordinates of brightest square center
- **Augmentation**: Geometric transforms preserve coordinate accuracy

### Data Statistics

- Training: 5000 synthetic images
- Validation: 500 synthetic images
- Batch Size: 32
- Total Iterations per Epoch: ~156 (training)

## Results and Performance

Expected performance metrics (after training):

- Mean pixel error: <10-15 pixels (on 256×256 images)
- Accuracy @10px: >80%
- Training time: ~30-60 minutes on NVIDIA GPU

## Troubleshooting

### GPU Out of Memory

- Reduce `batch_size` in config
- Reduce `img_size` (use 64 instead of 256)
- Reduce model hidden size

### Poor Convergence

- Increase training epochs
- Adjust learning rate
- Check data augmentation intensity
- Verify synthetic data quality

### CUDA/GPU Issues

- Verify CUDA is installed: `nvidia-smi`
- Set device to CPU: Change `device="cuda"` to `device="cpu"` in code
- Update PyTorch: `pip install --upgrade torch torchvision`

## Future Improvements

- [ ] Support for real image datasets (non-synthetic)
- [ ] Ensemble methods combining multiple models
- [ ] Attention-based loss weighting
- [ ] Real-time inference optimization
- [ ] Model quantization for mobile deployment
- [ ] Extended uncertainty estimation (Conformal Prediction)
- [ ] Integration with YOLOv8 for multi-object detection
- [ ] API endpoint for inference service

## References

- Vision Transformers Paper: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- Hugging Face Transformers: <https://huggingface.co/transformers/>
- Albumentations: <https://albumentations.ai/>
- PyTorch Documentation: <https://pytorch.org/docs/>

## License

This is a learning/prototype project. Feel free to use and modify as needed.

## Author

Asim Zahid - Personal Prototype Learning Project

## Contact & Support

For questions or issues, please refer to the inline code comments and docstrings in each module.
