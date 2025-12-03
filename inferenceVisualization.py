import matplotlib.pyplot as plt
import cv2
from ViTBackbone import ViTCoordinateRegressor
import torch
import numpy as np
from torch.utils.data import DataLoader


class BrightestPixelPredictor:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model = ViTCoordinateRegressor(use_pretrained=True)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def predict(self, image_array):
        """
        Predict brightest pixel in a 256x256 grayscale image

        Args:
            image_array: numpy array of shape (256, 256) in range [0, 255]

        Returns:
            (pred_x, pred_y): Predicted coordinates
            confidence: Optional confidence score
        """
        # Preprocess
        image_norm = image_array.astype(np.float32) / 255.0
        image_3ch = np.stack([image_norm, image_norm, image_norm], axis=0)
        image_tensor = torch.from_numpy(image_3ch).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            pred_norm = self.model(image_tensor)

        # Convert to pixel coordinates
        pred_norm = pred_norm.cpu().numpy()[0]
        pred_x = int(pred_norm[0] * 255)
        pred_y = int(pred_norm[1] * 255)

        # Clamp to image bounds
        pred_x = np.clip(pred_x, 0, 255)
        pred_y = np.clip(pred_y, 0, 255)

        return (pred_x, pred_y)

    def predict_batch(self, image_batch):
        """Predict for a batch of images"""
        batch_tensors = []
        for img in image_batch:
            img_norm = img.astype(np.float32) / 255.0
            img_3ch = np.stack([img_norm, img_norm, img_norm], axis=0)
            batch_tensors.append(img_3ch)

        batch_tensor = torch.from_numpy(np.array(batch_tensors)).to(self.device)

        with torch.no_grad():
            preds_norm = self.model(batch_tensor)

        preds_pixels = (preds_norm.cpu().numpy() * 255).astype(int)
        preds_pixels = np.clip(preds_pixels, 0, 255)

        return preds_pixels

    def visualize_prediction(
        self, image_array, pred_coords=None, true_coords=None, save_path=None
    ):
        """
        Visualize prediction on image

        Args:
            image_array: Original image array
            pred_coords: Predicted (x, y)
            true_coords: Ground truth (x, y)
            save_path: Path to save visualization
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Original image with predictions
        axes[0].imshow(image_array, cmap="gray", vmin=0, vmax=255)

        if pred_coords:
            axes[0].scatter(
                pred_coords[0],
                pred_coords[1],
                c="red",
                s=100,
                marker="x",
                label="Predicted",
            )
        if true_coords:
            axes[0].scatter(
                true_coords[0],
                true_coords[1],
                c="green",
                s=100,
                marker="o",
                label="True",
            )

        axes[0].set_title("Image with Predictions")
        axes[0].legend()
        axes[0].axis("off")

        # Zoomed region around prediction
        if pred_coords:
            zoom_size = 40
            x_min = max(0, pred_coords[0] - zoom_size)
            x_max = min(255, pred_coords[0] + zoom_size)
            y_min = max(0, pred_coords[1] - zoom_size)
            y_max = min(255, pred_coords[1] + zoom_size)

            zoom_region = image_array[y_min:y_max, x_min:x_max]
            axes[1].imshow(zoom_region, cmap="hot", aspect="auto")
            axes[1].set_title(
                f"Zoom around prediction\nPixel value: {image_array[pred_coords[1], pred_coords[0]]:.0f}"
            )
            axes[1].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()

    def evaluate_on_dataset(self, dataset):
        """Evaluate model on entire dataset"""
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        all_preds = []
        all_targets = []
        pixel_errors = []

        self.model.eval()
        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                preds = self.model(images)

                # Convert to pixel coordinates
                preds_pixels = preds * 255
                targets_pixels = targets * 255

                # Calculate distances
                distances = torch.sqrt(
                    torch.sum((preds_pixels - targets_pixels) ** 2, dim=1)
                )

                all_preds.append(preds_pixels.cpu().numpy())
                all_targets.append(targets_pixels.cpu().numpy())
                pixel_errors.extend(distances.cpu().numpy())

        # Statistics
        pixel_errors = np.array(pixel_errors)

        print("Evaluation Results:")
        print(f"  Mean Pixel Error: {pixel_errors.mean():.2f} px")
        print(f"  Median Pixel Error: {np.median(pixel_errors):.2f} px")
        print(f"  Std Pixel Error: {pixel_errors.std():.2f} px")
        print(f"  Max Pixel Error: {pixel_errors.max():.2f} px")
        print(f"  Accuracy @5px: {(pixel_errors <= 5).mean()*100:.1f}%")
        print(f"  Accuracy @10px: {(pixel_errors <= 10).mean()*100:.1f}%")

        return pixel_errors
