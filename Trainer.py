import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


class CoordinateRegressionTrainer:
    def __init__(self, model, device="cuda"):
        self.model = model.to(device)
        self.device = device

        # Loss functions with options
        self.mse_loss = nn.MSELoss()
        self.smooth_l1 = nn.SmoothL1Loss()

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.pixel_errors = []  # In actual pixels

    def train_epoch(self, dataloader, optimizer, scheduler=None, loss_type="mse"):
        self.model.train()
        epoch_loss = 0

        logger.info("Starting training epoch...")
        # logger.debug(f"Dataloader length: {len(dataloader)}")

        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            predictions = self.model(images)

            # Calculate loss
            if loss_type == "mse":
                loss = self.mse_loss(predictions, targets)
            elif loss_type == "smooth_l1":
                loss = self.smooth_l1(predictions, targets)
            elif loss_type == "combined":
                # Combine MSE with coordinate-specific penalty
                mse = self.mse_loss(predictions, targets)

                # Add penalty for predictions outside image bounds
                # (though sigmoid bounds to [0,1], we add small margin penalty)
                margin_penalty = torch.relu(predictions - 1.01) + torch.relu(
                    -predictions - 0.01
                )
                margin_penalty = margin_penalty.mean() * 0.1

                loss = mse + margin_penalty
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()

            # Log progress
            if batch_idx % 50 == 0:
                # Convert normalized coords back to pixels for error reporting
                pixel_error = self._calculate_pixel_error(predictions, targets)
                print(
                    f"  Batch {batch_idx}/{len(dataloader)}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Pixel Error: {pixel_error:.1f}"
                )

        return epoch_loss / len(dataloader)

    def _calculate_pixel_error(self, predictions, targets):
        """Calculate average pixel distance error"""
        # Convert from normalized [0,1] to pixel coordinates [0,255]
        pred_pixels = predictions * 255
        target_pixels = targets * 255

        # Euclidean distance
        distances = torch.sqrt(torch.sum((pred_pixels - target_pixels) ** 2, dim=1))
        return distances.mean().item()

    def validate(self, dataloader):
        self.model.eval()
        val_loss = 0
        pixel_errors = []

        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model(images)
                loss = self.mse_loss(predictions, targets)

                val_loss += loss.item()

                # Calculate pixel error
                pixel_error = self._calculate_pixel_error(predictions, targets)
                pixel_errors.append(pixel_error)

        avg_pixel_error = np.mean(pixel_errors)
        return val_loss / len(dataloader), avg_pixel_error

    def train(self, train_loader, val_loader, epochs=50, lr=1e-4, weight_decay=1e-4):

        # Optimizer with differential learning rates
        optimizer = optim.AdamW(
            [
                {
                    "params": self.model.vit.parameters(),
                    "lr": lr * 0.1,
                },  # Lower LR for pretrained
                {"params": self.model.regression_head.parameters(), "lr": lr},
            ],
            weight_decay=weight_decay,
        )

        # Scheduler
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Initial restart period
            T_mult=2,  # Period multiplier after each restart
            eta_min=1e-6,  # Minimum learning rate
        )

        print("Starting training...")
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(
                train_loader, optimizer, scheduler, loss_type="combined"
            )

            # Validate
            val_loss, pixel_error = self.validate(val_loader)

            # Update scheduler
            scheduler.step()

            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.pixel_errors.append(pixel_error)

            # Print progress
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  Val Pixel Error: {pixel_error:.2f} pixels")
            print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

            # Save best model
            if pixel_error == min(self.pixel_errors):
                torch.save(self.model.state_dict(), "best_model.pth")
                print("  * Best model saved!")

        self.plot_training_progress()

    def plot_training_progress(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss plot
        axes[0].plot(self.train_losses, label="Train Loss")
        axes[0].plot(self.val_losses, label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Pixel error plot
        axes[1].plot(self.pixel_errors)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Pixel Error")
        axes[1].set_title("Validation Pixel Error")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("training_progress.png", dpi=150)
        plt.show()
