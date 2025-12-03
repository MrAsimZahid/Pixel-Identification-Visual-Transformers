from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch


class BrightestPixelDataset(Dataset):
    def __init__(
        self,
        image_paths=None,
        annotations=None,
        train=True,
        synthetic_augmentation=False,
        synthetic_images=None,
        synthetic_anns=None,
    ):
        """
        Args:
            image_paths: List of image file paths
            annotations: Optional list of (x, y) coordinates
            train: Whether in training mode (for augmentation)
            synthetic_augmentation: Whether to generate synthetic bright spots
        """
        self.train = train
        self.synthetic_augmentation = synthetic_augmentation

        # Synthetic dataset flag
        self.use_synthetic = synthetic_images is not None

        self.synthetic_images = synthetic_images
        self.synthetic_anns = synthetic_anns

        # Real dataset paths
        self.image_paths = image_paths
        self.annotations = annotations
        self.train = train
        self.synthetic_augmentation = synthetic_augmentation

        # Basic transforms
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),  # Converts to [0, 1] range
            ]
        )

        # Image augmentations (geometric only - avoid brightness changes)
        self.augmentations = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=30, p=0.5, border_mode=0),  # Black borders
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.3
                ),
            ],
            keypoint_params=(
                A.KeypointParams(format="xy", remove_invisible=False) if train else None
            ),
        )

    def __len__(self):
        if self.use_synthetic:
            return len(self.synthetic_images)
        return len(self.image_paths)

    def _add_synthetic_bright_spot(self, image_array):
        """Add a synthetic bright spot for training augmentation"""
        if np.random.random() < 0.3 and self.synthetic_augmentation:
            h, w = image_array.shape

            # Create Gaussian bright spot
            center_x = np.random.randint(20, w - 20)
            center_y = np.random.randint(20, h - 20)
            intensity = np.random.uniform(0.8, 1.0)
            sigma = np.random.uniform(5, 15)

            y, x = np.ogrid[:h, :w]
            distance = (x - center_x) ** 2 + (y - center_y) ** 2
            bright_spot = intensity * np.exp(-distance / (2 * sigma**2))

            # Add to image with clipping
            image_array = np.clip(image_array + bright_spot, 0, 1)

            return image_array, (center_x, center_y)

        return image_array, None

    def __getitem__(self, idx):

        # --- SYNTHETIC MODE ---
        if self.use_synthetic:
            img = self.synthetic_images[idx]
            ann = self.synthetic_anns[idx]

            # Convert grayscale (H,W) → 3-channel tensor
            img_tensor = torch.from_numpy(
                np.stack([img / 255.0, img / 255.0, img / 255.0])
            ).float()

            coords_norm = torch.tensor(
                [ann[0] / 255.0, ann[1] / 255.0], dtype=torch.float32
            )

            return img_tensor, coords_norm

        # --- REAL MODE ---
        # Load image
        img_path = self.image_paths[idx]

        # Load as grayscale
        image = Image.open(img_path).convert("L")
        image_array = np.array(image, dtype=np.float32) / 255.0

        # Get or generate annotation
        if self.annotations is not None:
            target_x, target_y = self.annotations[idx]
        else:
            # Find brightest pixel
            flat_idx = np.argmax(image_array)
            target_y, target_x = np.unravel_index(flat_idx, image_array.shape)

        # Synthetic augmentation (add bright spot)
        synthetic_coords = None
        if self.train and self.synthetic_augmentation:
            image_array, synthetic_coords = self._add_synthetic_bright_spot(image_array)
            if synthetic_coords is not None:
                target_x, target_y = synthetic_coords

        # Convert to 3-channel (repeat grayscale)
        image_3ch = np.stack([image_array, image_array, image_array], axis=0)
        image_3ch = np.transpose(image_3ch, (1, 2, 0))  # HWC format for albumentations

        # Apply augmentations
        keypoints = [(target_x, target_y)]

        if self.train:
            augmented = self.augmentations(image=image_3ch, keypoints=keypoints)
            image_3ch = augmented["image"]
            target_x, target_y = augmented["keypoints"][0]
        else:
            # Ensure keypoints are within bounds
            target_x = np.clip(target_x, 0, 255)
            target_y = np.clip(target_y, 0, 255)

        # Convert to tensor
        image_tensor = torch.from_numpy(
            np.transpose(image_3ch, (2, 0, 1))  # CHW format
        ).float()

        # Normalize coordinates to [0, 1]
        coords_norm = torch.tensor(
            [target_x / 255.0, target_y / 255.0], dtype=torch.float32
        )

        return image_tensor, coords_norm
