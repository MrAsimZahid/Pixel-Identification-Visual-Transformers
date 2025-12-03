from SyntheticDataGeneration import SyntheticDataGenerator
from DataAugmentation import BrightestPixelDataset
from ViTBackbone import ViTCoordinateRegressor
from Trainer import CoordinateRegressionTrainer
from inferenceVisualization import BrightestPixelPredictor
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Configuration
    config = {
        "data_dir": "./data/train",
        "val_dir": "./data/val",
        "batch_size": 32,
        "epochs": 50,
        "learning_rate": 1e-4,
        "img_size": 64,
        "use_synthetic": False,
        "synthetic_samples": 5000,
        "save_dir": "./data/train_checker",
    }

    # Generate synthetic data if needed
    if config["use_synthetic"]:
        print("Generating synthetic dataset...")
        train_images, train_anns = SyntheticDataGenerator.generate_dataset(
            config["synthetic_samples"],
            save_dir=config["save_dir"] if config["save_dir"] else None,
        )
        val_images, val_anns = SyntheticDataGenerator.generate_dataset(500)

        logger.info("Generated synthetic dataset.")

        # Convert to dataset format
        train_dataset = BrightestPixelDataset(
            # list(range(len(train_images))),  # Dummy paths
            # train_anns,
            # train=True,
            # synthetic_augmentation=True,
            synthetic_images=train_images,
            synthetic_anns=train_anns,
            train=True,
            synthetic_augmentation=True,
        )
        val_dataset = BrightestPixelDataset(
            # list(range(len(val_images))), val_anns, train=False
            synthetic_images=val_images,
            synthetic_anns=val_anns,
            train=False,
        )

        logger.info("Created BrightestPixelDataset instances.")

        # Override __getitem__ for synthetic data
        def synthetic_getitem(self, idx):
            if self.train:
                img = train_images[idx]
                ann = train_anns[idx]
            else:
                img = val_images[idx]
                ann = val_anns[idx]

            # Convert to tensor format
            img_tensor = torch.from_numpy(
                np.stack([img / 255.0, img / 255.0, img / 255.0])
            ).float()

            coords_norm = torch.tensor(
                [ann[0] / 255.0, ann[1] / 255.0], dtype=torch.float32
            )

            return img_tensor, coords_norm

        train_dataset.__getitem__ = lambda idx: synthetic_getitem(train_dataset, idx)
        val_dataset.__getitem__ = lambda idx: synthetic_getitem(val_dataset, idx)

        logger.info("Overrode __getitem__ for synthetic data.")

    else:
        # Load real dataset
        train_dataset = BrightestPixelDataset(
            [f"{config['data_dir']}/{f}" for f in os.listdir(config["data_dir"])],
            train=True,
        )
        val_dataset = BrightestPixelDataset(
            [f"{config['val_dir']}/{f}" for f in os.listdir(config["val_dir"])],
            train=False,
        )

    logger.info("Creating DataLoaders...")
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    logger.info("Model Initialization...")
    # Initialize model
    model = ViTCoordinateRegressor(img_size=config["img_size"], use_pretrained=True)

    logger.info("Trainer Initialization...")
    # Initialize trainer
    trainer = CoordinateRegressionTrainer(model, device="cuda")

    logger.info("Starting Training...")
    # Train
    trainer.train(
        train_loader, val_loader, epochs=config["epochs"], lr=config["learning_rate"]
    )

    logger.info("Evaluating Model...")
    # Evaluate
    predictor = BrightestPixelPredictor("best_model.pth")
    pixel_errors = predictor.evaluate_on_dataset(val_dataset)

    # Visualize some predictions
    for i in range(3):
        sample_idx = np.random.randint(len(val_dataset))
        sample_img = val_images[sample_idx] if config["use_synthetic"] else None
        sample_ann = val_anns[sample_idx] if config["use_synthetic"] else None

        if sample_img is not None:
            pred_coords = predictor.predict(sample_img)
            predictor.visualize_prediction(
                sample_img,
                pred_coords=pred_coords,
                true_coords=sample_ann,
                save_path=f"prediction_sample_{i}.png",
            )


if __name__ == "__main__":
    main()
