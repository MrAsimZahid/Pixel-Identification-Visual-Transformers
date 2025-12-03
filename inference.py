import torch
from torchvision import transforms
import numpy as np
import cv2
import os
import logging

from ViTBackbone import ViTCoordinateRegressor
from inferenceVisualization import BrightestPixelPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------
#     IMAGE PREPROCESSING (same format as training)
# -----------------------------------------------------------

def load_and_preprocess_image(img_path, img_size=64):
    """
    Loads an image (grayscale or RGB), converts to tensor, normalizes,
    and repeats channels to form 3-channel input if needed.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")

    img = cv2.resize(img, (img_size, img_size))

    # Normalize to [0,1]
    img_norm = img.astype(np.float32) / 255.0

    # Make 3-channel (ViT expects RGB)
    img_3ch = np.stack([img_norm, img_norm, img_norm], axis=0)

    tensor = torch.tensor(img_3ch, dtype=torch.float32).unsqueeze(0)  # shape (1,3,H,W)
    return tensor, img


# -----------------------------------------------------------
#                SINGLE IMAGE INFERENCE
# -----------------------------------------------------------

def predict_single_image(img_path, model_path="best_model.pth", img_size=64):
    logger.info(f"Loading model from {model_path}...")
    predictor = BrightestPixelPredictor(model_path)

    logger.info(f"Loading image: {img_path}")
    tensor, raw_img = load_and_preprocess_image(img_path, img_size)

    pred_x, pred_y = predictor.predict(raw_img)   # predictor expects raw numpy image
    logger.info(f"Prediction: x={pred_x:.2f}, y={pred_y:.2f}")

    # Save a visualization
    save_path = f"{os.path.splitext(img_path)[0]}_prediction.png"
    predictor.visualize_prediction(
        raw_img,
        pred_coords=(pred_x, pred_y),
        true_coords=None,
        save_path=save_path,
    )

    logger.info(f"Saved visualization → {save_path}")
    return pred_x, pred_y


# -----------------------------------------------------------
#             BATCH INFERENCE ON A FOLDER
# -----------------------------------------------------------

def predict_folder(folder, model_path="best_model.pth", img_size=64):
    logger.info(f"Loading predictor: {model_path}")
    predictor = BrightestPixelPredictor(model_path)

    image_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    logger.info(f"Found {len(image_files)} images in '{folder}'")

    results = []

    for img_path in image_files:
        _, raw_img = load_and_preprocess_image(img_path, img_size)

        pred_x, pred_y = predictor.predict(raw_img)
        results.append((img_path, pred_x, pred_y))

        save_path = img_path.replace(".png", "_pred.png").replace(".jpg", "_pred.png")
        predictor.visualize_prediction(raw_img, (pred_x, pred_y), None, save_path)

        logger.info(f"{img_path} → ({pred_x:.1f}, {pred_y:.1f})")

    return results


# -----------------------------------------------------------
#                   MAIN (example usage)
# -----------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference for brightest pixel model")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--folder", type=str, help="Folder containing images")
    parser.add_argument("--model", type=str, default="best_model.pth")
    parser.add_argument("--img_size", type=int, default=64)
    args = parser.parse_args()

    if args.image:
        predict_single_image(args.image, args.model, args.img_size)

    elif args.folder:
        results = predict_folder(args.folder, args.model, args.img_size)
        print("\n=== Batch Results ===")
        for path, x, y in results:
            print(f"{path}: ({x:.1f}, {y:.1f})")

    else:
        print("Please specify --image or --folder")
