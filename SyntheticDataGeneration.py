import numpy as np
import cv2


class SyntheticDataGenerator:
    @staticmethod
    def generate_checkerboard_image(size=64, grid_size=8, intensity_range=(0.0, 1.0)):
        """
        Generate a checkerboard pattern with squares of different grayscale intensities.

        Args:
            size: Image size (size x size)
            grid_size: Number of squares per row/column
            intensity_range: Tuple (min_intensity, max_intensity) for random intensities

        Returns:
            image: Checkerboard pattern image
            (center_x, center_y): Coordinates of the brightest square's center
            squares_info: List of dictionaries with square information
        """
        # Create black image
        image = np.zeros((size, size))

        # Calculate square size
        square_size = size // grid_size

        # Generate random intensities for each square
        intensities = np.random.uniform(
            intensity_range[0], intensity_range[1], (grid_size, grid_size)
        )

        squares_info = []
        max_intensity = 0
        brightest_square_center = (0, 0)

        # Create checkerboard pattern
        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate square boundaries
                x_start = i * square_size
                x_end = (i + 1) * square_size
                y_start = j * square_size
                y_end = (j + 1) * square_size

                # Ensure we don't exceed image boundaries
                x_end = min(x_end, size)
                y_end = min(y_end, size)

                # Set the intensity for this square
                intensity = intensities[i, j]
                image[y_start:y_end, x_start:x_end] = intensity

                # Calculate center of this square
                center_x = (x_start + x_end) // 2
                center_y = (y_start + y_end) // 2

                # Store square information
                square_info = {
                    "top_left": (x_start, y_start),
                    "bottom_right": (x_end, y_end),
                    "center": (center_x, center_y),
                    "intensity": intensity,
                    "grid_position": (i, j),
                }
                squares_info.append(square_info)

                # Track brightest square
                if intensity > max_intensity:
                    max_intensity = intensity
                    brightest_square_center = (center_x, center_y)

        return image, brightest_square_center, squares_info

    # Alternative: Classic checkerboard with alternating black/white squares
    @staticmethod
    def generate_classic_checkerboard(
        size=256, grid_size=8, black_intensity=0.0, white_intensity=1.0
    ):
        """
        Generate a classic checkerboard pattern with alternating black and white squares.

        Args:
            size: Image size (size x size)
            grid_size: Number of squares per row/column
            black_intensity: Intensity for black squares (default 0.0)
            white_intensity: Intensity for white squares (default 1.0)

        Returns:
            image: Classic checkerboard image
            (center_x, center_y): Coordinates of a white square's center
            squares_info: List of dictionaries with square information
        """
        # Create base pattern
        image = np.zeros((size, size))
        square_size = size // grid_size

        squares_info = []

        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate square boundaries
                x_start = i * square_size
                x_end = min((i + 1) * square_size, size)
                y_start = j * square_size
                y_end = min((j + 1) * square_size, size)

                # Alternate colors
                if (i + j) % 2 == 0:
                    intensity = white_intensity
                else:
                    intensity = black_intensity

                image[y_start:y_end, x_start:x_end] = intensity

                # Calculate center
                center_x = (x_start + x_end) // 2
                center_y = (y_start + y_end) // 2

                squares_info.append(
                    {
                        "center": (center_x, center_y),
                        "intensity": intensity,
                        "grid_position": (i, j),
                    }
                )

        # Return center of first white square
        for square in squares_info:
            if square["intensity"] == white_intensity:
                brightest_center = square["center"]
                break
        else:
            brightest_center = (size // 2, size // 2)

        return image, brightest_center, squares_info

    @staticmethod
    def generate_image_with_single_pixel(size=256, x=None, y=None):
        """
        Generate synthetic image with a single white pixel at specified location.
        If no location is provided, a random position is chosen.

        Args:
            size: Image size (size x size)
            x: X-coordinate of the white pixel (optional)
            y: Y-coordinate of the white pixel (optional)

        Returns:
            image: 2D array with a single white pixel (1.0) and rest black (0.0)
            (brightest_x, brightest_y): Coordinates of the white pixel
            bright_spots: List containing the single spot coordinates
        """
        # Create black image
        image = np.zeros((size, size))

        # If coordinates not provided, choose random position
        if x is None or y is None:
            x = np.random.randint(0, size)
            y = np.random.randint(0, size)

        # Set the single pixel to white (value 1.0)
        image[y, x] = 1.0

        # Create bright spots list (containing just this one spot)
        bright_spots = [(x, y)]

        return image, (x, y), bright_spots

    @staticmethod
    def generate_image_with_bright_spot(size=256):
        """
        Generate synthetic image with controlled bright spot
        """
        # Start with random background noise
        image = np.random.normal(0.3, 0.1, (size, size))
        image = np.clip(image, 0, 1)

        # Add multiple bright spots
        num_spots = np.random.randint(1, 4)
        bright_spots = []

        for _ in range(num_spots):
            # Random position
            center_x = np.random.randint(30, size - 30)
            center_y = np.random.randint(30, size - 30)

            # Random parameters
            intensity = np.random.uniform(0.5, 1.0)
            sigma = np.random.uniform(5, 20)

            # Create Gaussian spot
            y, x = np.ogrid[:size, :size]
            distance = (x - center_x) ** 2 + (y - center_y) ** 2
            spot = intensity * np.exp(-distance / (2 * sigma**2))

            image = np.clip(image + spot, 0, 1)
            bright_spots.append((center_x, center_y))

        # Find actual brightest pixel (might be different from Gaussian center)
        flat_idx = np.argmax(image)
        brightest_y, brightest_x = np.unravel_index(flat_idx, image.shape)

        return image, (brightest_x, brightest_y), bright_spots

    @staticmethod
    def generate_dataset(n_samples=1000, save_dir=None):
        """
        Generate a synthetic dataset for training
        """
        images = []
        annotations = []

        for i in range(n_samples):
            image, brightest_coords, _ = (
                SyntheticDataGenerator.generate_checkerboard_image()
            )
            # (
            #     SyntheticDataGenerator.generate_image_with_bright_spot()
            # )

            # Scale to 0-255 and convert to uint8
            image_uint8 = (image * 255).astype(np.uint8)
            images.append(image_uint8)
            annotations.append(brightest_coords)

            # Optionally save
            if save_dir:
                cv2.imwrite(f"{save_dir}/image_{i:04d}.png", image_uint8)

                # Save annotation
                with open(f"{save_dir}/image_{i:04d}.txt", "w") as f:
                    f.write(f"{brightest_coords[0]},{brightest_coords[1]}")

        return images, annotations
