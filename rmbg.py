#!/usr/bin/env python3
"""
Script to add transparency channel to PNG images and make background transparent.

This script calculates the difference between each pixel and the background color,
then applies transparency based on the specified thresholds.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def calculate_color_distance(
    pixels: np.ndarray, background_color: tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Calculate the normalized distance between each pixel and the background color.

    Args:
        pixels: NumPy array of shape (H, W, 3) or (H, W, 4) containing RGB(A) values.
        background_color: The background color to compare against (default white).

    Returns:
        NumPy array of shape (H, W) with normalized distances in range [0, 1].
    """
    # Extract RGB channels only
    rgb = pixels[:, :, :3].astype(np.float32)
    bg = np.array(background_color, dtype=np.float32)

    # Calculate Euclidean distance and normalize
    # Maximum possible distance is sqrt(255^2 * 3) ≈ 441.67
    max_distance = np.sqrt(3 * (255**2))
    distance = np.sqrt(np.sum((rgb - bg) ** 2, axis=2)) / max_distance

    return distance


def apply_transparency(
    image: Image.Image,
    transparent_threshold: float = 0.1,
    opaque_threshold: float = 1.00,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Apply transparency to an image based on color distance from background.

    Pixels with distance below transparent_threshold become fully transparent.
    Pixels with distance above opaque_threshold become fully opaque.
    Pixels in between keep their original alpha values.

    Args:
        image: PIL Image object to process.
        transparent_threshold: Distance threshold below which pixels become transparent.
        opaque_threshold: Distance threshold above which pixels remain opaque.
        background_color: The background color to detect (default white).

    Returns:
        PIL Image with transparency applied.

    Raises:
        ValueError: If thresholds are invalid.
    """
    if not (0 <= transparent_threshold <= 1):
        raise ValueError("transparent_threshold must be between 0 and 1")
    if not (0 <= opaque_threshold <= 1):
        raise ValueError("opaque_threshold must be between 0 and 1")
    if transparent_threshold > opaque_threshold:
        raise ValueError("transparent_threshold must be <= opaque_threshold")

    # Convert to RGBA if necessary
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    pixels = np.array(image)
    distance = calculate_color_distance(pixels, background_color)

    # Get original alpha channel
    original_alpha = pixels[:, :, 3].copy()

    # Calculate alpha values based on thresholds
    # - Below transparent_threshold: fully transparent (alpha = 0)
    # - Above opaque_threshold: fully opaque (alpha = 255)
    # - In between: keep original alpha
    alpha = np.where(distance < transparent_threshold, 0, original_alpha)
    alpha = np.where(distance > opaque_threshold, 255, alpha)

    # Apply the new alpha channel
    pixels[:, :, 3] = alpha.astype(np.uint8)

    return Image.fromarray(pixels, mode="RGBA")


def process_image(
    input_path: str,
    output_path: str,
    transparent_threshold: float = 0.1,
    opaque_threshold: float = 1.00,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """
    Process an image file to add transparency.

    Args:
        input_path: Path to the input image file.
        output_path: Path to save the output image.
        transparent_threshold: Distance threshold for full transparency.
        opaque_threshold: Distance threshold for full opacity.
        background_color: The background color to detect.

    Raises:
        FileNotFoundError: If input file does not exist.
        ValueError: If input file is not a valid image.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"🚀 Processing image: {input_path}")
    print(f"   Transparent threshold: {transparent_threshold}")
    print(f"   Opaque threshold: {opaque_threshold}")
    print(f"   Background color: RGB{background_color}")

    try:
        image = Image.open(input_path)
    except Exception as e:
        raise ValueError(f"Failed to open image: {e}")

    result = apply_transparency(
        image,
        transparent_threshold=transparent_threshold,
        opaque_threshold=opaque_threshold,
        background_color=background_color,
    )

    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    result.save(output_path, format="PNG")
    print(f"✅ Image saved to: {output_path}")


def parse_color(color_str: str) -> tuple[int, int, int]:
    """
    Parse a color string in format 'R,G,B' to a tuple.

    Args:
        color_str: Color string like '255,255,255'.

    Returns:
        Tuple of (R, G, B) values.

    Raises:
        ValueError: If color format is invalid.
    """
    try:
        parts = [int(x.strip()) for x in color_str.split(",")]
        if len(parts) != 3:
            raise ValueError("Color must have exactly 3 components")
        if not all(0 <= p <= 255 for p in parts):
            raise ValueError("Color values must be between 0 and 255")
        return tuple(parts)  # type: ignore
    except Exception as e:
        raise ValueError(f"Invalid color format '{color_str}': {e}")


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Add transparency to PNG images by making background transparent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i input.png -o output.png
  %(prog)s -i input.png -o output.png --transparent 0.05 --opaque 0.5
  %(prog)s -i input.png -o output.png --background 0,0,0
        """,
    )

    parser.add_argument(
        "-i", "--input", required=True, help="Path to the input image file"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to save the output image"
    )
    parser.add_argument(
        "--transparent",
        type=float,
        default=0.1,
        help="Transparent threshold (0-1). Pixels with distance below this become transparent. Default: 0.1",
    )
    parser.add_argument(
        "--opaque",
        type=float,
        default=1.00,
        help="Opaque threshold (0-1). Pixels with distance above this remain opaque. Default: 1.00",
    )
    parser.add_argument(
        "--background",
        type=str,
        default="255,255,255",
        help="Background color to detect as 'R,G,B'. Default: 255,255,255 (white)",
    )

    args = parser.parse_args()

    try:
        background_color = parse_color(args.background)
        process_image(
            input_path=args.input,
            output_path=args.output,
            transparent_threshold=args.transparent,
            opaque_threshold=args.opaque,
            background_color=background_color,
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
