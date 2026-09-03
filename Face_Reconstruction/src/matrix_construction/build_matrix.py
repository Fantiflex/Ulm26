from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps


DEFAULT_INPUT_DIR = Path("data/matrix")
DEFAULT_OUTPUT_PATH = Path("data/matrix/matrix.jpg")
DEFAULT_TILE_SIZE = 800
DEFAULT_JPEG_QUALITY = 100
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(folder: Path) -> list[Path]:
    """Return all supported image files in the folder, sorted by name."""
    if not folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")

    images = [
        path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not images:
        raise FileNotFoundError(f"No image files found in: {folder}")

    return images


def square_grid_size(n_items: int) -> int:
    """Return the smallest square grid side that can hold n_items."""
    side = int(n_items ** 0.5)
    while side * side < n_items:
        side += 1
    return side


def fit_to_square(image: Image.Image, size: int, background: tuple[int, int, int]) -> Image.Image:
    """Resize an image so it fits in a square tile without distortion."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    new_image = Image.new("RGB", (size, size), background)

    if width >= height:
        resized_height = max(1, int(height * (size / width)))
        resized = image.resize((size, resized_height), Image.Resampling.LANCZOS)
        y_offset = (size - resized_height) // 2
        new_image.paste(resized, (0, y_offset))
    else:
        resized_width = max(1, int(width * (size / height)))
        resized = image.resize((resized_width, size), Image.Resampling.LANCZOS)
        x_offset = (size - resized_width) // 2
        new_image.paste(resized, (x_offset, 0))

    return new_image


def build_matrix(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    n_images: int | None = None,
    tile_size: int = DEFAULT_TILE_SIZE,
    background: tuple[int, int, int] = (255, 255, 255),
    quality: int = DEFAULT_JPEG_QUALITY,
) -> Path:
    """
    Build a square JPG montage made from up to N pictures in a folder.

    Parameters
    ----------
    input_dir:
        Folder containing pictures.
    output_path:
        Where the final JPG collage is saved.
    n_images:
        Number of pictures to include. If None, all pictures are used.
    tile_size:
        Pixel size of each image tile in the final collage.
    background:
        Background color used for padding and empty cells.
    """
    folder = Path(input_dir)
    images = list_images(folder)

    if n_images is not None:
        if n_images <= 0:
            raise ValueError("n_images must be a positive integer.")
        images = images[:n_images]

    if not images:
        raise ValueError(f"No usable images found in {folder}")

    grid_side = square_grid_size(len(images))
    canvas_size = grid_side * tile_size
    collage = Image.new("RGB", (canvas_size, canvas_size), background)

    for index, image_path in enumerate(images):
        row = index // grid_side
        col = index % grid_side

        x0 = col * tile_size
        y0 = row * tile_size

        with Image.open(image_path) as image:
            tile = fit_to_square(image, tile_size, background)
            collage.paste(tile, (x0, y0))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    collage.save(output, format="JPEG", quality=max(1, min(100, quality)))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a square image matrix from a folder of pictures and export as JPG."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing the input pictures (default: data/matrix)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JPG path (default: data/matrix/matrix.jpg)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of pictures to include. If omitted, all pictures are used.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help="Pixel size of each image tile in the final collage. Larger values keep more detail.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help="JPEG quality from 1 to 100. Higher values preserve more detail.",
    )

    args = parser.parse_args()

    output = build_matrix(
        input_dir=args.input_dir,
        output_path=args.output,
        n_images=args.n,
        tile_size=args.tile_size,
        quality=args.quality,
    )
    print(f"Matrix saved to: {output}")


if __name__ == "__main__":
    main()
