from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps


DEFAULT_INPUT_DIR = Path("data/matrix")
DEFAULT_OUTPUT_DIR = Path("data/matrix")
DEFAULT_OUTPUT_NAME = "matrix.jpg"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _iter_images(folder: Path) -> list[Path]:
    """Return image files in the folder, sorted by file name."""
    if not folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")

    files = [
        path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        raise FileNotFoundError(f"No supported image files found in: {folder}")

    return files


def _square_size(n: int) -> int:
    """Return the smallest integer k such that k^2 >= n."""
    side = int(n ** 0.5)
    while side * side < n:
        side += 1
    return side


def _load_and_resize(image_path: Path, target_size: int, fill_color: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Open an image and fit it into a square canvas of target_size."""
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        width, height = img.size
        max_dim = max(width, height)

        scaled = Image.new("RGB", (target_size, target_size), fill_color)

        if width >= height:
            new_height = max(1, int(height * (target_size / width)))
            resized = img.resize((target_size, new_height), Image.Resampling.LANCZOS)
            y_offset = (target_size - resized.height) // 2
            scaled.paste(resized, (0, y_offset))
        else:
            new_width = max(1, int(width * (target_size / height)))
            resized = img.resize((new_width, target_size), Image.Resampling.LANCZOS)
            x_offset = (target_size - resized.width) // 2
            scaled.paste(resized, (x_offset, 0))

        return scaled


def build_matrix(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_NAME,
    n_images: int | None = None,
    target_size: int = 512,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> Path:
    """
    Build a square image collage from up to N images in a folder.

    Parameters
    ----------
    input_dir:
        Directory containing the source images.
    output_path:
        Destination JPG file.
    n_images:
        Maximum number of images to use. If None, use all available images.
    target_size:
        Size of each individual thumbnail in the final collage.
    background_color:
        Background color for empty slots and padding.
    """
    source_dir = Path(input_dir)
    files = _iter_images(source_dir)

    if n_images is not None:
        if n_images <= 0:
            raise ValueError("n_images must be a positive integer.")
        files = files[:n_images]

    if not files:
        raise ValueError(f"No images available to build a matrix from {source_dir}")

    grid_side = _square_size(len(files))
    canvas_size = grid_side * target_size
    collage = Image.new("RGB", (canvas_size, canvas_size), background_color)

    for index, image_path in enumerate(files):
        row = index // grid_side
        col = index % grid_side

        x0 = col * target_size
        y0 = row * target_size

        item = _load_and_resize(image_path, target_size, background_color)
        collage.paste(item, (x0, y0))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    collage = collage.convert("RGB")
    collage.save(output, format="JPEG", quality=95)
    return output


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a square matrix collage from images in a folder and save it as JPG."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing the input images. Default: data/matrix",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_NAME,
        help="Output JPG file path. Default: data/matrix/matrix.jpg",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of images to use. If omitted, all images are used.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Size of each individual picture tile in pixels.",
    )

    args = parser.parse_args()

    out = build_matrix(
        input_dir=args.input_dir,
        output_path=args.output,
        n_images=args.n,
        target_size=args.tile_size,
    )
    print(f"Matrix saved to: {out}")


if __name__ == "__main__":
    main()
