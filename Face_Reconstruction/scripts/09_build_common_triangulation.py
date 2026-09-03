from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, QhullError


DEFAULT_ALIGNED_DIR = Path("data/interim/aligned_faces")
DEFAULT_OUTPUT_DIR = Path("data/interim/common_triangulation")
EXPECTED_LANDMARK_COUNT = 478

# The 478-point Face Landmarker output remains untouched in steps 06-08.
# For piecewise-affine morphing, we exclude points whose geometry varies for
# reasons that are not stable identity cues:
#   - the inner lip contour can become almost collinear when the mouth closes;
#   - iris positions vary with gaze and create very thin local triangles.
# The retained 448 points are selected identically for every face.
INNER_LIP_INDICES = frozenset(
    {
        13, 14, 78, 80, 81, 82, 87, 88, 95, 178,
        191, 308, 310, 311, 312, 317, 318, 324, 402, 415,
    }
)
IRIS_INDICES = frozenset(range(468, 478))
ADDITIONAL_UNSTABLE_INDICES = {
    323,
    361,
    454, 
    358,
}



EXCLUDED_LANDMARK_INDICES = (
    INNER_LIP_INDICES
    | IRIS_INDICES
    | ADDITIONAL_UNSTABLE_INDICES
)



def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one Delaunay triangulation on the mean aligned MediaPipe "
            "shape and validate the same topology on every aligned face."
        )
    )
    parser.add_argument(
        "--aligned-dir", type=Path, default=DEFAULT_ALIGNED_DIR,
        help=f"Directory produced by step 08. Default: {DEFAULT_ALIGNED_DIR}",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Triangulation output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--expected-landmarks", type=int, default=EXPECTED_LANDMARK_COUNT,
        help=f"Expected MediaPipe landmark count. Default: {EXPECTED_LANDMARK_COUNT}.",
    )
    parser.add_argument(
        "--boundary-margin", type=float, default=0.0,
        help="Inset in pixels for the eight canvas anchors. Default: 0.",
    )
    parser.add_argument(
        "--min-mean-triangle-area", type=float, default=0.05,
        help="Reject mean-shape triangles below this area in px². Default: 0.05.",
    )
    parser.add_argument(
        "--min-face-triangle-area", type=float, default=0.01,
        help="Per-face degeneracy threshold in px². Default: 0.01.",
    )
    parser.add_argument(
        "--max-flipped-fraction", type=float, default=0.0,
        help="Maximum fraction of triangles allowed to flip. Default: 0.",
    )
    parser.add_argument(
        "--max-degenerate-fraction", type=float, default=0.0,
        help="Maximum fraction of degenerate triangles. Default: 0.",
    )
    parser.add_argument(
        "--line-thickness", type=int, default=1,
        help="Thickness of the QC triangulation overlay. Default: 1.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace existing triangulation artifacts.",
    )
    parser.add_argument(
        "--fail-on-qc-error", action="store_true",
        help="Return status 1 if any aligned face fails topology QC.",
    )
    parser.add_argument(
    "--relative-area-tolerance",
    type=float,
    default=1e-4,
    help="Relative degeneracy tolerance based on squared edge length.",
    )
    parser.add_argument(
        "--min-triangle-quality",
        type=float,
        default=1e-3,
        help="Minimum dimensionless triangle quality.",
    )
    return parser.parse_args()


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).eq(1)
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y", "oui"}
    )


def resolve_path(raw: str, reference: Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    candidates = (Path.cwd() / path, reference.resolve().parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def load_alignment(aligned_dir: Path) -> tuple[pd.DataFrame, np.ndarray, int, int]:
    manifest_path = aligned_dir / "alignment_manifest.csv"
    mean_path = aligned_dir / "mean_aligned_landmarks.npy"
    config_path = aligned_dir / "alignment_config.json"
    for path in (manifest_path, mean_path, config_path):
        if not path.exists():
            raise FileNotFoundError(f"Required step-08 artifact not found: {path}")

    manifest = pd.read_csv(manifest_path)
    required = {"face_id", "aligned_image_path", "aligned_landmark_path"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Alignment manifest is missing columns: {sorted(missing)}")
    if "alignment_qc_pass" not in manifest.columns:
        raise ValueError("Alignment manifest has no alignment_qc_pass column.")
    manifest["face_id"] = manifest["face_id"].astype(str).str.strip()
    manifest["alignment_qc_pass"] = parse_bool(manifest["alignment_qc_pass"])
    manifest = manifest.loc[manifest["alignment_qc_pass"]].copy()
    manifest = manifest.sort_values("face_id", kind="stable").reset_index(drop=True)
    if manifest.empty:
        raise ValueError("No step-08 alignment passed QC.")
    if manifest["face_id"].duplicated().any():
        raise ValueError("Alignment manifest contains duplicate face_id values.")

    mean_landmarks = np.asarray(np.load(mean_path, allow_pickle=False), dtype=np.float64)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    width = int(config["output_width"])
    height = int(config["output_height"])
    return manifest, mean_landmarks, width, height


def build_selected_landmark_indices(expected_landmarks: int) -> np.ndarray:
    """Return stable MediaPipe indices, preserving their original order."""
    invalid = sorted(
        index
        for index in EXCLUDED_LANDMARK_INDICES
        if index < 0 or index >= expected_landmarks
    )
    if invalid:
        raise ValueError(
            "The configured exclusion set is incompatible with "
            f"--expected-landmarks={expected_landmarks}: {invalid}"
        )
    selected = np.asarray(
        [
            index
            for index in range(expected_landmarks)
            if index not in EXCLUDED_LANDMARK_INDICES
        ],
        dtype=np.int32,
    )
    if selected.size == 0:
        raise ValueError("Landmark selection is empty.")
    return selected


def boundary_points(width: int, height: int, margin: float) -> np.ndarray:
    left, top = margin, margin
    right, bottom = width - 1.0 - margin, height - 1.0 - margin
    if not (0.0 <= left < right and 0.0 <= top < bottom):
        raise ValueError("--boundary-margin leaves no valid canvas interior.")
    mid_x = (left + right) / 2.0
    mid_y = (top + bottom) / 2.0
    return np.array(
        [
            [left, top], [mid_x, top], [right, top], [right, mid_y],
            [right, bottom], [mid_x, bottom], [left, bottom], [left, mid_y],
        ],
        dtype=np.float64,
    )


def signed_double_areas(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a = points[triangles[:, 0]]
    b = points[triangles[:, 1]]
    c = points[triangles[:, 2]]
    return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (
        b[:, 1] - a[:, 1]
    ) * (c[:, 0] - a[:, 0])


def orient_triangles_ccw(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    result = np.asarray(triangles, dtype=np.int32).copy()
    negative = signed_double_areas(points, result) < 0.0
    result[negative, 1], result[negative, 2] = (
        result[negative, 2].copy(), result[negative, 1].copy()
    )
    return result


def sha256_arrays(points: np.ndarray, triangles: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(points, dtype="<f4").tobytes(order="C"))
    digest.update(np.asarray(triangles, dtype="<i4").tobytes(order="C"))
    return digest.hexdigest()


def triangle_geometry(
    points: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = points[triangles]

    ab = vertices[:, 1] - vertices[:, 0]
    ac = vertices[:, 2] - vertices[:, 0]
    bc = vertices[:, 2] - vertices[:, 1]

    signed_area = (
        ab[:, 0] * ac[:, 1]
        - ab[:, 1] * ac[:, 0]
    ) / 2.0

    squared_lengths = np.stack(
        [
            np.sum(ab * ab, axis=1),
            np.sum(ac * ac, axis=1),
            np.sum(bc * bc, axis=1),
        ],
        axis=1,
    )

    max_edge_squared = squared_lengths.max(axis=1)

    # 1 = triangle équilatéral ; 0 = triangle aplati
    quality = (
        4.0
        * np.sqrt(3.0)
        * np.abs(signed_area)
        / np.maximum(squared_lengths.sum(axis=1), 1e-12)
    )

    return signed_area, max_edge_squared, quality

def validate_face_topology(
    points: np.ndarray,
    reference_points: np.ndarray,
    triangles: np.ndarray,
    selected_indices: np.ndarray,
    *,
    min_absolute_area: float,
    relative_area_tolerance: float,
    min_triangle_quality: float,
    max_flipped_fraction: float,
    max_degenerate_fraction: float,
) -> dict[str, Any]:
    face_area, max_edge_squared, quality = triangle_geometry(
        points,
        triangles,
    )
    reference_area, _, _ = triangle_geometry(
        reference_points,
        triangles,
    )

    area_tolerance = np.maximum(
        min_absolute_area,
        relative_area_tolerance * max_edge_squared,
    )

    degenerate = (
        (np.abs(face_area) <= area_tolerance)
        | (quality < min_triangle_quality)
    )

    # On ne classe pas comme inversé un triangle déjà quasi dégénéré.
    flipped = (
        (face_area * reference_area < 0.0)
        & ~degenerate
    )

    unstable = degenerate | flipped
    problem_triangle_indices = np.flatnonzero(unstable)

    if len(problem_triangle_indices) > 0:
        problem_point_indices = np.unique(
            triangles[problem_triangle_indices].ravel()
        )
    else:
        problem_point_indices = np.array([], dtype=int)

    n_landmarks = len(selected_indices)

    problem_mediapipe_indices: list[int | str] = []

    for point_index in problem_point_indices:
        point_index = int(point_index)

        if point_index < n_landmarks:
            problem_mediapipe_indices.append(
                int(selected_indices[point_index])
            )
        else:
            anchor_index = point_index - n_landmarks
            problem_mediapipe_indices.append(
                f"anchor_{anchor_index}"
            )

    flipped_fraction = float(flipped.mean())
    degenerate_fraction = float(degenerate.mean())

    reasons: list[str] = []
    if flipped_fraction > max_flipped_fraction:
        reasons.append("flipped_triangles")
    if degenerate_fraction > max_degenerate_fraction:
        reasons.append("degenerate_triangles")

    return {
        "n_triangles": int(len(triangles)),
        "n_flipped_triangles": int(flipped.sum()),
        "flipped_triangle_fraction": flipped_fraction,
        "flipped_triangle_indices": json.dumps(
            np.flatnonzero(flipped).tolist()
        ),
        "n_degenerate_triangles": int(degenerate.sum()),
        "degenerate_triangle_fraction": degenerate_fraction,
        "degenerate_triangle_indices": json.dumps(
            np.flatnonzero(degenerate).tolist()
        ),
        "unstable_triangle_indices": json.dumps(
            np.flatnonzero(unstable).tolist()
        ),
        "min_absolute_triangle_area_px2": float(
            np.abs(face_area).min()
        ),
        "min_triangle_quality": float(quality.min()),
        "median_triangle_quality": float(np.median(quality)),
        "topology_qc_pass": not reasons,
        "topology_qc_reason": (
            "ok" if not reasons else ";".join(reasons)
        ),
        
        "problem_point_indices": json.dumps(
            problem_point_indices.tolist()
        ),
        "problem_mediapipe_indices": json.dumps(
            problem_mediapipe_indices
        ),
        "problem_triangle_indices": json.dumps(
            problem_triangle_indices.tolist()
        ),
    }


def draw_overlay(
    image: np.ndarray,
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    thickness: int,
    landmark_count: int,
) -> np.ndarray:
    overlay = image.copy()
    for triangle in triangles:
        polygon = np.rint(points[triangle]).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [polygon], True, (0, 220, 0), thickness, cv2.LINE_AA)
    for x, y in points[:landmark_count]:
        cv2.circle(overlay, (int(round(x)), int(round(y))), 1, (0, 80, 255), -1)
    return overlay


def ensure_writable(paths: list[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Triangulation outputs already exist. Use --overwrite: " + ", ".join(existing)
        )


def main() -> int:
    args = parse_arguments()
    if args.expected_landmarks <= 0:
        raise ValueError("--expected-landmarks must be positive.")
    if args.min_mean_triangle_area < 0 or args.min_face_triangle_area < 0:
        raise ValueError("Triangle-area thresholds cannot be negative.")
    if not 0 <= args.max_flipped_fraction <= 1:
        raise ValueError("--max-flipped-fraction must be between 0 and 1.")
    if not 0 <= args.max_degenerate_fraction <= 1:
        raise ValueError("--max-degenerate-fraction must be between 0 and 1.")
    if args.line_thickness <= 0:
        raise ValueError("--line-thickness must be positive.")

    aligned_dir = args.aligned_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    manifest, mean_landmarks, width, height = load_alignment(aligned_dir)
    expected_shape = (args.expected_landmarks, 2)
    if mean_landmarks.shape != expected_shape:
        raise ValueError(f"Expected mean shape {expected_shape}, got {mean_landmarks.shape}.")
    if not np.isfinite(mean_landmarks).all():
        raise ValueError("Mean landmarks contain non-finite coordinates.")

    selected_indices = build_selected_landmark_indices(args.expected_landmarks)
    excluded_indices = np.asarray(
        sorted(EXCLUDED_LANDMARK_INDICES), dtype=np.int32
    )
    selected_mean_landmarks = mean_landmarks[selected_indices]
    anchors = boundary_points(width, height, args.boundary_margin)
    common_points = np.vstack([selected_mean_landmarks, anchors])
    if len(np.unique(np.round(common_points, decimals=6), axis=0)) != len(common_points):
        raise ValueError("Common points contain duplicates at 1e-6 pixel precision.")
    try:
        triangles = Delaunay(common_points).simplices.astype(np.int32)
    except QhullError as error:
        raise RuntimeError(f"Delaunay triangulation failed: {error}") from error
    triangles = orient_triangles_ccw(common_points, triangles)
    mean_areas = signed_double_areas(common_points, triangles) / 2.0
    if np.any(mean_areas <= args.min_mean_triangle_area):
        count = int(np.sum(mean_areas <= args.min_mean_triangle_area))
        raise ValueError(
            f"Mean triangulation contains {count} triangles at or below "
            f"{args.min_mean_triangle_area} px²."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    points_path = output_dir / "common_points.npy"
    triangles_path = output_dir / "common_triangles.npy"
    selected_indices_path = output_dir / "selected_landmark_indices.npy"
    excluded_indices_path = output_dir / "excluded_landmark_indices.npy"
    manifest_path = output_dir / "triangulation_manifest.csv"
    config_path = output_dir / "triangulation_config.json"
    overlay_path = output_dir / "common_triangulation_overlay.png"
    ensure_writable(
        [
            points_path,
            triangles_path,
            selected_indices_path,
            excluded_indices_path,
            manifest_path,
            config_path,
            overlay_path,
        ],
        args.overwrite,
    )

    rows: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        landmark_path = resolve_path(str(row.aligned_landmark_path), aligned_dir / "alignment_manifest.csv")
        result: dict[str, Any] = {
            "face_id": str(row.face_id),
            "aligned_landmark_path": landmark_path.as_posix(),
            "topology_qc_pass": False,
            "topology_qc_reason": "not_processed",
            "n_triangles": len(triangles),
            "n_flipped_triangles": np.nan,
            "flipped_triangle_fraction": np.nan,
            "n_degenerate_triangles": np.nan,
            "degenerate_triangle_fraction": np.nan,
            "min_absolute_triangle_area_px2": np.nan,
            "median_absolute_triangle_area_px2": np.nan,
            "error_type": "",
            "error_message": "",
        }
        try:
            landmarks = np.asarray(np.load(landmark_path, allow_pickle=False), dtype=np.float64)
            if landmarks.shape != expected_shape:
                raise ValueError(f"Expected {expected_shape}, got {landmarks.shape}.")
            if not np.isfinite(landmarks).all():
                raise ValueError("Non-finite aligned landmarks.")
            selected_landmarks = landmarks[selected_indices]
            face_points = np.vstack([selected_landmarks, anchors])
            result.update(
                validate_face_topology(
                    face_points,
                    common_points,
                    triangles,
                    selected_indices,
                    min_absolute_area=args.min_face_triangle_area,
                    relative_area_tolerance=args.relative_area_tolerance,
                    min_triangle_quality=args.min_triangle_quality,
                    max_flipped_fraction=args.max_flipped_fraction,
                    max_degenerate_fraction=args.max_degenerate_fraction,
                )
)
        except Exception as error:
            result.update(
                {
                    "topology_qc_reason": "processing_exception",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
        rows.append(result)

    results = pd.DataFrame(rows)
    np.save(points_path, common_points.astype(np.float32), allow_pickle=False)
    np.save(triangles_path, triangles, allow_pickle=False)
    np.save(selected_indices_path, selected_indices, allow_pickle=False)
    np.save(excluded_indices_path, excluded_indices, allow_pickle=False)
    results.to_csv(manifest_path, index=False)

    mean_image_path = aligned_dir / "mean_aligned_face.png"
    mean_image = cv2.imread(str(mean_image_path), cv2.IMREAD_COLOR)
    if mean_image is None:
        raise ValueError(f"OpenCV could not read mean aligned face: {mean_image_path}")
    if mean_image.shape[:2] != (height, width):
        raise ValueError(
            f"Mean image shape {mean_image.shape[:2]} does not match {(height, width)}."
        )
    overlay = draw_overlay(
        mean_image,
        common_points,
        triangles,
        thickness=args.line_thickness,
        landmark_count=len(selected_indices),
    )
    if not cv2.imwrite(str(overlay_path), overlay):
        raise OSError(f"OpenCV could not save: {overlay_path}")

    config = {
        "algorithm": "scipy.spatial.Delaunay",
        "landmark_topology": "mediapipe_face_landmarker_478",
        "landmark_selection": "stable_448_without_inner_lips_or_irises",
        "triangulation_reference": "selected_mean_aligned_landmarks",
        "canvas_width": width,
        "canvas_height": height,
        "n_detected_landmarks": args.expected_landmarks,
        "n_selected_landmarks": int(len(selected_indices)),
        "n_excluded_landmarks": int(len(excluded_indices)),
        "selected_landmark_indices_file": selected_indices_path.name,
        "excluded_landmark_indices_file": excluded_indices_path.name,
        "inner_lip_indices_excluded": sorted(INNER_LIP_INDICES),
        "iris_indices_excluded": sorted(IRIS_INDICES),
        "n_boundary_anchors": len(anchors),
        "boundary_anchor_indices": list(
            range(len(selected_indices), len(common_points))
        ),
        "boundary_points": anchors.tolist(),
        "n_triangles": len(triangles),
        "min_mean_triangle_area_px2": float(mean_areas.min()),
        "median_mean_triangle_area_px2": float(np.median(mean_areas)),
        "min_face_triangle_area_threshold_px2": args.min_face_triangle_area,
        "max_flipped_fraction": args.max_flipped_fraction,
        "max_degenerate_fraction": args.max_degenerate_fraction,
        "topology_sha256": sha256_arrays(common_points, triangles),
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    passed = int(results["topology_qc_pass"].sum())
    print("\nCommon triangulation completed")
    print(
        f"Landmarks: {args.expected_landmarks} detected -> "
        f"{len(selected_indices)} selected -> {len(excluded_indices)} excluded"
    )
    print(f"Points: {len(common_points)} ({len(selected_indices)} landmarks + 8 anchors)")
    print(f"Triangles: {len(triangles)}")
    print(f"Face topology QC passed: {passed}/{len(results)}")
    print(f"QC overlay: {overlay_path}")
    print(f"Selected landmark indices: {selected_indices_path}")
    print(f"Manifest: {manifest_path}")
    failures = results.loc[~results["topology_qc_pass"]]
    if not failures.empty:
        print("\nTopology QC failures:")
        print(
            failures[
                ["face_id", "topology_qc_reason", "n_flipped_triangles",
                 "n_degenerate_triangles", "error_type", "error_message"]
            ].to_string(index=False)
        )
    return 1 if args.fail_on_qc_error and passed != len(results) else 0


if __name__ == "__main__":
    sys.exit(main())
