from pathlib import Path

import numpy as np
import pandas as pd


from minority_estimation.config import CELL_PX

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FACE_LUMINANCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "faces_only"
    / "face_luminance.csv"
)



df = pd.read_csv(FACE_LUMINANCE_PATH)

df = df.loc[
    df["face_detected"] == True
].copy()


print("\n=== FACE AREA DISTRIBUTION ===")

print(
    df["face_area_px"].describe()
)


median_face_area = float(
    df["face_area_px"].median()
)

mean_face_area = float(
    df["face_area_px"].mean()
)


median_equivalent_radius = np.sqrt(
    median_face_area / np.pi
)

mean_equivalent_radius = np.sqrt(
    mean_face_area / np.pi
)


median_radius_ratio = (
    median_equivalent_radius
    / CELL_PX
)

mean_radius_ratio = (
    mean_equivalent_radius
    / CELL_PX
)


print("\n=== EQUIVALENT CIRCLE ===")

print(
    f"Median face area: "
    f"{median_face_area:.1f} px²"
)

print(
    f"Mean face area: "
    f"{mean_face_area:.1f} px²"
)

print(
    f"Radius from median area: "
    f"{median_equivalent_radius:.2f} px"
)

print(
    f"Radius from mean area: "
    f"{mean_equivalent_radius:.2f} px"
)

print(
    f"Recommended radius ratio "
    f"(median-based): "
    f"{median_radius_ratio:.3f}"
)

print(
    f"Mean-based radius ratio: "
    f"{mean_radius_ratio:.3f}"
)