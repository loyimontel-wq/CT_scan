#!/usr/bin/env python3

import re
import numpy as np
import tifffile
import pyvista as pv

from pathlib import Path


# ==========================================================
# 1. INPUT FOLDER
# ==========================================================

folder = Path(
    "/home/loyiso/Desktop/CT_scan/sim_poly/recon"
)


# ==========================================================
# 2. FIND AND LOAD TIFF DATA
# ==========================================================

if not folder.is_dir():
    raise NotADirectoryError(f"Folder does not exist: {folder}")


# Find TIFF files
files = list(folder.glob("*.tif")) + list(folder.glob("*.tiff"))


if not files:
    raise FileNotFoundError(
        f"No TIFF files found inside: {folder}"
    )


# Natural sorting for numbered filenames
def natural_sort_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


files = sorted(files, key=natural_sort_key)


print("\nTIFF files found:")

for file in files:
    print(" -", file.name)


# ==========================================================
# 3. LOAD THE VOLUME
# ==========================================================

if len(files) == 1:

    print("\nLoading single TIFF file / stack...")

    volume = tifffile.imread(files[0])

else:

    print("\nLoading multiple TIFF slices...")

    slices = [tifffile.imread(file) for file in files]

    volume = np.stack(slices, axis=0)

print("\n========== VOXEL ANALYSIS ==========")

percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]

for p in percentiles:
    print(f"{p}% percentile: {np.percentile(volume, p):.2f}")

print("\nNon-zero voxels:", np.count_nonzero(volume))
print("Total voxels:", volume.size)

print(
    "Percentage of non-zero voxels:",
    100 * np.count_nonzero(volume) / volume.size
)

print("\n========== SLICE CHECK ==========")

for z in range(volume.shape[0]):
    current_slice = volume[z]

    print(
        f"Slice {z:03d}: "
        f"min={current_slice.min()}, "
        f"max={current_slice.max()}, "
        f"mean={current_slice.mean():.2f}, "
        f"nonzero={np.count_nonzero(current_slice)}"
    )
# ==========================================================
# 4. CHECK THE DATA
# ==========================================================

print("\n========== VOLUME INFORMATION ==========")

print("Volume shape (Z, Y, X):", volume.shape)
print("Data type:", volume.dtype)
print("Minimum value:", volume.min())
print("Maximum value:", volume.max())

if volume.ndim != 3:
    raise ValueError(
        f"Expected a 3D volume, but received {volume.ndim} dimensions."
    )

nz, ny, nx = volume.shape

print("Number of slices:", nz)
print("Image height:", ny)
print("Image width:", nx)

# ==========================================================
# 5. PREPARE VOLUME
# ==========================================================

# Convert to floating point for visualization
volume_float = volume.astype(np.float32)

# Use percentile limits to avoid extreme values
vmin, vmax = np.percentile(volume_float, [1, 99])

print("Display range:", vmin, vmax)

if vmax <= vmin:
    raise ValueError("Volume has no usable intensity range.")

grid = pv.ImageData()

nz, ny, nx = volume.shape

grid.dimensions = (nx, ny, nz)
grid.spacing = (1.0, 1.0, 1.0)

# Flatten in VTK point order
grid.point_data["intensity"] = volume_float.ravel(order="C")

# Create a surface from the reconstructed volume

threshold = np.percentile(volume, 95)

print("Surface threshold:", threshold)

surface = grid.contour(
    isosurfaces=[threshold],
    scalars="intensity"
)

print("Surface points:", surface.n_points)
print("Surface cells:", surface.n_cells)

plotter = pv.Plotter()

if surface.n_points > 0:
    plotter.add_mesh(
        surface,
        color="lightgray",
        show_edges=False
    )
else:
    print("No surface found at this threshold.")

plotter.add_axes()
plotter.show()


import matplotlib.pyplot as plt

# Select representative slices
indices = [0, 10, 15, 20, 25, 30, 40, 50,
           60, 64, 70, 80, 90, 100, 110, 117, 127]

fig, axes = plt.subplots(3, 6, figsize=(15, 9))

for ax, index in zip(axes.flat, indices):
    ax.imshow(volume[index], cmap="gray")
    ax.set_title(f"Slice {index}")
    ax.axis("off")

# Hide unused subplot
for ax in axes.flat[len(indices):]:
    ax.axis("off")

plt.tight_layout()
plt.show()