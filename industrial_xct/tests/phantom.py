"""Analytic cone-beam projections of spheres for a given ASTRA-style vector geometry.
Used only by tests: exact chord lengths, so no discretisation error from a projector."""
import numpy as np


def sphere_projections(vectors, n_rows, n_cols, spheres):
    """spheres: list of (cx, cy, cz, radius, mu). Returns (N, H, W) float32 line integrals."""
    N = vectors.shape[0]
    out = np.zeros((N, n_rows, n_cols), np.float32)
    r = np.arange(n_rows)[:, None, None] - (n_rows - 1) / 2
    c = np.arange(n_cols)[None, :, None] - (n_cols - 1) / 2
    for k in range(N):
        s, d, u, v = vectors[k, 0:3], vectors[k, 3:6], vectors[k, 6:9], vectors[k, 9:12]
        pix = d + c * u + r * v                                   # (H, W, 3)
        pix = pix.reshape(n_rows, n_cols, 3)
        dirv = pix - s
        dirv /= np.linalg.norm(dirv, axis=2, keepdims=True)
        acc = np.zeros((n_rows, n_cols))
        for (cx, cy, cz, rad, mu) in spheres:
            sc = np.array([cx, cy, cz]) - s
            proj = dirv @ sc
            perp2 = np.sum(sc * sc) - proj ** 2
            chord = 2 * np.sqrt(np.clip(rad ** 2 - perp2, 0, None))
            acc += mu * chord * (proj > 0)
        out[k] = acc
    return out
