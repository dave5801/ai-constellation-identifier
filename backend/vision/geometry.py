from __future__ import annotations

from itertools import combinations
from math import dist

import numpy as np
from skimage.transform import AffineTransform


def normalize_points(points: np.ndarray) -> np.ndarray:
    center = np.mean(points, axis=0)
    centered = points - center
    scale = max(np.linalg.norm(centered, axis=1).max(), 1e-6)
    return centered / scale


def triangle_signature(points: np.ndarray, indices: tuple[int, int, int]) -> np.ndarray:
    p0, p1, p2 = points[list(indices)]
    lengths = sorted([dist(p0, p1), dist(p1, p2), dist(p0, p2)])
    baseline = max(lengths[-1], 1e-6)
    return np.array([lengths[0] / baseline, lengths[1] / baseline], dtype=np.float32)


def iter_triangle_indices(point_count: int) -> list[tuple[int, int, int]]:
    return list(combinations(range(point_count), 3))


def estimate_affine_transform(
    template_points: np.ndarray,
    template_triangle: tuple[int, int, int],
    candidate_points: np.ndarray,
    candidate_triangle: tuple[int, int, int],
) -> tuple[AffineTransform | None, np.ndarray | None]:
    src = template_points[list(template_triangle)]
    dst = candidate_points[list(candidate_triangle)]
    model = AffineTransform.from_estimate(src, dst)
    if model is None:
        return None, None

    return model, np.ones(len(src), dtype=bool)


def score_transformed_template(
    transformed_points: np.ndarray,
    candidate_points: np.ndarray,
) -> float:
    errors = np.linalg.norm(
        transformed_points[:, None, :] - candidate_points[None, :, :],
        axis=2,
    )
    nearest = np.min(errors, axis=1)
    distance_scale = max(np.linalg.norm(candidate_points.std(axis=0)), 1.0)
    matched_ratio = float(np.mean(nearest < 18))
    geometry_score = max(0.0, 1.0 - float(np.mean(nearest) / (distance_scale * 2.5)))
    return (matched_ratio * 0.65) + (geometry_score * 0.35)
