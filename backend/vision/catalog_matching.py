from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from vision.catalog_projection import catalog_star_magnitudes, project_catalog_stars
from vision.geometry import (
    estimate_affine_transform,
    iter_triangle_indices,
    normalize_points,
    triangle_signature,
)
from vision.models import MatchResult, Star


def candidate_stars(stars: list[Star], limit: int = 30) -> list[Star]:
    return stars[:limit]


def candidate_points(stars: list[Star], limit: int = 30) -> np.ndarray:
    if not stars:
        return np.empty((0, 2), dtype=np.float32)
    return np.array([[star["x"], star["y"]] for star in candidate_stars(stars, limit)], dtype=np.float32)


def candidate_brightness(stars: list[Star], limit: int = 30) -> np.ndarray:
    if not stars:
        return np.empty((0,), dtype=np.float32)
    return np.array([star["brightness"] for star in candidate_stars(stars, limit)], dtype=np.float32)


def magnitude_to_relative_flux(magnitudes: np.ndarray) -> np.ndarray:
    flux = np.power(10.0, -0.4 * magnitudes)
    total = max(float(np.sum(flux)), 1e-6)
    return (flux / total).astype(np.float32)


def normalize_detected_brightness(brightness: np.ndarray) -> np.ndarray:
    clipped = np.clip(brightness.astype(np.float32), 1e-6, None)
    total = max(float(np.sum(clipped)), 1e-6)
    return clipped / total


def unique_nearest_neighbor_matches(
    transformed_catalog_points: np.ndarray,
    detected_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    neighbors = NearestNeighbors(n_neighbors=1)
    neighbors.fit(detected_points)
    distances, indices = neighbors.kneighbors(transformed_catalog_points)

    pairings = sorted(
        (
            float(distance),
            catalog_idx,
            int(detected_idx),
        )
        for catalog_idx, (distance, detected_idx) in enumerate(zip(distances[:, 0], indices[:, 0], strict=False))
    )

    matched_catalog_indices: list[int] = []
    matched_detected_indices: list[int] = []
    used_detected: set[int] = set()

    for _, catalog_idx, detected_idx in pairings:
        if detected_idx in used_detected:
            continue
        matched_catalog_indices.append(catalog_idx)
        matched_detected_indices.append(detected_idx)
        used_detected.add(detected_idx)

    return np.array(matched_catalog_indices, dtype=np.int32), np.array(matched_detected_indices, dtype=np.int32)


def score_geometric_residuals(
    transformed_catalog_points: np.ndarray,
    detected_points: np.ndarray,
    matched_catalog_indices: np.ndarray,
    matched_detected_indices: np.ndarray,
) -> tuple[float, float]:
    if len(matched_catalog_indices) == 0:
        return 0.0, 0.0

    matched_catalog_points = transformed_catalog_points[matched_catalog_indices]
    matched_detected_points = detected_points[matched_detected_indices]
    residuals = np.linalg.norm(matched_catalog_points - matched_detected_points, axis=1)

    field_scale = max(np.linalg.norm(detected_points.std(axis=0)), 1.0)
    max_residual = field_scale * 0.35
    geometric_score = max(0.0, 1.0 - float(np.mean(residuals) / max_residual))
    coverage_score = len(matched_catalog_indices) / max(len(transformed_catalog_points), 1)
    return geometric_score, coverage_score


def score_brightness_consistency(
    catalog_magnitudes: np.ndarray,
    detected_brightness: np.ndarray,
    matched_catalog_indices: np.ndarray,
    matched_detected_indices: np.ndarray,
) -> float:
    if len(matched_catalog_indices) < 2:
        return 0.0

    catalog_flux = magnitude_to_relative_flux(catalog_magnitudes[matched_catalog_indices])
    image_flux = normalize_detected_brightness(detected_brightness[matched_detected_indices])
    difference = np.mean(np.abs(catalog_flux - image_flux))
    return max(0.0, 1.0 - float(difference * 2.5))


def score_catalog_alignment(
    transformed_catalog_points: np.ndarray,
    detected_points: np.ndarray,
    catalog_magnitudes: np.ndarray,
    detected_brightness: np.ndarray,
) -> float:
    matched_catalog_indices, matched_detected_indices = unique_nearest_neighbor_matches(
        transformed_catalog_points,
        detected_points,
    )
    geometric_score, coverage_score = score_geometric_residuals(
        transformed_catalog_points,
        detected_points,
        matched_catalog_indices,
        matched_detected_indices,
    )
    brightness_score = score_brightness_consistency(
        catalog_magnitudes,
        detected_brightness,
        matched_catalog_indices,
        matched_detected_indices,
    )
    return (geometric_score * 0.5) + (coverage_score * 0.3) + (brightness_score * 0.2)


def match_catalog_entry(stars: list[Star], catalog_entry: dict) -> MatchResult | None:
    image_points = candidate_points(stars)
    detected_brightness = candidate_brightness(stars)
    if len(image_points) < 3:
        return None

    catalog_points = project_catalog_stars(catalog_entry)
    catalog_magnitudes = catalog_star_magnitudes(catalog_entry)
    catalog_norm = normalize_points(catalog_points)
    catalog_triangles = iter_triangle_indices(len(catalog_norm))
    image_triangles = iter_triangle_indices(len(image_points))

    best_score = 0.0
    best_transform = None

    for catalog_triangle in catalog_triangles:
        catalog_signature = triangle_signature(catalog_norm, catalog_triangle)
        for image_triangle in image_triangles:
            image_norm = normalize_points(image_points[list(image_triangle)])
            image_signature = triangle_signature(image_norm, (0, 1, 2))
            if np.linalg.norm(catalog_signature - image_signature) > 0.18:
                continue

            model, inliers = estimate_affine_transform(
                catalog_norm,
                catalog_triangle,
                image_points,
                image_triangle,
            )
            if model is None or inliers is None:
                continue

            transformed = model(catalog_norm)
            score = score_catalog_alignment(
                transformed,
                image_points,
                catalog_magnitudes,
                detected_brightness,
            )

            if score > best_score:
                best_score = score
                best_transform = model

    if best_transform is None or best_score < 0.5:
        return None

    transformed_full = best_transform(catalog_norm)
    return MatchResult(
        name=catalog_entry["name"],
        confidence=min(0.99, round(best_score, 4)),
        transformed_points=[tuple(point) for point in transformed_full.tolist()],
        connections=[tuple(connection) for connection in catalog_entry["connections"]],
        color=tuple(catalog_entry.get("color", [0, 255, 0])),
    )
