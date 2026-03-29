from __future__ import annotations

import numpy as np
from skimage.transform import SimilarityTransform
from scipy.optimize import linear_sum_assignment

from vision.catalog_projection import catalog_star_magnitudes, project_catalog_stars
from vision.geometry import (
    estimate_affine_transform,
    iter_triangle_indices,
    normalize_points,
    triangle_signature,
)
from vision.matcher_config import (
    ASPECT_RATIO_LOG_TOLERANCE,
    BRIGHT_STAR_COUNT_TOLERANCE,
    DETECTED_BRIGHTNESS_QUANTILE,
    LARGE_PATTERN_MIN_STARS,
    MEDIUM_PATTERN_MIN_STARS,
    MAX_CANDIDATE_STARS,
    MAX_TRIANGLE_SEED_STARS,
    MIN_COMPATIBILITY_SCORE,
    MIN_LARGE_PATTERN_MATCHED_STARS,
    MIN_MATCH_COVERAGE,
    MIN_LARGE_PATTERN_COVERAGE,
    MIN_MEDIUM_PATTERN_MATCHED_STARS,
    MIN_MATCH_SCORE,
    TRIANGLE_SIGNATURE_DISTANCE_LIMIT,
)
from vision.models import CatalogMatchEvaluation, MatchResult, Star


def candidate_stars(stars: list[Star], limit: int = MAX_CANDIDATE_STARS) -> list[Star]:
    return stars[:limit]


def candidate_points(stars: list[Star], limit: int = MAX_CANDIDATE_STARS) -> np.ndarray:
    if not stars:
        return np.empty((0, 2), dtype=np.float32)
    return np.array([[star["x"], star["y"]] for star in candidate_stars(stars, limit)], dtype=np.float32)


def candidate_brightness(stars: list[Star], limit: int = MAX_CANDIDATE_STARS) -> np.ndarray:
    if not stars:
        return np.empty((0,), dtype=np.float32)
    return np.array([star["brightness"] for star in candidate_stars(stars, limit)], dtype=np.float32)


def triangle_seed_indices_for_catalog(catalog_magnitudes: np.ndarray) -> list[int]:
    if catalog_magnitudes.size <= MAX_TRIANGLE_SEED_STARS:
        return list(range(int(catalog_magnitudes.size)))
    ranked = np.argsort(catalog_magnitudes)
    return [int(index) for index in ranked[:MAX_TRIANGLE_SEED_STARS]]


def triangle_seed_indices_for_stars(stars: list[Star], limit: int = MAX_TRIANGLE_SEED_STARS) -> list[int]:
    return list(range(min(len(stars), limit)))


def required_matched_star_count(pattern_size: int) -> int:
    if pattern_size >= LARGE_PATTERN_MIN_STARS:
        return MIN_LARGE_PATTERN_MATCHED_STARS
    if pattern_size >= MEDIUM_PATTERN_MIN_STARS:
        return MIN_MEDIUM_PATTERN_MATCHED_STARS
    return 3


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
    if len(transformed_catalog_points) == 0 or len(detected_points) == 0:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)

    distance_matrix = np.linalg.norm(
        transformed_catalog_points[:, None, :] - detected_points[None, :, :],
        axis=2,
    )
    catalog_indices, detected_indices = linear_sum_assignment(distance_matrix)

    if len(catalog_indices) == 0:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)

    field_scale = max(np.linalg.norm(detected_points.std(axis=0)), 1.0)
    max_distance = field_scale * 0.4
    kept_pairs = [
        (int(catalog_idx), int(detected_idx))
        for catalog_idx, detected_idx in zip(catalog_indices, detected_indices, strict=False)
        if distance_matrix[catalog_idx, detected_idx] <= max_distance
    ]
    if not kept_pairs:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)

    return (
        np.array([catalog_idx for catalog_idx, _ in kept_pairs], dtype=np.int32),
        np.array([detected_idx for _, detected_idx in kept_pairs], dtype=np.int32),
    )


def pattern_aspect_ratio(points: np.ndarray) -> float:
    if len(points) < 2:
        return 1.0
    centered = points - np.mean(points, axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    major = max(float(singular_values[0]), 1e-6)
    minor = max(float(singular_values[-1]), 1e-6)
    return major / minor


def bright_detected_star_count(detected_brightness: np.ndarray) -> int:
    if detected_brightness.size == 0:
        return 0
    threshold = float(np.quantile(detected_brightness, DETECTED_BRIGHTNESS_QUANTILE))
    return int(np.sum(detected_brightness >= threshold))


def bright_catalog_star_count(catalog_magnitudes: np.ndarray) -> int:
    if catalog_magnitudes.size == 0:
        return 0
    threshold = float(np.quantile(catalog_magnitudes, 1.0 - DETECTED_BRIGHTNESS_QUANTILE))
    return int(np.sum(catalog_magnitudes <= threshold))


def early_reject_catalog_entry(
    image_points: np.ndarray,
    detected_brightness: np.ndarray,
    catalog_points: np.ndarray,
    catalog_magnitudes: np.ndarray,
) -> bool:
    if len(image_points) < 3 or len(catalog_points) < 3:
        return True

    if len(image_points) <= 12 and len(catalog_points) + 2 < len(image_points):
        return True

    detected_bright_count = bright_detected_star_count(detected_brightness)
    catalog_bright_count = bright_catalog_star_count(catalog_magnitudes)
    if abs(detected_bright_count - catalog_bright_count) > BRIGHT_STAR_COUNT_TOLERANCE:
        return True

    image_ratio = pattern_aspect_ratio(normalize_points(image_points))
    catalog_ratio = pattern_aspect_ratio(normalize_points(catalog_points))
    ratio_distance = abs(np.log(max(image_ratio, 1e-6)) - np.log(max(catalog_ratio, 1e-6)))
    return ratio_distance > ASPECT_RATIO_LOG_TOLERANCE


def compatibility_score(
    image_points: np.ndarray,
    detected_brightness: np.ndarray,
    catalog_points: np.ndarray,
    catalog_magnitudes: np.ndarray,
) -> tuple[float, str | None]:
    if len(image_points) < 3 or len(catalog_points) < 3:
        return 0.0, "too_few_points"

    size_penalty = 0.0
    if len(image_points) <= 12 and len(catalog_points) + 2 < len(image_points):
        size_penalty = 0.35

    detected_bright_count = bright_detected_star_count(detected_brightness)
    catalog_bright_count = bright_catalog_star_count(catalog_magnitudes)
    bright_delta = abs(detected_bright_count - catalog_bright_count)
    bright_score = max(0.0, 1.0 - (bright_delta / max(BRIGHT_STAR_COUNT_TOLERANCE + 1, 1)))

    image_ratio = pattern_aspect_ratio(normalize_points(image_points))
    catalog_ratio = pattern_aspect_ratio(normalize_points(catalog_points))
    ratio_distance = abs(np.log(max(image_ratio, 1e-6)) - np.log(max(catalog_ratio, 1e-6)))
    aspect_score = max(0.0, 1.0 - (ratio_distance / max(ASPECT_RATIO_LOG_TOLERANCE, 1e-6)))

    score = max(0.0, (bright_score * 0.45) + (aspect_score * 0.55) - size_penalty)
    rejection_reason = None
    if score < 0.2:
        rejection_reason = "low_compatibility"
    return score, rejection_reason


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
    catalog_coverage = len(matched_catalog_indices) / max(len(transformed_catalog_points), 1)
    relevant_detected_count = min(len(detected_points), len(transformed_catalog_points) + 2)
    detected_coverage = len(matched_detected_indices) / max(relevant_detected_count, 1)
    coverage_score = (catalog_coverage * 0.45) + (detected_coverage * 0.55)
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
) -> tuple[float, float, float, float, int, np.ndarray, np.ndarray]:
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
    score = (geometric_score * 0.3) + (coverage_score * 0.3) + (brightness_score * 0.4)
    return (
        score,
        geometric_score,
        coverage_score,
        brightness_score,
        len(matched_catalog_indices),
        matched_catalog_indices,
        matched_detected_indices,
    )


def evaluate_catalog_entry(
    stars: list[Star],
    catalog_entry: dict,
    *,
    cluster_id: int = 0,
    prepared_catalog: dict | None = None,
) -> CatalogMatchEvaluation | None:
    image_points = candidate_points(stars, limit=MAX_CANDIDATE_STARS)
    detected_brightness = candidate_brightness(stars, limit=MAX_CANDIDATE_STARS)
    if len(image_points) < 3:
        return None

    catalog_points = (
        prepared_catalog["projected_points"] if prepared_catalog is not None else project_catalog_stars(catalog_entry)
    )
    catalog_magnitudes = (
        prepared_catalog["magnitudes"] if prepared_catalog is not None else catalog_star_magnitudes(catalog_entry)
    )
    soft_compatibility, rejection_reason = compatibility_score(
        image_points,
        detected_brightness,
        catalog_points,
        catalog_magnitudes,
    )
    if soft_compatibility < MIN_COMPATIBILITY_SCORE:
        rejection_reason = rejection_reason or "low_compatibility"
        return CatalogMatchEvaluation(
            name=catalog_entry["name"],
            confidence=soft_compatibility,
            accepted=False,
            cluster_id=cluster_id,
            matched_star_count=0,
            geometric_score=0.0,
            coverage_score=0.0,
            brightness_score=0.0,
            compatibility_score=soft_compatibility,
            rejection_reason=rejection_reason,
            transformed_points=[],
            connections=[tuple(connection) for connection in catalog_entry["connections"]],
            color=tuple(catalog_entry.get("color", [255, 255, 255])),
        )
    if early_reject_catalog_entry(image_points, detected_brightness, catalog_points, catalog_magnitudes):
        rejection_reason = rejection_reason or "early_rejection"

    catalog_norm = (
        prepared_catalog["normalized_points"] if prepared_catalog is not None else normalize_points(catalog_points)
    )
    catalog_triangles = (
        prepared_catalog["triangles"]
        if prepared_catalog is not None
        else iter_triangle_indices(len(triangle_seed_indices_for_catalog(catalog_magnitudes)))
    )
    catalog_seed_indices = (
        prepared_catalog["seed_indices"]
        if prepared_catalog is not None
        else triangle_seed_indices_for_catalog(catalog_magnitudes)
    )
    image_seed_indices = triangle_seed_indices_for_stars(candidate_stars(stars, MAX_CANDIDATE_STARS))
    image_triangles = iter_triangle_indices(len(image_seed_indices))

    best_score = 0.0
    best_transform = None
    best_coverage = 0.0
    best_geometric = 0.0
    best_brightness = 0.0
    best_matched_count = 0

    for catalog_triangle in catalog_triangles:
        mapped_catalog_triangle = tuple(catalog_seed_indices[index] for index in catalog_triangle)
        catalog_signature = triangle_signature(catalog_norm, mapped_catalog_triangle)
        for image_triangle in image_triangles:
            mapped_image_triangle = tuple(image_seed_indices[index] for index in image_triangle)
            image_norm = normalize_points(image_points[list(mapped_image_triangle)])
            image_signature = triangle_signature(image_norm, (0, 1, 2))
            if np.linalg.norm(catalog_signature - image_signature) > TRIANGLE_SIGNATURE_DISTANCE_LIMIT:
                continue

            model, inliers = estimate_affine_transform(
                catalog_norm,
                mapped_catalog_triangle,
                image_points,
                mapped_image_triangle,
            )
            if model is None or inliers is None:
                continue

            transformed = model(catalog_norm)
            (
                score,
                geometric_score,
                coverage,
                brightness_score,
                matched_star_count,
                matched_catalog_indices,
                matched_detected_indices,
            ) = score_catalog_alignment(
                transformed,
                image_points,
                catalog_magnitudes,
                detected_brightness,
            )

            if matched_star_count >= 3:
                refined_model = SimilarityTransform.from_estimate(
                    catalog_norm[matched_catalog_indices],
                    image_points[matched_detected_indices],
                )
                if refined_model is not None:
                    refined_transformed = refined_model(catalog_norm)
                    (
                        refined_score,
                        refined_geometric_score,
                        refined_coverage,
                        refined_brightness_score,
                        refined_matched_star_count,
                        _,
                        _,
                    ) = score_catalog_alignment(
                        refined_transformed,
                        image_points,
                        catalog_magnitudes,
                        detected_brightness,
                    )
                    if refined_score >= score:
                        model = refined_model
                        transformed = refined_transformed
                        score = refined_score
                        geometric_score = refined_geometric_score
                        coverage = refined_coverage
                        brightness_score = refined_brightness_score
                        matched_star_count = refined_matched_star_count

            score = (score * 0.85) + (soft_compatibility * 0.15)

            if score > best_score:
                best_score = score
                best_transform = model
                best_coverage = coverage
                best_geometric = geometric_score
                best_brightness = brightness_score
                best_matched_count = matched_star_count

    transformed_full = best_transform(catalog_norm).tolist() if best_transform is not None else []
    accepted = (
        best_transform is not None
        and best_matched_count >= required_matched_star_count(len(catalog_entry["stars"]))
        and best_score >= MIN_MATCH_SCORE
        and best_coverage >= MIN_MATCH_COVERAGE
        and (
            len(catalog_entry["stars"]) < LARGE_PATTERN_MIN_STARS
            or best_coverage >= MIN_LARGE_PATTERN_COVERAGE
        )
    )
    if not accepted and rejection_reason is None:
        if best_transform is None:
            rejection_reason = "no_alignment"
        elif best_matched_count < required_matched_star_count(len(catalog_entry["stars"])):
            rejection_reason = "too_few_matched_stars"
        elif best_score < MIN_MATCH_SCORE:
            rejection_reason = "low_score"
        elif best_coverage < MIN_MATCH_COVERAGE:
            rejection_reason = "low_coverage"
        else:
            rejection_reason = "large_pattern_low_coverage"

    return CatalogMatchEvaluation(
        name=catalog_entry["name"],
        confidence=min(0.99, round(best_score, 4)),
        accepted=accepted,
        cluster_id=cluster_id,
        matched_star_count=best_matched_count,
        geometric_score=round(best_geometric, 4),
        coverage_score=round(best_coverage, 4),
        brightness_score=round(best_brightness, 4),
        compatibility_score=round(soft_compatibility, 4),
        rejection_reason=None if accepted else rejection_reason,
        transformed_points=[tuple(point) for point in transformed_full],
        connections=[tuple(connection) for connection in catalog_entry["connections"]],
        color=tuple(catalog_entry.get("color", [0, 255, 0])),
    )


def match_catalog_entry(stars: list[Star], catalog_entry: dict) -> MatchResult | None:
    evaluation = evaluate_catalog_entry(stars, catalog_entry)
    if evaluation is None or not evaluation.accepted:
        return None

    return MatchResult(
        name=evaluation.name,
        confidence=evaluation.confidence,
        transformed_points=evaluation.transformed_points,
        connections=evaluation.connections,
        color=evaluation.color,
    )
