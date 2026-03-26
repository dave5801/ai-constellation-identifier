from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import gaussian_laplace, maximum_filter

from vision.models import BrightObject, Star


@dataclass
class StarCandidate:
    x: float
    y: float
    brightness: float
    peak: float
    contrast: float
    sharpness: float
    spread: float
    response: float
    confidence: float = 0.0


def normalize_grayscale_image(image: np.ndarray) -> np.ndarray:
    image_float = image.astype(np.float32)
    if image_float.max() > 1.0:
        image_float /= 255.0

    min_value = float(np.min(image_float))
    max_value = float(np.max(image_float))
    if max_value - min_value < 1e-6:
        return np.zeros_like(image_float, dtype=np.float32)

    normalized = (image_float - min_value) / (max_value - min_value)
    return np.clip(normalized, 0.0, 1.0)


def preprocess_for_star_detection(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized = normalize_grayscale_image(image)
    denoised = cv2.GaussianBlur(normalized, (0, 0), sigmaX=1.0, sigmaY=1.0)
    log_response = -gaussian_laplace(denoised, sigma=1.0)
    log_response = np.clip(log_response, 0.0, None).astype(np.float32)
    return denoised, log_response


def adaptive_peak_mask(denoised: np.ndarray, response: np.ndarray) -> np.ndarray:
    local_max = maximum_filter(response, size=5, mode="nearest")
    response_mean = float(np.mean(response))
    response_std = float(np.std(response))
    brightness_mean = float(np.mean(denoised))
    brightness_std = float(np.std(denoised))

    response_threshold = response_mean + (1.4 * response_std)
    brightness_threshold = brightness_mean + (0.75 * brightness_std)

    return (
        (response == local_max)
        & (response > response_threshold)
        & (denoised > brightness_threshold)
    )


def extract_candidate_coordinates(mask: np.ndarray, response: np.ndarray, limit: int = 200) -> list[tuple[int, int]]:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return []

    ranked = sorted(
        ((float(response[y, x]), int(y), int(x)) for y, x in coordinates),
        reverse=True,
    )
    return [(y, x) for _, y, x in ranked[:limit]]


def candidate_patch(image: np.ndarray, y: int, x: int, radius: int = 3) -> tuple[np.ndarray, int, int]:
    y0 = max(y - radius, 0)
    y1 = min(y + radius + 1, image.shape[0])
    x0 = max(x - radius, 0)
    x1 = min(x + radius + 1, image.shape[1])
    return image[y0:y1, x0:x1], y0, x0


def patch_spread(weights: np.ndarray) -> float:
    if float(np.sum(weights)) <= 1e-6:
        return 0.0

    y_idx, x_idx = np.indices(weights.shape, dtype=np.float32)
    total = float(np.sum(weights))
    centroid_x = float(np.sum(x_idx * weights) / total)
    centroid_y = float(np.sum(y_idx * weights) / total)
    variance = np.sum(weights * ((x_idx - centroid_x) ** 2 + (y_idx - centroid_y) ** 2)) / total
    return float(np.sqrt(max(variance, 0.0)))


def refine_subpixel_centroid(
    patch: np.ndarray,
    y0: int,
    x0: int,
    local_baseline: float,
) -> tuple[float, float]:
    weights = np.clip(patch - local_baseline, 0.0, None)
    if float(np.sum(weights)) <= 1e-6:
        return float(x0 + (patch.shape[1] - 1) / 2.0), float(y0 + (patch.shape[0] - 1) / 2.0)

    y_idx, x_idx = np.indices(weights.shape, dtype=np.float32)
    total = float(np.sum(weights))
    refined_x = float(x0 + np.sum(x_idx * weights) / total)
    refined_y = float(y0 + np.sum(y_idx * weights) / total)
    return refined_x, refined_y


def build_candidate(
    denoised: np.ndarray,
    response: np.ndarray,
    y: int,
    x: int,
    border_margin: int = 4,
) -> StarCandidate | None:
    if (
        y < border_margin
        or x < border_margin
        or y >= denoised.shape[0] - border_margin
        or x >= denoised.shape[1] - border_margin
    ):
        return None

    patch, y0, x0 = candidate_patch(denoised, y, x, radius=3)
    response_patch, _, _ = candidate_patch(response, y, x, radius=3)
    if patch.shape[0] < 5 or patch.shape[1] < 5:
        return None

    peak = float(denoised[y, x])
    local_mean = float(np.mean(patch))
    local_median = float(np.median(patch))
    contrast = peak - local_mean
    sharpness = peak - local_median
    weights = np.clip(patch - local_mean, 0.0, None)
    spread = patch_spread(weights)
    response_peak = float(response[y, x])
    high_intensity_fraction = float(np.mean(patch > max(local_mean + contrast * 0.6, local_mean + 0.03)))

    if contrast < 0.025:
        return None
    if sharpness < 0.02:
        return None
    if response_peak < float(np.mean(response_patch)) + 0.01:
        return None
    if spread < 0.45 or spread > 2.6:
        return None
    if high_intensity_fraction > 0.8:
        return None

    refined_x, refined_y = refine_subpixel_centroid(patch, y0, x0, local_mean)
    brightness = float(np.mean(np.sort(patch, axis=None)[-3:]))

    return StarCandidate(
        x=refined_x,
        y=refined_y,
        brightness=brightness,
        peak=peak,
        contrast=contrast,
        sharpness=sharpness,
        spread=spread,
        response=response_peak,
    )


def normalize_feature(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lower = float(np.min(values))
    upper = float(np.max(values))
    if upper - lower < 1e-6:
        return np.ones_like(values, dtype=np.float32)
    return ((values - lower) / (upper - lower)).astype(np.float32)


def score_candidates(candidates: list[StarCandidate]) -> list[StarCandidate]:
    if not candidates:
        return []

    brightness = normalize_feature(np.array([candidate.brightness for candidate in candidates], dtype=np.float32))
    contrast = normalize_feature(np.array([candidate.contrast for candidate in candidates], dtype=np.float32))
    sharpness = normalize_feature(np.array([candidate.sharpness for candidate in candidates], dtype=np.float32))
    response = normalize_feature(np.array([candidate.response for candidate in candidates], dtype=np.float32))
    spread = np.array([candidate.spread for candidate in candidates], dtype=np.float32)

    ideal_spread = 1.2
    spread_score = 1.0 - np.clip(np.abs(spread - ideal_spread) / 1.8, 0.0, 1.0)

    raw_scores = (brightness * 0.35) + (contrast * 0.25) + (sharpness * 0.2) + (response * 0.1) + (spread_score * 0.1)
    normalized_scores = normalize_feature(raw_scores)

    for candidate, score in zip(candidates, normalized_scores, strict=False):
        candidate.confidence = float(np.clip(score, 0.0, 1.0))

    return sorted(candidates, key=lambda candidate: candidate.confidence, reverse=True)


def serialize_candidates(candidates: list[StarCandidate]) -> list[dict[str, float]]:
    return [
        {
            "x": candidate.x,
            "y": candidate.y,
            "brightness": candidate.brightness,
            "confidence": candidate.confidence,
        }
        for candidate in candidates
    ]


def detect_stars(
    image: np.ndarray,
    max_stars: int = 40,
    debug: bool = False,
) -> list[dict[str, float]] | dict[str, Any]:
    denoised, response = preprocess_for_star_detection(image)
    peak_mask = adaptive_peak_mask(denoised, response)
    raw_coordinates = extract_candidate_coordinates(peak_mask, response, limit=200)

    raw_candidates = [
        {
            "x": float(x),
            "y": float(y),
            "brightness": float(denoised[y, x]),
            "confidence": 0.0,
        }
        for y, x in raw_coordinates
    ]

    filtered_candidates = [
        candidate
        for coordinate in raw_coordinates
        if (candidate := build_candidate(denoised, response, coordinate[0], coordinate[1])) is not None
    ]
    ranked_candidates = score_candidates(filtered_candidates)[:max_stars]
    stars = serialize_candidates(ranked_candidates)

    if debug:
        return {
            "stars": stars,
            "raw_candidates": raw_candidates,
            "filtered_candidates": serialize_candidates(filtered_candidates),
        }

    return stars


def find_possible_planets(stars: list[Star]) -> list[BrightObject]:
    if not stars:
        return []

    brightness_values = np.array([star["brightness"] for star in stars], dtype=np.float32)
    _, median_brightness, std_brightness = sigma_clipped_stats(brightness_values, sigma=2.0)
    bright_cutoff = float(median_brightness + (2.5 * std_brightness))

    return [
        {
            "x": star["x"],
            "y": star["y"],
            "brightness": star["brightness"],
        }
        for star in stars
        if star["brightness"] >= bright_cutoff and star["confidence"] >= 0.45
    ]
