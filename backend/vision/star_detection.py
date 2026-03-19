from __future__ import annotations

import cv2
import numpy as np
from astropy.stats import sigma_clipped_stats

from vision.models import BrightObject, DetectionResult, Star

def build_blob_detector() -> cv2.SimpleBlobDetector:
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 3
    params.maxArea = 250
    params.filterByCircularity = False
    params.filterByColor = True
    params.blobColor = 255
    params.filterByConvexity = False
    params.filterByInertia = False
    params.minThreshold = 10
    params.maxThreshold = 255
    params.thresholdStep = 5
    return cv2.SimpleBlobDetector_create(params)


def estimate_star_brightness(grayscale: np.ndarray, x: float, y: float) -> float:
    ix, iy = int(round(x)), int(round(y))
    x0 = max(ix - 3, 0)
    x1 = min(ix + 4, grayscale.shape[1])
    y0 = max(iy - 3, 0)
    y1 = min(iy + 4, grayscale.shape[0])
    return float(np.mean(grayscale[y0:y1, x0:x1]))


def extract_star_candidates(
    keypoints: list[cv2.KeyPoint],
    grayscale: np.ndarray,
) -> list[Star]:
    stars: list[Star] = []

    for point in keypoints:
        x, y = point.pt
        stars.append(
            {
                "x": float(x),
                "y": float(y),
                "size": float(point.size),
                "brightness": estimate_star_brightness(grayscale, x, y),
            }
        )

    return stars


def find_possible_planets(stars: list[Star]) -> list[BrightObject]:
    if not stars:
        return []

    brightness_values = np.array([star["brightness"] for star in stars], dtype=np.float32)
    _, median_brightness, std_brightness = sigma_clipped_stats(brightness_values, sigma=2.0)
    bright_cutoff = float(median_brightness + 2.0 * std_brightness)
    large_star_cutoff = float(np.percentile([star["size"] for star in stars], 80))

    return [
        star
        for star in stars
        if star["brightness"] >= bright_cutoff and star["size"] >= large_star_cutoff
    ]


def detect_stars(processed: dict[str, np.ndarray]) -> DetectionResult:
    detector = build_blob_detector()
    thresholded = processed["thresholded"]
    grayscale = processed["grayscale"]
    keypoints = detector.detect(thresholded)
    stars = extract_star_candidates(keypoints, grayscale)

    if not stars:
        return DetectionResult(stars=[], possible_planets=[])

    possible_planets = find_possible_planets(stars)
    stars.sort(key=lambda star: star["brightness"], reverse=True)
    return DetectionResult(stars=stars, possible_planets=possible_planets)
