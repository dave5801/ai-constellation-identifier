from __future__ import annotations

import base64

import cv2
import numpy as np
from fastapi import HTTPException

from vision.models import DetectionResult, MatchResult


def encode_image(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image.")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def draw_constellation_outline(
    image: np.ndarray,
    match: MatchResult,
    color: tuple[int, int, int],
) -> None:
    points = np.array(match.transformed_points, dtype=np.float32)
    if len(points) < 3:
        return

    hull = cv2.convexHull(points.astype(np.int32))
    cv2.polylines(image, [hull], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

    x, y, width, height = cv2.boundingRect(hull)
    cv2.rectangle(
        image,
        (x, y),
        (x + width, y + height),
        color,
        1,
        cv2.LINE_AA,
    )


def draw_detected_stars(image: np.ndarray, detection: DetectionResult) -> None:
    for star in detection.stars:
        center = (int(round(star["x"])), int(round(star["y"])))
        radius = max(3, int(round(star["size"] / 2)))
        cv2.circle(image, center, radius, (0, 255, 255), 1)


def draw_possible_planets(image: np.ndarray, detection: DetectionResult) -> None:
    for planet in detection.possible_planets:
        center = (int(round(planet["x"])), int(round(planet["y"])))
        cv2.circle(image, center, 10, (255, 140, 0), 2)
        cv2.putText(
            image,
            "Possible planet",
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 140, 0),
            1,
            cv2.LINE_AA,
        )


def draw_constellation_matches(image: np.ndarray, matches: list[MatchResult]) -> None:
    for match in matches:
        color = tuple(int(channel) for channel in match.color)
        draw_constellation_outline(image, match, color)

        for start_idx, end_idx in match.connections:
            start = tuple(int(round(value)) for value in match.transformed_points[start_idx])
            end = tuple(int(round(value)) for value in match.transformed_points[end_idx])
            cv2.line(image, start, end, color, 2, cv2.LINE_AA)

        anchor = tuple(int(round(value)) for value in match.transformed_points[0])
        label = f"{match.name} ({match.confidence:.2f})"
        cv2.putText(
            image,
            label,
            (anchor[0] + 6, anchor[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_annotations(
    image: np.ndarray,
    detection: DetectionResult,
    matches: list[MatchResult],
) -> np.ndarray:
    annotated = image.copy()
    draw_detected_stars(annotated, detection)
    draw_possible_planets(annotated, detection)
    draw_constellation_matches(annotated, matches)
    return annotated
