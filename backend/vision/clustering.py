from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from vision.matcher_config import MAX_CLUSTER_COUNT, MAX_CLUSTER_STARS
from vision.models import Star


def star_points(stars: list[Star]) -> np.ndarray:
    if not stars:
        return np.empty((0, 2), dtype=np.float32)
    return np.array([[star["x"], star["y"]] for star in stars], dtype=np.float32)


def adaptive_cluster_radius(points: np.ndarray, width: int, height: int) -> float:
    if len(points) < 3:
        return max(min(width, height) * 0.08, 24.0)

    neighbor_count = min(4, len(points))
    neighbors = NearestNeighbors(n_neighbors=neighbor_count)
    neighbors.fit(points)
    distances, _ = neighbors.kneighbors(points)
    reference_distance = float(np.median(distances[:, -1]))
    lower_bound = max(min(width, height) * 0.03, 18.0)
    upper_bound = max(min(width, height) * 0.18, lower_bound)
    return float(np.clip(reference_distance * 1.8, lower_bound, upper_bound))


def cluster_compactness(stars: list[Star]) -> float:
    if len(stars) < 2:
        return 0.0
    points = star_points(stars)
    centered = points - np.mean(points, axis=0, keepdims=True)
    radius = float(np.mean(np.linalg.norm(centered, axis=1)))
    return 1.0 / max(radius, 1.0)


def cluster_rank(stars: list[Star]) -> tuple[float, float, float]:
    mean_confidence = float(np.mean([star.get("confidence", 0.0) for star in stars])) if stars else 0.0
    mean_brightness = float(np.mean([star["brightness"] for star in stars])) if stars else 0.0
    return mean_confidence, cluster_compactness(stars), mean_brightness


def cluster_star_fields(stars: list[Star], width: int, height: int) -> list[list[Star]]:
    if len(stars) < 3:
        return [stars] if stars else []

    points = star_points(stars)
    eps = adaptive_cluster_radius(points, width, height)
    labels = DBSCAN(eps=eps, min_samples=3).fit_predict(points)

    clusters: list[list[Star]] = []
    for label in sorted(set(labels)):
        if label == -1:
            continue
        cluster = [star for star, assigned in zip(stars, labels, strict=False) if assigned == label]
        if len(cluster) >= 3:
            ranked_cluster = sorted(cluster, key=lambda star: star.get("confidence", 0.0), reverse=True)
            clusters.append(ranked_cluster[:MAX_CLUSTER_STARS])

    if not clusters:
        return [stars[: min(len(stars), 12)]]

    clusters.sort(key=cluster_rank, reverse=True)

    unique_clusters: list[list[Star]] = []
    seen_signatures: set[tuple[tuple[int, int], ...]] = set()
    for cluster in clusters:
        signature = tuple(sorted((int(round(star["x"])), int(round(star["y"]))) for star in cluster))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        unique_clusters.append(cluster)

    return unique_clusters[:MAX_CLUSTER_COUNT]
