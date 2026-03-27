from __future__ import annotations

from dataclasses import dataclass


Star = dict[str, float]
BrightObject = dict[str, float]


@dataclass
class DetectionResult:
    stars: list[Star]
    possible_planets: list[BrightObject]


@dataclass
class MatchResult:
    name: str
    confidence: float
    transformed_points: list[tuple[float, float]]
    connections: list[tuple[int, int]]
    color: tuple[int, int, int]


@dataclass
class CatalogMatchEvaluation:
    name: str
    confidence: float
    accepted: bool
    cluster_id: int
    matched_star_count: int
    geometric_score: float
    coverage_score: float
    brightness_score: float
    compatibility_score: float
    rejection_reason: str | None
    transformed_points: list[tuple[float, float]]
    connections: list[tuple[int, int]]
    color: tuple[int, int, int]
