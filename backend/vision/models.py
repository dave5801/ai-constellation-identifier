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
