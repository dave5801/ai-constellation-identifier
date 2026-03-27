from __future__ import annotations

from pydantic import BaseModel


class ConstellationResponse(BaseModel):
    name: str
    confidence: float


class CatalogScoreResponse(BaseModel):
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


class DebugInfoResponse(BaseModel):
    detected_stars: list[dict[str, float]]
    raw_candidates: list[dict[str, float]]
    filtered_candidates: list[dict[str, float]]
    catalog_scores: list[CatalogScoreResponse]


class IdentifyResponse(BaseModel):
    constellations: list[ConstellationResponse]
    stars_detected: int
    possible_planets: list[dict[str, float]]
    annotated_image: str
    debug: DebugInfoResponse | None = None
