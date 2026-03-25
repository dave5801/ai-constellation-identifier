from __future__ import annotations

from pydantic import BaseModel


class ConstellationResponse(BaseModel):
    name: str
    confidence: float


class IdentifyResponse(BaseModel):
    constellations: list[ConstellationResponse]
    stars_detected: int
    possible_planets: list[dict[str, float]]
    annotated_image: str
