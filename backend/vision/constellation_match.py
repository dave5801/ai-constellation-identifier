from __future__ import annotations

import numpy as np
from vision.geometry import (
    estimate_affine_transform,
    iter_triangle_indices,
    normalize_points,
    score_transformed_template,
    triangle_signature,
)
from vision.models import MatchResult, Star


class ConstellationMatcher:
    def __init__(self, templates: list[dict]) -> None:
        self.templates = templates

    def _candidate_points(self, stars: list[Star], limit: int = 30) -> np.ndarray:
        if not stars:
            return np.empty((0, 2), dtype=np.float32)
        points = np.array([[star["x"], star["y"]] for star in stars[:limit]], dtype=np.float32)
        return points

    def _match_template(self, stars: list[Star], template: dict) -> MatchResult | None:
        candidate_points = self._candidate_points(stars)
        if len(candidate_points) < 3:
            return None

        template_points = np.array(template["points"], dtype=np.float32)
        template_norm = normalize_points(template_points)

        template_triangles = iter_triangle_indices(len(template_norm))
        candidate_triangles = iter_triangle_indices(len(candidate_points))

        best_score = 0.0
        best_transform = None

        for template_triangle in template_triangles:
            signature = triangle_signature(template_norm, template_triangle)
            for candidate_triangle in candidate_triangles:
                candidate_norm = normalize_points(candidate_points[list(candidate_triangle)])
                candidate_signature = triangle_signature(candidate_norm, (0, 1, 2))
                if np.linalg.norm(signature - candidate_signature) > 0.18:
                    continue

                model, inliers = estimate_affine_transform(
                    template_norm,
                    template_triangle,
                    candidate_points,
                    candidate_triangle,
                )
                if model is None or inliers is None:
                    continue

                transformed = model(template_norm)
                score = score_transformed_template(transformed, candidate_points)

                if score > best_score:
                    best_score = score
                    best_transform = model

        if best_transform is None or best_score < 0.45:
            return None

        transformed_full = best_transform(template_norm)

        return MatchResult(
            name=template["name"],
            confidence=min(0.99, round(best_score, 4)),
            transformed_points=[tuple(point) for point in transformed_full.tolist()],
            connections=[tuple(connection) for connection in template["connections"]],
            color=tuple(template.get("color", [0, 255, 0])),
        )

    def match(self, stars: list[Star], width: int, height: int) -> list[MatchResult]:
        del width, height

        results = []
        for template in self.templates:
            match = self._match_template(stars, template)
            if match:
                results.append(match)

        results.sort(key=lambda item: item.confidence, reverse=True)
        return results[:3]
