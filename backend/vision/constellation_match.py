from __future__ import annotations

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from vision.geometry import (
    estimate_affine_transform,
    iter_triangle_indices,
    normalize_points,
    score_transformed_template,
    triangle_signature,
)
from vision.models import MatchResult, Star


class ConstellationMatcher:
    def __init__(self, catalog: list[dict]) -> None:
        self.catalog = catalog

    def _candidate_points(self, stars: list[Star], limit: int = 30) -> np.ndarray:
        if not stars:
            return np.empty((0, 2), dtype=np.float32)
        points = np.array([[star["x"], star["y"]] for star in stars[:limit]], dtype=np.float32)
        return points

    def _catalog_points(self, catalog_entry: dict) -> np.ndarray:
        sky_coords = SkyCoord(
            ra=[star["ra_deg"] for star in catalog_entry["stars"]] * u.deg,
            dec=[star["dec_deg"] for star in catalog_entry["stars"]] * u.deg,
            frame="icrs",
        )
        wrapped_ra = sky_coords.ra.wrap_at(180 * u.deg)
        center_ra = np.mean(wrapped_ra.deg)
        center_dec = np.mean(sky_coords.dec.deg)
        delta_ra = wrapped_ra.deg - center_ra
        cos_dec = np.cos(np.deg2rad(center_dec))
        x = delta_ra * cos_dec
        y = sky_coords.dec.deg - center_dec
        return np.column_stack([x, y]).astype(np.float32)

    def _match_catalog_entry(self, stars: list[Star], catalog_entry: dict) -> MatchResult | None:
        candidate_points = self._candidate_points(stars)
        if len(candidate_points) < 3:
            return None

        catalog_points = self._catalog_points(catalog_entry)
        catalog_norm = normalize_points(catalog_points)

        template_triangles = iter_triangle_indices(len(catalog_norm))
        candidate_triangles = iter_triangle_indices(len(candidate_points))

        best_score = 0.0
        best_transform = None

        for template_triangle in template_triangles:
            signature = triangle_signature(catalog_norm, template_triangle)
            for candidate_triangle in candidate_triangles:
                candidate_norm = normalize_points(candidate_points[list(candidate_triangle)])
                candidate_signature = triangle_signature(candidate_norm, (0, 1, 2))
                if np.linalg.norm(signature - candidate_signature) > 0.18:
                    continue

                model, inliers = estimate_affine_transform(
                    catalog_norm,
                    template_triangle,
                    candidate_points,
                    candidate_triangle,
                )
                if model is None or inliers is None:
                    continue

                transformed = model(catalog_norm)
                score = score_transformed_template(transformed, candidate_points)

                if score > best_score:
                    best_score = score
                    best_transform = model

        if best_transform is None or best_score < 0.45:
            return None

        transformed_full = best_transform(catalog_norm)

        return MatchResult(
            name=catalog_entry["name"],
            confidence=min(0.99, round(best_score, 4)),
            transformed_points=[tuple(point) for point in transformed_full.tolist()],
            connections=[tuple(connection) for connection in catalog_entry["connections"]],
            color=tuple(catalog_entry.get("color", [0, 255, 0])),
        )

    def match(self, stars: list[Star], width: int, height: int) -> list[MatchResult]:
        del width, height

        results = []
        for catalog_entry in self.catalog:
            match = self._match_catalog_entry(stars, catalog_entry)
            if match:
                results.append(match)

        results.sort(key=lambda item: item.confidence, reverse=True)
        return results[:3]
